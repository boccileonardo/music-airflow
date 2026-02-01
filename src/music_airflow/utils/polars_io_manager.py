"""
IO Manager for handling Polars DataFrame/LazyFrame and JSON read/write operations.

This module provides centralized interfaces for Parquet, Delta, and JSON I/O operations.
Supports both DataFrame and LazyFrame with automatic collection when needed.
Supports both local storage and Google Cloud Storage (GCS) for Delta tables.

GCS Configuration:
    Set the GCS_BUCKET_URI in .env file to enable cloud storage.
    Example: GCS_BUCKET_URI=gs://my-bucket/data

    Authentication uses Application Default Credentials (ADC):
    - Locally: Run `gcloud auth application-default login`
    - On GCP: Automatically uses service account credentials

    Alternatively, set GOOGLE_APPLICATION_CREDENTIALS to path of a service account JSON.
"""

import os
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
import polars as pl

load_dotenv()


def get_gcs_storage_options() -> dict[str, str] | None:
    """
    Get GCS storage options for delta-rs.

    Supports:
    - Application Default Credentials (GOOGLE_APPLICATION_CREDENTIALS env var)
    - No explicit config if running on GCP (uses instance credentials)
    """
    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if creds_path:
        return {"google_service_account": creds_path}
    # Let delta-rs auto-detect ADC (works with `gcloud auth application-default login`)
    return None


class JSONIOManager:
    """
    Manager for reading and writing JSON files.

    Uses Polars native JSON I/O for efficient reading.
    Supports both local storage and Google Cloud Storage (GCS).

    Methods:
        write_json: Write list/dict data to JSON format.
        read_json: Read JSON file using Polars read_json (returns LazyFrame).
    """

    def __init__(self, medallion_layer: str = "bronze"):
        """
        Initialize JSON IO Manager.

        Args:
            medallion_layer: Medallion architecture layer (bronze, silver, or gold)
        """
        if medallion_layer not in ["bronze", "silver", "gold"]:
            raise ValueError(
                f"medallion_layer must be 'bronze', 'silver', or 'gold', got '{medallion_layer}'"
            )
        self.medallion_layer = medallion_layer

        # Check for GCS configuration
        gcs_bucket_uri = os.getenv("GCS_BUCKET_URI")
        if gcs_bucket_uri:
            self.base_uri = f"{gcs_bucket_uri.rstrip('/')}/{medallion_layer}"
            self.is_cloud = True
        else:
            self.base_uri = str(
                Path(__file__).parent.parent.parent.parent / "data" / medallion_layer
            )
            self.is_cloud = False

    @property
    def base_dir(self) -> Path:
        """Return base directory as Path (for local storage only)."""
        if self.is_cloud:
            raise ValueError("base_dir not available for cloud storage. Use base_uri.")
        return Path(self.base_uri)

    @base_dir.setter
    def base_dir(self, value: Path) -> None:
        """Set base directory (for testing)."""
        self.base_uri = str(value)
        self.is_cloud = False

    def _get_file_uri(self, filename: str) -> str:
        """Get full URI for a file."""
        if self.is_cloud:
            return f"{self.base_uri}/{filename}"
        return str(Path(self.base_uri) / filename)

    def write_json(self, data: list | dict, filename: str, **kwargs) -> dict[str, Any]:
        """
        Write data to JSON format.

        Args:
            data: List or dict to write
            filename: Output filename (relative to medallion layer directory)
            **kwargs: Additional arguments passed to json.dump()

        Returns:
            Metadata dict with path and record count
        """
        import json

        file_uri = self._get_file_uri(filename)

        if self.is_cloud:
            import gcsfs

            fs = gcsfs.GCSFileSystem()
            with fs.open(file_uri, "w") as f:
                json.dump(data, f, **kwargs)
        else:
            path = Path(file_uri)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                json.dump(data, f, **kwargs)

        # Calculate record count
        if isinstance(data, list):
            record_count = len(data)
        elif isinstance(data, dict):
            record_count = 1
        else:
            record_count = 0

        return {
            "path": file_uri,
            "filename": filename,
            "rows": record_count,
            "format": "json",
            "medallion_layer": self.medallion_layer,
            "storage": "gcs" if self.is_cloud else "local",
        }

    def read_json(self, filename: str, **kwargs) -> pl.LazyFrame:
        """
        Read JSON file using Polars native read_json.

        Args:
            filename: Input filename (relative to medallion layer directory)
            **kwargs: Additional arguments passed to pl.read_json()

        Returns:
            Polars LazyFrame
        """
        file_uri = self._get_file_uri(filename)

        if self.is_cloud:
            import gcsfs

            fs = gcsfs.GCSFileSystem()
            with fs.open(file_uri, "r") as f:
                import json

                data = json.load(f)
            return pl.DataFrame(data).lazy()
        else:
            return pl.read_json(file_uri, **kwargs).lazy()


class PolarsParquetIOManager:
    """
    Manager for reading and writing Polars DataFrames/LazyFrames to Parquet format.

    Automatically handles LazyFrame collection when needed.
    Methods:
        write_parquet: Write DataFrame or LazyFrame to Parquet format.
        read_parquet: Read DataFrame or LazyFrame from Parquet format.
    """

    def __init__(self, medallion_layer: str = "bronze"):
        """
        Initialize Parquet IO Manager.

        Args:
            medallion_layer: Medallion architecture layer (bronze, silver, or gold)
        """
        if medallion_layer not in ["bronze", "silver", "gold"]:
            raise ValueError(
                f"medallion_layer must be 'bronze', 'silver', or 'gold', got '{medallion_layer}'"
            )
        self.medallion_layer = medallion_layer
        self.base_dir = (
            Path(__file__).parent.parent.parent.parent / "data" / medallion_layer
        )

    def write_parquet(
        self, df: pl.DataFrame | pl.LazyFrame, filename: str, **kwargs
    ) -> dict:
        """
        Write DataFrame or LazyFrame to Parquet format.

        Args:
            df: Polars DataFrame or LazyFrame to write
            filename: Output filename (relative to medallion layer directory)
            **kwargs: Additional arguments passed to write_parquet() or sink_parquet()

        Returns:
            Metadata dict with path, rows, and schema
        """
        path = self.base_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)

        def collect_metadata(df: pl.LazyFrame | pl.DataFrame) -> tuple[int, dict]:
            """Helper to collect row count and schema from LazyFrame."""
            if isinstance(df, pl.DataFrame):
                return df.select(pl.len()).item(), df.schema
            try:
                rows = df.select(pl.len()).collect(engine="streaming").item()
            except Exception:
                rows = df.select(pl.len()).collect().item()
            schema = df.collect_schema()
            return rows, schema

        # Collect LazyFrame if needed using streaming engine with fallback
        if isinstance(df, pl.LazyFrame):
            try:
                df.sink_parquet(path, **kwargs)
                rows, schema = collect_metadata(df)
            except Exception:
                try:
                    df = df.collect(engine="streaming")
                except Exception:
                    df = df.collect()
                df.write_parquet(path, **kwargs)
                rows, schema = collect_metadata(df)
        else:
            df.write_parquet(path, **kwargs)
            rows, schema = collect_metadata(df)

        return {
            "path": str(path.absolute()),
            "filename": filename,
            "rows": rows,
            "schema": {name: str(dtype) for name, dtype in schema.items()},
            "format": "parquet",
            "medallion_layer": self.medallion_layer,
        }

    def read_parquet(self, filename: str, **kwargs) -> pl.LazyFrame:
        """
        Read DataFrame or LazyFrame from Parquet format.

        Args:
            filename: Input filename (relative to medallion layer directory)
            lazy: If True (default), return LazyFrame. If False, return DataFrame.
            **kwargs: Additional arguments passed to pl.scan_parquet() or pl.read_parquet()

        Returns:
            Polars LazyFrame (default) or DataFrame
        """
        path = self.base_dir / filename
        return pl.scan_parquet(path, **kwargs)


class PolarsDeltaIOManager:
    """
    Manager for reading and writing Polars DataFrames to Delta Lake format.

    Automatically handles LazyFrame collection (Delta Lake requires materialized DataFrames).
    Supports merge operations for upserts with partition key optimization.
    Supports both local storage and Google Cloud Storage (GCS).

    Storage Configuration:
        - Local: Default, stores in ./data/{medallion_layer}/
        - GCS: Set GCS_BUCKET_URI env var (e.g., gs://my-bucket/data)

    Methods:
        write_delta: Write DataFrame or LazyFrame to Delta Lake with merge/upsert support.
        read_delta: Read DataFrame or LazyFrame from Delta Lake.
    """

    def __init__(self, medallion_layer: str = "silver"):
        """
        Initialize Delta IO Manager.

        Args:
            medallion_layer: Medallion architecture layer (bronze, silver, or gold)
        """
        if medallion_layer not in ["bronze", "silver", "gold"]:
            raise ValueError(
                f"medallion_layer must be 'bronze', 'silver', or 'gold', got '{medallion_layer}'"
            )
        self.medallion_layer = medallion_layer

        # Check for GCS configuration
        gcs_bucket_uri = os.getenv("GCS_BUCKET_URI")
        if gcs_bucket_uri:
            # GCS storage: gs://bucket-name/path -> gs://bucket-name/path/{layer}
            self.base_uri = f"{gcs_bucket_uri.rstrip('/')}/{medallion_layer}"
            self.is_cloud = True
            self.storage_options = get_gcs_storage_options()
        else:
            # Local storage fallback
            self.base_uri = str(
                Path(__file__).parent.parent.parent.parent / "data" / medallion_layer
            )
            self.is_cloud = False
            self.storage_options = None

    def _get_table_uri(self, table_name: str) -> str:
        """Get the full URI for a table."""
        if self.is_cloud:
            return f"{self.base_uri}/{table_name}"
        return str(Path(self.base_uri) / table_name)

    def write_delta(
        self,
        df: pl.DataFrame | pl.LazyFrame,
        table_name: str,
        mode: Literal["merge", "append", "overwrite", "error"] = "merge",
        predicate: str | None = None,
        partition_by: str | list[str] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Write DataFrame or LazyFrame to Delta Lake format with merge/upsert support.

        Args:
            df: Polars DataFrame or LazyFrame to write
            table_name: Table name (relative to medallion layer directory)
            mode: Write mode - 'merge' for upsert, 'append', 'overwrite', or 'error'
            predicate: Merge predicate for upsert (e.g., "s.id = t.id AND s.username = t.username")
                      Required when mode='merge'. Use 's' for source and 't' for target alias.
            partition_by: Column name(s) to partition by. Passed to delta_write_options.
            **kwargs: Additional arguments passed to write_delta()

        Returns:
            Metadata dict with path, rows, schema, and format information
        """
        # Collect LazyFrame to DataFrame (Delta Lake doesn't support sink operations)
        if isinstance(df, pl.LazyFrame):
            try:
                df = df.collect(engine="streaming")
            except Exception:
                df = df.collect()

        table_uri = self._get_table_uri(table_name)

        # For local storage, ensure directory exists
        if not self.is_cloud:
            Path(table_uri).mkdir(parents=True, exist_ok=True)

        # Prepare delta_write_options
        delta_write_options = kwargs.pop("delta_write_options", {})
        if partition_by:
            delta_write_options["partition_by"] = partition_by

        # Check if table exists
        table_exists = self._table_exists_at_uri(table_uri)
        merge_metrics = None

        if mode == "merge":
            if not predicate:
                raise ValueError("predicate is required when mode='merge'")

            # If table doesn't exist, create it with initial write
            if not table_exists:
                df.write_delta(
                    table_uri,
                    mode="overwrite",
                    delta_write_options=delta_write_options
                    if delta_write_options
                    else None,
                    storage_options=self.storage_options,
                    **kwargs,
                )
            else:
                # Setup merge options
                delta_merge_options = {
                    "predicate": predicate,
                    "source_alias": "s",
                    "target_alias": "t",
                }

                # Perform merge operation and capture metrics
                merge_metrics = (
                    df.write_delta(
                        table_uri,
                        mode="merge",
                        delta_merge_options=delta_merge_options,
                        storage_options=self.storage_options,
                        **kwargs,
                    )
                    .when_matched_update_all()
                    .when_not_matched_insert_all()
                    .execute()
                )
        else:
            # Direct write for append/overwrite/error modes
            df.write_delta(
                table_uri,
                mode=mode,
                delta_write_options=delta_write_options
                if delta_write_options
                else None,
                storage_options=self.storage_options,
                **kwargs,
            )

        # Collect metadata
        rows = len(df)
        schema = {name: str(dtype) for name, dtype in df.schema.items()}

        result = {
            "path": table_uri,
            "table_name": table_name,
            "rows": rows,
            "schema": schema,
            "format": "delta",
            "medallion_layer": self.medallion_layer,
            "mode": mode,
            "storage": "gcs" if self.is_cloud else "local",
        }

        # Add merge metrics if available
        if merge_metrics:
            result["merge_metrics"] = merge_metrics

        return result

    def _table_exists_at_uri(self, table_uri: str) -> bool:
        """Check if a Delta table exists at the given URI."""
        if self.is_cloud:
            # For GCS, try to read table metadata
            try:
                from deltalake import DeltaTable

                DeltaTable(table_uri, storage_options=self.storage_options)
                return True
            except Exception:
                return False
        else:
            # Local path check
            path = Path(table_uri)
            return path.exists() and (path / "_delta_log").exists()

    def table_exists(self, table_name: str) -> bool:
        """
        Check if a Delta table exists.

        Args:
            table_name: Table name (relative to medallion layer directory)

        Returns:
            True if table exists, False otherwise
        """
        table_uri = self._get_table_uri(table_name)
        return self._table_exists_at_uri(table_uri)

    def read_delta(self, table_name: str, **kwargs) -> pl.LazyFrame:
        """
        Read DataFrame or LazyFrame from Delta Lake.

        Args:
            table_name: Table name (relative to medallion layer directory)
            **kwargs: Additional arguments passed to pl.scan_delta()

        Returns:
            Polars LazyFrame
        """
        table_uri = self._get_table_uri(table_name)
        return pl.scan_delta(table_uri, storage_options=self.storage_options, **kwargs)
