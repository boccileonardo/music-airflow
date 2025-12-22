"""
IO Manager for handling Polars DataFrame/LazyFrame and JSON read/write operations.

This module provides centralized interfaces for Parquet and JSON I/O operations.
Supports both DataFrame and LazyFrame with automatic collection when needed.
"""

from pathlib import Path
from typing import Any
import polars as pl


class JSONIOManager:
    """
    Manager for reading and writing JSON files.

    Uses Polars native JSON I/O for efficient reading.
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
        self.base_dir = (
            Path(__file__).parent.parent.parent.parent / "data" / medallion_layer
        )

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

        path = self.base_dir / filename
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
            "path": str(path.absolute()),
            "rows": record_count,
            "format": "json",
            "medallion_layer": self.medallion_layer,
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
        path = self.base_dir / filename
        return pl.read_json(path, **kwargs).lazy()


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
            "rows": rows,
            "schema": {name: str(dtype) for name, dtype in schema.items()},
            "format": "parquet",
            "medallion_layer": self.medallion_layer,
        }

    def read_parquet(
        self, filename: str, lazy: bool = True, **kwargs
    ) -> pl.DataFrame | pl.LazyFrame:
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
        if lazy:
            return pl.scan_parquet(path, **kwargs)
        return pl.read_parquet(path, **kwargs)
