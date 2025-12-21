"""
IO Manager for handling Polars DataFrame/LazyFrame read/write operations.

This module provides a centralized interface for Parquet I/O operations.
Supports both DataFrame and LazyFrame with automatic collection when needed.
"""

from pathlib import Path
import polars as pl


class PolarsParquetIOManager:
    """
    Manager for reading and writing Polars DataFrames/LazyFrames to Parquet format.

    Automatically handles LazyFrame collection when needed.
    Methods:
        write_parquet: Write DataFrame or LazyFrame to Parquet format.
        read_parquet: Read DataFrame or LazyFrame from Parquet format.
    """

    @staticmethod
    def write_parquet(
        df: pl.DataFrame | pl.LazyFrame, path: str | Path, **kwargs
    ) -> dict:
        """
        Write DataFrame or LazyFrame to Parquet format.

        Args:
            df: Polars DataFrame or LazyFrame to write
            path: Output file path
            **kwargs: Additional arguments passed to write_parquet() or sink_parquet()

        Returns:
            Metadata dict with path, rows, and schema
        """
        path = Path(path)
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
        }

    @staticmethod
    def read_parquet(
        path: str | Path, lazy: bool = True, **kwargs
    ) -> pl.DataFrame | pl.LazyFrame:
        """
        Read DataFrame or LazyFrame from Parquet format.

        Args:
            path: Input file path
            lazy: If True (default), return LazyFrame. If False, return DataFrame.
            **kwargs: Additional arguments passed to pl.scan_parquet() or pl.read_parquet()

        Returns:
            Polars LazyFrame (default) or DataFrame
        """
        if lazy:
            return pl.scan_parquet(path, **kwargs)
        return pl.read_parquet(path, **kwargs)
