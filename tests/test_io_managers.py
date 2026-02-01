"""
Tests for Polars IO Managers.

Tests JSONIOManager, PolarsParquetIOManager, and PolarsDeltaIOManager
for reading, writing, and merging data.
"""

import os
from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest

from music_airflow.utils.polars_io_manager import (
    JSONIOManager,
    PolarsParquetIOManager,
    PolarsDeltaIOManager,
    get_gcs_storage_options,
)


@pytest.fixture
def test_data_dir(tmp_path: Path) -> Path:
    """Create a temporary data directory for testing."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "bronze").mkdir()
    (data_dir / "silver").mkdir()
    (data_dir / "gold").mkdir()
    return data_dir


@pytest.fixture(autouse=True)
def clear_gcs_env(monkeypatch):
    """Clear GCS environment variables for all tests."""
    monkeypatch.delenv("GCS_BUCKET_URI", raising=False)


class TestJSONIOManager:
    """Test JSON IO Manager."""

    def test_initialization_valid_layers(self):
        """Test initialization with valid medallion layers."""
        for layer in ["bronze", "silver", "gold"]:
            manager = JSONIOManager(medallion_layer=layer)
            assert manager.medallion_layer == layer

    def test_initialization_invalid_layer(self):
        """Test initialization with invalid medallion layer."""
        with pytest.raises(ValueError, match="medallion_layer must be"):
            JSONIOManager(medallion_layer="invalid")

    def test_write_json_list(self, test_data_dir):
        """Test writing a list to JSON."""
        manager = JSONIOManager(medallion_layer="bronze")
        manager.base_dir = test_data_dir / "bronze"

        data = [{"name": "track1"}, {"name": "track2"}]
        result = manager.write_json(data, "test/data.json")

        assert result["rows"] == 2
        assert result["format"] == "json"
        assert result["medallion_layer"] == "bronze"
        assert "test/data.json" in result["filename"]

        # Verify file exists
        assert Path(result["path"]).exists()

    def test_write_json_dict(self, test_data_dir):
        """Test writing a dict to JSON."""
        manager = JSONIOManager(medallion_layer="bronze")
        manager.base_dir = test_data_dir / "bronze"

        data = {"key": "value"}
        result = manager.write_json(data, "test/data.json")

        assert result["rows"] == 1
        assert result["format"] == "json"

    def test_write_json_creates_dirs(self, test_data_dir):
        """Test that write_json creates parent directories."""
        manager = JSONIOManager(medallion_layer="bronze")
        manager.base_dir = test_data_dir / "bronze"

        data = [{"test": "data"}]
        result = manager.write_json(data, "deeply/nested/path/data.json")

        assert Path(result["path"]).exists()
        assert Path(result["path"]).parent.name == "path"

    def test_read_json(self, test_data_dir):
        """Test reading JSON file."""
        manager = JSONIOManager(medallion_layer="bronze")
        manager.base_dir = test_data_dir / "bronze"

        # Write data first
        data = [
            {"name": "track1", "artist": "artist1"},
            {"name": "track2", "artist": "artist2"},
        ]
        manager.write_json(data, "test/tracks.json")

        # Read it back
        df = manager.read_json("test/tracks.json")

        assert isinstance(df, pl.LazyFrame)
        collected = df.collect()
        assert len(collected) == 2
        assert collected["name"].to_list() == ["track1", "track2"]


class TestPolarsParquetIOManager:
    """Test Parquet IO Manager."""

    def test_initialization_valid_layers(self):
        """Test initialization with valid medallion layers."""
        for layer in ["bronze", "silver", "gold"]:
            manager = PolarsParquetIOManager(medallion_layer=layer)
            assert manager.medallion_layer == layer

    def test_initialization_invalid_layer(self):
        """Test initialization with invalid medallion layer."""
        with pytest.raises(ValueError, match="medallion_layer must be"):
            PolarsParquetIOManager(medallion_layer="invalid")

    def test_write_parquet_dataframe(self, test_data_dir):
        """Test writing a DataFrame to Parquet."""
        manager = PolarsParquetIOManager(medallion_layer="silver")
        manager.base_dir = test_data_dir / "silver"

        df = pl.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
        result = manager.write_parquet(df, "test/data.parquet")

        assert result["rows"] == 3
        assert result["format"] == "parquet"
        assert result["medallion_layer"] == "silver"
        assert "col1" in result["schema"]
        assert "col2" in result["schema"]

        # Verify file exists
        assert Path(result["path"]).exists()

    def test_write_parquet_lazyframe(self, test_data_dir):
        """Test writing a LazyFrame to Parquet."""
        manager = PolarsParquetIOManager(medallion_layer="silver")
        manager.base_dir = test_data_dir / "silver"

        df = pl.LazyFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
        result = manager.write_parquet(df, "test/data.parquet")

        assert result["rows"] == 3
        assert result["format"] == "parquet"

        # Verify file exists
        assert Path(result["path"]).exists()

    def test_read_parquet(self, test_data_dir):
        """Test reading Parquet file."""
        manager = PolarsParquetIOManager(medallion_layer="silver")
        manager.base_dir = test_data_dir / "silver"

        # Write data first
        df = pl.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
        manager.write_parquet(df, "test/data.parquet")

        # Read it back
        result = manager.read_parquet("test/data.parquet")

        assert isinstance(result, pl.LazyFrame)
        collected = result.collect()
        assert len(collected) == 3
        assert collected["col1"].to_list() == [1, 2, 3]


class TestPolarsDeltaIOManager:
    """Test Delta Lake IO Manager."""

    def test_initialization_valid_layers(self):
        """Test initialization with valid medallion layers."""
        for layer in ["bronze", "silver", "gold"]:
            manager = PolarsDeltaIOManager(medallion_layer=layer)
            assert manager.medallion_layer == layer

    def test_initialization_invalid_layer(self):
        """Test initialization with invalid medallion layer."""
        with pytest.raises(ValueError, match="medallion_layer must be"):
            PolarsDeltaIOManager(medallion_layer="invalid")

    def test_write_delta_overwrite(self, test_data_dir):
        """Test writing a DataFrame to Delta with overwrite mode."""
        manager = PolarsDeltaIOManager(medallion_layer="silver")
        manager.base_uri = str(test_data_dir / "silver")

        df = pl.DataFrame({"id": [1, 2, 3], "value": ["a", "b", "c"]})
        result = manager.write_delta(df, "test_table", mode="overwrite")

        assert result["rows"] == 3
        assert result["format"] == "delta"
        assert result["medallion_layer"] == "silver"
        assert result["mode"] == "overwrite"
        assert result["table_name"] == "test_table"
        assert "id" in result["schema"]
        assert result["storage"] == "local"

        # Verify Delta table exists
        delta_log = Path(result["path"]) / "_delta_log"
        assert delta_log.exists()

    def test_write_delta_lazyframe(self, test_data_dir):
        """Test writing a LazyFrame to Delta."""
        manager = PolarsDeltaIOManager(medallion_layer="silver")
        manager.base_uri = str(test_data_dir / "silver")

        df = pl.LazyFrame({"id": [1, 2, 3], "value": ["a", "b", "c"]})
        result = manager.write_delta(df, "test_table", mode="overwrite")

        assert result["rows"] == 3
        assert result["format"] == "delta"

    def test_write_delta_merge_initial(self, test_data_dir):
        """Test merge mode with initial write (table doesn't exist)."""
        manager = PolarsDeltaIOManager(medallion_layer="silver")
        manager.base_uri = str(test_data_dir / "silver")

        df = pl.DataFrame({"id": [1, 2], "value": ["a", "b"]})
        result = manager.write_delta(
            df,
            "test_table",
            mode="merge",
            predicate="s.id = t.id",
        )

        assert result["rows"] == 2
        assert result["mode"] == "merge"
        # Initial write shouldn't have merge metrics
        assert "merge_metrics" not in result

    def test_write_delta_merge_requires_predicate(self, test_data_dir):
        """Test that merge mode requires a predicate."""
        manager = PolarsDeltaIOManager(medallion_layer="silver")
        manager.base_uri = str(test_data_dir / "silver")

        df = pl.DataFrame({"id": [1, 2], "value": ["a", "b"]})

        with pytest.raises(ValueError, match="predicate is required"):
            manager.write_delta(df, "test_table", mode="merge")

    def test_write_delta_merge_update_and_insert(self, test_data_dir):
        """Test merge with updates and inserts."""
        manager = PolarsDeltaIOManager(medallion_layer="silver")
        manager.base_uri = str(test_data_dir / "silver")

        # Initial data
        df1 = pl.DataFrame({"id": [1, 2], "value": ["a", "b"]})
        manager.write_delta(df1, "test_table", mode="merge", predicate="s.id = t.id")

        # Merge with 1 update and 1 insert
        df2 = pl.DataFrame({"id": [2, 3], "value": ["b_updated", "c"]})
        result = manager.write_delta(
            df2, "test_table", mode="merge", predicate="s.id = t.id"
        )

        # Should have merge metrics
        assert "merge_metrics" in result
        metrics = result["merge_metrics"]
        assert metrics["num_source_rows"] == 2
        assert metrics["num_target_rows_inserted"] == 1  # id=3
        assert metrics["num_target_rows_updated"] == 1  # id=2

        # Verify final state
        final_df = pl.read_delta(str(test_data_dir / "silver" / "test_table"))
        assert len(final_df) == 3
        assert set(final_df["id"].to_list()) == {1, 2, 3}
        # Verify update happened
        assert final_df.filter(pl.col("id") == 2)["value"][0] == "b_updated"

    def test_write_delta_with_partitioning(self, test_data_dir):
        """Test writing Delta with partitioning."""
        manager = PolarsDeltaIOManager(medallion_layer="silver")
        manager.base_uri = str(test_data_dir / "silver")

        df = pl.DataFrame(
            {
                "user": ["user1", "user1", "user2"],
                "id": [1, 2, 3],
                "value": ["a", "b", "c"],
            }
        )
        result = manager.write_delta(
            df, "test_table", mode="overwrite", partition_by="user"
        )

        assert result["rows"] == 3

        # Verify partitioned directories exist
        table_path = Path(result["path"])
        partitions = list(table_path.glob("user=*"))
        assert len(partitions) == 2

    def test_read_delta(self, test_data_dir):
        """Test reading Delta table."""
        manager = PolarsDeltaIOManager(medallion_layer="silver")
        manager.base_uri = str(test_data_dir / "silver")

        # Write data first
        df = pl.DataFrame({"id": [1, 2, 3], "value": ["a", "b", "c"]})
        manager.write_delta(df, "test_table", mode="overwrite")

        # Read it back
        result = manager.read_delta("test_table")

        assert isinstance(result, pl.LazyFrame)
        collected = result.collect()
        assert len(collected) == 3
        assert collected["id"].to_list() == [1, 2, 3]

    def test_write_delta_append(self, test_data_dir):
        """Test appending to Delta table."""
        manager = PolarsDeltaIOManager(medallion_layer="silver")
        manager.base_uri = str(test_data_dir / "silver")

        # Initial data
        df1 = pl.DataFrame({"id": [1, 2], "value": ["a", "b"]})
        manager.write_delta(df1, "test_table", mode="overwrite")

        # Append more data
        df2 = pl.DataFrame({"id": [3, 4], "value": ["c", "d"]})
        result = manager.write_delta(df2, "test_table", mode="append")

        assert result["mode"] == "append"

        # Verify final count
        final_df = pl.read_delta(str(test_data_dir / "silver" / "test_table"))
        assert len(final_df) == 4

    def test_table_exists(self, test_data_dir):
        """Test table_exists method."""
        manager = PolarsDeltaIOManager(medallion_layer="silver")
        manager.base_uri = str(test_data_dir / "silver")

        # Table doesn't exist yet
        assert not manager.table_exists("test_table")

        # Create table
        df = pl.DataFrame({"id": [1, 2]})
        manager.write_delta(df, "test_table", mode="overwrite")

        # Now it exists
        assert manager.table_exists("test_table")

    def test_gcs_configuration(self):
        """Test GCS configuration via environment variable."""
        with patch.dict(os.environ, {"GCS_BUCKET_URI": "gs://my-bucket/data"}):
            manager = PolarsDeltaIOManager(medallion_layer="silver")
            assert manager.is_cloud
            assert manager.base_uri == "gs://my-bucket/data/silver"
            assert manager._get_table_uri("test") == "gs://my-bucket/data/silver/test"

    def test_local_configuration_default(self):
        """Test local configuration when GCS_BUCKET_URI is not set."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove GCS_BUCKET_URI if present
            os.environ.pop("GCS_BUCKET_URI", None)
            manager = PolarsDeltaIOManager(medallion_layer="gold")
            assert not manager.is_cloud
            assert "gold" in manager.base_uri
            assert manager.storage_options is None


class TestGCSStorageOptions:
    """Test GCS storage options helper."""

    def test_with_service_account_env(self):
        """Test storage options with GOOGLE_APPLICATION_CREDENTIALS set."""
        with patch.dict(
            os.environ, {"GOOGLE_APPLICATION_CREDENTIALS": "/path/to/creds.json"}
        ):
            options = get_gcs_storage_options()
            assert options == {"google_service_account": "/path/to/creds.json"}

    def test_without_service_account_env(self):
        """Test storage options without GOOGLE_APPLICATION_CREDENTIALS."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS", None)
            options = get_gcs_storage_options()
            assert options is None
