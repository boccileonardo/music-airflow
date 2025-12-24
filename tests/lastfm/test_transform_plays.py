"""
Tests for plays transformation logic.

Tests transformation functions including raw-to-structured conversion,
edge cases, and data type handling.
"""

import datetime as dt
from pathlib import Path
from unittest.mock import patch

import polars as pl

from music_airflow.transform.plays import (
    transform_plays_raw_to_structured,
    transform_plays_to_silver,
)


class TestTransformPlaysRawToStructured:
    """Test transform_plays_raw_to_structured function."""

    def test_basic_transformation(self):
        """Test basic transformation of raw tracks."""
        # Create sample raw data matching Last.fm API structure
        raw_data = {
            "name": ["Track 1", "Track 2"],
            "mbid": ["mbid1", "mbid2"],
            "url": ["url1", "url2"],
            "date": [
                {"uts": "1609459200", "#text": "01 Jan 2021"},
                {"uts": "1609545600", "#text": "02 Jan 2021"},
            ],
            "artist": [
                {"name": "Artist 1", "mbid": "artist_mbid1"},
                {"name": "Artist 2", "mbid": "artist_mbid2"},
            ],
            "album": [
                {"#text": "Album 1", "mbid": "album_mbid1"},
                {"#text": "Album 2", "mbid": "album_mbid2"},
            ],
        }

        df = pl.LazyFrame(raw_data)
        result = transform_plays_raw_to_structured(df, "testuser").collect()

        # Check columns
        expected_cols = [
            "username",
            "scrobbled_at",
            "scrobbled_at_utc",
            "track_name",
            "track_mbid",
            "track_url",
            "artist_name",
            "album_name",
            "album_mbid",
        ]
        assert result.columns == expected_cols

        # Check data
        assert result["username"].to_list() == ["testuser", "testuser"]
        assert result["track_name"].to_list() == ["Track 1", "Track 2"]
        assert result["artist_name"].to_list() == ["Artist 1", "Artist 2"]
        assert result["album_name"].to_list() == ["Album 1", "Album 2"]

        # Check timestamps
        assert result["scrobbled_at"].to_list() == [1609459200, 1609545600]

        # Check datetime conversion (compare as timestamps since timezone repr may differ)
        dt_values = result["scrobbled_at_utc"].to_list()
        assert (
            dt_values[0].timestamp()
            == dt.datetime(2021, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc).timestamp()
        )
        assert (
            dt_values[1].timestamp()
            == dt.datetime(2021, 1, 2, 0, 0, 0, tzinfo=dt.timezone.utc).timestamp()
        )

    def test_missing_mbids(self):
        """Test handling of missing MBIDs (converted to None)."""
        raw_data = {
            "name": ["Track 1"],
            "mbid": [""],
            "url": ["url1"],
            "date": [{"uts": "1609459200", "#text": "01 Jan 2021"}],
            "artist": [{"name": "Artist 1", "mbid": None}],
            "album": [{"#text": "Album 1", "mbid": None}],
        }

        df = pl.LazyFrame(raw_data)
        result = transform_plays_raw_to_structured(df, "testuser").collect()

        # Empty string MBIDs should be converted to None
        assert result["track_mbid"].to_list() == [None]
        assert result["album_mbid"].to_list() == [None]

    def test_sorting_by_timestamp(self):
        """Test that results are sorted by scrobbled_at."""
        # Create data in reverse chronological order
        raw_data = {
            "name": ["Track 3", "Track 1", "Track 2"],
            "mbid": ["mbid3", "mbid1", "mbid2"],
            "url": ["url3", "url1", "url2"],
            "date": [
                {"uts": "1609632000", "#text": "03 Jan 2021"},
                {"uts": "1609459200", "#text": "01 Jan 2021"},
                {"uts": "1609545600", "#text": "02 Jan 2021"},
            ],
            "artist": [
                {"name": "Artist 3", "mbid": "artist_mbid3"},
                {"name": "Artist 1", "mbid": "artist_mbid1"},
                {"name": "Artist 2", "mbid": "artist_mbid2"},
            ],
            "album": [
                {"#text": "Album 3", "mbid": "album_mbid3"},
                {"#text": "Album 1", "mbid": "album_mbid1"},
                {"#text": "Album 2", "mbid": "album_mbid2"},
            ],
        }

        df = pl.LazyFrame(raw_data)
        result = transform_plays_raw_to_structured(df, "testuser").collect()

        # Should be sorted chronologically
        assert result["track_name"].to_list() == ["Track 1", "Track 2", "Track 3"]
        assert result["scrobbled_at"].to_list() == [1609459200, 1609545600, 1609632000]

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        # For empty dataframes, we need to provide proper schema with structs
        raw_data = pl.LazyFrame(
            schema={
                "name": pl.Utf8,
                "mbid": pl.Utf8,
                "url": pl.Utf8,
                "date": pl.Struct(
                    [pl.Field("uts", pl.Utf8), pl.Field("#text", pl.Utf8)]
                ),
                "artist": pl.Struct(
                    [pl.Field("name", pl.Utf8), pl.Field("mbid", pl.Utf8)]
                ),
                "album": pl.Struct(
                    [pl.Field("#text", pl.Utf8), pl.Field("mbid", pl.Utf8)]
                ),
            }
        )

        result = transform_plays_raw_to_structured(raw_data, "testuser").collect()

        assert len(result) == 0
        assert result.columns == [
            "username",
            "scrobbled_at",
            "scrobbled_at_utc",
            "track_name",
            "track_mbid",
            "track_url",
            "artist_name",
            "album_name",
            "album_mbid",
        ]


class TestTransformPlaysToSilver:
    """Test transform_plays_to_silver integration function."""

    def test_successful_transformation(self, test_data_dir, monkeypatch):
        """Test successful transformation from bronze to silver Delta table."""
        # Setup: Create bronze data
        bronze_dir = test_data_dir / "bronze" / "plays" / "testuser"
        bronze_dir.mkdir(parents=True, exist_ok=True)
        silver_dir = test_data_dir / "silver"
        silver_dir.mkdir(parents=True, exist_ok=True)

        import json

        tracks = [
            {
                "name": "Track 1",
                "mbid": "mbid1",
                "url": "url1",
                "date": {"uts": "1609459200", "#text": "01 Jan 2021"},
                "artist": {"name": "Artist 1", "mbid": "artist_mbid1"},
                "album": {"#text": "Album 1", "mbid": "album_mbid1"},
            },
            {
                "name": "Track 2",
                "mbid": "mbid2",
                "url": "url2",
                "date": {"uts": "1609462800", "#text": "01 Jan 2021"},
                "artist": {"name": "Artist 2", "mbid": "artist_mbid2"},
                "album": {"#text": "Album 2", "mbid": "album_mbid2"},
            },
        ]

        bronze_file = bronze_dir / "20210101.json"
        with open(bronze_file, "w") as f:
            json.dump(tracks, f)

        # Patch both IO managers' base_dir after instantiation
        with (
            patch("music_airflow.transform.plays.JSONIOManager") as mock_json_io,
            patch(
                "music_airflow.transform.plays.PolarsDeltaIOManager"
            ) as mock_delta_io,
        ):
            # Create real instances but override base_dir
            from music_airflow.utils.polars_io_manager import (
                JSONIOManager,
                PolarsDeltaIOManager,
            )

            json_mgr = JSONIOManager(medallion_layer="bronze")
            json_mgr.base_dir = test_data_dir / "bronze"

            delta_mgr = PolarsDeltaIOManager(medallion_layer="silver")
            delta_mgr.base_dir = test_data_dir / "silver"

            mock_json_io.return_value = json_mgr
            mock_delta_io.return_value = delta_mgr

            # Execute transformation
            fetch_metadata = {
                "filename": "plays/testuser/20210101.json",
                "username": "testuser",
                "from_datetime": "2021-01-01T00:00:00+00:00",
                "to_datetime": "2021-01-02T00:00:00+00:00",
            }

            result = transform_plays_to_silver(fetch_metadata)

        # Verify result metadata
        assert result["rows"] == 2
        assert result["username"] == "testuser"
        assert result["format"] == "delta"
        assert result["medallion_layer"] == "silver"
        assert result["table_name"] == "plays"
        assert result["mode"] == "merge"

        # Verify Delta table was created
        delta_table_path = Path(result["path"])
        assert delta_table_path.exists()

        # Verify content from Delta table
        df = pl.read_delta(str(delta_table_path))
        assert len(df) == 2
        assert df["track_name"].to_list() == ["Track 1", "Track 2"]
        assert df["username"].to_list() == ["testuser", "testuser"]

    def test_empty_bronze_file_returns_skipped(self, test_data_dir):
        """Test that empty bronze file returns skipped result."""
        # Setup: Create empty bronze data
        bronze_dir = test_data_dir / "bronze" / "plays" / "testuser"
        bronze_dir.mkdir(parents=True, exist_ok=True)

        import json

        bronze_file = bronze_dir / "20210101.json"
        with open(bronze_file, "w") as f:
            json.dump([], f)

        with patch("music_airflow.transform.plays.JSONIOManager") as mock_json_io:
            from music_airflow.utils.polars_io_manager import JSONIOManager

            json_mgr = JSONIOManager(medallion_layer="bronze")
            json_mgr.base_dir = test_data_dir / "bronze"
            mock_json_io.return_value = json_mgr

            # Execute transformation
            fetch_metadata = {
                "filename": "plays/testuser/20210101.json",
                "username": "testuser",
                "from_datetime": "2021-01-01T00:00:00+00:00",
                "to_datetime": "2021-01-02T00:00:00+00:00",
            }

            result = transform_plays_to_silver(fetch_metadata)

        # Should return empty/skipped result
        assert result["skipped"] is True
        assert result["rows"] == 0
        assert result["path"] is None
        assert result["reason"] == "No tracks in time range"

    def test_multiple_users_in_single_table(self, test_data_dir):
        """Test that multiple users write to the same Delta table with partitioning."""
        bronze_dir = test_data_dir / "bronze" / "plays"
        bronze_dir.mkdir(parents=True, exist_ok=True)
        silver_dir = test_data_dir / "silver"
        silver_dir.mkdir(parents=True, exist_ok=True)

        import json

        # Create bronze data for user1
        user1_dir = bronze_dir / "user1"
        user1_dir.mkdir(parents=True, exist_ok=True)
        tracks_user1 = [
            {
                "name": "Track User1",
                "mbid": "mbid1",
                "url": "url1",
                "date": {"uts": "1672531200", "#text": "01 Jan 2023"},
                "artist": {"name": "Artist 1", "mbid": "artist_mbid1"},
                "album": {"#text": "Album 1", "mbid": "album_mbid1"},
            }
        ]
        with open(user1_dir / "20230101.json", "w") as f:
            json.dump(tracks_user1, f)

        # Create bronze data for user2
        user2_dir = bronze_dir / "user2"
        user2_dir.mkdir(parents=True, exist_ok=True)
        tracks_user2 = [
            {
                "name": "Track User2",
                "mbid": "mbid2",
                "url": "url2",
                "date": {"uts": "1672531200", "#text": "01 Jan 2023"},
                "artist": {"name": "Artist 2", "mbid": "artist_mbid2"},
                "album": {"#text": "Album 2", "mbid": "album_mbid2"},
            }
        ]
        with open(user2_dir / "20230101.json", "w") as f:
            json.dump(tracks_user2, f)

        with (
            patch("music_airflow.transform.plays.JSONIOManager") as mock_json_io,
            patch(
                "music_airflow.transform.plays.PolarsDeltaIOManager"
            ) as mock_delta_io,
        ):
            from music_airflow.utils.polars_io_manager import (
                JSONIOManager,
                PolarsDeltaIOManager,
            )

            json_mgr = JSONIOManager(medallion_layer="bronze")
            json_mgr.base_dir = test_data_dir / "bronze"

            delta_mgr = PolarsDeltaIOManager(medallion_layer="silver")
            delta_mgr.base_dir = test_data_dir / "silver"

            mock_json_io.return_value = json_mgr
            mock_delta_io.return_value = delta_mgr

            # Process user1
            result1 = transform_plays_to_silver(
                {
                    "filename": "plays/user1/20230101.json",
                    "username": "user1",
                    "from_datetime": "2023-01-01T00:00:00+00:00",
                    "to_datetime": "2023-01-02T00:00:00+00:00",
                }
            )

            # Process user2
            result2 = transform_plays_to_silver(
                {
                    "filename": "plays/user2/20230101.json",
                    "username": "user2",
                    "from_datetime": "2023-01-01T00:00:00+00:00",
                    "to_datetime": "2023-01-02T00:00:00+00:00",
                }
            )

        # Both should write to same table
        assert result1["table_name"] == "plays"
        assert result2["table_name"] == "plays"
        assert result1["path"] == result2["path"]

        # Verify both users' data is in the table
        df = pl.read_delta(str(result2["path"]))
        assert len(df) == 2
        usernames = df["username"].unique().sort().to_list()
        assert usernames == ["user1", "user2"]

    def test_preserves_datetime_metadata(self, test_data_dir):
        """Test that from/to datetimes are preserved in result."""
        bronze_dir = test_data_dir / "bronze" / "plays" / "testuser"
        bronze_dir.mkdir(parents=True, exist_ok=True)
        silver_dir = test_data_dir / "silver"
        silver_dir.mkdir(parents=True, exist_ok=True)

        import json

        tracks = [
            {
                "name": "Track",
                "mbid": "mbid",
                "url": "url",
                "date": {"uts": "1609459200", "#text": "01 Jan 2021"},
                "artist": {"name": "Artist", "mbid": "artist_mbid"},
                "album": {"#text": "Album", "mbid": "album_mbid"},
            }
        ]

        bronze_file = bronze_dir / "20210101.json"
        with open(bronze_file, "w") as f:
            json.dump(tracks, f)

        with (
            patch("music_airflow.transform.plays.JSONIOManager") as mock_json_io,
            patch(
                "music_airflow.transform.plays.PolarsDeltaIOManager"
            ) as mock_delta_io,
        ):
            from music_airflow.utils.polars_io_manager import (
                JSONIOManager,
                PolarsDeltaIOManager,
            )

            json_mgr = JSONIOManager(medallion_layer="bronze")
            json_mgr.base_dir = test_data_dir / "bronze"

            delta_mgr = PolarsDeltaIOManager(medallion_layer="silver")
            delta_mgr.base_dir = test_data_dir / "silver"

            mock_json_io.return_value = json_mgr
            mock_delta_io.return_value = delta_mgr

            fetch_metadata = {
                "filename": "plays/testuser/20210101.json",
                "username": "testuser",
                "from_datetime": "2021-01-01T00:00:00+00:00",
                "to_datetime": "2021-01-02T00:00:00+00:00",
            }

            result = transform_plays_to_silver(fetch_metadata)

        assert result["from_datetime"] == "2021-01-01T00:00:00+00:00"
        assert result["to_datetime"] == "2021-01-02T00:00:00+00:00"


class TestDeltaMergeMetrics:
    """Test Delta table merge metrics capture and partition handling."""

    def test_merge_with_updates_and_inserts(self, test_data_dir):
        """Test merge operation captures correct metrics for updates and inserts."""
        from music_airflow.utils.polars_io_manager import PolarsDeltaIOManager

        delta_mgr = PolarsDeltaIOManager(medallion_layer="silver")
        delta_mgr.base_dir = test_data_dir / "silver"

        # Initial data
        df1 = pl.DataFrame(
            {
                "username": ["user1", "user1", "user2"],
                "scrobbled_at": [1000, 2000, 3000],
                "track_name": ["Track A", "Track B", "Track C"],
            }
        )

        delta_mgr.write_delta(
            df1,
            table_name="plays",
            mode="merge",
            predicate="s.username = t.username AND s.scrobbled_at = t.scrobbled_at",
            partition_by="username",
        )

        # New data: 1 update (user1, 1000), 2 new inserts
        df2 = pl.DataFrame(
            {
                "username": ["user1", "user1", "user2"],
                "scrobbled_at": [1000, 4000, 5000],  # 1000 exists, 4000 & 5000 are new
                "track_name": [
                    "Track A Updated",
                    "Track D",
                    "Track E",
                ],
            }
        )

        result = delta_mgr.write_delta(
            df2,
            table_name="plays",
            mode="merge",
            predicate="s.username = t.username AND s.scrobbled_at = t.scrobbled_at",
            partition_by="username",
        )

        # Check merge metrics
        assert "merge_metrics" in result
        metrics = result["merge_metrics"]
        assert metrics["num_source_rows"] == 3
        assert metrics["num_target_rows_inserted"] == 2  # 2 new tracks
        assert metrics["num_target_rows_updated"] == 1  # 1 updated track
        assert metrics["num_target_rows_deleted"] == 0

        # Verify final table content
        final_df = pl.read_delta(str(test_data_dir / "silver" / "plays")).sort(
            "scrobbled_at"
        )
        assert len(final_df) == 5  # 3 original + 2 new
        # Check the update happened
        track_a = final_df.filter(pl.col("scrobbled_at") == 1000)
        assert track_a["track_name"].to_list() == ["Track A Updated"]

    def test_partition_isolation_in_merge(self, test_data_dir):
        """Test that merge only affects files in relevant partitions."""
        from music_airflow.utils.polars_io_manager import PolarsDeltaIOManager

        delta_mgr = PolarsDeltaIOManager(medallion_layer="silver")
        delta_mgr.base_dir = test_data_dir / "silver"

        # Create data for multiple users (partitions)
        df1 = pl.DataFrame(
            {
                "username": ["user1", "user1", "user2", "user2", "user3"],
                "scrobbled_at": [1000, 2000, 3000, 4000, 5000],
                "track_name": ["A1", "A2", "B1", "B2", "C1"],
            }
        )

        delta_mgr.write_delta(
            df1,
            table_name="plays",
            mode="merge",
            predicate="s.username = t.username AND s.scrobbled_at = t.scrobbled_at",
            partition_by="username",
        )

        # Update only user2 partition
        df2 = pl.DataFrame(
            {
                "username": ["user2", "user2"],
                "scrobbled_at": [3000, 6000],  # 3000 update, 6000 insert
                "track_name": ["B1 Updated", "B3"],
            }
        )

        result = delta_mgr.write_delta(
            df2,
            table_name="plays",
            mode="merge",
            predicate="s.username = t.username AND s.scrobbled_at = t.scrobbled_at",
            partition_by="username",
        )

        # Check merge metrics show partition isolation
        metrics = result["merge_metrics"]
        assert metrics["num_source_rows"] == 2
        assert metrics["num_target_rows_inserted"] == 1
        assert metrics["num_target_rows_updated"] == 1

        # num_target_rows_copied represents rows from the affected partition (user2)
        # that weren't matched in the merge (B2 with scrobbled_at=4000)
        assert metrics["num_target_rows_copied"] == 1  # B2 copied through

        # num_output_rows is the total for the affected partition after merge
        assert metrics["num_output_rows"] == 3  # B1 Updated, B2, B3

        # Files metrics: should show targeted file operations
        # num_target_files_scanned includes all partitions scanned (3 partitions = 3 files)
        assert metrics["num_target_files_scanned"] == 3
        # Only user2 partition file is rewritten
        assert metrics["num_target_files_added"] == 1
        assert metrics["num_target_files_removed"] == 1

        # Verify other partitions unchanged
        final_df = pl.read_delta(str(test_data_dir / "silver" / "plays"))
        user1_data = final_df.filter(pl.col("username") == "user1").sort("scrobbled_at")
        assert user1_data["track_name"].to_list() == ["A1", "A2"]  # Unchanged

        user2_data = final_df.filter(pl.col("username") == "user2").sort("scrobbled_at")
        assert user2_data["track_name"].to_list() == ["B1 Updated", "B2", "B3"]

        user3_data = final_df.filter(pl.col("username") == "user3")
        assert user3_data["track_name"].to_list() == ["C1"]  # Unchanged

    def test_merge_metrics_included_in_transform_result(self, test_data_dir):
        """Test that transform_plays_to_silver includes merge metrics on actual merges."""
        import json

        bronze_dir = test_data_dir / "bronze" / "plays" / "testuser"
        bronze_dir.mkdir(parents=True, exist_ok=True)
        silver_dir = test_data_dir / "silver"
        silver_dir.mkdir(parents=True, exist_ok=True)

        # First write - create initial data
        tracks1 = [
            {
                "name": "Track 1",
                "mbid": "mbid1",
                "url": "url1",
                "date": {"uts": "1609459200", "#text": "01 Jan 2021"},
                "artist": {"name": "Artist 1", "mbid": "artist_mbid1"},
                "album": {"#text": "Album 1", "mbid": "album_mbid1"},
            }
        ]

        bronze_file1 = bronze_dir / "20210101.json"
        with open(bronze_file1, "w") as f:
            json.dump(tracks1, f)

        with (
            patch("music_airflow.transform.plays.JSONIOManager") as mock_json_io,
            patch(
                "music_airflow.transform.plays.PolarsDeltaIOManager"
            ) as mock_delta_io,
        ):
            from music_airflow.utils.polars_io_manager import (
                JSONIOManager,
                PolarsDeltaIOManager,
            )

            json_mgr = JSONIOManager(medallion_layer="bronze")
            json_mgr.base_dir = test_data_dir / "bronze"

            delta_mgr = PolarsDeltaIOManager(medallion_layer="silver")
            delta_mgr.base_dir = test_data_dir / "silver"

            mock_json_io.return_value = json_mgr
            mock_delta_io.return_value = delta_mgr

            # Initial write (no merge metrics expected)
            fetch_metadata1 = {
                "filename": "plays/testuser/20210101.json",
                "username": "testuser",
                "from_datetime": "2021-01-01T00:00:00+00:00",
                "to_datetime": "2021-01-02T00:00:00+00:00",
            }

            result1 = transform_plays_to_silver(fetch_metadata1)
            assert "merge_metrics" not in result1  # Initial write has no merge metrics

            # Second write with 1 update and 1 insert - should have merge metrics
            tracks2 = [
                {
                    "name": "Track 1 Updated",  # Update existing
                    "mbid": "mbid1",
                    "url": "url1",
                    "date": {"uts": "1609459200", "#text": "01 Jan 2021"},
                    "artist": {"name": "Artist 1", "mbid": "artist_mbid1"},
                    "album": {"#text": "Album 1", "mbid": "album_mbid1"},
                },
                {
                    "name": "Track 2",  # New insert
                    "mbid": "mbid2",
                    "url": "url2",
                    "date": {"uts": "1609462800", "#text": "01 Jan 2021"},
                    "artist": {"name": "Artist 2", "mbid": "artist_mbid2"},
                    "album": {"#text": "Album 2", "mbid": "album_mbid2"},
                },
            ]

            bronze_file2 = bronze_dir / "20210102.json"
            with open(bronze_file2, "w") as f:
                json.dump(tracks2, f)

            fetch_metadata2 = {
                "filename": "plays/testuser/20210102.json",
                "username": "testuser",
                "from_datetime": "2021-01-02T00:00:00+00:00",
                "to_datetime": "2021-01-03T00:00:00+00:00",
            }

            result2 = transform_plays_to_silver(fetch_metadata2)

        # Verify merge metrics are present for actual merge
        assert "merge_metrics" in result2
        metrics = result2["merge_metrics"]
        assert "num_source_rows" in metrics
        assert "num_target_rows_inserted" in metrics
        assert "num_target_rows_updated" in metrics
        assert metrics["num_source_rows"] == 2
        assert metrics["num_target_rows_inserted"] == 1  # Track 2 is new
        assert metrics["num_target_rows_updated"] == 1  # Track 1 is updated
