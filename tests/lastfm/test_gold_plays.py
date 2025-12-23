"""
Tests for gold layer play aggregations.
"""

import datetime as dt
from unittest.mock import patch

import polars as pl
import pytest

from music_airflow.transform.gold_plays import (
    _compute_artist_aggregations,
    _compute_track_aggregations,
    compute_artist_play_counts,
    compute_track_play_counts,
)


@pytest.fixture
def sample_plays_df():
    """Create sample plays data for testing."""
    return pl.LazyFrame(
        {
            "username": ["user1", "user1", "user1", "user1", "user2", "user2"],
            "scrobbled_at_utc": [
                dt.datetime(2025, 1, 1, tzinfo=dt.timezone.utc),
                dt.datetime(2025, 1, 5, tzinfo=dt.timezone.utc),
                dt.datetime(2025, 1, 10, tzinfo=dt.timezone.utc),
                dt.datetime(2025, 1, 15, tzinfo=dt.timezone.utc),
                dt.datetime(2025, 1, 1, tzinfo=dt.timezone.utc),
                dt.datetime(2025, 1, 20, tzinfo=dt.timezone.utc),
            ],
            "track_name": ["Song A", "Song A", "Song B", "Song A", "Song C", "Song C"],
            "track_mbid": ["", "", "", "", "", ""],
            "artist_name": [
                "Artist X",
                "Artist X",
                "Artist Y",
                "Artist X",
                "Artist Z",
                "Artist Z",
            ],
            "artist_mbid": ["", "", "", "", "", ""],
            "album_name": [
                "Album 1",
                "Album 1",
                "Album 2",
                "Album 1",
                "Album 3",
                "Album 3",
            ],
        }
    )


def test_compute_artist_aggregations(sample_plays_df):
    """Test artist aggregation logic."""
    execution_date = dt.datetime(2025, 1, 21, tzinfo=dt.timezone.utc)

    result_df = _compute_artist_aggregations(sample_plays_df, execution_date).collect()

    # Check schema
    assert "username" in result_df.columns
    assert "artist_name" in result_df.columns
    assert "play_count" in result_df.columns
    assert "first_played_on" in result_df.columns
    assert "last_played_on" in result_df.columns
    assert "recency_score" in result_df.columns
    assert "days_since_last_play" in result_df.columns

    # Check user1 - Artist X (3 plays)
    user1_artist_x = result_df.filter(
        (pl.col("username") == "user1") & (pl.col("artist_name") == "Artist X")
    )
    assert len(user1_artist_x) == 1
    assert user1_artist_x["play_count"][0] == 3
    assert user1_artist_x["days_since_last_play"][0] == 6  # Jan 21 - Jan 15

    # Check user1 - Artist Y (1 play)
    user1_artist_y = result_df.filter(
        (pl.col("username") == "user1") & (pl.col("artist_name") == "Artist Y")
    )
    assert len(user1_artist_y) == 1
    assert user1_artist_y["play_count"][0] == 1

    # Check user2 - Artist Z (2 plays)
    user2_artist_z = result_df.filter(
        (pl.col("username") == "user2") & (pl.col("artist_name") == "Artist Z")
    )
    assert len(user2_artist_z) == 1
    assert user2_artist_z["play_count"][0] == 2
    assert user2_artist_z["days_since_last_play"][0] == 1  # Jan 21 - Jan 20


def test_compute_track_aggregations(sample_plays_df):
    """Test track aggregation logic."""
    execution_date = dt.datetime(2025, 1, 21, tzinfo=dt.timezone.utc)

    result_df = _compute_track_aggregations(sample_plays_df, execution_date).collect()

    # Check schema
    assert "username" in result_df.columns
    assert "track_name" in result_df.columns
    assert "artist_name" in result_df.columns
    assert "play_count" in result_df.columns
    assert "recency_score" in result_df.columns

    # Check user1 - Song A (3 plays)
    user1_song_a = result_df.filter(
        (pl.col("username") == "user1") & (pl.col("track_name") == "Song A")
    )
    assert len(user1_song_a) == 1
    assert user1_song_a["play_count"][0] == 3
    assert user1_song_a["days_since_last_play"][0] == 6  # Jan 21 - Jan 15

    # Check user2 - Song C (2 plays)
    user2_song_c = result_df.filter(
        (pl.col("username") == "user2") & (pl.col("track_name") == "Song C")
    )
    assert len(user2_song_c) == 1
    assert user2_song_c["play_count"][0] == 2


def test_recency_score_decay(sample_plays_df):
    """Test that recency score properly decays with time."""
    execution_date = dt.datetime(2025, 1, 21, tzinfo=dt.timezone.utc)

    result_df = _compute_track_aggregations(sample_plays_df, execution_date).collect()

    # Song A: plays on Jan 1, 5, 15 (20, 16, 6 days ago)
    # Song B: play on Jan 10 (11 days ago)
    # Song A should have higher recency score (more recent plays)
    user1_song_a = result_df.filter(
        (pl.col("username") == "user1") & (pl.col("track_name") == "Song A")
    )
    user1_song_b = result_df.filter(
        (pl.col("username") == "user1") & (pl.col("track_name") == "Song B")
    )

    # Song A has 3 plays with most recent being 6 days ago
    # Song B has 1 play 11 days ago
    # Song A should have higher recency score
    assert user1_song_a["recency_score"][0] > user1_song_b["recency_score"][0]


def test_aggregations_with_lookback_window(sample_plays_df):
    """Test that lookback window filters old plays correctly."""
    execution_date = dt.datetime(2025, 1, 21, tzinfo=dt.timezone.utc)

    # Apply 10-day lookback (only plays after Jan 11)
    cutoff_date = execution_date - pl.duration(days=10)
    filtered_df = sample_plays_df.filter(pl.col("scrobbled_at_utc") >= cutoff_date)

    result_df = _compute_track_aggregations(filtered_df, execution_date).collect()

    # Only Jan 15 and Jan 20 plays should be included
    # user1 - Song A should have 1 play (Jan 15)
    user1_song_a = result_df.filter(
        (pl.col("username") == "user1") & (pl.col("track_name") == "Song A")
    )
    assert user1_song_a["play_count"][0] == 1

    # user1 - Song B should not appear (Jan 10 is outside window)
    user1_song_b = result_df.filter(
        (pl.col("username") == "user1") & (pl.col("track_name") == "Song B")
    )
    assert len(user1_song_b) == 0


def test_first_and_last_played_dates(sample_plays_df):
    """Test that first and last played dates are correct."""
    execution_date = dt.datetime(2025, 1, 21, tzinfo=dt.timezone.utc)

    result_df = _compute_artist_aggregations(sample_plays_df, execution_date).collect()

    # Artist X: first = Jan 1, last = Jan 15
    artist_x = result_df.filter(
        (pl.col("username") == "user1") & (pl.col("artist_name") == "Artist X")
    )
    assert artist_x["first_played_on"][0] == dt.datetime(
        2025, 1, 1, tzinfo=dt.timezone.utc
    )
    assert artist_x["last_played_on"][0] == dt.datetime(
        2025, 1, 15, tzinfo=dt.timezone.utc
    )


class TestComputeArtistPlayCountsIntegration:
    """Integration tests for compute_artist_play_counts function."""

    def test_compute_artist_play_counts(self, test_data_dir):
        """Test full pipeline for computing artist play counts."""
        # Setup: Create silver plays table
        silver_dir = test_data_dir / "silver"
        silver_dir.mkdir(parents=True, exist_ok=True)
        gold_dir = test_data_dir / "gold"
        gold_dir.mkdir(parents=True, exist_ok=True)

        # Create sample plays data
        plays_df = pl.DataFrame(
            {
                "username": ["user1", "user1", "user1", "user2"],
                "scrobbled_at_utc": [
                    dt.datetime(2025, 1, 1, tzinfo=dt.timezone.utc),
                    dt.datetime(2025, 1, 5, tzinfo=dt.timezone.utc),
                    dt.datetime(2025, 1, 10, tzinfo=dt.timezone.utc),
                    dt.datetime(2025, 1, 15, tzinfo=dt.timezone.utc),
                ],
                "track_name": ["Song A", "Song B", "Song A", "Song C"],
                "track_mbid": ["", "", "", ""],
                "artist_name": ["Artist X", "Artist X", "Artist X", "Artist Y"],
                "artist_mbid": ["", "", "", ""],
                "album_name": ["Album 1", "Album 1", "Album 1", "Album 2"],
            }
        )

        # Write to silver Delta table
        plays_df.write_delta(str(silver_dir / "plays"))

        # Patch IO managers to use test directories
        with (
            patch("music_airflow.transform.gold_plays.PolarsDeltaIOManager") as mock_io,
        ):
            from music_airflow.utils.polars_io_manager import PolarsDeltaIOManager

            # Create mock that returns real managers with test paths
            def create_io_manager(medallion_layer):
                mgr = PolarsDeltaIOManager(medallion_layer=medallion_layer)
                mgr.base_dir = test_data_dir / medallion_layer
                return mgr

            mock_io.side_effect = create_io_manager

            execution_date = dt.datetime(2025, 1, 20, tzinfo=dt.timezone.utc)
            result = compute_artist_play_counts(execution_date)

        # Verify result metadata
        assert result["table_name"] == "artist_play_count"
        assert result["format"] == "delta"
        assert result["medallion_layer"] == "gold"
        assert result["mode"] == "overwrite"
        assert "execution_date" in result

        # Verify gold table was created
        gold_table_path = gold_dir / "artist_play_count"
        assert gold_table_path.exists()

        # Verify content
        gold_df = pl.read_delta(str(gold_table_path))
        assert len(gold_df) == 2  # 2 user-artist combinations
        assert "play_count" in gold_df.columns
        assert "recency_score" in gold_df.columns
        assert "days_since_last_play" in gold_df.columns


class TestComputeTrackPlayCountsIntegration:
    """Integration tests for compute_track_play_counts function."""

    def test_compute_track_play_counts(self, test_data_dir):
        """Test full pipeline for computing track play counts."""
        # Setup: Create silver plays table
        silver_dir = test_data_dir / "silver"
        silver_dir.mkdir(parents=True, exist_ok=True)
        gold_dir = test_data_dir / "gold"
        gold_dir.mkdir(parents=True, exist_ok=True)

        # Create sample plays data
        plays_df = pl.DataFrame(
            {
                "username": ["user1", "user1", "user1", "user2"],
                "scrobbled_at_utc": [
                    dt.datetime(2025, 1, 1, tzinfo=dt.timezone.utc),
                    dt.datetime(2025, 1, 5, tzinfo=dt.timezone.utc),
                    dt.datetime(2025, 1, 10, tzinfo=dt.timezone.utc),
                    dt.datetime(2025, 1, 15, tzinfo=dt.timezone.utc),
                ],
                "track_name": ["Song A", "Song B", "Song A", "Song C"],
                "track_mbid": ["", "", "", ""],
                "artist_name": ["Artist X", "Artist X", "Artist X", "Artist Y"],
                "artist_mbid": ["", "", "", ""],
                "album_name": ["Album 1", "Album 1", "Album 1", "Album 2"],
            }
        )

        # Write to silver Delta table
        plays_df.write_delta(str(silver_dir / "plays"))

        # Patch IO managers to use test directories
        with (
            patch("music_airflow.transform.gold_plays.PolarsDeltaIOManager") as mock_io,
        ):
            from music_airflow.utils.polars_io_manager import PolarsDeltaIOManager

            # Create mock that returns real managers with test paths
            def create_io_manager(medallion_layer):
                mgr = PolarsDeltaIOManager(medallion_layer=medallion_layer)
                mgr.base_dir = test_data_dir / medallion_layer
                return mgr

            mock_io.side_effect = create_io_manager

            execution_date = dt.datetime(2025, 1, 20, tzinfo=dt.timezone.utc)
            result = compute_track_play_counts(execution_date)

        # Verify result metadata
        assert result["table_name"] == "track_play_count"
        assert result["format"] == "delta"
        assert result["medallion_layer"] == "gold"
        assert result["mode"] == "overwrite"
        assert "execution_date" in result

        # Verify gold table was created
        gold_table_path = gold_dir / "track_play_count"
        assert gold_table_path.exists()

        # Verify content
        gold_df = pl.read_delta(str(gold_table_path))
        assert len(gold_df) == 3  # 3 user-track combinations
        assert "play_count" in gold_df.columns
        assert "recency_score" in gold_df.columns
        assert "days_since_last_play" in gold_df.columns
