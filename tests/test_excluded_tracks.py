"""
Tests for excluded track and artist recommendation functionality.
"""

import polars as pl
import pytest
from music_airflow.app.excluded_tracks import (
    write_excluded_track,
    read_excluded_tracks,
    write_excluded_artist,
    read_excluded_artists,
)
from music_airflow.utils.polars_io_manager import PolarsDeltaIOManager


@pytest.fixture
def temp_gold_dir(tmp_path, monkeypatch):
    """Create a temporary gold directory for testing."""
    gold_dir = tmp_path / "data" / "gold"
    gold_dir.mkdir(parents=True, exist_ok=True)

    # Monkey-patch the base_dir in PolarsDeltaIOManager
    original_init = PolarsDeltaIOManager.__init__

    def mock_init(self, medallion_layer: str = "silver"):
        if medallion_layer not in ["bronze", "silver", "gold"]:
            raise ValueError(
                f"medallion_layer must be 'bronze', 'silver', or 'gold', got '{medallion_layer}'"
            )
        self.medallion_layer = medallion_layer
        self.base_dir = tmp_path / "data" / medallion_layer

    monkeypatch.setattr(PolarsDeltaIOManager, "__init__", mock_init)

    yield gold_dir

    # Restore original __init__
    monkeypatch.setattr(PolarsDeltaIOManager, "__init__", original_init)


def test_write_excluded_track_creates_table(temp_gold_dir):
    """Test that writing an excluded track creates the table."""
    result = write_excluded_track(
        username="testuser",
        track_id="track1|artist1",
        track_name="Test Track",
        artist_name="Test Artist",
    )

    assert result["table_name"] == "excluded_recommendations"
    assert result["rows"] == 1
    assert result["format"] == "delta"

    # Verify table was created
    table_path = temp_gold_dir / "excluded_recommendations"
    assert table_path.exists()
    assert (table_path / "_delta_log").exists()


def test_write_excluded_track_merge_updates_timestamp(temp_gold_dir):
    """Test that writing the same track again updates the timestamp."""
    # First write
    write_excluded_track(
        username="testuser",
        track_id="track1|artist1",
        track_name="Test Track",
        artist_name="Test Artist",
    )

    # Read back
    excluded = read_excluded_tracks("testuser").collect()
    assert len(excluded) == 1
    first_timestamp = excluded["excluded_at"][0]

    # Wait a bit and write again
    import time

    time.sleep(0.1)

    write_excluded_track(
        username="testuser",
        track_id="track1|artist1",
        track_name="Test Track",
        artist_name="Test Artist",
    )

    # Read back again
    excluded = read_excluded_tracks("testuser").collect()
    assert len(excluded) == 1  # Still only one row (merge, not append)
    second_timestamp = excluded["excluded_at"][0]

    # Timestamp should be updated
    assert second_timestamp > first_timestamp


def test_read_excluded_tracks_filters_by_username(temp_gold_dir):
    """Test that read_excluded_tracks returns only tracks for the specified user."""
    # Write tracks for two users
    write_excluded_track(
        username="user1",
        track_id="track1|artist1",
        track_name="Track 1",
        artist_name="Artist 1",
    )
    write_excluded_track(
        username="user2",
        track_id="track2|artist2",
        track_name="Track 2",
        artist_name="Artist 2",
    )
    write_excluded_track(
        username="user1",
        track_id="track3|artist3",
        track_name="Track 3",
        artist_name="Artist 3",
    )

    # Read tracks for user1
    user1_excluded = read_excluded_tracks("user1").collect()
    assert len(user1_excluded) == 2
    assert all(user1_excluded["username"] == "user1")

    # Read tracks for user2
    user2_excluded = read_excluded_tracks("user2").collect()
    assert len(user2_excluded) == 1
    assert all(user2_excluded["username"] == "user2")


def test_read_excluded_tracks_empty_table(temp_gold_dir):
    """Test that read_excluded_tracks returns empty LazyFrame when table doesn't exist."""
    excluded = read_excluded_tracks("nonexistent_user").collect()

    assert len(excluded) == 0
    assert excluded.schema == {
        "username": pl.String,
        "track_id": pl.String,
        "track_name": pl.String,
        "artist_name": pl.String,
        "excluded_at": pl.Datetime(time_zone="UTC"),
    }


def test_excluded_track_schema(temp_gold_dir):
    """Test that excluded tracks have the correct schema."""
    write_excluded_track(
        username="testuser",
        track_id="track1|artist1",
        track_name="Test Track",
        artist_name="Test Artist",
    )

    excluded = read_excluded_tracks("testuser").collect()

    assert excluded.schema == {
        "username": pl.String,
        "track_id": pl.String,
        "track_name": pl.String,
        "artist_name": pl.String,
        "excluded_at": pl.Datetime(time_zone="UTC"),
    }

    # Check that timestamp is timezone-aware (UTC)
    assert excluded["excluded_at"][0].tzinfo is not None
    assert str(excluded["excluded_at"][0].tzinfo) == "UTC"


def test_exclusion_matching_by_name_not_id(temp_gold_dir):
    """
    Test that exclusions match by track_name + artist_name, not track_id.

    This is important because track_id can change if MBID is added later,
    but track_name + artist_name remain stable.
    """
    # Exclude a track with one track_id format
    write_excluded_track(
        username="testuser",
        track_id="track_name|artist_name",  # Synthetic ID without MBID
        track_name="Great Song",
        artist_name="Amazing Artist",
    )

    # Create candidate pool with same track but different track_id (has MBID)
    candidates = pl.DataFrame(
        {
            "track_id": ["mbid123", "mbid456", "track2|artist2"],
            "track_name": ["Great Song", "Other Song", "Track 2"],
            "artist_name": ["Amazing Artist", "Different Artist", "Artist 2"],
            "score": [10, 8, 6],
        }
    ).lazy()

    # Load excluded tracks
    excluded = read_excluded_tracks("testuser")

    # Filter candidates using name-based matching (not ID-based)
    filtered = candidates.join(
        excluded.select(["track_name", "artist_name"]),
        on=["track_name", "artist_name"],
        how="anti",
    ).collect()

    # Should exclude "Great Song" by "Amazing Artist" even though track_id differs
    assert len(filtered) == 2
    assert "Great Song" not in filtered["track_name"].to_list()
    assert set(filtered["track_name"].to_list()) == {"Other Song", "Track 2"}


def test_write_excluded_artist_creates_table(temp_gold_dir):
    """Test that writing an excluded artist creates the table."""
    result = write_excluded_artist(
        username="testuser",
        artist_name="Blocked Artist",
    )

    assert result["table_name"] == "excluded_artists"
    assert result["rows"] == 1
    assert result["format"] == "delta"

    # Verify table was created
    table_path = temp_gold_dir / "excluded_artists"
    assert table_path.exists()
    assert (table_path / "_delta_log").exists()


def test_write_excluded_artist_merge_updates_timestamp(temp_gold_dir):
    """Test that writing the same artist again updates the timestamp."""
    # First write
    write_excluded_artist(
        username="testuser",
        artist_name="Blocked Artist",
    )

    # Read back
    excluded = read_excluded_artists("testuser").collect()
    assert len(excluded) == 1
    first_timestamp = excluded["excluded_at"][0]

    # Wait a bit and write again
    import time

    time.sleep(0.1)

    write_excluded_artist(
        username="testuser",
        artist_name="Blocked Artist",
    )

    # Read back again
    excluded = read_excluded_artists("testuser").collect()
    assert len(excluded) == 1  # Still only one row (merge, not append)
    second_timestamp = excluded["excluded_at"][0]

    # Timestamp should be updated
    assert second_timestamp > first_timestamp


def test_read_excluded_artists_filters_by_username(temp_gold_dir):
    """Test that read_excluded_artists returns only artists for the specified user."""
    # Write artists for two users
    write_excluded_artist(username="user1", artist_name="Artist A")
    write_excluded_artist(username="user2", artist_name="Artist B")
    write_excluded_artist(username="user1", artist_name="Artist C")

    # Read artists for user1
    user1_excluded = read_excluded_artists("user1").collect()
    assert len(user1_excluded) == 2
    assert all(user1_excluded["username"] == "user1")
    assert set(user1_excluded["artist_name"].to_list()) == {"Artist A", "Artist C"}

    # Read artists for user2
    user2_excluded = read_excluded_artists("user2").collect()
    assert len(user2_excluded) == 1
    assert all(user2_excluded["username"] == "user2")
    assert user2_excluded["artist_name"][0] == "Artist B"


def test_read_excluded_artists_empty_table(temp_gold_dir):
    """Test that read_excluded_artists returns empty LazyFrame when table doesn't exist."""
    excluded = read_excluded_artists("nonexistent_user").collect()

    assert len(excluded) == 0
    assert excluded.schema == {
        "username": pl.String,
        "artist_name": pl.String,
        "excluded_at": pl.Datetime(time_zone="UTC"),
    }


def test_artist_exclusion_filters_all_tracks(temp_gold_dir):
    """Test that excluding an artist filters out all their tracks from candidates."""
    # Exclude an artist
    write_excluded_artist(
        username="testuser",
        artist_name="Blocked Artist",
    )

    # Create candidate pool with multiple tracks from blocked artist
    candidates = pl.DataFrame(
        {
            "track_id": ["track1", "track2", "track3", "track4"],
            "track_name": ["Song 1", "Song 2", "Song 3", "Song 4"],
            "artist_name": [
                "Blocked Artist",
                "Good Artist",
                "Blocked Artist",
                "Another Artist",
            ],
            "score": [10, 8, 7, 6],
        }
    ).lazy()

    # Load excluded artists
    excluded_artists = read_excluded_artists("testuser")

    # Filter candidates by excluded artists
    filtered = candidates.join(
        excluded_artists.select("artist_name"),
        on="artist_name",
        how="anti",
    ).collect()

    # Should exclude all tracks from "Blocked Artist"
    assert len(filtered) == 2
    assert "Blocked Artist" not in filtered["artist_name"].to_list()
    assert set(filtered["artist_name"].to_list()) == {"Good Artist", "Another Artist"}
