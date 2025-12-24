"""
Tests for candidate track generation logic.

Tests the three candidate generation strategies and merging:
- Similar artist candidates
- Similar tag candidates
- Deep cut candidates
- Merged unified candidates
"""

import polars as pl
import pytest

from music_airflow.transform.candidate_generation import (
    generate_deep_cut_candidates,
    generate_similar_artist_candidates,
    generate_similar_tag_candidates,
    merge_candidate_sources,
)


@pytest.fixture
def setup_test_data(tmp_path):
    """
    Create test silver layer data for candidate generation with synthetic IDs.

    Creates:
    - plays: User play history with track_id
    - artists: Artist metadata with tags and artist_id
    - tracks: Track metadata with tags, popularity, track_id, artist_id
    """
    # Create test data directories
    silver_path = tmp_path / "silver"
    silver_path.mkdir()

    # Sample artists with tags and IDs
    artists_data = pl.DataFrame(
        {
            "artist_name": [
                "Artist A",
                "Artist B",
                "Artist C",
                "Artist D",
                "Artist E",
            ],
            "artist_mbid": ["mbid-a", "mbid-b", "", "", "mbid-e"],
            "listeners": [10000, 8000, 6000, 4000, 2000],
            "playcount": [50000, 40000, 30000, 20000, 10000],
            "tags": [
                "rock,alternative,indie",
                "rock,indie,pop",
                "electronic,ambient,chill",
                "rock,punk,alternative",
                "jazz,blues,soul",
            ],
            "bio_summary": ["Bio A", "Bio B", "Bio C", "Bio D", "Bio E"],
            "artist_url": ["url-a", "url-b", "url-c", "url-d", "url-e"],
        }
    ).with_columns(
        # Create artist_id (use MBID or fallback to name)
        pl.when(pl.col("artist_mbid") != "")
        .then(pl.col("artist_mbid"))
        .otherwise(pl.col("artist_name"))
        .alias("artist_id")
    )

    # Sample tracks with IDs
    tracks_data = pl.DataFrame(
        {
            "track_name": [
                "Track A1",
                "Track A2",
                "Track A3 Obscure",
                "Track B1",
                "Track B2",
                "Track C1",
                "Track C2",
                "Track D1",
                "Track D2 Obscure",
                "Track E1",
            ],
            "track_mbid": [
                "track-mbid-0",
                "track-mbid-1",
                "",
                "track-mbid-3",
                "",
                "",
                "",
                "track-mbid-7",
                "",
                "",
            ],
            "artist_name": [
                "Artist A",
                "Artist A",
                "Artist A",
                "Artist B",
                "Artist B",
                "Artist C",
                "Artist C",
                "Artist D",
                "Artist D",
                "Artist E",
            ],
            "artist_mbid": [
                "mbid-a",
                "mbid-a",
                "mbid-a",
                "mbid-b",
                "mbid-b",
                "",
                "",
                "",
                "",
                "mbid-e",
            ],
            "album_name": [
                "Album A",
                "Album A",
                "Album A",
                "Album B",
                "Album B",
                "Album C",
                "Album C",
                "Album D",
                "Album D",
                "Album E",
            ],
            "duration_ms": [180000] * 10,
            "listeners": [10000, 8000, 500, 9000, 7000, 6000, 5000, 8500, 300, 2000],
            "playcount": [
                50000,
                40000,
                1000,
                45000,
                35000,
                30000,
                25000,
                42000,
                800,
                10000,
            ],
            "tags": [
                "rock,alternative,indie",
                "rock,indie",
                "rock,alternative",
                "rock,indie,pop",
                "indie,pop",
                "electronic,ambient",
                "chill,ambient",
                "rock,punk,alternative",
                "punk,alternative",
                "jazz,blues",
            ],
            "track_url": [f"url-track-{i}" for i in range(10)],
        }
    ).with_columns(
        # Create artist_id
        pl.when(pl.col("artist_mbid") != "")
        .then(pl.col("artist_mbid"))
        .otherwise(pl.col("artist_name"))
        .alias("artist_id"),
        # Create track_id: prefer MBID when available, else synthetic
        pl.when((pl.col("track_mbid").is_not_null()) & (pl.col("track_mbid") != ""))
        .then(pl.col("track_mbid"))
        .otherwise(
            pl.concat_str([pl.col("track_name"), pl.col("artist_name")], separator="|")
        )
        .alias("track_id"),
    )

    # Sample user plays with IDs
    plays_data = pl.DataFrame(
        {
            "scrobbled_at": [1700000000, 1700000100],
            "scrobbled_at_utc": pl.Series(
                [1700000000, 1700000100], dtype=pl.Int64
            ).cast(pl.Datetime("us", "UTC")),
            "track_name": ["Track A1", "Track A2"],
            "track_mbid": ["track-mbid-0", "track-mbid-1"],
            "track_url": ["url-track-0", "url-track-1"],
            "artist_name": ["Artist A", "Artist A"],
            "album_name": ["Album A", "Album A"],
            "album_mbid": ["album-mbid-a", "album-mbid-a"],
            "username": ["test_user", "test_user"],
        }
    ).with_columns(
        # Create track_id: prefer MBID when available, else synthetic
        pl.when((pl.col("track_mbid").is_not_null()) & (pl.col("track_mbid") != ""))
        .then(pl.col("track_mbid"))
        .otherwise(
            pl.concat_str([pl.col("track_name"), pl.col("artist_name")], separator="|")
        )
        .alias("track_id")
    )

    # Write to Delta tables
    artists_path = str(silver_path / "artists")
    tracks_path = str(silver_path / "tracks")
    plays_path = str(silver_path / "plays")

    artists_data.write_delta(artists_path, mode="overwrite")
    tracks_data.write_delta(tracks_path, mode="overwrite")
    plays_data.write_delta(plays_path, mode="overwrite")

    return {
        "silver_path": str(silver_path),
        "artists_path": artists_path,
        "tracks_path": tracks_path,
        "plays_path": plays_path,
    }


def test_generate_similar_artist_candidates(setup_test_data, monkeypatch):
    """
    Test similar artist candidate generation with ID-based joins.

    Verifies:
    - Uses artist_id and track_id for joins
    - Finds artists with overlapping tags
    - Excludes already played tracks
    - Returns LazyFrame
    """
    # Store original scan_delta
    original_scan_delta = pl.scan_delta

    # Patch delta paths to use test data
    def mock_scan_delta(path):
        if "artists" in str(path):
            return original_scan_delta(setup_test_data["artists_path"])
        elif "tracks" in str(path):
            return original_scan_delta(setup_test_data["tracks_path"])
        elif "plays" in str(path):
            return original_scan_delta(setup_test_data["plays_path"])
        raise ValueError(f"Unknown path: {path}")

    monkeypatch.setattr("polars.scan_delta", mock_scan_delta)

    # Generate candidates
    result_lf = generate_similar_artist_candidates(username="test_user")

    # Verify returns LazyFrame
    assert isinstance(result_lf, pl.LazyFrame)

    # Collect and verify
    candidates = result_lf.collect()

    # Should have some candidates
    assert len(candidates) > 0

    # Should have required columns including IDs
    required_cols = [
        "username",
        "track_id",
        "artist_id",
        "track_name",
        "artist_name",
        "score",
    ]
    for col in required_cols:
        assert col in candidates.columns

    # Should not include played tracks (test data has MBIDs)
    played_track_ids = ["track-mbid-0", "track-mbid-1"]
    candidate_track_ids = candidates["track_id"].to_list()
    for track_id in played_track_ids:
        assert track_id not in candidate_track_ids


def test_generate_similar_tag_candidates(setup_test_data, monkeypatch):
    """
    Test similar tag candidate generation.

    Verifies:
    - Uses track_id for joins
    - Finds tracks with matching tags
    - Excludes already played tracks
    - Returns LazyFrame
    """
    original_scan_delta = pl.scan_delta

    def mock_scan_delta(path):
        if "tracks" in str(path):
            return original_scan_delta(setup_test_data["tracks_path"])
        elif "plays" in str(path):
            return original_scan_delta(setup_test_data["plays_path"])
        raise ValueError(f"Unknown path: {path}")

    monkeypatch.setattr("polars.scan_delta", mock_scan_delta)

    result_lf = generate_similar_tag_candidates(username="test_user")

    assert isinstance(result_lf, pl.LazyFrame)

    candidates = result_lf.collect()
    assert len(candidates) > 0

    # Verify IDs present
    assert "track_id" in candidates.columns
    assert "artist_id" in candidates.columns

    # Should not include played tracks (test data has MBIDs)
    played_track_ids = ["track-mbid-0", "track-mbid-1"]
    candidate_track_ids = candidates["track_id"].to_list()
    for track_id in played_track_ids:
        assert track_id not in candidate_track_ids


def test_generate_deep_cut_candidates(setup_test_data, monkeypatch):
    """
    Test deep cut candidate generation.

    Verifies:
    - Uses artist_id and track_id for joins
    - Finds obscure tracks from known artists
    - Respects listener range
    - Returns LazyFrame
    """
    original_scan_delta = pl.scan_delta

    def mock_scan_delta(path):
        if "tracks" in str(path):
            return original_scan_delta(setup_test_data["tracks_path"])
        elif "plays" in str(path):
            return original_scan_delta(setup_test_data["plays_path"])
        elif "artists" in str(path):
            return original_scan_delta(setup_test_data["artists_path"])
        raise ValueError(f"Unknown path: {path}")

    monkeypatch.setattr("polars.scan_delta", mock_scan_delta)

    result_lf = generate_deep_cut_candidates(
        username="test_user", min_listeners=100, max_listeners=1000
    )

    assert isinstance(result_lf, pl.LazyFrame)

    candidates = result_lf.collect()
    assert len(candidates) > 0

    # Should find Track A3 Obscure (500 listeners from Artist A)
    track_names = candidates["track_name"].to_list()
    assert "Track A3 Obscure" in track_names

    # Should not include played tracks
    assert "Track A1" not in track_names
    assert "Track A2" not in track_names

    # Verify obscurity range
    assert all((candidates["listeners"] >= 100) & (candidates["listeners"] <= 1000))


def test_merge_candidate_sources(setup_test_data, monkeypatch):
    """
    Test merging of candidate sources with type indicators.

    Verifies:
    - Creates one-hot encoded type columns
    - Deduplicates by track_id
    - Preserves all source flags for duplicates
    """
    original_scan_delta = pl.scan_delta

    def mock_scan_delta(path):
        if "artists" in str(path):
            return original_scan_delta(setup_test_data["artists_path"])
        elif "tracks" in str(path):
            return original_scan_delta(setup_test_data["tracks_path"])
        elif "plays" in str(path):
            return original_scan_delta(setup_test_data["plays_path"])
        raise ValueError(f"Unknown path: {path}")

    monkeypatch.setattr("polars.scan_delta", mock_scan_delta)

    # Generate all three types
    similar_artists_lf = generate_similar_artist_candidates(username="test_user")
    similar_tags_lf = generate_similar_tag_candidates(username="test_user")
    deep_cuts_lf = generate_deep_cut_candidates(username="test_user")

    # Merge
    merged_lf = merge_candidate_sources(
        similar_artists_lf, similar_tags_lf, deep_cuts_lf
    )

    assert isinstance(merged_lf, pl.LazyFrame)

    merged = merged_lf.collect()

    # Should have type indicator columns
    type_cols = ["similar_artist", "similar_tag", "deep_cut_same_artist"]
    for col in type_cols:
        assert col in merged.columns

    # Should have deduplicated by track_id
    track_ids = merged["track_id"].to_list()
    assert len(track_ids) == len(set(track_ids))

    # At least one type should be True for each track
    for i in range(len(merged)):
        row = merged.row(i, named=True)
        assert (
            row["similar_artist"] or row["similar_tag"] or row["deep_cut_same_artist"]
        )
