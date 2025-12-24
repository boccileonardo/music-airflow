"""
Extract dimension data (tracks, artists) from Last.fm API.

Fetches metadata for tracks and artists that appear in user play history.
Dimensions are global (not per-user) - the same track/artist has same metadata regardless of who listened.
Supports incremental extraction - only fetches new tracks/artists not already in bronze layer.
"""

from datetime import datetime
from typing import Any
import time

from airflow.exceptions import AirflowSkipException
from deltalake.exceptions import TableNotFoundError

from music_airflow.lastfm_client import LastFMClient
from music_airflow.utils.polars_io_manager import PolarsDeltaIOManager, JSONIOManager


def extract_tracks_to_bronze() -> dict[str, Any]:
    """
    Extract track metadata from Last.fm API for new tracks across all users.

    Reads silver plays table for all users, identifies tracks not yet in bronze,
    and fetches metadata (tags, listeners, playcount) from Last.fm API.
    Rate limited to avoid API throttling.

    Returns:
        Metadata dict with:
        - filename: Path to saved JSON file
        - path: Absolute path to file
        - rows: Number of tracks fetched
        - format: "json"
        - medallion_layer: "bronze"
        - tracks_fetched: Number of new tracks

    Raises:
        AirflowSkipException: If no new tracks to fetch
        ValueError: If API key not configured
    """
    # Read silver plays to get unique tracks across all users
    try:
        plays_io = PolarsDeltaIOManager(medallion_layer="silver")
        plays_lf = plays_io.read_delta("plays")
    except (FileNotFoundError, TableNotFoundError):
        # No plays data yet - nothing to process
        raise AirflowSkipException("No plays data available yet - run plays DAG first")

    # Get unique tracks (keep as LazyFrame)
    unique_tracks_lf = plays_lf.select(
        ["track_name", "artist_name", "track_mbid"]
    ).unique()

    # Check which tracks already exist in silver
    try:
        silver_io = PolarsDeltaIOManager(medallion_layer="silver")
        existing_tracks_lf = silver_io.read_delta("tracks").select(
            ["track_name", "artist_name"]
        )
        # Anti-join to find new tracks
        new_tracks_lf = unique_tracks_lf.join(
            existing_tracks_lf,
            on=["track_name", "artist_name"],
            how="anti",
        )
    except (FileNotFoundError, TableNotFoundError):
        # No existing tracks table - all tracks are new
        new_tracks_lf = unique_tracks_lf

    # Only collect once to preserve row order across columns
    new_tracks_df = new_tracks_lf.select(
        "track_name", "artist_name", "track_mbid"
    ).collect()
    track_names = new_tracks_df["track_name"].to_list()
    artist_names = new_tracks_df["artist_name"].to_list()
    track_mbids = new_tracks_df["track_mbid"].to_list()

    track_count = len(track_names)
    if track_count == 0:
        raise AirflowSkipException("No new tracks to fetch")

    print(f"Fetching metadata for {track_count} new tracks...")

    # Fetch metadata for new tracks
    client = LastFMClient()
    tracks_data = []

    for idx in range(track_count):
        track_name = track_names[idx]
        artist_name = artist_names[idx]
        track_mbid = track_mbids[idx]

        try:
            track_info = client.get_track_info(
                track=track_name,
                artist=artist_name,
                mbid=track_mbid if track_mbid else None,
            )
            # Preserve the MBID from plays rather than the one from API response
            # (API often returns empty MBID even when we query with one)
            if track_mbid:
                track_info["mbid"] = track_mbid
            tracks_data.append(track_info)

            # Progress logging every 10 tracks
            if (idx + 1) % 10 == 0:
                print(
                    f"Progress: {idx + 1}/{track_count} tracks fetched ({len(tracks_data)} successful)"
                )

            # Rate limiting: 5 requests per second max
            if (idx + 1) % 5 == 0:
                time.sleep(1.0)
            else:
                time.sleep(0.2)

        except (ValueError, KeyError) as e:
            # Log error but continue with other tracks
            print(f"Failed to fetch {track_name} by {artist_name}: {e}")
            continue

    success_count = len(tracks_data)
    print(f"Successfully fetched {success_count}/{track_count} tracks")

    if not tracks_data:
        raise AirflowSkipException(
            f"No track metadata successfully fetched (0/{track_count} succeeded)"
        )

    # Save to bronze as JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"tracks_{timestamp}.json"

    io_manager = JSONIOManager(medallion_layer="bronze")
    tracks_path = io_manager.base_dir / "tracks"
    tracks_path.mkdir(parents=True, exist_ok=True)
    metadata = io_manager.write_json(tracks_data, f"tracks/{filename}")

    return {
        **metadata,
        "tracks_fetched": len(tracks_data),
    }


def extract_artists_to_bronze() -> dict[str, Any]:
    """
    Extract artist metadata from Last.fm API for new artists across all users.

    Reads silver plays table for all users, identifies artists not yet in bronze,
    and fetches metadata (tags, listeners, playcount, bio) from Last.fm API.
    Rate limited to avoid API throttling.

    Returns:
        Metadata dict with:
        - filename: Path to saved JSON file
        - path: Absolute path to file
        - rows: Number of artists fetched
        - format: "json"
        - medallion_layer: "bronze"
        - artists_fetched: Number of new artists

    Raises:
        AirflowSkipException: If no new artists to fetch
        ValueError: If API key not configured
    """
    # Read silver plays to get unique artists across all users
    try:
        plays_io = PolarsDeltaIOManager(medallion_layer="silver")
        plays_lf = plays_io.read_delta("plays")
    except (FileNotFoundError, TableNotFoundError):
        # No plays data yet - nothing to process
        raise AirflowSkipException("No plays data available yet - run plays DAG first")

    # Get unique artists (keep as LazyFrame)
    unique_artists_lf = plays_lf.select(["artist_name"]).unique()

    # Check which artists already exist in silver (transform writes to silver, not bronze)
    try:
        silver_io = PolarsDeltaIOManager(medallion_layer="silver")
        existing_artists_lf = silver_io.read_delta("artists").select(["artist_name"])
        # Anti-join to find new artists
        new_artists_lf = unique_artists_lf.join(
            existing_artists_lf,
            on=["artist_name"],
            how="anti",
        )
    except (FileNotFoundError, TableNotFoundError):
        # No existing artists table - all artists are new
        new_artists_lf = unique_artists_lf

    # Collect artist names
    new_artists_df = new_artists_lf.collect()
    artist_names = new_artists_df["artist_name"].to_list()

    artist_count = len(artist_names)
    if artist_count == 0:
        raise AirflowSkipException("No new artists to fetch")

    print(f"Fetching metadata for {artist_count} new artists...")

    # Fetch metadata for new artists
    client = LastFMClient()
    artists_data = []

    for idx in range(artist_count):
        artist_name = artist_names[idx]

        try:
            artist_info = client.get_artist_info(artist=artist_name)
            artists_data.append(artist_info)

            # Progress logging every 10 artists
            if (idx + 1) % 10 == 0:
                print(
                    f"Progress: {idx + 1}/{artist_count} artists fetched ({len(artists_data)} successful)"
                )

            # Rate limiting: 5 requests per second max
            if (idx + 1) % 5 == 0:
                time.sleep(1.0)
            else:
                time.sleep(0.2)

        except (ValueError, KeyError) as e:
            # Log error but continue with other artists
            print(f"Failed to fetch {artist_name}: {e}")
            continue

    success_count = len(artists_data)
    print(f"Successfully fetched {success_count}/{artist_count} artists")

    if not artists_data:
        raise AirflowSkipException(
            f"No artist metadata successfully fetched (0/{artist_count} succeeded)"
        )

    # Save to bronze as JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"artists_{timestamp}.json"

    io_manager = JSONIOManager(medallion_layer="bronze")
    artists_path = io_manager.base_dir / "artists"
    artists_path.mkdir(parents=True, exist_ok=True)
    metadata = io_manager.write_json(artists_data, f"artists/{filename}")

    return {
        **metadata,
        "artists_fetched": len(artists_data),
    }
