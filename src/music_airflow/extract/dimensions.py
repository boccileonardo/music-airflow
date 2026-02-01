"""
Extract dimension data (tracks, artists) from Last.fm API.

Fetches metadata for tracks and artists that appear in user play history.
Dimensions are global (not per-user) - the same track/artist has same metadata regardless of who listened.
Supports incremental extraction - only fetches new tracks/artists not already in bronze layer.
"""

from datetime import datetime
from typing import Any

import asyncio
from airflow.exceptions import AirflowSkipException
from deltalake.exceptions import TableNotFoundError

from music_airflow.lastfm_client import LastFMClient
from music_airflow.utils.polars_io_manager import PolarsDeltaIOManager, JSONIOManager
from music_airflow.utils.lastfm_scraper import LastFMScraper
from music_airflow.utils.text_normalization import normalize_text
from music_airflow.utils.ytmusic_search import search_youtube_url
import polars as pl


async def extract_tracks_to_bronze() -> dict[str, Any]:
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

    # Get unique tracks from plays (keep as LazyFrame)
    unique_tracks_lf = plays_lf.select(["track_name", "artist_name"]).unique()

    # Also check for tracks from candidate_generation (gold layer)
    # Candidates include track_name and artist_name for API calls
    try:
        gold_io = PolarsDeltaIOManager(medallion_layer="gold")
        candidates_lf = gold_io.read_delta("track_candidates")

        # Extract track names from candidates (use original names for API calls)
        candidate_tracks_lf = (
            candidates_lf.select(["track_name", "artist_name"])
            .filter(
                pl.col("track_name").is_not_null() & pl.col("artist_name").is_not_null()
            )
            .unique()
        )
        # Union with play tracks
        unique_tracks_lf = pl.concat([unique_tracks_lf, candidate_tracks_lf]).unique()
    except (FileNotFoundError, TableNotFoundError):
        # No candidates yet, just use tracks from plays
        pass

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
    new_tracks_df: pl.DataFrame = new_tracks_lf.select(
        "track_name", "artist_name"
    ).collect(engine="streaming")
    track_names: list = new_tracks_df["track_name"].to_list()
    artist_names: list = new_tracks_df["artist_name"].to_list()

    track_count = len(track_names)
    if track_count == 0:
        raise AirflowSkipException("No new tracks to fetch")

    print(f"Fetching metadata for {track_count} new tracks...")

    # Fetch metadata for new tracks and search for popular versions
    async with LastFMClient() as client:
        # Create all track info tasks
        track_tasks = [
            client.get_track_info(
                track=track_names[idx],
                artist=artist_names[idx],
            )
            for idx in range(track_count)
        ]

        print(f"Fetching info for {track_count} tracks concurrently...")
        all_track_info = await asyncio.gather(*track_tasks, return_exceptions=True)

        # Process results
        tracks_data = []
        for idx, track_info in enumerate(all_track_info):
            if isinstance(track_info, Exception):
                print(
                    f"Failed to fetch {track_names[idx]} by {artist_names[idx]}: {track_info}"
                )
                continue

            # Type guard: track_info is dict[str, Any] here
            if not isinstance(track_info, dict):
                print(
                    f"Unexpected type for track {track_names[idx]}: {type(track_info)}"
                )
                continue

            try:
                tracks_data.append(track_info)
            except (ValueError, KeyError) as e:
                print(
                    f"Failed to process {track_names[idx]} by {artist_names[idx]}: {e}"
                )
                continue

        success_count = len(tracks_data)
        print(f"Successfully fetched {success_count}/{track_count} tracks")

        if not tracks_data:
            raise AirflowSkipException(
                f"No track metadata successfully fetched (0/{track_count} succeeded)"
            )

        # Find the most popular version of each track for streaming links
        # The track from get_track_info may be an obscure remaster with no YouTube link
        # We search using NORMALIZED names to find the canonical version (most listeners)
        print(f"Finding popular versions for {len(tracks_data)} tracks...")

        search_tasks = [
            client.search_track(
                track=normalize_text(track.get("name", "")),
                artist=normalize_text(track.get("artist", {}).get("name", "")),
                limit=1,
            )
            for track in tracks_data
        ]
        search_results = await asyncio.gather(*search_tasks, return_exceptions=True)

    # Build mapping from original track to popular version URL
    popular_urls: list[str] = []
    for idx, result in enumerate(search_results):
        original_url = tracks_data[idx].get("url", "")
        if isinstance(result, Exception) or not result:
            # Fall back to original URL
            popular_urls.append(original_url)
        elif isinstance(result, list) and len(result) > 0:
            # Use the most popular version's URL
            popular_urls.append(result[0].get("url", original_url))
        else:
            popular_urls.append(original_url)

    # Enrich with streaming links from the popular version's Last.fm page
    print(f"Fetching streaming links for {len(tracks_data)} tracks...")
    valid_urls = [url for url in popular_urls if url]

    async with LastFMScraper() as scraper:
        streaming_links_list = await scraper.get_streaming_links_batch(valid_urls)

    # Build mapping from URL to streaming links
    streaming_links = dict(zip(valid_urls, streaming_links_list))

    # Merge streaming links into track data using popular version URLs
    for idx, track in enumerate(tracks_data):
        popular_url = popular_urls[idx]
        if popular_url and popular_url in streaming_links:
            links = streaming_links[popular_url]
            track["youtube_url"] = links.get("youtube_url")
            track["spotify_url"] = links.get("spotify_url")
        else:
            track["youtube_url"] = None
            track["spotify_url"] = None

    enriched_count = sum(
        1 for t in tracks_data if t.get("youtube_url") or t.get("spotify_url")
    )
    print(f"Enriched {enriched_count}/{len(tracks_data)} tracks with streaming links")

    # Fallback: Use YTMusic search for tracks without YouTube URLs
    # Use NORMALIZED names to find canonical versions, not obscure remasters
    missing_youtube = [
        (idx, t) for idx, t in enumerate(tracks_data) if not t.get("youtube_url")
    ]
    if missing_youtube:
        print(
            f"Searching YTMusic for {len(missing_youtube)} tracks without YouTube URLs..."
        )
        ytmusic_found = 0
        for idx, track in missing_youtube:
            track_name = normalize_text(track.get("name", ""))
            artist_name = normalize_text(track.get("artist", {}).get("name", ""))
            if track_name and artist_name:
                youtube_url = search_youtube_url(track_name, artist_name)
                if youtube_url:
                    track["youtube_url"] = youtube_url
                    ytmusic_found += 1
        print(f"Found {ytmusic_found}/{len(missing_youtube)} tracks via YTMusic search")

    final_youtube_count = sum(1 for t in tracks_data if t.get("youtube_url"))
    print(f"Final YouTube coverage: {final_youtube_count}/{len(tracks_data)} tracks")

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


async def extract_artists_to_bronze() -> dict[str, Any]:
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
    new_artists_df: pl.DataFrame = new_artists_lf.collect(engine="streaming")
    artist_names: list = new_artists_df["artist_name"].to_list()
    artist_count = len(artist_names)
    if artist_count == 0:
        raise AirflowSkipException("No new artists to fetch")

    print(f"Fetching metadata for {artist_count} new artists...")

    # Fetch metadata for new artists
    async with LastFMClient() as client:
        # Create all artist info tasks
        artist_tasks = [
            client.get_artist_info(artist=artist_names[idx])
            for idx in range(artist_count)
        ]

        print(f"Fetching info for {artist_count} artists concurrently...")
        all_artist_info = await asyncio.gather(*artist_tasks, return_exceptions=True)

        # Process results
        artists_data = []
        for idx, artist_info in enumerate(all_artist_info):
            if isinstance(artist_info, Exception):
                print(f"Failed to fetch artist {artist_names[idx]}: {artist_info}")
                continue

            try:
                artists_data.append(artist_info)
            except (ValueError, KeyError) as e:
                print(f"Failed to process artist {artist_names[idx]}: {e}")
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
