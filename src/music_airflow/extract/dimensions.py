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
from music_airflow.utils.spotify_search import search_spotify_url, is_spotify_configured
import polars as pl
import logging

logger = logging.getLogger(__name__)


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

    logger.info(f"Fetching metadata for {track_count} new tracks...")

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

        logger.info(f"Fetching info for {track_count} tracks concurrently...")
        all_track_info = await asyncio.gather(*track_tasks, return_exceptions=True)

        # Process results
        tracks_data = []
        for idx, track_info in enumerate(all_track_info):
            if isinstance(track_info, Exception):
                logger.info(
                    f"Failed to fetch {track_names[idx]} by {artist_names[idx]}: {track_info}"
                )
                continue

            # Type guard: track_info is dict[str, Any] here
            if not isinstance(track_info, dict):
                logger.info(
                    f"Unexpected type for track {track_names[idx]}: {type(track_info)}"
                )
                continue

            try:
                tracks_data.append(track_info)
            except (ValueError, KeyError) as e:
                logger.info(
                    f"Failed to process {track_names[idx]} by {artist_names[idx]}: {e}"
                )
                continue

        success_count = len(tracks_data)
        logger.info(f"Successfully fetched {success_count}/{track_count} tracks")

        if not tracks_data:
            raise AirflowSkipException(
                f"No track metadata successfully fetched (0/{track_count} succeeded)"
            )

        # Find the most popular version of each track for streaming links
        # The track from get_track_info may be an obscure remaster with no YouTube link
        # We search using NORMALIZED names to find the canonical version (most listeners)
        logger.info(f"Finding popular versions for {len(tracks_data)} tracks...")

        search_tasks = [
            client.search_track(
                track=normalize_text(track.get("name", "")),
                artist=normalize_text(track.get("artist", {}).get("name", "")),
                limit=1,
            )
            for track in tracks_data
        ]
        search_results = await asyncio.gather(*search_tasks, return_exceptions=True)

    # Build mapping from original track to popular version URL (used for Last.fm fallback)
    popular_urls: list[str] = []
    for idx, result in enumerate(search_results):
        original_url = tracks_data[idx].get("url", "")
        if isinstance(result, Exception) or not result:
            popular_urls.append(original_url)
        elif isinstance(result, list) and len(result) > 0:
            popular_urls.append(result[0].get("url", original_url))
        else:
            popular_urls.append(original_url)

    # Primary: Use YTMusic search for YouTube URLs (faster than scraping, better audio results)
    # YTMusic filters for audio-only versions, avoiding music videos
    logger.info(f"Searching YTMusic for {len(tracks_data)} tracks...")
    ytmusic_found = 0
    for idx, track in enumerate(tracks_data):
        track_name = normalize_text(track.get("name", ""))
        artist_name = normalize_text(track.get("artist", {}).get("name", ""))
        if track_name and artist_name:
            youtube_url = search_youtube_url(track_name, artist_name)
            if youtube_url:
                track["youtube_url"] = youtube_url
                ytmusic_found += 1
            else:
                track["youtube_url"] = None
        else:
            track["youtube_url"] = None
        track["spotify_url"] = (
            None  # Will be populated from Spotify search or Last.fm fallback
        )
    logger.info(f"Found {ytmusic_found}/{len(tracks_data)} tracks via YTMusic search")

    # Primary for Spotify: Use Spotipy search (more reliable than Last.fm scraping)
    spotify_configured = is_spotify_configured()
    spotipy_found = 0
    if spotify_configured:
        logger.info(f"Searching Spotify for {len(tracks_data)} tracks...")
        for idx, track in enumerate(tracks_data):
            track_name = normalize_text(track.get("name", ""))
            artist_name = normalize_text(track.get("artist", {}).get("name", ""))
            if track_name and artist_name:
                spotify_url = search_spotify_url(track_name, artist_name)
                if spotify_url:
                    track["spotify_url"] = spotify_url
                    spotipy_found += 1
        logger.info(
            f"Found {spotipy_found}/{len(tracks_data)} tracks via Spotify search"
        )
    else:
        logger.info("Spotify credentials not configured, will use Last.fm fallback")

    # Fallback: Scrape Last.fm pages for tracks missing YouTube or Spotify URLs
    # Last.fm pages sometimes have music video links, so we only use as fallback for YouTube
    missing_youtube = [
        (idx, t) for idx, t in enumerate(tracks_data) if not t.get("youtube_url")
    ]
    missing_spotify = [
        (idx, t) for idx, t in enumerate(tracks_data) if not t.get("spotify_url")
    ]
    valid_urls = [url for url in popular_urls if url]

    # Only scrape Last.fm if there are missing URLs
    if missing_youtube or missing_spotify:
        logger.info(
            f"Scraping Last.fm for {len(valid_urls)} tracks ({len(missing_spotify)} missing Spotify, {len(missing_youtube)} missing YouTube)..."
        )
        async with LastFMScraper() as scraper:
            streaming_links_list = await scraper.get_streaming_links_batch(valid_urls)

        streaming_links = dict(zip(valid_urls, streaming_links_list))

        # Merge Last.fm streaming links: Spotify/YouTube only if missing from primary search
        lastfm_youtube_found = 0
        lastfm_spotify_found = 0
        for idx, track in enumerate(tracks_data):
            popular_url = popular_urls[idx]
            if popular_url and popular_url in streaming_links:
                links = streaming_links[popular_url]
                # Only take Spotify URL if Spotipy didn't find one
                if not track.get("spotify_url") and links.get("spotify_url"):
                    track["spotify_url"] = links["spotify_url"]
                    lastfm_spotify_found += 1
                # Only take YouTube URL if YTMusic didn't find one
                if not track.get("youtube_url") and links.get("youtube_url"):
                    track["youtube_url"] = links["youtube_url"]
                    lastfm_youtube_found += 1

        logger.info(
            f"Last.fm scraping fallback: {lastfm_spotify_found} Spotify URLs, {lastfm_youtube_found} YouTube URLs"
        )
    else:
        logger.info("No missing streaming links, skipping Last.fm scraping")

    final_youtube_count = sum(1 for t in tracks_data if t.get("youtube_url"))
    final_spotify_count = sum(1 for t in tracks_data if t.get("spotify_url"))
    logger.info(
        f"Final coverage: {final_youtube_count}/{len(tracks_data)} YouTube, {final_spotify_count}/{len(tracks_data)} Spotify"
    )

    # Save to bronze as JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"tracks_{timestamp}.json"

    io_manager = JSONIOManager(medallion_layer="bronze")
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

    logger.info(f"Fetching metadata for {artist_count} new artists...")

    # Fetch metadata for new artists
    async with LastFMClient() as client:
        # Create all artist info tasks
        artist_tasks = [
            client.get_artist_info(artist=artist_names[idx])
            for idx in range(artist_count)
        ]

        logger.info(f"Fetching info for {artist_count} artists concurrently...")
        all_artist_info = await asyncio.gather(*artist_tasks, return_exceptions=True)

        # Process results
        artists_data = []
        for idx, artist_info in enumerate(all_artist_info):
            if isinstance(artist_info, Exception):
                logger.info(
                    f"Failed to fetch artist {artist_names[idx]}: {artist_info}"
                )
                continue

            try:
                artists_data.append(artist_info)
            except (ValueError, KeyError) as e:
                logger.info(f"Failed to process artist {artist_names[idx]}: {e}")
                continue

    success_count = len(artists_data)
    logger.info(f"Successfully fetched {success_count}/{artist_count} artists")

    if not artists_data:
        raise AirflowSkipException(
            f"No artist metadata successfully fetched (0/{artist_count} succeeded)"
        )

    # Save to bronze as JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"artists_{timestamp}.json"

    io_manager = JSONIOManager(medallion_layer="bronze")
    metadata = io_manager.write_json(artists_data, f"artists/{filename}")

    return {
        **metadata,
        "artists_fetched": len(artists_data),
    }
