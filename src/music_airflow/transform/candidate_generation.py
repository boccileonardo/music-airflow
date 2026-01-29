"""
Candidate track generation for music recommendation system.

Generates candidate track lists using Last.fm API similarity:
- Similar artist tracks: Top tracks from artists similar to user's plays (via artist.getSimilar)
- Similar tag tracks: Tracks appearing under multiple tags from user's tag profile (via tag.getTopTracks)
- Deep cut tracks: Obscure tracks from top albums by user's favorite artists (via artist.getTopAlbums)
- Old favorites: Tracks user played in past but not recently (based on half_life from dim tables)

Returns DataFrames for silver-layer storage; DAG consolidates into single gold table.
All candidates are saved (no limits), relying on incremental processing to avoid reprocessing.
"""

from deltalake.exceptions import TableNotFoundError
from datetime import datetime
from typing import Any
import logging

import asyncio
import polars as pl

from music_airflow.lastfm_client import LastFMClient
from music_airflow.utils.polars_io_manager import PolarsDeltaIOManager
from music_airflow.utils.text_normalization import generate_canonical_track_id

logger = logging.getLogger(__name__)

__all__ = [
    "generate_similar_artist_candidates",
    "generate_similar_tag_candidates",
    "generate_deep_cut_candidates",
    "generate_old_favorites_candidates",
    "merge_candidate_sources",
]


async def _gather_with_progress(
    tasks: list,
    description: str,
    progress_interval: int = 60,
) -> list:
    """
    Execute async tasks with periodic progress logging.

    Args:
        tasks: List of coroutines to execute
        description: Description for log messages
        progress_interval: Seconds between progress updates

    Returns:
        List of results from asyncio.gather
    """
    total = len(tasks)
    done_event = asyncio.Event()

    async def log_progress():
        elapsed = 0
        while not done_event.is_set():
            try:
                await asyncio.wait_for(done_event.wait(), timeout=progress_interval)
                break  # Event was set, exit immediately
            except asyncio.TimeoutError:
                # Timeout reached, log progress and continue
                elapsed += progress_interval
                logger.info(
                    f"{description}: {elapsed}s elapsed, waiting for {total} tasks..."
                )

    # Start progress logger
    progress_task = asyncio.create_task(log_progress())

    try:
        # Execute all tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results
    finally:
        # Stop progress logger
        done_event.set()
        await progress_task


def _resolve_track_ids(
    track_data: list[dict[str, Any]],
    delta_mgr: PolarsDeltaIOManager,
) -> pl.DataFrame:
    """
    Generate canonical track IDs for track data.

    Uses text normalization to create consistent track_ids regardless
    of recording version (remastered, live, etc.).

    Args:
        track_data: List of dicts with track_name and artist_name
        delta_mgr: IO manager (not used anymore, kept for compatibility)

    Returns:
        DataFrame with canonical track_id added
    """
    if not track_data:
        return pl.DataFrame()

    # Create DataFrame
    df = pl.DataFrame(track_data)

    # Generate canonical track_id from normalized names
    df = df.with_columns(
        pl.struct(["track_name", "artist_name"])
        .map_elements(
            lambda x: generate_canonical_track_id(x["track_name"], x["artist_name"]),
            return_dtype=pl.Utf8,
        )
        .alias("track_id")
    )

    return df


async def generate_similar_artist_candidates(
    username: str,
    min_listeners: int = 1000,
    similarity_threshold: float = 0.9,
    artist_sample_rate: float = 0.2,
) -> dict[str, Any]:
    """
    Generate candidate tracks from artists similar to user's played artists (via Last.fm API).

    Strategy:
    1. Get all unique artists played by user
    2. Sample artists to limit API requests
    3. For each artist, call artist.getSimilar
    4. Filter out clones/false positives (match > similarity_threshold)
    5. For each similar artist, call artist.getTopTracks
    6. Exclude tracks user has already played
    7. Save all candidates (incremental processing avoids reprocessing)

    Saves results to data/silver/candidate_similar_artist/ Delta table.

    Args:
        username: Target user
        min_listeners: Minimum listeners for candidate tracks (quality filter)
        similarity_threshold: Exclude artists with match > this (likely duplicates)
        artist_sample_rate: Fraction of user's artists to sample (0.2 = 20%)

    Returns:
        Metadata dict with path, rows, table_name
    """
    delta_mgr_silver = PolarsDeltaIOManager(medallion_layer="silver")
    delta_mgr_gold = PolarsDeltaIOManager(medallion_layer="gold")

    # Load user's plays
    plays_lf = delta_mgr_silver.read_delta("plays").filter(
        pl.col("username") == username
    )
    tracks_lf = delta_mgr_silver.read_delta("tracks")
    artists_lf = delta_mgr_silver.read_delta("artists")

    # Load user's artist play counts for scoring
    artist_play_counts_lf = (
        delta_mgr_gold.read_delta("artist_play_count")
        .filter(pl.col("username") == username)
        .select(["artist_id", "play_count"])
        .rename({"play_count": "user_artist_play_count"})
    )

    # Get unique artists user has played (id + name + play count)
    played_artists = (
        plays_lf.select("track_id")
        .unique()
        .join(tracks_lf.select("track_id", "artist_id"), on="track_id", how="left")
        .join(
            artists_lf.select("artist_id", "artist_name"),
            on="artist_id",
            how="left",
        )
        .join(
            artist_play_counts_lf,
            on="artist_id",
            how="left",
        )
        .select(["artist_id", "artist_name", "user_artist_play_count"])
        .unique()
        .filter(
            pl.col("artist_id").is_not_null()
            & pl.col("artist_name").is_not_null()
            & pl.col("user_artist_play_count").is_not_null()
        )
    )

    # Incremental processing: exclude artists already used as a source for this user
    try:
        processed_artists = (
            delta_mgr_silver.read_delta("candidate_similar_artist")
            .filter(pl.col("username") == username)
            .select("source_artist_id")
            .unique()
            .rename({"source_artist_id": "artist_id"})
        )
        artists_to_process = played_artists.join(
            processed_artists, on="artist_id", how="anti"
        )
    except (TableNotFoundError, FileNotFoundError):
        # Table may not exist on first run; process all played artists
        artists_to_process = played_artists

    # Collect first to enable sampling
    total_artists_lf = artists_to_process.select(pl.len())
    total_artists = total_artists_lf.collect().item()
    logger.info(f"Found {total_artists} source artists to process for {username}")

    artists_to_process = artists_to_process.collect(engine="streaming")

    # Apply sampling if we have many artists (reduces API load)
    if total_artists > 30:
        logger.info(
            f"Sampling from artist list to limit API requests (rate={artist_sample_rate})"
        )
        artists_to_process = artists_to_process.sample(fraction=artist_sample_rate)
        total_artists = len(artists_to_process)
        logger.info(f"Remaining artists after sampling: {total_artists}")

    # Prepare set of played track_ids for exclusion
    played_track_ids_df = (
        plays_lf.select("track_id").unique().collect(engine="streaming")
    )
    played_track_ids_set = set(played_track_ids_df["track_id"].to_list())

    # Collect candidate tracks
    all_candidates: list[dict[str, Any]] = []

    async with LastFMClient() as client:
        # Fetch all similar artists concurrently
        artist_names = artists_to_process["artist_name"].to_list()
        artist_ids = artists_to_process["artist_id"].to_list()

        similar_artist_tasks = [
            client.get_similar_artists(artist_name, limit=20)
            for artist_name in artist_names
        ]
        logger.info(
            f"Fetching similar artists for {len(similar_artist_tasks)} artists..."
        )
        all_similar_artists = await _gather_with_progress(
            similar_artist_tasks,
            description="Fetching similar artists",
        )
        logger.info("Completed fetching similar artists")

        # Process results and gather top track requests
        top_track_tasks = []
        task_metadata = []  # Store (source_artist_name, source_artist_id) for each task

        for idx, similar_artists in enumerate(all_similar_artists):
            if isinstance(similar_artists, Exception):
                logger.warning(
                    f"Failed to fetch similar artists for {artist_names[idx]}: {similar_artists}"
                )
                continue

            # Filter out clones/false positives
            filtered_similar = [
                a
                for a in similar_artists
                if float(a.get("match", 0)) <= similarity_threshold
            ]

            # Queue top track requests for each similar artist
            for similar_artist in filtered_similar:
                similar_artist_name = similar_artist.get("name")
                similarity_score = float(similar_artist.get("match", 0))
                if not similar_artist_name:
                    continue

                top_track_tasks.append(
                    client.get_artist_top_tracks(similar_artist_name, limit=50)
                )
                task_metadata.append(
                    {
                        "source_artist_name": artist_names[idx],
                        "source_artist_id": artist_ids[idx],
                        "similar_artist_name": similar_artist_name,
                        "similarity_score": similarity_score,
                        "user_artist_play_count": artists_to_process[
                            "user_artist_play_count"
                        ][idx],
                    }
                )

        # Fetch all top tracks concurrently
        logger.info(
            f"Fetching top tracks for {len(top_track_tasks)} similar artists..."
        )
        all_top_tracks = await _gather_with_progress(
            top_track_tasks,
            description="Fetching top tracks",
        )
        logger.info("Completed fetching top tracks")

        # Process all results
        total_tracks_examined = 0
        tracks_filtered_by_min_listeners = 0
        tracks_filtered_by_already_played = 0
        top_tracks_failures = 0

        for top_tracks, metadata in zip(all_top_tracks, task_metadata):
            if isinstance(top_tracks, Exception):
                top_tracks_failures += 1
                logger.warning(
                    f"Failed to fetch top tracks for {metadata['similar_artist_name']}: {top_tracks}"
                )
                continue

            total_tracks_examined += len(top_tracks)

            for track in top_tracks:
                track_name = track.get("name")
                artist_info = track.get("artist", {})
                if isinstance(artist_info, dict):
                    artist_name_track = artist_info.get(
                        "name", metadata["similar_artist_name"]
                    )
                else:
                    artist_name_track = metadata["similar_artist_name"]

                listeners = int(track.get("listeners", 0))
                playcount = int(track.get("playcount", 0))

                # Apply quality filter
                if listeners < min_listeners:
                    tracks_filtered_by_min_listeners += 1
                    continue

                # Create canonical track ID using text normalization
                track_id = generate_canonical_track_id(track_name, artist_name_track)

                # Skip if already played
                if track_id in played_track_ids_set:
                    tracks_filtered_by_already_played += 1
                    continue

                # Score = similarity * user's play count of source artist * track global playcount
                # This prioritizes: tracks from similar artists to heavily played artists
                score = (
                    metadata["similarity_score"]
                    * metadata["user_artist_play_count"]
                    * playcount
                )

                all_candidates.append(
                    {
                        "username": username,
                        "track_name": track_name,
                        "artist_name": artist_name_track,
                        "similarity": metadata["similarity_score"],
                        "score": score,
                        "source_artist_id": metadata["source_artist_id"],
                    }
                )

        logger.info(
            f"Similar-artist generation for {username}: "
            f"examined {total_tracks_examined} tracks from {len(top_track_tasks)} similar artists, "
            f"filtered {tracks_filtered_by_min_listeners} by min_listeners ({min_listeners}), "
            f"filtered {tracks_filtered_by_already_played} already played, "
            f"{top_tracks_failures} top tracks failures, "
            f"result: {len(all_candidates)} candidates"
        )

    # Convert to DataFrame and deduplicate
    if not all_candidates:
        # Create empty DataFrame with correct schema
        df = pl.DataFrame(
            schema={
                "username": pl.String,
                "track_id": pl.String,
                "similarity": pl.Float64,
                "score": pl.Float64,
                "source_artist_id": pl.String,
            }
        )
    else:
        # Resolve track IDs from dimension table
        df = _resolve_track_ids(all_candidates, delta_mgr_silver)
        df = (
            df.unique(subset=["track_id"])  # dedup by track
            .sort("score", descending=True)
            .select(
                [
                    "username",
                    "track_id",
                    "similarity",
                    "score",
                    "source_artist_id",
                ]
            )
        )

    # Write to silver Delta table with incremental upsert
    write_meta = delta_mgr_silver.write_delta(
        df,
        table_name="candidate_similar_artist",
        mode="merge",
        predicate="s.track_id = t.track_id AND s.username = t.username",
        partition_by="username",
    )

    return {
        "path": write_meta["path"],
        "rows": write_meta["rows"],
        "table_name": write_meta["table_name"],
    }


async def generate_similar_tag_candidates(
    username: str,
    top_tags_count: int = 100,
    tracks_per_tag: int = 500,
    min_tag_matches: int = 2,
) -> dict[str, Any]:
    """
    Generate candidate tracks matching user's tag profile (via Last.fm API).

    Strategy:
    1. Build user's tag profile: count tag frequency across played artists
    2. Select top N most frequent tags (represents core taste)
    3. For each tag, get top tracks from Last.fm
    4. Track which tags each track appears under
    5. Score by: number of matching tags + average rank
    6. Keep only tracks appearing under min_tag_matches or more tags
    7. Exclude tracks user has already played
    8. Save all candidates (incremental processing avoids reprocessing)

    Saves results to data/silver/candidate_similar_tag/ Delta table.

    Args:
        username: Target user
        top_tags_count: Number of most frequent tags to use (default 30)
        tracks_per_tag: Max tracks to fetch per tag (default 50)
        min_tag_matches: Minimum tags a track must appear under (default 2)

    Returns:
        Metadata dict with path, rows, table_name
    """
    delta_mgr_silver = PolarsDeltaIOManager(medallion_layer="silver")

    # Load data
    plays = delta_mgr_silver.read_delta("plays").filter(pl.col("username") == username)
    tracks = delta_mgr_silver.read_delta("tracks")
    artists = delta_mgr_silver.read_delta("artists")

    # Build user's tag profile with frequency counts
    tag_profile = (
        plays.select("track_id")
        .unique()
        .join(
            tracks.select("track_id", "artist_id"),
            on="track_id",
            how="left",
        )
        .join(
            artists.select("artist_id", "tags"),
            on="artist_id",
            how="left",
        )
        .filter(pl.col("tags").is_not_null() & (pl.col("tags") != ""))
        .with_columns(pl.col("tags").str.split(",").alias("tag_list"))
        .explode("tag_list")
        .with_columns(pl.col("tag_list").str.strip_chars().alias("tag"))
        .group_by("tag")
        .agg(pl.len().alias("tag_count"))
        .sort("tag_count", descending=True)
        .limit(top_tags_count)
    )

    # Incremental: check which tags have been processed
    try:
        processed_tags_df = (
            delta_mgr_silver.read_delta("candidate_similar_tag")
            .filter(pl.col("username") == username)
            .select("source_tags")
            .unique()
            .collect()
        )
        # source_tags is a comma-separated string, need to parse it
        processed_tags_set = set()
        for row in processed_tags_df.iter_rows():
            if row[0]:
                processed_tags_set.update(row[0].split(","))

        # Get current tag profile as set
        current_tags_df = tag_profile.collect()
        current_tags_set = set(current_tags_df["tag"].to_list())

        # Only reprocess if tag profile has changed significantly (>20% new tags)
        new_tags = current_tags_set - processed_tags_set
        if len(new_tags) / len(current_tags_set) < 0.2:
            logger.info(f"Tag profile mostly unchanged for {username}, skipping")
            return {
                "path": "data/silver/candidate_similar_tag",
                "rows": 0,
                "table_name": "candidate_similar_tag",
            }
    except (TableNotFoundError, FileNotFoundError):
        # First run, process all
        pass

    tag_profile_df = tag_profile.collect()
    top_tags = tag_profile_df["tag"].to_list()

    logger.info(f"Using top {len(top_tags)} tags for {username}: {top_tags[:10]}...")

    # Get played track IDs to exclude
    played_track_ids = plays.select("track_id").unique().collect(engine="streaming")
    played_track_ids_set = set(played_track_ids["track_id"].to_list())

    # Collect tracks with tag associations
    # track_id -> {track_info, tags: [tag1, tag2, ...], ranks: [rank1, rank2, ...]}
    track_tag_map: dict[str, dict[str, Any]] = {}

    async with LastFMClient() as client:
        # Fetch all tag top tracks concurrently
        tag_tasks = [
            client.get_tag_top_tracks(tag, limit=tracks_per_tag) for tag in top_tags
        ]
        logger.info(f"Fetching top tracks for {len(top_tags)} tags...")
        all_tag_tracks = await _gather_with_progress(
            tag_tasks,
            description="Fetching tag top tracks",
        )
        logger.info("Completed fetching tag top tracks")

        # Process all results
        total_tracks_examined = 0
        tracks_filtered_by_already_played = 0
        tag_fetch_failures = 0

        for tag, top_tracks in zip(top_tags, all_tag_tracks):
            if isinstance(top_tracks, Exception):
                tag_fetch_failures += 1
                logger.warning(
                    f"Failed to fetch top tracks for tag '{tag}': {top_tracks}"
                )
                continue

            total_tracks_examined += len(top_tracks)

            for track in top_tracks:
                track_name = track.get("name")

                artist_info = track.get("artist", {})
                if isinstance(artist_info, dict):
                    artist_name = artist_info.get("name", "")
                else:
                    artist_name = str(artist_info) if artist_info else ""

                # Generate canonical track ID
                track_id = generate_canonical_track_id(track_name, artist_name)

                # Skip if already played
                if track_id in played_track_ids_set:
                    tracks_filtered_by_already_played += 1
                    continue

                # Add or update track in map
                if track_id not in track_tag_map:
                    track_tag_map[track_id] = {
                        "username": username,
                        "track_name": track_name,
                        "artist_name": artist_name,
                        "tags": [],
                    }

                track_tag_map[track_id]["tags"].append(tag)

        logger.info(
            f"Tag generation for {username}: "
            f"examined {total_tracks_examined} tracks from {len(top_tags)} tags, "
            f"filtered {tracks_filtered_by_already_played} already played, "
            f"{tag_fetch_failures} tag fetch failures, "
            f"result: {len(track_tag_map)} unique tracks before min_tag_matches filter"
        )

    # Filter tracks by minimum tag matches and calculate scores
    all_candidates = []
    tracks_filtered_by_min_tag_matches = 0

    for track_id, track_data in track_tag_map.items():
        tag_match_count = len(track_data["tags"])

        if tag_match_count < min_tag_matches:
            tracks_filtered_by_min_tag_matches += 1
            continue

        # Score: tag matches
        score = tag_match_count

        all_candidates.append(
            {
                "username": track_data["username"],
                "track_name": track_data["track_name"],
                "artist_name": track_data["artist_name"],
                "tag_match_count": tag_match_count,
                "score": score,
                "source_tags": ",".join(track_data["tags"]),
            }
        )

    logger.info(
        f"Filtered {tracks_filtered_by_min_tag_matches} by min_tag_matches ({min_tag_matches}), "
        f"final result: {len(all_candidates)} candidates"
    )

    # Convert to DataFrame
    if not all_candidates:
        df = pl.DataFrame(
            schema={
                "username": pl.String,
                "track_id": pl.String,
                "tag_match_count": pl.Int64,
                "score": pl.Float64,
                "source_tags": pl.String,
            }
        )
    else:
        # Resolve track IDs from dimension table
        df = _resolve_track_ids(all_candidates, delta_mgr_silver)
        df = (
            df.with_columns(
                pl.col("username").cast(pl.String),
                pl.col("track_id").cast(pl.String),
                pl.col("tag_match_count").cast(pl.Int64),
                pl.col("score").cast(pl.Float64),
                pl.col("source_tags").cast(pl.String),
            )
            .sort("score", descending=True)
            .select(
                [
                    "username",
                    "track_id",
                    "tag_match_count",
                    "score",
                    "source_tags",
                ]
            )
        )

    # Write to silver Delta table (overwrite for this user since we process full tag profile)
    write_meta = delta_mgr_silver.write_delta(
        df,
        table_name="candidate_similar_tag",
        mode="merge",
        predicate="s.username = t.username",
        partition_by="username",
    )

    return {
        "path": write_meta["path"],
        "rows": write_meta["rows"],
        "table_name": write_meta["table_name"],
    }


async def generate_deep_cut_candidates(
    username: str,
    min_listeners: int = 100,
    top_artists_count: int = 30,
) -> dict[str, Any]:
    """
    Generate track candidates from user's top artists that they haven't played (via Last.fm API).

    Strategy:
    1. Get artists user has played, ranked by play count
    2. Take top N artists (highest play count)
    3. For each artist, call artist.getTopAlbums
    4. Filter albums with minimum listener threshold (quality filter)
    5. For each album, call album.getInfo to get tracklist
    6. Collect all tracks from these albums
    7. Exclude tracks user has already played
    8. Save all candidates (incremental processing avoids reprocessing)

    Saves results to data/silver/candidate_deep_cut/ Delta table.

    Args:
        username: Target user
        min_listeners: Minimum album listeners (quality threshold)
        top_artists_count: Number of top artists to process

    Returns:
        Metadata dict with path, rows, table_name
    """
    delta_mgr_silver = PolarsDeltaIOManager(medallion_layer="silver")
    delta_mgr_gold = PolarsDeltaIOManager(medallion_layer="gold")

    # Load data
    plays_lf = delta_mgr_silver.read_delta("plays").filter(
        pl.col("username") == username
    )
    artists_lf = delta_mgr_silver.read_delta("artists")

    # Load user's artist play counts from gold table
    artist_play_counts_lf = (
        delta_mgr_gold.read_delta("artist_play_count")
        .filter(pl.col("username") == username)
        .select(["artist_id", "play_count"])
        .rename({"play_count": "user_artist_play_count"})
    )

    # Get user's top artists by play count (id + name + play count from gold)
    # Filter to only artists with valid data
    top_artists_lf = (
        artist_play_counts_lf.join(
            artists_lf.select("artist_id", "artist_name"),
            on="artist_id",
            how="left",
        )
        .filter(pl.col("artist_id").is_not_null() & pl.col("artist_name").is_not_null())
        .sort("user_artist_play_count", descending=True)
        .limit(top_artists_count)
    )

    # Incremental: exclude artists already processed
    try:
        processed_deep_lf = (
            delta_mgr_silver.read_delta("candidate_deep_cut")
            .filter(pl.col("username") == username)
            .select("source_artist_id")
            .unique()
            .rename({"source_artist_id": "artist_id"})
        )
        artists_to_process_df = top_artists_lf.join(
            processed_deep_lf, on="artist_id", how="anti"
        ).collect(engine="streaming")
    except (TableNotFoundError, FileNotFoundError):
        artists_to_process_df = top_artists_lf.collect(engine="streaming")

    total_artists = len(artists_to_process_df)
    logger.info(
        f"Found {total_artists} deep-cut source artists to process for {username}"
    )
    if total_artists > 50:
        logger.info("Sampling deep-cut artists to limit API requests")
        # Use random sampling instead of hash/mod to avoid always skipping same artists
        artists_to_process_df = artists_to_process_df.sample(fraction=0.5, seed=None)

    # Get played track IDs to exclude
    played_track_ids_df = (
        plays_lf.select("track_id").unique().collect(engine="streaming")
    )
    played_track_ids_set = set(played_track_ids_df["track_id"].to_list())

    # Collect candidate tracks
    all_candidates = []

    async with LastFMClient() as client:
        # Fetch all artist top albums concurrently
        artist_names = artists_to_process_df["artist_name"].to_list()
        artist_ids = artists_to_process_df["artist_id"].to_list()

        album_tasks = [
            client.get_artist_top_albums(artist_name, limit=15)
            for artist_name in artist_names
        ]
        logger.info(f"Fetching top albums for {len(artist_names)} artists...")
        all_artist_albums = await _gather_with_progress(
            album_tasks,
            description="Fetching artist albums",
        )
        logger.info("Completed fetching top albums")

        # Prepare album info requests
        album_info_tasks = []
        album_metadata = []  # Store context for each album request
        total_albums_received = 0
        albums_filtered_by_min_listeners = 0

        for idx, top_albums in enumerate(all_artist_albums):
            if isinstance(top_albums, Exception):
                logger.warning(
                    f"Failed to fetch albums for {artist_names[idx]}: {top_albums}"
                )
                continue

            # Type guard: top_albums is list here
            if not isinstance(top_albums, list):
                logger.warning(f"Unexpected type for albums: {type(top_albums)}")
                continue

            total_albums_received += len(top_albums)
            # Filter albums with reasonable listener counts
            filtered_albums = [
                album
                for album in top_albums
                if int(album.get("playcount", 0)) >= min_listeners
            ]
            albums_filtered_by_min_listeners += len(top_albums) - len(filtered_albums)

            # Get user's play count for this artist
            user_artist_play_count = artists_to_process_df["user_artist_play_count"][
                idx
            ]

            # Queue album info requests (limit to top 10 albums)
            for album in filtered_albums[:10]:
                album_name = album.get("name")
                if not album_name:
                    continue

                album_info_tasks.append(
                    client.get_album_info(album_name, artist_names[idx])
                )
                album_metadata.append(
                    {
                        "artist_name": artist_names[idx],
                        "artist_id": artist_ids[idx],
                        "album_name": album_name,
                        "album_playcount": int(album.get("playcount", 0)),
                        "user_artist_play_count": user_artist_play_count,
                    }
                )

        # Fetch all album info concurrently
        logger.info(
            f"Fetching track info for {len(album_info_tasks)} albums "
            f"(received {total_albums_received} total, filtered {albums_filtered_by_min_listeners} by min_listeners={min_listeners})..."
        )
        all_album_info = await _gather_with_progress(
            album_info_tasks,
            description="Fetching album track info",
        )
        logger.info("Completed fetching album info")

        # Process all results
        total_tracks_examined = 0
        tracks_filtered_by_album_listeners = 0
        tracks_filtered_by_already_played = 0
        album_info_failures = 0

        for album_info, metadata in zip(all_album_info, album_metadata):
            if isinstance(album_info, Exception):
                album_info_failures += 1
                continue

            tracks = album_info.get("tracks", {})

            # Handle different response formats
            if isinstance(tracks, dict):
                track_list = tracks.get("track", [])
            else:
                track_list = tracks

            if isinstance(track_list, dict):
                track_list = [track_list]

            total_tracks_examined += len(track_list)

            for track in track_list:
                track_name = track.get("name")

                album_listeners = metadata["album_playcount"]
                if album_listeners < min_listeners:
                    tracks_filtered_by_album_listeners += 1
                    continue

                # Generate canonical track ID
                track_id = generate_canonical_track_id(
                    track_name, metadata["artist_name"]
                )

                if track_id in played_track_ids_set:
                    tracks_filtered_by_already_played += 1
                    continue

                # Score = user's artist play count * album global popularity
                # This prioritizes deep cuts from artists the user actually likes
                score = float(metadata["user_artist_play_count"] * album_listeners)

                all_candidates.append(
                    {
                        "username": username,
                        "track_name": track_name,
                        "artist_name": metadata["artist_name"],
                        "album_name": metadata["album_name"],
                        "score": score,
                        "source_artist_id": metadata["artist_id"],
                    }
                )

        logger.info(
            f"Deep-cut generation for {username}: "
            f"examined {total_tracks_examined} tracks from {len(album_info_tasks)} albums, "
            f"filtered {tracks_filtered_by_album_listeners} by min_listeners ({min_listeners}), "
            f"filtered {tracks_filtered_by_already_played} already played, "
            f"{album_info_failures} album info failures, "
            f"result: {len(all_candidates)} candidates"
        )

    # Convert to DataFrame and deduplicate
    if not all_candidates:
        # Create empty DataFrame with correct schema
        df = pl.DataFrame(
            schema={
                "username": pl.String,
                "track_id": pl.String,
                "album_name": pl.String,
                "score": pl.Float64,
                "source_artist_id": pl.String,
            }
        )
    else:
        # Resolve track IDs from dimension table
        df = _resolve_track_ids(all_candidates, delta_mgr_silver)
        df = (
            df.with_columns(
                pl.col("username").cast(pl.String),
                pl.col("track_id").cast(pl.String),
                pl.col("album_name").cast(pl.String),
                pl.col("score").cast(pl.Float64),
                pl.col("source_artist_id").cast(pl.String),
            )
            .unique(subset=["track_id"])
            .sort("score", descending=True)
            .select(
                [
                    "username",
                    "track_id",
                    "album_name",
                    "score",
                    "source_artist_id",
                ]
            )
        )

    # Write to silver Delta table (merge for incrementality)
    write_meta = delta_mgr_silver.write_delta(
        df,
        table_name="candidate_deep_cut",
        mode="merge",
        predicate="s.track_id = t.track_id AND s.username = t.username",
        partition_by="username",
    )

    return {
        "path": write_meta["path"],
        "rows": write_meta["rows"],
        "table_name": write_meta["table_name"],
    }


def generate_old_favorites_candidates(
    username: str,
    min_days_since_last_play: int = 90,
    max_candidates: int = 500,
) -> dict[str, Any]:
    """
    Generate candidate tracks from user's old play history (tracks not played recently).

    Strategy:
    1. Get all tracks user has played with play timestamps
    2. Calculate days since last play for each track
    3. Filter tracks not played in min_days_since_last_play days
    4. Score by: play count (popularity) + recency decay (half-life)
    5. Return top N candidates

    This is the "remind" mode: rediscover forgotten favorites.

    Saves results to data/silver/candidate_old_favorites/ Delta table.

    Args:
        username: Target user
        min_days_since_last_play: Minimum days since last play (default 90)
        max_candidates: Maximum candidates to return (default 500)

    Returns:
        Metadata dict with path, rows, table_name
    """
    from datetime import timezone

    delta_mgr_silver = PolarsDeltaIOManager(medallion_layer="silver")

    # Load plays for user
    plays_lf = delta_mgr_silver.read_delta("plays").filter(
        pl.col("username") == username
    )

    # Load user dimension to get per-user half-life
    dim_users_lf = delta_mgr_silver.read_delta("dim_users")
    user_half_life = (
        dim_users_lf.filter(pl.col("username") == username)
        .select("user_half_life_days")
        .collect()
    )

    if len(user_half_life) == 0:
        logger.warning(
            f"No user dimension found for {username}, using default half-life"
        )
        half_life_days = 30.0  # MIN_HALF_LIFE_DAYS from dimensions.py
    else:
        half_life_days = float(user_half_life["user_half_life_days"].item())

    # Get tracks with play statistics
    now = datetime.now(timezone.utc)
    track_stats = (
        plays_lf.group_by("track_id")
        .agg(
            pl.len().alias("play_count"),
            pl.col("scrobbled_at_utc").max().alias("last_played_at"),
        )
        .with_columns(
            # Calculate days since last play
            ((pl.lit(now.timestamp()) - pl.col("last_played_at").dt.epoch()) / 86400)
            .cast(pl.Int64)
            .alias("days_since_last_play")
        )
        .filter(pl.col("days_since_last_play") >= min_days_since_last_play)
    )

    # Score with half-life decay: play_count * exp(-days / half_life)
    # Use per-user half-life (calculated as listening_span / 3, min 30 days)
    scored_tracks = track_stats.with_columns(
        (
            pl.col("play_count")
            * pl.lit(2).pow(-pl.col("days_since_last_play") / half_life_days)
        ).alias("score")
    )

    # Get top candidates
    candidates = (
        scored_tracks.sort("score", descending=True)
        .limit(max_candidates)
        .with_columns(pl.lit(username).alias("username"))
        .select(["username", "track_id", "score", "play_count", "days_since_last_play"])
        .collect()
    )

    logger.info(
        f"Found {len(candidates)} old favorite candidates for {username} "
        f"(not played in {min_days_since_last_play}+ days)"
    )

    # Write to silver Delta table (overwrite per user)
    if len(candidates) == 0:
        # Create empty DataFrame with correct schema
        candidates = pl.DataFrame(
            schema={
                "username": pl.String,
                "track_id": pl.String,
                "score": pl.Float64,
                "play_count": pl.Int64,
                "days_since_last_play": pl.Int64,
            }
        )

    write_meta = delta_mgr_silver.write_delta(
        candidates,
        table_name="candidate_old_favorites",
        mode="merge",
        predicate="s.username = t.username",
        partition_by="username",
    )

    return {
        "path": write_meta["path"],
        "rows": write_meta["rows"],
        "table_name": write_meta["table_name"],
    }


def merge_candidate_sources(
    username: str,
    tracks_per_source: int = 500,
) -> dict[str, Any]:
    """
    Merge candidate sources from silver tables into single gold table.

    Reads candidates from silver tables and creates a unified candidate table
    with one-hot encoded source columns:
    - similar_artist: bool
    - similar_tag: bool
    - deep_cut_same_artist: bool
    - old_favorite: bool

    Normalizes scores within each source (min-max to 0-1 range), limits each source
    to top N tracks, filters out tracks user has already played (except old_favorites),
    and deduplicates tracks that appear in multiple sources.

    Note: Metadata columns (track_name, artist_name, etc.) are NOT included in the gold
    table, as recommendations may not exist in dimension tables yet.

    Saves result to data/gold/track_candidates/ Delta table.

    Args:
        username: Target user
        tracks_per_source: Maximum tracks to include from each source (default 500)

    Returns:
        Metadata dict with path, rows, table_name
    """
    # IO managers for silver and gold layers
    delta_mgr_silver = PolarsDeltaIOManager(medallion_layer="silver")
    delta_mgr_gold = PolarsDeltaIOManager(medallion_layer="gold")

    # Load from silver candidate tables
    similar_artists_lf = delta_mgr_silver.read_delta("candidate_similar_artist")
    similar_tags_lf = delta_mgr_silver.read_delta("candidate_similar_tag")
    deep_cuts_lf = delta_mgr_silver.read_delta("candidate_deep_cut")

    # Try to load old_favorites (may not exist initially)
    try:
        old_favorites_lf = delta_mgr_silver.read_delta("candidate_old_favorites")
    except (TableNotFoundError, FileNotFoundError):
        old_favorites_lf = None

    # Load plays to filter out already-played tracks
    plays_lf = delta_mgr_silver.read_delta("plays")
    played_track_ids = (
        plays_lf.filter(pl.col("username") == username).select("track_id").unique()
    )

    # Load tracks dimension for streaming links
    tracks_dim_lf = delta_mgr_silver.read_delta("tracks")

    # Process each source: normalize scores, limit, and filter played tracks
    def process_source(
        source_lf,
        source_name: str,
        filter_played: bool = True,
    ):
        """Normalize scores using percentile ranks, limit to top N, optionally filter played tracks."""
        filtered = source_lf.filter(pl.col("username") == username)

        # Filter out already-played tracks (except for old_favorites which ARE played tracks)
        if filter_played:
            filtered = filtered.join(played_track_ids, on="track_id", how="anti")

        # Normalize scores using percentile rank (0-1) to avoid outlier domination
        # This is more robust than min-max normalization
        filtered = (
            filtered.with_columns(pl.col("score").rank(method="average").alias("rank"))
            .with_columns(
                (pl.col("rank") / pl.col("rank").max()).alias("normalized_score")
            )
            .drop("rank")
        )

        # Limit to top N tracks by original score
        filtered = filtered.sort("score", descending=True).limit(tracks_per_source)

        # Add source flags
        filtered = filtered.with_columns(
            pl.lit(source_name == "similar_artist").alias("similar_artist"),
            pl.lit(source_name == "similar_tag").alias("similar_tag"),
            pl.lit(source_name == "deep_cut").alias("deep_cut_same_artist"),
            pl.lit(source_name == "old_favorite").alias("old_favorite"),
        )

        # Select standard columns (no metadata columns)
        return filtered.select(
            [
                "username",
                "track_id",
                "normalized_score",
                "similar_artist",
                "similar_tag",
                "deep_cut_same_artist",
                "old_favorite",
            ]
        )

    similar_artists_processed = process_source(
        similar_artists_lf,
        "similar_artist",
        filter_played=True,
    )
    similar_tags_processed = process_source(
        similar_tags_lf,
        "similar_tag",
        filter_played=True,
    )
    deep_cuts_processed = process_source(
        deep_cuts_lf,
        "deep_cut",
        filter_played=True,
    )

    # Process old_favorites if available (don't filter played - they ARE played tracks)
    sources_to_concat = [
        similar_artists_processed,
        similar_tags_processed,
        deep_cuts_processed,
    ]

    if old_favorites_lf is not None:
        old_favorites_processed = process_source(
            old_favorites_lf,
            "old_favorite",
            filter_played=False,  # Don't filter - these ARE old plays
        )
        sources_to_concat.append(old_favorites_processed)

    # Concatenate all sources
    all_candidates = pl.concat(sources_to_concat)

    # Deduplicate by track_id, aggregating source flags and summing normalized scores
    # Summing scores rewards tracks recommended by multiple generators (consensus)
    # Note: We don't join with tracks dimension for metadata, as candidates may not exist there yet
    merged_lf = (
        all_candidates.group_by("username", "track_id")
        .agg(
            [
                pl.max("similar_artist").alias("similar_artist"),
                pl.max("similar_tag").alias("similar_tag"),
                pl.max("deep_cut_same_artist").alias("deep_cut_same_artist"),
                pl.max("old_favorite").alias("old_favorite"),
                pl.sum("normalized_score").alias("score"),  # Sum to reward consensus
            ]
        )
        .sort("score", descending=True)
    )

    # Note: We no longer enrich tracks here to avoid circular dependency.
    # New tracks discovered in candidates will be enriched by the weekly
    # dimension update DAG which reads from gold/track_candidates.
    # This creates a 1-7 day delay for full metadata, which is acceptable.

    # Join with tracks dimension to add streaming links and track metadata
    tracks_with_links = tracks_dim_lf.select(
        [
            "track_id",
            "youtube_url",
            "spotify_url",
        ]
    )

    merged_with_links = merged_lf.join(
        tracks_with_links,
        on="track_id",
        how="left",
    )

    # CRITICAL: Deduplicate by youtube_url at the gold layer
    # Different track versions (e.g., "Song" vs "Song (Remastered)") may map to the same
    # YouTube video. We deduplicate here to ensure downstream consumers (Streamlit app)
    # get clean data with no duplicate videos. Keep highest-scored track for each video.
    # Also filter out tracks without youtube_url since they cannot be played.
    merged_deduped = (
        merged_with_links.filter(pl.col("youtube_url").is_not_null())
        .sort("score", descending=True)
        .unique(subset=["username", "youtube_url"], keep="first")
    )

    # Write gold candidate table with streaming links (clean, deduplicated data)
    df = merged_deduped.collect()  # type: ignore[attr-defined]
    write_meta = delta_mgr_gold.write_delta(
        df,
        table_name="track_candidates",
        mode="overwrite",
        partition_by="username",
    )

    return {
        "path": write_meta["path"],
        "rows": write_meta["rows"],
        "table_name": write_meta["table_name"],
    }
