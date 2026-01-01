"""
Candidate track generation for music recommendation system.

Generates candidate track lists using Last.fm API similarity:
- Similar artist tracks: Top tracks from artists similar to user's plays (via artist.getSimilar)
- Similar tag tracks: Top tracks with tags matching/similar to user's library (via tag.getSimilar)
- Deep cut tracks: Obscure tracks from top albums by user's favorite artists (via artist.getTopAlbums)

Returns DataFrames for silver-layer storage; DAG consolidates into single gold table.
"""
# todo: review and adjust the 2 other functions that have bugs:
# similar tag error
# bad id in deep cuts

from deltalake.exceptions import TableNotFoundError
from typing import Any

import polars as pl

from music_airflow.lastfm_client import LastFMClient
from music_airflow.utils.polars_io_manager import PolarsDeltaIOManager

__all__ = [
    "generate_similar_artist_candidates",
    "generate_similar_tag_candidates",
    "generate_deep_cut_candidates",
    "merge_candidate_sources",
]


def generate_similar_artist_candidates(
    username: str,
    min_listeners: int = 1000,
    max_candidates_per_artist: int = 10,
    max_total_candidates: int = 500,
    similarity_threshold: float = 0.9,
    artist_sample_rate: float = 0.3,
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

    Saves results to data/silver/candidate_similar_artist/ Delta table.

    Args:
        username: Target user
        min_listeners: Minimum listeners for candidate tracks (quality filter)
        max_candidates_per_artist: Max tracks per similar artist
        max_total_candidates: Maximum total candidates to return
        similarity_threshold: Exclude artists with match > this (likely duplicates)
        artist_sample_rate: Fraction of user's artists to sample (0.3 = 30%)

    Returns:
        Metadata dict with path, rows, table_name
    """
    client = LastFMClient()
    delta_mgr_silver = PolarsDeltaIOManager(medallion_layer="silver")

    # Load user's plays
    plays_lf = delta_mgr_silver.read_delta("plays").filter(
        pl.col("username") == username
    )
    tracks_lf = delta_mgr_silver.read_delta("tracks")
    artists_lf = delta_mgr_silver.read_delta("artists")

    # Get unique artists user has played (id + name)
    played_artists = (
        plays_lf.select("track_id")
        .unique()
        .join(tracks_lf.select("track_id", "artist_id"), on="track_id", how="left")
        .join(
            artists_lf.select("artist_id", "artist_name"),
            on="artist_id",
            how="left",
        )
        .select(["artist_id", "artist_name"])
        .unique()
        .filter(pl.col("artist_id").is_not_null() & pl.col("artist_name").is_not_null())
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

    total_artists = artists_to_process.select(pl.len()).collect().item()
    print(f"Found {total_artists} source artists to process for {username}")

    # Sampling only if we have more than 50 artists to process
    if total_artists > 50:
        print("Sampling from artist list to limit API requests")
        artists_to_process = artists_to_process.filter(
            pl.col("artist_name").hash().mod(100) < int(artist_sample_rate * 100)
        ).collect(engine="streaming")
        sampled_count = artists_to_process.select(pl.len()).item()
        # Update total count to reflect sampled set for accurate progress logs
        total_artists = sampled_count
        print(f"Remaining artists after sampling: {sampled_count}")
    else:
        artists_to_process = artists_to_process.collect(engine="streaming")

    # Prepare set of played track_ids for exclusion
    played_track_ids_df = (
        plays_lf.select("track_id").unique().collect(engine="streaming")
    )
    played_track_ids_set = set(played_track_ids_df["track_id"].to_list())

    # Collect candidate tracks
    all_candidates: list[dict[str, Any]] = []
    processed_count = 0

    for artist_name, artist_id in zip(
        artists_to_process["artist_name"].to_list(),
        artists_to_process["artist_id"].to_list(),
    ):
        # Get similar artists
        similar_artists = client.get_similar_artists(artist_name, limit=30)

        # Filter out clones/false positives
        filtered_similar = [
            a
            for a in similar_artists
            if float(a.get("match", 0)) <= similarity_threshold
        ]

        # Get top tracks from each similar artist (limit to top 10 similar)
        for similar_artist in filtered_similar[:10]:
            similar_artist_name = similar_artist.get("name")
            if not similar_artist_name:
                continue

            top_tracks = client.get_artist_top_tracks(
                similar_artist_name, limit=max_candidates_per_artist
            )

            for track in top_tracks:
                track_name = track.get("name")
                track_mbid = track.get("mbid")
                if track_mbid == "":
                    track_mbid = None
                artist_info = track.get("artist", {})
                if isinstance(artist_info, dict):
                    artist_name_track = artist_info.get("name", similar_artist_name)
                    artist_mbid = artist_info.get("mbid")
                else:
                    artist_name_track = similar_artist_name
                    artist_mbid = None

                listeners = int(track.get("listeners", 0))
                playcount = int(track.get("playcount", 0))

                # Apply quality filter
                if listeners < min_listeners:
                    continue

                # Create track ID consistent with plays: prefer MBID else "track|artist"
                track_id = (
                    track_mbid if track_mbid else f"{track_name}|{artist_name_track}"
                )

                # Skip if already played
                if track_id in played_track_ids_set:
                    continue

                all_candidates.append(
                    {
                        "username": username,
                        "track_id": track_id,
                        "track_name": track_name,
                        "track_mbid": track_mbid,
                        "artist_name": artist_name_track,
                        "artist_mbid": artist_mbid,
                        "listeners": listeners,
                        "playcount": playcount,
                        "score": playcount,
                        "source_artist_name": artist_name,
                        "source_artist_id": artist_id,
                    }
                )

        processed_count += 1
        if processed_count % 5 == 0 or processed_count == total_artists:
            print(
                f"Processed {processed_count}/{total_artists} similar-artist sources for {username}"
            )

    # Convert to DataFrame and deduplicate
    if not all_candidates:
        # Create empty DataFrame with correct schema
        df = pl.DataFrame(
            schema={
                "username": pl.String,
                "track_id": pl.String,
                "track_name": pl.String,
                "track_mbid": pl.String,
                "artist_name": pl.String,
                "artist_mbid": pl.String,
                "listeners": pl.Int64,
                "playcount": pl.Int64,
                "score": pl.Int64,
                "source_artist_name": pl.String,
                "source_artist_id": pl.String,
            }
        )
    else:
        df = pl.DataFrame(all_candidates)
        df = (
            df.unique(subset=["track_id"])  # dedup by track
            .sort("score", descending=True)
            .limit(max_total_candidates)
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


def generate_similar_tag_candidates(
    username: str,
    min_listeners: int = 1000,
    max_candidates_per_tag: int = 10,
    max_total_candidates: int = 500,
    tag_sample_rate: float = 0.5,
) -> dict[str, Any]:
    """
    Generate candidate tracks with tags similar to user's library (via Last.fm API).

    Strategy:
    1. Extract tags from user's played tracks/artists
    2. Sample tags for diversity
    3. For each tag, call tag.getSimilar to expand tag set
    4. For each tag (user + similar), call tag.getTopTracks
    5. Exclude tracks user has already played

    Saves results to data/silver/candidate_similar_tag/ Delta table.

    Args:
        username: Target user
        min_listeners: Minimum listeners for candidate tracks
        max_candidates_per_tag: Max tracks per tag
        max_total_candidates: Maximum number of candidates
        tag_sample_rate: Fraction of tags to sample (0.5 = 50%)

    Returns:
        Metadata dict with path, rows, table_name
    """
    client = LastFMClient()
    delta_mgr_silver = PolarsDeltaIOManager(medallion_layer="silver")

    # Load data
    plays_lf = delta_mgr_silver.read_delta("plays").filter(
        pl.col("username") == username
    )
    tracks_lf = delta_mgr_silver.read_delta("tracks")
    artists_lf = delta_mgr_silver.read_delta("artists")

    # Get user's played tracks and extract tags
    # First join with tracks to get artist id, then join with artists to get tags
    plays_with_tracks = (
        plays_lf.select("track_id")
        .unique()
        .join(
            tracks_lf.select("track_id", "artist_id", "tags"),
            on="track_id",
            how="left",
        )
    )

    plays_with_artist_tags = (
        plays_with_tracks.select("artist_id")
        .unique()
        .join(
            artists_lf.select("artist_id", "tags"),
            on="artist_id",
            how="left",
        )
        .select("tags")
    )

    plays_with_track_tags = plays_with_tracks.select("tags")

    # Combine track and artist tags lazily, explode to individual tags
    user_tags = pl.concat(
        [plays_with_track_tags, plays_with_artist_tags], how="vertical"
    )
    user_tags = (
        user_tags.filter(pl.col("tags").is_not_null() & (pl.col("tags") != ""))
        .with_columns(pl.col("tags").str.split(",").alias("tag_list"))
        .explode("tag_list")
        .select(pl.col("tag_list").str.strip_chars().alias("tag"))
        .unique()
    )

    # Incremental: exclude tags already processed for this user
    try:
        existing_tag_lf = delta_mgr_silver.read_delta("candidate_similar_tag").filter(
            pl.col("username") == username
        )
        # Guard against older schema without 'source_tag'
        existing_schema = existing_tag_lf.collect_schema()
        if "source_tag" in existing_schema:
            processed_tags = (
                existing_tag_lf.select("source_tag")
                .unique()
                .rename({"source_tag": "tag"})
            )
            tags_to_process = user_tags.join(processed_tags, on="tag", how="anti")
        else:
            tags_to_process = user_tags
    except (TableNotFoundError, FileNotFoundError):
        tags_to_process = user_tags

    # Count tags and optionally sample (>50)
    total_tags = tags_to_process.select(pl.len()).collect(engine="streaming").item()
    print(f"Found {total_tags} source tags to process for {username}")
    if total_tags > 50:
        print("Sampling from tag list to limit API requests")
        original_tags = tags_to_process
        tags_to_process = tags_to_process.filter(
            pl.col("tag").hash().mod(100) < int(tag_sample_rate * 100)
        ).collect(engine="streaming")
        total_tags = tags_to_process.select(pl.len()).item()
        if total_tags == 0:
            print("Sampling produced 0 tags; falling back to top 50 unsampled tags")
            tags_to_process = original_tags.collect(engine="streaming").head(50)
            total_tags = tags_to_process.select(pl.len()).item()
        print(f"Remaining tags after sampling: {total_tags}")
    else:
        tags_to_process = tags_to_process.collect(engine="streaming")

    # Expand each source tag with its similar tags on the fly
    all_candidates = []
    processed_count = 0

    # Get played track IDs to exclude
    played_track_ids_df = (
        plays_lf.select("track_id").unique().collect(engine="streaming")
    )
    played_track_ids_set = set(played_track_ids_df["track_id"].to_list())

    for tag in tags_to_process["tag"].to_list():
        similar_tags = client.get_similar_tags(tag)
        expanded = [tag]
        for similar_tag in similar_tags[:5]:
            tag_name = similar_tag.get("name")
            if tag_name:
                expanded.append(tag_name)

        for t in expanded:
            top_tracks = client.get_tag_top_tracks(t, limit=max_candidates_per_tag)
            for track in top_tracks:
                track_name = track.get("name")
                track_mbid = track.get("mbid", "")
                if track_mbid == "":
                    track_mbid = None
                artist_info = track.get("artist", {})
                if isinstance(artist_info, dict):
                    artist_name = artist_info.get("name", "")
                    artist_mbid = artist_info.get("mbid", "")
                else:
                    artist_name = str(artist_info) if artist_info else ""
                    artist_mbid = ""

                listeners = int(track.get("listeners", 0))
                playcount = int(track.get("playcount", 0))
                if listeners < min_listeners:
                    continue

                track_id = track_mbid if track_mbid else f"{track_name}|{artist_name}"
                if track_id in played_track_ids_set:
                    continue

                all_candidates.append(
                    {
                        "username": username,
                        "track_id": track_id,
                        "track_name": track_name,
                        "track_mbid": track_mbid,
                        "artist_name": artist_name,
                        "artist_mbid": artist_mbid,
                        "listeners": listeners,
                        "playcount": playcount,
                        "score": playcount,
                        "source_tag": tag,
                    }
                )

        processed_count += 1
        if processed_count % 5 == 0 or processed_count == total_tags:
            print(
                f"Processed {processed_count}/{total_tags} source tags for {username}"
            )

    # Convert to DataFrame and deduplicate
    if not all_candidates:
        # Create empty DataFrame with correct schema
        df = pl.DataFrame(
            schema={
                "username": pl.String,
                "track_id": pl.String,
                "track_name": pl.String,
                "track_mbid": pl.String,
                "artist_name": pl.String,
                "artist_mbid": pl.String,
                "listeners": pl.Int64,
                "playcount": pl.Int64,
                "score": pl.Int64,
                "source_tag": pl.String,
            }
        )
    else:
        df = pl.DataFrame(all_candidates)
        df = (
            df.unique(subset=["track_id"])
            .sort("score", descending=True)
            .limit(max_total_candidates)
        )

    # Ensure stable schema types (avoid Null dtypes)
    df = df.with_columns(
        pl.col("username").cast(pl.String),
        pl.col("track_id").cast(pl.String),
        pl.col("track_name").cast(pl.String),
        pl.col("track_mbid").cast(pl.String),
        pl.col("artist_name").cast(pl.String),
        pl.col("artist_mbid").cast(pl.String),
        pl.col("listeners").cast(pl.Int64),
        pl.col("playcount").cast(pl.Int64),
        pl.col("score").cast(pl.Int64),
        pl.col("source_tag").cast(pl.String),
    )

    # Write to silver Delta table (merge for incrementality)
    write_meta = delta_mgr_silver.write_delta(
        df,
        table_name="candidate_similar_tag",
        mode="merge",
        predicate="s.track_id = t.track_id AND s.username = t.username",
        partition_by="username",
    )

    return {
        "path": write_meta["path"],
        "rows": write_meta["rows"],
        "table_name": write_meta["table_name"],
    }


def generate_deep_cut_candidates(
    username: str,
    min_listeners: int = 100,
    max_listeners: int = 50000,
    max_candidates: int = 300,
    top_artists_count: int = 30,
) -> dict[str, Any]:
    """
    Generate deep cut (obscure) track candidates from user's top artists (via Last.fm API).

    Strategy:
    1. Get artists user has played, ranked by play count
    2. Take top N artists (highest play count)
    3. For each artist, call artist.getTopAlbums
    4. Filter albums with reasonable listener counts (exclude bad editions)
    5. For each album, call album.getInfo to get tracklist
    6. Find lesser-played tracks from these albums
    7. Exclude tracks user has already played

    Saves results to data/silver/candidate_deep_cut/ Delta table.

    Args:
        username: Target user
        min_listeners: Minimum listeners (quality threshold)
        max_listeners: Maximum listeners (obscurity threshold)
        max_candidates: Maximum number of candidates
        top_artists_count: Number of top artists to process

    Returns:
        Metadata dict with path, rows, table_name
    """
    client = LastFMClient()
    delta_mgr_silver = PolarsDeltaIOManager(medallion_layer="silver")

    # Load data
    plays_lf = delta_mgr_silver.read_delta("plays").filter(
        pl.col("username") == username
    )
    tracks_lf = delta_mgr_silver.read_delta("tracks")
    artists_lf = delta_mgr_silver.read_delta("artists")

    # Get user's top artists by play count (id + name)
    top_artists_lf = (
        plays_lf.select("track_id")
        .join(tracks_lf.select("track_id", "artist_id"), on="track_id", how="left")
        .join(
            artists_lf.select("artist_id", "artist_name"),
            on="artist_id",
            how="left",
        )
        .filter(pl.col("artist_id").is_not_null() & pl.col("artist_name").is_not_null())
        .group_by(["artist_id", "artist_name"])
        .agg(pl.len().alias("play_count"))
        .sort("play_count", descending=True)
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
    print(f"Found {total_artists} deep-cut source artists to process for {username}")
    if total_artists > 50:
        print("Sampling deep-cut artists to limit API requests")
        artists_to_process_df = artists_to_process_df.filter(
            pl.col("artist_name").hash().mod(100) < 50
        )

    # Get played track IDs to exclude
    played_track_ids_df = (
        plays_lf.select("track_id").unique().collect(engine="streaming")
    )
    played_track_ids_set = set(played_track_ids_df["track_id"].to_list())

    # Collect candidate tracks
    all_candidates = []

    processed_count = 0
    for artist_name, artist_id in zip(
        artists_to_process_df["artist_name"].to_list(),
        artists_to_process_df["artist_id"].to_list(),
    ):
        # Get top albums for artist
        top_albums = client.get_artist_top_albums(artist_name, limit=15)

        # Filter albums with reasonable listener counts
        filtered_albums = [
            album
            for album in top_albums
            if int(album.get("playcount", 0)) >= min_listeners
        ]

        # Process each album
        for album in filtered_albums[:10]:  # Limit to top 10 albums
            album_name = album.get("name")
            if not album_name:
                continue

            # Get album info with tracklist
            album_info = client.get_album_info(album_name, artist_name)
            tracks = album_info.get("tracks", {})

            # Handle different response formats
            if isinstance(tracks, dict):
                track_list = tracks.get("track", [])
            else:
                track_list = tracks

            if isinstance(track_list, dict):
                track_list = [track_list]

            for track in track_list:
                track_name = track.get("name")
                track_mbid = track.get("mbid", "")
                if track_mbid == "":
                    track_mbid = None

                album_listeners = int(album.get("playcount", 0))
                if not (min_listeners <= album_listeners <= max_listeners):
                    continue

                track_id = track_mbid if track_mbid else f"{track_name}|{artist_name}"
                if track_id in played_track_ids_set:
                    continue

                artist_mbid = (
                    album.get("artist", {}).get("mbid", "")
                    if isinstance(album.get("artist"), dict)
                    else ""
                )

                all_candidates.append(
                    {
                        "username": username,
                        "track_id": track_id,
                        "track_name": track_name,
                        "track_mbid": track_mbid,
                        "artist_name": artist_name,
                        "artist_mbid": artist_mbid,
                        "album_name": album_name,
                        "listeners": album_listeners,
                        "playcount": album_listeners,
                        "score": 1.0 / (album_listeners + 1),
                        "source_artist_name": artist_name,
                        "source_artist_id": artist_id,
                    }
                )

        processed_count += 1
        if processed_count % 5 == 0 or processed_count == total_artists:
            print(
                f"Processed {processed_count}/{total_artists} deep-cut artists for {username}"
            )

    # Convert to DataFrame and deduplicate
    if not all_candidates:
        # Create empty DataFrame with correct schema
        df = pl.DataFrame(
            schema={
                "username": pl.String,
                "track_id": pl.String,
                "track_name": pl.String,
                "track_mbid": pl.String,
                "artist_name": pl.String,
                "artist_mbid": pl.String,
                "album_name": pl.String,
                "listeners": pl.Int64,
                "playcount": pl.Int64,
                "score": pl.Float64,
                "source_artist_name": pl.String,
                "source_artist_id": pl.String,
            }
        )
    else:
        df = pl.DataFrame(all_candidates)
        df = (
            df.unique(subset=["track_id"])
            .sort("score", descending=True)
            .limit(max_candidates)
        )

    # Ensure stable schema types (avoid Null dtypes)
    df = df.with_columns(
        pl.col("username").cast(pl.String),
        pl.col("track_id").cast(pl.String),
        pl.col("track_name").cast(pl.String),
        pl.col("track_mbid").cast(pl.String),
        pl.col("artist_name").cast(pl.String),
        pl.col("artist_mbid").cast(pl.String),
        pl.col("album_name").cast(pl.String),
        pl.col("listeners").cast(pl.Int64),
        pl.col("playcount").cast(pl.Int64),
        pl.col("score").cast(pl.Float64),
        pl.col("source_artist_name").cast(pl.String),
        pl.col("source_artist_id").cast(pl.String),
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


def merge_candidate_sources(username: str) -> dict[str, Any]:
    """
    Merge candidate sources from silver tables into single gold table.

    Reads candidates from silver tables and creates a unified candidate table
    with one-hot encoded source columns:
    - similar_artist: bool
    - similar_tag: bool
    - deep_cut_same_artist: bool

    Deduplicates tracks that appear in multiple sources, preserving all source flags.
    Saves result to data/gold/track_candidates/ Delta table.

    Args:
        username: Target user

    Returns:
        Metadata dict with path, rows, table_name
    """
    # IO managers for silver and gold layers
    delta_mgr_silver = PolarsDeltaIOManager(medallion_layer="silver")
    delta_mgr_gold = PolarsDeltaIOManager(medallion_layer="gold")

    # Load from silver tables
    similar_artists_lf = delta_mgr_silver.read_delta("candidate_similar_artist")
    similar_tags_lf = delta_mgr_silver.read_delta("candidate_similar_tag")
    deep_cuts_lf = delta_mgr_silver.read_delta("candidate_deep_cut")

    # Normalize schemas - deep_cuts has album_name, others don't
    # Add missing columns with defaults
    similar_artists_typed = similar_artists_lf.with_columns(
        pl.lit(None).cast(pl.String).alias("album_name")
        if "album_name" not in similar_artists_lf.collect_schema().names()
        else pl.col("album_name"),
        pl.lit(True).alias("similar_artist"),
        pl.lit(False).alias("similar_tag"),
        pl.lit(False).alias("deep_cut_same_artist"),
    )

    similar_tags_typed = similar_tags_lf.with_columns(
        pl.lit(None).cast(pl.String).alias("album_name")
        if "album_name" not in similar_tags_lf.collect_schema().names()
        else pl.col("album_name"),
        pl.lit(False).alias("similar_artist"),
        pl.lit(True).alias("similar_tag"),
        pl.lit(False).alias("deep_cut_same_artist"),
    )

    deep_cuts_typed = deep_cuts_lf.with_columns(
        pl.lit(False).alias("similar_artist"),
        pl.lit(False).alias("similar_tag"),
        pl.lit(True).alias("deep_cut_same_artist"),
    )

    # Concatenate all sources
    all_candidates = pl.concat(
        [similar_artists_typed, similar_tags_typed, deep_cuts_typed]
    )

    # Deduplicate by track_id, aggregating source flags with max (True if any source had it)
    merged = (
        all_candidates.group_by(
            "username",
            "track_id",
            "track_name",
            "track_mbid",
            "artist_name",
            "artist_mbid",
        )
        .agg(
            pl.first("album_name").alias("album_name"),
            pl.first("listeners").alias("listeners"),
            pl.first("playcount").alias("playcount"),
            pl.max("similar_artist").alias("similar_artist"),
            pl.max("similar_tag").alias("similar_tag"),
            pl.max("deep_cut_same_artist").alias("deep_cut_same_artist"),
            pl.max("score").alias("max_score"),
        )
        .sort("max_score", descending=True)
        .drop("max_score")
    )

    # Write to gold Delta table
    df = merged.collect()
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
