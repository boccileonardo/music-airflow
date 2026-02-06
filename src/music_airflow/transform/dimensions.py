"""
Transform dimension data (tracks, artists) from bronze to silver layer.

Cleans and structures raw Last.fm metadata for analytics.
"""

from datetime import datetime
import logging
from typing import Any

import polars as pl
from airflow.exceptions import AirflowSkipException
from deltalake.exceptions import TableNotFoundError

from music_airflow.utils.polars_io_manager import JSONIOManager, PolarsDeltaIOManager
from music_airflow.utils.text_normalization import (
    generate_canonical_track_id_expr,
    generate_canonical_artist_id_expr,
    is_music_video_expr,
)

logger = logging.getLogger(__name__)

# Minimum half-life for new users (30 days)
MIN_HALF_LIFE_DAYS = 30.0

__all__ = [
    "transform_tracks_to_silver",
    "transform_artists_to_silver",
    "compute_dim_users",
    "_transform_tracks_raw_to_structured",
    "_transform_artists_raw_to_structured",
    "_deduplicate_tracks",
    "_deduplicate_artists",
]


def transform_tracks_to_silver(fetch_metadata: dict[str, Any]) -> dict[str, Any]:
    """
    Transform raw track metadata from bronze to structured Delta table in silver layer.

    Reads raw JSON track data from bronze, extracts and flattens relevant fields,
    and merges into silver Delta table using canonical track_id for deduplication.

    Args:
        fetch_metadata: Metadata from extraction containing filename, tracks_fetched

    Returns:
        Metadata dict with:
        - path: Path to Delta table
        - table_name: "tracks"
        - rows: Number of tracks processed
        - schema: Column names and types
        - format: "delta"
        - medallion_layer: "silver"
        - mode: "merge"
        - merge_metrics: Dict with merge statistics

    Raises:
        AirflowSkipException: If no tracks to process
    """
    # Read raw JSON using Polars
    io_manager = JSONIOManager(medallion_layer="bronze")
    filename = fetch_metadata["filename"]
    tracks_lf = io_manager.read_json(filename)

    # Apply transformations (includes streaming links from bronze enrichment)
    df = _transform_tracks_raw_to_structured(tracks_lf)

    df = _deduplicate_tracks(df)
    # Note: Removed _union_enriched_recommended_tracks() to eliminate circular dependency
    # New tracks from candidates will be discovered via fetch_tracks() reading gold/track_candidates

    # Write to silver layer Delta table with merge/upsert
    table_name = "tracks"
    predicate = """
        s.track_id = t.track_id
    """

    silver_io = PolarsDeltaIOManager(medallion_layer="silver")
    write_metadata = silver_io.write_delta(
        df,
        table_name=table_name,
        mode="merge",
        predicate=predicate,
    )

    return write_metadata


def transform_artists_to_silver(fetch_metadata: dict[str, Any]) -> dict[str, Any]:
    """
    Transform raw artist metadata from bronze to structured Delta table in silver layer.

    Reads raw JSON artist data from bronze, extracts and flattens relevant fields,
    and merges into silver Delta table using canonical artist_id for deduplication.

    Args:
        fetch_metadata: Metadata from extraction containing filename, artists_fetched

    Returns:
        Metadata dict with:
        - path: Path to Delta table
        - table_name: "artists"
        - rows: Number of artists processed
        - schema: Column names and types
        - format: "delta"
        - medallion_layer: "silver"
        - mode: "merge"
        - merge_metrics: Dict with merge statistics

    Raises:
        AirflowSkipException: If no artists to process
    """
    # Read raw JSON using Polars
    io_manager = JSONIOManager(medallion_layer="bronze")
    filename = fetch_metadata["filename"]
    artists_lf = io_manager.read_json(filename)

    # Apply transformations
    df = _transform_artists_raw_to_structured(artists_lf)
    df = _deduplicate_artists(df)

    # Exclude artists with very low listener counts (likely invalid)
    df = df.filter(pl.col("listeners") >= 1000)

    # Write to silver layer Delta table with merge/upsert
    table_name = "artists"
    predicate = """
       s.artist_id = t.artist_id
    """

    silver_io = PolarsDeltaIOManager(medallion_layer="silver")
    write_metadata = silver_io.write_delta(
        df,
        table_name=table_name,
        mode="merge",
        predicate=predicate,
    )

    return write_metadata


def _transform_tracks_raw_to_structured(raw_tracks: pl.LazyFrame) -> pl.LazyFrame:
    """
    Transform raw Last.fm track metadata to structured format.

    Extracts relevant fields from nested JSON structure:
    - Basic info: name, artist, album, duration
    - Popularity metrics: listeners, playcount
    - Tags (flattened to comma-separated string)
    - URLs

    Args:
        raw_tracks: LazyFrame with raw JSON structure from Last.fm API

    Returns:
        Transformed LazyFrame with flattened columns
    """
    # Check input schema for streaming links BEFORE transformations
    input_schema_names = raw_tracks.collect_schema().names()
    has_youtube = "youtube_url" in input_schema_names
    has_spotify = "spotify_url" in input_schema_names

    # Check if toptags.tag has actual data (not List(Null))
    # When all tracks have empty tag lists, Polars infers List(Null) which causes
    # StructFieldNotFoundError when trying to access struct.field("name")
    input_schema = raw_tracks.collect_schema()
    toptags_dtype = input_schema.get("toptags")
    has_tags_data = False
    if toptags_dtype is not None:
        # Check if the tag field inside toptags struct is not List(Null)
        toptags_str = str(toptags_dtype)
        has_tags_data = "List(Null)" not in toptags_str

    df = raw_tracks.with_columns(
        [
            # Basic info
            pl.col("name").alias("track_name"),
            pl.col("url").alias("track_url"),
            pl.col("duration").cast(pl.Int64).alias("duration_ms"),
            # Artist info (nested struct)
            pl.col("artist").struct.field("name").alias("artist_name"),
            # Popularity metrics
            pl.col("listeners").cast(pl.Int64).alias("listeners"),
            pl.col("playcount").cast(pl.Int64).alias("playcount"),
        ]
    )

    # Add tags column - only attempt to parse if data exists
    if has_tags_data:
        df = df.with_columns(
            pl.when(pl.col("toptags").is_not_null())
            .then(
                pl.col("toptags")
                .struct.field("tag")
                .list.head(5)
                .list.eval(pl.element().struct.field("name"))
                .list.join(", ")
            )
            .otherwise(None)
            .alias("tags")
        )
    else:
        # All tags are empty, just use None
        df = df.with_columns(pl.lit(None, dtype=pl.String).alias("tags"))

    # Add streaming links if available (enriched in bronze extraction)
    # Cast to String to avoid Null type issues when all values are null
    if has_youtube:
        df = df.with_columns(pl.col("youtube_url").cast(pl.String).alias("youtube_url"))
    else:
        df = df.with_columns(pl.lit(None, dtype=pl.String).alias("youtube_url"))

    if has_spotify:
        df = df.with_columns(pl.col("spotify_url").cast(pl.String).alias("spotify_url"))
    else:
        df = df.with_columns(pl.lit(None, dtype=pl.String).alias("spotify_url"))

    df = df.select(
        [
            "track_name",
            "artist_name",
            "duration_ms",
            "listeners",
            "playcount",
            "tags",
            "track_url",
            "youtube_url",
            "spotify_url",
        ]
    )

    return df


def _transform_artists_raw_to_structured(raw_artists: pl.LazyFrame) -> pl.LazyFrame:
    """
    Transform raw Last.fm artist metadata to structured format.

    Extracts relevant fields from nested JSON structure:
    - Basic info: name
    - Popularity metrics: listeners, playcount
    - Tags (flattened to comma-separated string)
    - Bio summary (first 500 chars)
    - URLs

    Args:
        raw_artists: LazyFrame with raw JSON structure from Last.fm API

    Returns:
        Transformed LazyFrame with flattened columns
    """
    df = raw_artists.with_columns(
        [
            # Basic info
            pl.col("name").alias("artist_name"),
            pl.col("url").alias("artist_url"),
            # Popularity metrics (nested in stats struct)
            pl.when(pl.col("stats").is_not_null())
            .then(pl.col("stats").struct.field("listeners").cast(pl.Int64))
            .otherwise(None)
            .alias("listeners"),
            pl.when(pl.col("stats").is_not_null())
            .then(pl.col("stats").struct.field("playcount").cast(pl.Int64))
            .otherwise(None)
            .alias("playcount"),
            # Tags - extract top 5 tag names as comma-separated string
            pl.when(pl.col("tags").is_not_null())
            .then(
                pl.col("tags")
                .struct.field("tag")
                .list.head(5)
                .list.eval(pl.element().struct.field("name"))
                .list.join(", ")
            )
            .otherwise(None)
            .alias("tags"),
            # Bio summary
            pl.when(pl.col("bio").is_not_null())
            .then(pl.col("bio").struct.field("summary"))
            .otherwise(None)
            .alias("bio_summary"),
        ]
    ).select(
        [
            "artist_name",
            "listeners",
            "playcount",
            "tags",
            "bio_summary",
            "artist_url",
        ]
    )

    return df


def _deduplicate_tracks(tracks_lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Deduplicate tracks using fuzzy text matching on normalized names.

    Uses canonical track IDs based on normalized track + artist names.
    When multiple versions exist (live, remastered, etc.):
    - Prefers non-music-video versions
    - Prefers versions with highest playcount
    - Keeps best metadata from chosen version
    - Uses max playcount across all versions

    Args:
        tracks_lf: LazyFrame with potentially duplicate tracks

    Returns:
        Deduplicated LazyFrame with canonical track_id and best metadata
    """
    # Check which streaming URL columns exist in the schema
    schema_names = tracks_lf.collect_schema().names()
    has_youtube = "youtube_url" in schema_names
    has_spotify = "spotify_url" in schema_names

    # Add helper columns for deduplication using native Polars expressions
    tracks_with_helpers = tracks_lf.with_columns(
        # Generate canonical track_id from normalized names
        generate_canonical_track_id_expr("track_name", "artist_name").alias("track_id"),
        # Detect music videos
        is_music_video_expr("track_name").alias("is_music_video"),
        # Fill null playcount for sorting
        pl.col("playcount").fill_null(0).alias("playcount_filled"),
    )

    # Sort: non-videos first, then by playcount descending
    # This ensures we pick the best version of each track
    sorted_tracks = tracks_with_helpers.sort(
        [
            pl.col("is_music_video"),  # False (non-video) first
            pl.col("playcount_filled"),
        ],
        descending=[False, True],
    )

    # Build aggregation list - take first (best) of each group
    agg_list = [
        pl.first("track_name").alias("track_name"),
        pl.first("artist_name").alias("artist_name"),
        pl.first("duration_ms").alias("duration_ms"),
        pl.first("track_url").alias("track_url"),
        pl.first("tags").alias("tags"),
        # Take max for popularity metrics across all versions
        pl.max("listeners").alias("listeners"),
        pl.max("playcount").alias("playcount"),
    ]

    # Add streaming URLs if they exist - cast to ensure proper type
    if has_youtube:
        agg_list.append(pl.first("youtube_url").cast(pl.String).alias("youtube_url"))
    if has_spotify:
        agg_list.append(pl.first("spotify_url").cast(pl.String).alias("spotify_url"))

    # Group by canonical track_id
    deduplicated = sorted_tracks.group_by("track_id").agg(agg_list)

    # Generate artist_id using native Polars normalization
    deduplicated = deduplicated.with_columns(
        generate_canonical_artist_id_expr("artist_name").alias("artist_id")
    )

    return deduplicated


def _deduplicate_artists(artists_lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Deduplicate artists using fuzzy text matching on normalized names.

    Uses canonical artist IDs based on normalized artist names.
    When multiple entries exist, keeps best metadata and max popularity metrics.

    Args:
        artists_lf: LazyFrame with potentially duplicate artists

    Returns:
        Deduplicated LazyFrame with canonical artist_id
    """
    # Generate canonical artist_id using native Polars normalization
    artists_with_id = artists_lf.with_columns(
        generate_canonical_artist_id_expr("artist_name").alias("artist_id")
    )

    # Sort by playcount descending to get best version first
    sorted_artists = artists_with_id.sort("playcount", descending=True, nulls_last=True)

    # Group by canonical artist_id, take first (best) of each
    deduplicated = sorted_artists.group_by("artist_id").agg(
        [
            pl.first("artist_name").alias("artist_name"),
            pl.first("artist_url").alias("artist_url"),
            pl.first("tags").alias("tags"),
            pl.first("bio_summary").alias("bio_summary"),
            # Take max for popularity metrics
            pl.max("listeners").alias("listeners"),
            pl.max("playcount").alias("playcount"),
        ]
    )

    return deduplicated


def compute_dim_users(execution_date: datetime) -> dict[str, Any]:
    """
    Compute user dimension table with listening profile metadata.

    Creates a dimension table with per-user metrics:
    - Listening span (first to last play dates)
    - User-specific half-life for recency calculations
    - Total plays count

    Half-life is calculated as: max(listening_span_days / 3, 30 days)
    This ensures:
    - New users (30 day history): 30 day half-life
    - Users with 6 months history: 60 day half-life
    - Users with 1 year history: 120 day half-life

    Args:
        execution_date: Reference date for calculations (typically DAG run date)

    Returns:
        Metadata dict with:
        - path: Path to Delta table
        - table_name: "dim_users"
        - rows: Number of unique users
        - schema: Column schema
        - format: "delta"
        - medallion_layer: "silver"
        - execution_date: Reference date used

    Raises:
        AirflowSkipException: If no plays data available yet
    """
    # Read silver plays table
    try:
        io_manager = PolarsDeltaIOManager(medallion_layer="silver")
        plays_lf: pl.LazyFrame = io_manager.read_delta("plays")
    except (FileNotFoundError, TableNotFoundError):
        # No plays data yet - nothing to process
        raise AirflowSkipException("No plays data available yet - run plays DAG first")

    # Compute user-level aggregations
    dim_users_lf = (
        plays_lf.group_by("username")
        .agg(
            [
                pl.col("scrobbled_at_utc").min().alias("first_play_date"),
                pl.col("scrobbled_at_utc").max().alias("last_play_date"),
                pl.len().alias("total_plays"),
            ]
        )
        .with_columns(
            [
                # Listening span in days
                (pl.col("last_play_date") - pl.col("first_play_date"))
                .dt.total_days()
                .alias("listening_span_days"),
            ]
        )
        .with_columns(
            [
                # Half-life = max(listening_span / 3, 30 days)
                pl.max_horizontal(
                    pl.col("listening_span_days") / 3.0, pl.lit(MIN_HALF_LIFE_DAYS)
                ).alias("user_half_life_days"),
            ]
        )
        .sort("username")
    )

    # Write to silver layer
    silver_io = PolarsDeltaIOManager(medallion_layer="silver")
    write_metadata = silver_io.write_delta(
        dim_users_lf,
        table_name="dim_users",
        mode="overwrite",  # Full refresh
    )

    return {
        **write_metadata,
        "execution_date": execution_date.isoformat(),
    }
