"""
Transform dimension data (tracks, artists) from bronze to silver layer.

Cleans and structures raw Last.fm metadata for analytics.
"""

from typing import Any

import polars as pl

from music_airflow.utils.polars_io_manager import JSONIOManager, PolarsDeltaIOManager

__all__ = [
    "transform_tracks_to_silver",
    "transform_artists_to_silver",
    "_transform_tracks_raw_to_structured",
    "_transform_artists_raw_to_structured",
]


def transform_tracks_to_silver(fetch_metadata: dict[str, Any]) -> dict[str, Any]:
    """
    Transform raw track metadata from bronze to structured Delta table in silver layer.

    Reads raw JSON track data from bronze, extracts and flattens relevant fields,
    and merges into silver Delta table. Uses upsert based on MBID when available,
    otherwise falls back to track_name + artist_name + album_name.

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

    # Apply transformation (keeps as LazyFrame)
    df = _transform_tracks_raw_to_structured(tracks_lf)

    # Write to silver layer Delta table with merge/upsert
    # Use MBID when available, otherwise use track_name + artist_name + album_name for uniqueness
    # This handles live versions, remixes, etc better than just track + artist
    table_name = "tracks"
    predicate = """
        (s.track_mbid != '' AND s.track_mbid = t.track_mbid) OR
        (s.track_mbid = '' AND t.track_mbid = '' AND
         s.track_name = t.track_name AND
         s.artist_name = t.artist_name AND
         COALESCE(s.album_name, '') = COALESCE(t.album_name, ''))
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
    and merges into silver Delta table. Uses upsert based on MBID when available.

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

    # Apply transformation (keeps as LazyFrame)
    df = _transform_artists_raw_to_structured(artists_lf)

    # Write to silver layer Delta table with merge/upsert
    # Use MBID when available, otherwise use artist_name
    table_name = "artists"
    predicate = """
        (s.artist_mbid != '' AND s.artist_mbid = t.artist_mbid) OR
        (s.artist_mbid = '' AND t.artist_mbid = '' AND s.artist_name = t.artist_name)
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
    - MBIDs for linking
    - Popularity metrics: listeners, playcount
    - Tags (flattened to comma-separated string)
    - URLs

    Args:
        raw_tracks: LazyFrame with raw JSON structure from Last.fm API

    Returns:
        Transformed LazyFrame with flattened columns
    """
    df = raw_tracks.with_columns(
        [
            # Basic info
            pl.col("name").alias("track_name"),
            pl.col("mbid").alias("track_mbid"),
            pl.col("url").alias("track_url"),
            pl.col("duration").cast(pl.Int64).alias("duration_ms"),
            # Artist info (nested struct)
            pl.col("artist").struct.field("name").alias("artist_name"),
            pl.col("artist").struct.field("mbid").alias("artist_mbid"),
            # Album info (nested struct)
            pl.when(pl.col("album").is_not_null())
            .then(pl.col("album").struct.field("title"))
            .otherwise(None)
            .alias("album_name"),
            # Popularity metrics
            pl.col("listeners").cast(pl.Int64).alias("listeners"),
            pl.col("playcount").cast(pl.Int64).alias("playcount"),
            # Tags - extract top 5 tag names as comma-separated string
            pl.when(pl.col("toptags").is_not_null())
            .then(
                pl.col("toptags")
                .struct.field("tag")
                .list.head(5)
                .list.eval(pl.element().struct.field("name"))
                .list.join(", ")
            )
            .otherwise(None)
            .alias("tags"),
        ]
    ).select(
        [
            "track_name",
            "track_mbid",
            "artist_name",
            "artist_mbid",
            "album_name",
            "duration_ms",
            "listeners",
            "playcount",
            "tags",
            "track_url",
        ]
    )

    return df


def _transform_artists_raw_to_structured(raw_artists: pl.LazyFrame) -> pl.LazyFrame:
    """
    Transform raw Last.fm artist metadata to structured format.

    Extracts relevant fields from nested JSON structure:
    - Basic info: name, mbid
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
            pl.col("mbid").alias("artist_mbid"),
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
            # Bio summary (truncate to 500 chars)
            pl.when(pl.col("bio").is_not_null())
            .then(pl.col("bio").struct.field("summary").str.slice(0, 500))
            .otherwise(None)
            .alias("bio_summary"),
        ]
    ).select(
        [
            "artist_name",
            "artist_mbid",
            "listeners",
            "playcount",
            "tags",
            "bio_summary",
            "artist_url",
        ]
    )

    return df
