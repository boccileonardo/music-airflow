"""
Plays transformation logic.

Pure transformation functions for converting raw Last.fm play data
from bronze to silver layer using Delta Lake.
"""

from typing import Any
import polars as pl

from music_airflow.utils.polars_io_manager import JSONIOManager, PolarsDeltaIOManager


def transform_plays_to_silver(fetch_metadata: dict[str, Any]) -> dict[str, Any]:
    """
    Transform raw JSON plays from bronze to structured Delta table in silver layer.

    Reads raw data from bronze layer, applies transformations, and merges into silver
    Delta table partitioned by username. Uses upsert to avoid duplicates.
    Returns empty metadata dict if no tracks found.

    Args:
        fetch_metadata: Metadata from extraction containing filename, username, timestamps

    Returns:
        Metadata dict with path, table_name, rows, schema, format, medallion_layer, username, from/to datetimes
        Or dict with skipped=True if no tracks in time range
    """
    # Read raw JSON using Polars
    io_manager = JSONIOManager(medallion_layer="bronze")
    filename = fetch_metadata["filename"]
    tracks_df = io_manager.read_json(filename)
    username = fetch_metadata.get("username", "unknown")

    # Check if empty by trying to select a column
    row_count = tracks_df.select(pl.len()).collect().item()
    if row_count == 0:
        # No tracks in this interval - return empty result
        return {
            "path": None,
            "rows": 0,
            "schema": {},
            "skipped": True,
            "reason": "No tracks in time range",
        }

    # Apply transformation using extracted business logic
    df = transform_plays_raw_to_structured(tracks_df, username)

    # Write to silver layer Delta table with merge/upsert
    # Use a single table for all users, partitioned by username
    table_name = "plays"

    # Define merge predicate: match on username and scrobbled_at (unique identifier for a play)
    predicate = "s.username = t.username AND s.scrobbled_at = t.scrobbled_at"

    io_manager = PolarsDeltaIOManager(medallion_layer="silver")
    write_metadata = io_manager.write_delta(
        df,
        table_name=table_name,
        mode="merge",
        predicate=predicate,
        partition_by="username",
    )

    # Return io_manager metadata with additional context
    return {
        **write_metadata,
        "username": username,
        "from_datetime": fetch_metadata["from_datetime"],
        "to_datetime": fetch_metadata["to_datetime"],
    }


def transform_plays_raw_to_structured(
    raw_tracks: pl.LazyFrame, username: str
) -> pl.LazyFrame:
    """
    Transform raw Last.fm API plays to structured format.

    Extracts relevant fields, flattens nested structures (artist, album),
    and standardizes column names for analytics.

    Args:
        raw_tracks: LazyFrame with raw JSON structure from Last.fm API
        username: Last.fm username for this data

    Returns:
        Transformed LazyFrame with columns:
        - username: str
        - scrobbled_at: int (Unix timestamp)
        - scrobbled_at_utc: datetime (UTC)
        - track_name, track_mbid, track_url: str
        - artist_name, artist_mbid: str
        - album_name, album_mbid: str
    """
    df = (
        raw_tracks.with_columns(
            [
                # User identifier
                pl.lit(username).alias("username"),
                # Core identifiers - parse timestamp from date.uts (Unix seconds)
                pl.col("date").struct.field("uts").cast(pl.Int64).alias("scrobbled_at"),
                pl.col("date")
                .struct.field("uts")
                .cast(pl.Int64)
                .mul(1000)  # Convert seconds to milliseconds
                .cast(pl.Datetime("ms"))
                .dt.replace_time_zone("UTC")
                .cast(pl.Datetime("us", "UTC"))
                .alias("scrobbled_at_utc"),
                # Track info
                pl.col("name").alias("track_name"),
                pl.col("mbid").alias("track_mbid"),
                pl.col("url").alias("track_url"),
                # Artist info - extract from nested struct
                # Artist has "name" field in the struct
                pl.col("artist").struct.field("name").alias("artist_name"),
                pl.col("artist")
                .struct.field("mbid")
                .fill_null("")
                .alias("artist_mbid"),
                # Album info - extract from nested struct
                # Album has "#text" field in the struct
                pl.col("album").struct.field("#text").alias("album_name"),
                pl.col("album").struct.field("mbid").fill_null("").alias("album_mbid"),
            ]
        )
        .select(
            [
                "username",
                "scrobbled_at",
                "scrobbled_at_utc",
                "track_name",
                "track_mbid",
                "track_url",
                "artist_name",
                "artist_mbid",
                "album_name",
                "album_mbid",
            ]
        )
        .sort("scrobbled_at")
    )

    return df
