"""
Scrobbles transformation logic.

Pure transformation functions for converting raw Last.fm scrobble data
to structured formats suitable for analytics.
"""

import polars as pl


def transform_scrobbles_raw_to_structured(
    raw_tracks: pl.LazyFrame, username: str
) -> pl.LazyFrame:
    """
    Transform raw Last.fm API scrobbles to structured format.

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
                # Core identifiers - parse timestamp from date.uts
                pl.col("date").struct.field("uts").cast(pl.Int64).alias("scrobbled_at"),
                pl.col("date")
                .struct.field("uts")
                .cast(pl.Int64)
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
