"""
Gold layer fact table transformations for play aggregations.

Creates aggregate fact tables from silver plays with recency measures
for recommendation systems. Uses user dimension table for per-user half-life.
"""

from datetime import datetime
from typing import Any

import polars as pl

from music_airflow.utils.polars_io_manager import PolarsDeltaIOManager


def compute_artist_play_counts(execution_date: datetime) -> dict[str, Any]:
    """
    Compute artist play counts per user with recency measures from silver plays.

    Aggregates all historical plays to create per-user artist statistics with
    per-user recency decay. Uses dim_users for user-specific half-life values.

    Recency measures:
    - last_played_on: Most recent play timestamp (for time capsule filtering)
    - recency_score: Exponential decay score sum(exp(-days_ago / half_life))
    - recency_score_normalized: recency_score / play_count (prevents feedback loops)
    - days_since_last_play: Days between execution_date and last play

    Args:
        execution_date: Reference date for recency calculations (typically DAG run date)

    Returns:
        Metadata dict with:
        - path: Path to Delta table
        - table_name: "artist_play_count"
        - rows: Number of user-artist combinations
        - schema: Column schema
        - format: "delta"
        - medallion_layer: "gold"
        - execution_date: Reference date used
    """
    # Read silver tables
    io_manager = PolarsDeltaIOManager(medallion_layer="silver")
    plays_lf: pl.LazyFrame = io_manager.read_delta("plays")  # type: ignore[assignment]
    dim_users_lf: pl.LazyFrame = io_manager.read_delta("dim_users")  # type: ignore[assignment]
    artists_lf: pl.LazyFrame = io_manager.read_delta("artists")  # type: ignore[assignment]

    # Compute aggregations with per-user recency measures
    gold_lf = _compute_artist_aggregations(
        plays_lf, dim_users_lf, artists_lf, execution_date
    )

    # Write to gold layer
    gold_io_manager = PolarsDeltaIOManager(medallion_layer="gold")
    write_metadata = gold_io_manager.write_delta(
        gold_lf,
        table_name="artist_play_count",
        mode="overwrite",  # Full refresh - recompute all aggregates
        partition_by="username",
    )

    return {
        **write_metadata,
        "execution_date": execution_date.isoformat(),
    }


def compute_track_play_counts(execution_date: datetime) -> dict[str, Any]:
    """
    Compute track play counts per user with recency measures from silver plays.

    Aggregates all historical plays to create per-user track statistics with
    per-user recency decay. Uses dim_users for user-specific half-life values.

    Recency measures:
    - last_played_on: Most recent play timestamp (for time capsule filtering)
    - recency_score: Exponential decay score sum(exp(-days_ago / half_life))
    - recency_score_normalized: recency_score / play_count (prevents feedback loops)
    - days_since_last_play: Days between execution_date and last play

    Args:
        execution_date: Reference date for recency calculations (typically DAG run date)

    Returns:
        Metadata dict with:
        - path: Path to Delta table
        - table_name: "track_play_count"
        - rows: Number of user-track combinations
        - schema: Column schema
        - format: "delta"
        - medallion_layer: "gold"
        - execution_date: Reference date used
    """
    # Read silver tables
    io_manager = PolarsDeltaIOManager(medallion_layer="silver")
    plays_lf: pl.LazyFrame = io_manager.read_delta("plays")  # type: ignore[assignment]
    dim_users_lf: pl.LazyFrame = io_manager.read_delta("dim_users")  # type: ignore[assignment]

    # Compute aggregations with per-user recency measures
    gold_lf = _compute_track_aggregations(plays_lf, dim_users_lf, execution_date)

    # Write to gold layer
    gold_io_manager = PolarsDeltaIOManager(medallion_layer="gold")
    write_metadata = gold_io_manager.write_delta(
        gold_lf,
        table_name="track_play_count",
        mode="overwrite",  # Full refresh - recompute all aggregates
        partition_by="username",
    )

    return {
        **write_metadata,
        "execution_date": execution_date.isoformat(),
    }


def _compute_artist_aggregations(
    plays_lf: pl.LazyFrame,
    dim_users_lf: pl.LazyFrame,
    artists_lf: pl.LazyFrame,
    execution_date: datetime,
) -> pl.LazyFrame:
    """
    Compute artist-level aggregations with per-user recency measures.

    Groups plays by username and artist, calculating:
    - Total play count
    - First and last play dates
    - Normalized recency score (prevents feedback loops)
    - Days since last play

    Enriches with artist_mbid from silver/artists dimension.

    Args:
        plays_lf: Silver plays LazyFrame
        dim_users_lf: User dimension LazyFrame with half-life values
        artists_lf: Artists dimension LazyFrame with artist_mbid
        execution_date: Reference date for recency calculations

    Returns:
        Aggregated LazyFrame with columns:
        - username, artist_name, artist_mbid
        - play_count, first_played_on, last_played_on
        - recency_score, days_since_last_play
    """
    # Join plays with user half-life from dim_users
    df = plays_lf.join(
        dim_users_lf.select(["username", "user_half_life_days"]), on="username"
    )

    # Add days_ago column for recency calculation
    df = df.with_columns(
        [
            # Days between each play and execution_date
            (
                pl.lit(execution_date).cast(pl.Datetime("us", "UTC"))
                - pl.col("scrobbled_at_utc")
            )
            .dt.total_days()
            .alias("days_ago"),
        ]
    )

    # Group by user and artist, compute aggregations
    agg_lf = (
        df.group_by("username", "artist_name")
        .agg(
            [
                pl.len().alias("play_count"),
                pl.col("scrobbled_at_utc").min().alias("first_played_on"),
                pl.col("scrobbled_at_utc").max().alias("last_played_on"),
                # Normalized recency score: average exponential decay per play
                # Prevents feedback loops by normalizing for play frequency
                (
                    (-(pl.col("days_ago") / pl.col("user_half_life_days"))).exp().sum()
                    / pl.len()
                ).alias("recency_score"),
            ]
        )
        .with_columns(
            [
                # Days since last play
                (
                    pl.lit(execution_date).cast(pl.Datetime("us", "UTC"))
                    - pl.col("last_played_on")
                )
                .dt.total_days()
                .cast(pl.Int32)
                .alias("days_since_last_play"),
            ]
        )
    )

    # Join with artists dimension to get proper artist_mbid
    # Use left join to keep all artists even if not in dimension table
    agg_lf = agg_lf.join(
        artists_lf.select(["artist_name", "artist_mbid"]),
        on="artist_name",
        how="left",
    ).with_columns(
        [
            # Fill null artist_mbid with empty string
            pl.col("artist_mbid").fill_null(""),
        ]
    )

    # Final sort and column order
    agg_lf = agg_lf.select(
        [
            "username",
            "artist_name",
            "artist_mbid",
            "play_count",
            "first_played_on",
            "last_played_on",
            "recency_score",
            "days_since_last_play",
        ]
    ).sort("username", "artist_name")

    return agg_lf


def _compute_track_aggregations(
    plays_lf: pl.LazyFrame,
    dim_users_lf: pl.LazyFrame,
    execution_date: datetime,
) -> pl.LazyFrame:
    """
    Compute track-level aggregations with per-user recency measures.

    Groups plays by username and track, calculating:
    - Total play count
    - First and last play dates
    - Normalized recency score (prevents feedback loops)
    - Days since last play

    Args:
        plays_lf: Silver plays LazyFrame
        dim_users_lf: User dimension LazyFrame with half-life values
        execution_date: Reference date for recency calculations

    Returns:
        Aggregated LazyFrame with columns:
        - username, track_name, track_mbid, artist_name, album_name
        - play_count, first_played_on, last_played_on
        - recency_score, days_since_last_play
    """
    # Join plays with user half-life from dim_users
    df = plays_lf.join(
        dim_users_lf.select(["username", "user_half_life_days"]), on="username"
    )

    # Add days_ago column for recency calculation
    df = df.with_columns(
        [
            # Days between each play and execution_date
            (
                pl.lit(execution_date).cast(pl.Datetime("us", "UTC"))
                - pl.col("scrobbled_at_utc")
            )
            .dt.total_days()
            .alias("days_ago"),
        ]
    )

    # Group by user and track, compute aggregations
    agg_lf = (
        df.group_by("username", "track_name", "track_mbid", "artist_name", "album_name")
        .agg(
            [
                pl.len().alias("play_count"),
                pl.col("scrobbled_at_utc").min().alias("first_played_on"),
                pl.col("scrobbled_at_utc").max().alias("last_played_on"),
                # Normalized recency score: average exponential decay per play
                # Prevents feedback loops by normalizing for play frequency
                (
                    (-(pl.col("days_ago") / pl.col("user_half_life_days"))).exp().sum()
                    / pl.len()
                ).alias("recency_score"),
            ]
        )
        .with_columns(
            [
                # Days since last play
                (
                    pl.lit(execution_date).cast(pl.Datetime("us", "UTC"))
                    - pl.col("last_played_on")
                )
                .dt.total_days()
                .cast(pl.Int32)
                .alias("days_since_last_play"),
            ]
        )
        .select(
            [
                "username",
                "track_name",
                "track_mbid",
                "artist_name",
                "album_name",
                "play_count",
                "first_played_on",
                "last_played_on",
                "recency_score",
                "days_since_last_play",
            ]
        )
        .sort("username", "track_name")
    )

    return agg_lf
