"""
Gold layer fact table transformations for play aggregations.

Creates aggregate fact tables from silver plays with recency measures
for recommendation systems. Uses dynamic per-user half-life based on listening history.
"""

from datetime import datetime
from typing import Any

import polars as pl

from music_airflow.utils.polars_io_manager import PolarsDeltaIOManager


# Minimum half-life for new users (30 days)
MIN_HALF_LIFE_DAYS = 30.0


def compute_artist_play_counts(execution_date: datetime) -> dict[str, Any]:
    """
    Compute artist play counts per user with dynamic recency measures from silver plays.

    Aggregates all historical plays to create per-user artist statistics with
    per-user recency decay. Half-life is calculated per user based on their listening
    history span, with a minimum of 30 days for new users.

    Recency measures:
    - last_played_on: Most recent play timestamp (for time capsule filtering)
    - recency_score: Exponential decay score (higher = more recent)
      Formula: sum(exp(-days_ago / half_life))
      Half-life = max(listening_span_days / 3, 30 days)
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
    # Read silver plays table
    io_manager = PolarsDeltaIOManager(medallion_layer="silver")
    plays_lf: pl.LazyFrame = io_manager.read_delta("plays")  # type: ignore[assignment]

    # Compute aggregations with dynamic per-user recency measures
    gold_lf = _compute_artist_aggregations(plays_lf, execution_date)

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
    Compute track play counts per user with dynamic recency measures from silver plays.

    Aggregates all historical plays to create per-user track statistics with
    per-user recency decay. Half-life is calculated per user based on their listening
    history span, with a minimum of 30 days for new users.

    Recency measures:
    - last_played_on: Most recent play timestamp (for time capsule filtering)
    - recency_score: Exponential decay score (higher = more recent)
      Formula: sum(exp(-days_ago / half_life))
      Half-life = max(listening_span_days / 3, 30 days)
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
    # Read silver plays table
    io_manager = PolarsDeltaIOManager(medallion_layer="silver")
    plays_lf: pl.LazyFrame = io_manager.read_delta("plays")  # type: ignore[assignment]

    # Compute aggregations with dynamic per-user recency measures
    gold_lf = _compute_track_aggregations(plays_lf, execution_date)

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
    plays_lf: pl.LazyFrame, execution_date: datetime
) -> pl.LazyFrame:
    """
    Compute artist-level aggregations with dynamic per-user recency measures.

    Groups plays by username and artist, calculating:
    - Total play count
    - First and last play dates
    - Recency score with per-user exponential decay
    - Days since last play

    Half-life is calculated per user as: max(listening_span_days / 3, 30)
    This means:
    - New users (30 day history): 30 day half-life
    - Users with 6 months history: 60 day half-life
    - Users with 1 year history: 120 day half-life

    Args:
        plays_lf: Silver plays LazyFrame
        execution_date: Reference date for recency calculations

    Returns:
        Aggregated LazyFrame with columns:
        - username, artist_name, artist_mbid
        - play_count, first_played_on, last_played_on
        - recency_score, days_since_last_play
        - user_half_life_days
    """
    # Calculate per-user listening span and half-life
    user_span_lf = (
        plays_lf.group_by("username")
        .agg(
            [
                pl.col("scrobbled_at_utc").min().alias("user_first_play"),
                pl.col("scrobbled_at_utc").max().alias("user_last_play"),
            ]
        )
        .with_columns(
            [
                # Listening span in days
                (pl.col("user_last_play") - pl.col("user_first_play"))
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
        .select(["username", "user_half_life_days"])
    )

    # Join plays with user half-life
    df = plays_lf.join(user_span_lf, on="username")

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
        df.group_by("username", "artist_name", "artist_mbid")
        .agg(
            [
                pl.len().alias("play_count"),
                pl.col("scrobbled_at_utc").min().alias("first_played_on"),
                pl.col("scrobbled_at_utc").max().alias("last_played_on"),
                pl.col("user_half_life_days").first().alias("user_half_life_days"),
                # Recency score: sum of exponential decay for each play
                # exp(-days_ago / user_half_life)
                (-(pl.col("days_ago") / pl.col("user_half_life_days")))
                .exp()
                .sum()
                .alias("recency_score"),
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
        .sort("username", "artist_name")
    )

    return agg_lf


def _compute_track_aggregations(
    plays_lf: pl.LazyFrame, execution_date: datetime
) -> pl.LazyFrame:
    """
    Compute track-level aggregations with dynamic per-user recency measures.

    Groups plays by username and track, calculating:
    - Total play count
    - First and last play dates
    - Recency score with per-user exponential decay
    - Days since last play

    Half-life is calculated per user as: max(listening_span_days / 3, 30)

    Args:
        plays_lf: Silver plays LazyFrame
        execution_date: Reference date for recency calculations

    Returns:
        Aggregated LazyFrame with columns:
        - username, track_name, track_mbid, artist_name, album_name
        - play_count, first_played_on, last_played_on
        - recency_score, days_since_last_play
        - user_half_life_days
    """
    # Calculate per-user listening span and half-life
    user_span_lf = (
        plays_lf.group_by("username")
        .agg(
            [
                pl.col("scrobbled_at_utc").min().alias("user_first_play"),
                pl.col("scrobbled_at_utc").max().alias("user_last_play"),
            ]
        )
        .with_columns(
            [
                # Listening span in days
                (pl.col("user_last_play") - pl.col("user_first_play"))
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
        .select(["username", "user_half_life_days"])
    )

    # Join plays with user half-life
    df = plays_lf.join(user_span_lf, on="username")

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
                pl.col("user_half_life_days").first().alias("user_half_life_days"),
                # Recency score: sum of exponential decay for each play
                # exp(-days_ago / user_half_life)
                (-(pl.col("days_ago") / pl.col("user_half_life_days")))
                .exp()
                .sum()
                .alias("recency_score"),
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
        .sort("username", "track_name")
    )

    return agg_lf
