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
    plays_lf = io_manager.read_delta("plays")

    # Handle case when dim_users doesn't exist (bootstrapping)
    if io_manager.table_exists("dim_users"):
        dim_users_lf = io_manager.read_delta("dim_users")
    else:
        # Create fallback with default half-life for all users
        from music_airflow.transform.dimensions import MIN_HALF_LIFE_DAYS

        users_lf = plays_lf.select("username").unique()
        dim_users_lf = users_lf.with_columns(
            pl.lit(MIN_HALF_LIFE_DAYS).alias("user_half_life_days")
        )

    # Handle case when tracks/artists don't exist (bootstrapping)
    if not io_manager.table_exists("tracks") or not io_manager.table_exists("artists"):
        # Return empty result - cannot compute artist aggregations without dimensions
        gold_io_manager = PolarsDeltaIOManager(medallion_layer="gold")
        empty_df = pl.DataFrame(
            schema={
                "username": pl.String,
                "artist_id": pl.String,
                "artist_name": pl.String,
                "play_count": pl.Int64,
                "first_played_on": pl.Datetime("us", "UTC"),
                "last_played_on": pl.Datetime("us", "UTC"),
                "recency_score": pl.Float64,
                "days_since_last_play": pl.Int32,
            }
        )
        write_metadata = gold_io_manager.write_delta(
            empty_df,
            table_name="artist_play_count",
            mode="overwrite",
            partition_by="username",
        )
        return {
            **write_metadata,
            "execution_date": execution_date.isoformat(),
            "skipped": True,
            "reason": "tracks or artists table not yet available",
        }

    tracks_lf = io_manager.read_delta("tracks")
    artists_lf = io_manager.read_delta("artists")

    # Compute aggregations with per-user recency measures
    gold_lf = _compute_artist_aggregations(
        plays_lf, dim_users_lf, tracks_lf, artists_lf, execution_date
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

    # Handle case when dim_users doesn't exist (bootstrapping)
    if io_manager.table_exists("dim_users"):
        dim_users_lf: pl.LazyFrame = io_manager.read_delta("dim_users")  # type: ignore[assignment]
    else:
        # Create fallback with default half-life for all users
        from music_airflow.transform.dimensions import MIN_HALF_LIFE_DAYS

        users_lf = plays_lf.select("username").unique()
        dim_users_lf = users_lf.with_columns(
            pl.lit(MIN_HALF_LIFE_DAYS).alias("user_half_life_days")
        )

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
    tracks_lf: pl.LazyFrame,
    artists_lf: pl.LazyFrame,
    execution_date: datetime,
) -> pl.LazyFrame:
    """
    Compute artist-level aggregations with per-user recency measures.

    Groups plays by username and artist_id,
    calculating:
    - Total play count
    - First and last play dates
    - Normalized recency score (prevents feedback loops)
    - Days since last play

    Args:
        plays_lf: Silver plays LazyFrame with track_id
        dim_users_lf: User dimension LazyFrame with half-life values
        tracks_lf: Tracks dimension LazyFrame with artist_id
        artist_lf: Artists dimension LazyFrame with artist_name
        execution_date: Reference date for recency calculations

    Returns:
        Aggregated LazyFrame with columns:
        - username, artist_id, artist_name
        - play_count, first_played_on, last_played_on
        - recency_score, days_since_last_play
    """
    # Enrich plays with proper artist_id from tracks dimension
    df = plays_lf.join(
        tracks_lf.select(["track_id", "artist_id"]),
        on="track_id",
        how="left",
    )

    # Join with user half-life from dim_users
    df = df.join(
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

    # Group by user and artist_id, compute aggregations
    agg_lf = (
        df.group_by("username", "artist_id")
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

    # Join with artists dimension to get artist_name
    agg_lf = agg_lf.join(
        artists_lf.select(["artist_id", "artist_name"]).unique(),
        on="artist_id",
        how="left",
    )

    # Final sort and column order
    agg_lf = agg_lf.select(
        [
            "username",
            "artist_id",
            "artist_name",
            "play_count",
            "first_played_on",
            "last_played_on",
            "recency_score",
            "days_since_last_play",
        ]
    ).sort("username", "artist_id")

    return agg_lf


def _compute_track_aggregations(
    plays_lf: pl.LazyFrame,
    dim_users_lf: pl.LazyFrame,
    execution_date: datetime,
) -> pl.LazyFrame:
    """
    Compute track-level aggregations with per-user recency measures.

    Groups plays by username and track_id, calculating:
    - Total play count
    - First and last play dates
    - Normalized recency score (prevents feedback loops)
    - Days since last play

    Args:
        plays_lf: Silver plays LazyFrame with track_id
        dim_users_lf: User dimension LazyFrame with half-life values

    Returns:
        Aggregated LazyFrame with columns:
        - username, track_id, track_name, artist_name
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

    # Group by user and track_id (coalesced ID), compute aggregations
    agg_lf = (
        df.group_by("username", "track_id")
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
                # Keep name columns for reference (should be consistent within track_id)
                pl.col("track_name").first().alias("track_name"),
                pl.col("artist_name").first().alias("artist_name"),
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
                "track_id",
                "track_name",
                "artist_name",
                "play_count",
                "first_played_on",
                "last_played_on",
                "recency_score",
                "days_since_last_play",
            ]
        )
        .sort("username", "track_id")
    )

    return agg_lf
