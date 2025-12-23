"""
Gold Layer DAG - Compute aggregate fact tables with recency measures.

This DAG computes gold layer aggregations from silver plays:
- artist_play_count: Per-user artist play statistics
- track_play_count: Per-user track play statistics

Both tables include recency measures for time capsule recommendations.

Configuration:
- Runs when plays asset is updated (asset-scheduled)
- Full refresh (recomputes all aggregates)
- Uses dynamic per-user half-life for recency scoring
"""

import datetime as dt
from typing import Any

from airflow.sdk import Asset, dag, task

# Asset produced by lastfm_plays DAG - triggers this DAG when updated
plays_asset = Asset("delta://data/silver/plays")

# Assets produced by this DAG
artist_play_count_asset = Asset("delta://data/gold/artist_play_count")
track_play_count_asset = Asset("delta://data/gold/track_play_count")


@dag(
    schedule=[plays_asset],  # Run when plays asset is updated
    start_date=dt.datetime(2025, 11, 1, tzinfo=dt.timezone.utc),
    catchup=False,  # Only process latest date
    max_active_runs=1,
    tags=["gold", "aggregations"],
    doc_md=__doc__,
)
def gold_play_aggregations():
    """
    Compute gold layer aggregate fact tables with recency measures.

    Reads silver plays and creates:
    - artist_play_count (username, artist, counts, recency)
    - track_play_count (username, track, counts, recency)
    """

    @task(multiple_outputs=False, outlets=[artist_play_count_asset])
    def compute_artist_aggregations() -> dict[str, Any]:
        """
        Compute artist play counts with dynamic per-user recency measures.

        Aggregates all historical plays to create per-user artist statistics.
        Full refresh - recomputes entire table.

        Uses data_interval_start from Airflow context as reference for recency calculations.

        Returns:
            Metadata dict with path, rows, table_name, execution_date
        """
        from airflow.sdk import get_current_context
        from music_airflow.transform import compute_artist_play_counts

        context = get_current_context()
        data_interval_start = context.get("data_interval_start")

        if data_interval_start is None:
            # For asset-triggered DAGs, use current datetime as reference
            execution_date = dt.datetime.now(tz=dt.timezone.utc)
        else:
            # Convert Pendulum DateTime to Python datetime
            execution_date = data_interval_start.replace(tzinfo=dt.timezone.utc)

        return compute_artist_play_counts(execution_date=execution_date)

    @task(multiple_outputs=False, outlets=[track_play_count_asset])
    def compute_track_aggregations() -> dict[str, Any]:
        """
        Compute track play counts with dynamic per-user recency measures.

        Aggregates all historical plays to create per-user track statistics.
        Full refresh - recomputes entire table.

        Uses data_interval_start from Airflow context as reference for recency calculations.

        Returns:
            Metadata dict with path, rows, table_name, execution_date
        """
        from airflow.sdk import get_current_context
        from music_airflow.transform import compute_track_play_counts

        context = get_current_context()
        data_interval_start = context.get("data_interval_start")

        if data_interval_start is None:
            # For asset-triggered DAGs, use current datetime as reference
            execution_date = dt.datetime.now(tz=dt.timezone.utc)
        else:
            # Convert Pendulum DateTime to Python datetime
            execution_date = data_interval_start.replace(tzinfo=dt.timezone.utc)

        return compute_track_play_counts(execution_date=execution_date)

    # Run both aggregations in parallel
    compute_artist_aggregations()
    compute_track_aggregations()


# Instantiate the DAG
gold_play_aggregations()
