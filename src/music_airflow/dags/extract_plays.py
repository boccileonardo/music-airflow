"""
Last.fm DAG - Fetch and store user listening history.

This DAG fetches play data from Last.fm API and stores it as the base fact table.
Supports both backfill and incremental loads.

Data Flow:
1. Determine time range (backfill or incremental based on last successful run)
2. Extract plays from Last.fm API to bronze layer with pagination
3. Transform raw JSON to structured Polars LazyFrame in silver layer
4. Write to Parquet using io_manager (only metadata passed via XCom)

Configuration:
- Add Last.fm usernames to LASTFM_USERNAMES list in `music_airflow/utils/constants.py`
- API credentials loaded from .env file via lastfm_client
"""

import datetime as dt
from typing import Any

from airflow.sdk import Asset, dag, task

from music_airflow.utils.constants import LAST_FM_USERNAMES, DAG_START_DATE
from music_airflow.extract import extract_plays_to_bronze
from music_airflow.transform import transform_plays_to_silver

# Define the plays asset - used for scheduling downstream DAGs
plays_asset = Asset("delta://data/silver/plays")


@dag(
    schedule="@daily",
    start_date=DAG_START_DATE,
    catchup=True,  # backfill from start date
    max_active_runs=1,  # Prevent concurrent runs
    tags=["lastfm", "bronze", "silver"],
    doc_md=__doc__,
)
def lastfm_plays():
    """
    Fetch and store Last.fm play history for multiple users.

    This DAG incrementally builds a local copy of listening history for all
    configured users in LASTFM_USERNAMES list.
    With catchup=True, it backfills from start_date.
    On each run, it fetches plays for the previous day relative to data_interval_start.
    """

    @task(multiple_outputs=False)
    def fetch_plays(
        username: str,
    ) -> dict[str, Any]:
        """
        Extract plays from Last.fm API to bronze layer for the previous day.

        For a daily DAG running on date T, this extracts data from T-1 00:00:00 to T 00:00:00.
        Handles pagination automatically and persists raw data to bronze layer.

        Uses data_interval_start from Airflow context to determine the date range.

        Args:
            username: Last.fm username to fetch data for

        Returns:
            Metadata dict with path, filename, rows, format, medallion_layer, username, from/to datetimes
            Or dict with skipped=True if no plays found for the date range
        """
        import asyncio
        from airflow.sdk import get_current_context

        # Get data_interval_start from Airflow context
        context = get_current_context()
        data_interval_start = context["data_interval_start"]

        if data_interval_start is None:
            raise ValueError("data_interval_start not available in context")

        # Convert Pendulum DateTime to Python datetime and fetch data for the previous day
        # For a DAG run on date T, fetch data from T-1 00:00:00 to T 00:00:00
        interval_start = data_interval_start.replace(tzinfo=dt.timezone.utc)
        from_dt = interval_start - dt.timedelta(days=1)
        to_dt = interval_start

        return asyncio.run(extract_plays_to_bronze(username, from_dt, to_dt))

    @task(multiple_outputs=False, outlets=[plays_asset])
    def transform_and_save(fetch_metadata: dict[str, Any]) -> dict[str, Any]:
        """
        Transform raw JSON plays from bronze to structured Delta in silver layer.

        Reads raw data from bronze layer, extracts relevant fields, flattens nested structures,
        and saves to silver layer as Delta. This task produces the plays asset.

        Args:
            fetch_metadata: Metadata from extraction containing filename, username, timestamps

        Returns:
            Metadata dict with path, filename, rows, schema, format, medallion_layer, username, from/to datetimes
        """
        return transform_plays_to_silver(fetch_metadata)

    # Define task dependencies - process each user independently
    # Use expand() for dynamic task mapping across multiple users
    fetch_results = fetch_plays.expand(username=LAST_FM_USERNAMES)
    transform_and_save.expand(fetch_metadata=fetch_results)


# Instantiate the DAG
lastfm_plays()
