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

from airflow.sdk import dag, task

from music_airflow.utils.constants import LAST_FM_USERNAMES
from music_airflow.extract import extract_plays_to_bronze
from music_airflow.transform import transform_plays_to_silver


@dag(
    schedule="@daily",
    start_date=dt.datetime(2025, 11, 1, tzinfo=dt.timezone.utc),
    catchup=True,  # backfill from start date
    max_active_runs=1,  # Prevent concurrent runs
    tags=["lastfm", "bronze"],
    doc_md=__doc__,
)
def lastfm_plays():
    """
    Fetch and store Last.fm play history for multiple users.

    This DAG incrementally builds a local copy of listening history for all
    configured users in LASTFM_USERNAMES list.
    With catchup=True, it backfills from start_date.
    On each run, it fetches plays for the previous day relative to logical_date.
    """

    @task(multiple_outputs=False)
    def fetch_plays(
        username: str,
        logical_date: dt.datetime,
    ) -> dict[str, Any]:
        """
        Extract plays from Last.fm API to bronze layer for the previous day.

        For a daily DAG running on date T, this extracts data from T-1 00:00:00 to T 00:00:00.
        Handles pagination automatically and persists raw data to bronze layer.

        Args:
            username: Last.fm username to fetch data for
            logical_date: Logical date of the DAG run (T)

        Returns:
            Metadata dict with path, filename, rows, format, medallion_layer, username, from/to datetimes

        Raises:
            AirflowSkipException: If no plays found for the date range
        """
        # Validate context parameters
        if logical_date is None:
            raise ValueError("logical_date must be provided")

        # Fetch data for the previous day (logical_date - 1)
        # For a DAG run on date T, fetch data from T-1 00:00:00 to T 00:00:00
        from_dt = logical_date - dt.timedelta(days=1)
        to_dt = logical_date

        return extract_plays_to_bronze(username, from_dt, to_dt)

    @task(multiple_outputs=False)
    def transform_and_save(fetch_metadata: dict[str, Any]) -> dict[str, Any]:
        """
        Transform raw JSON plays from bronze to structured Parquet in silver layer.

        Reads raw data from bronze layer, extracts relevant fields, flattens nested structures,
        and saves to silver layer as Parquet.

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
