"""
Last.fm Scrobbles DAG - Fetch and store user listening history.

This DAG fetches scrobble data from Last.fm API and stores it as the base fact table.
Supports both backfill and incremental loads.

Data Flow:
1. Determine time range (backfill or incremental based on last successful run)
2. Fetch scrobbles from Last.fm API with pagination
3. Transform raw JSON to structured Polars LazyFrame
4. Write to Parquet using io_manager (only metadata passed via XCom)

Configuration:
- Add Last.fm usernames to LASTFM_USERNAMES list in `music_airflow/utils/constants.py`
- API credentials loaded from .env file via lastfm_client
"""

import datetime as dt
from pathlib import Path
from typing import Any

import polars as pl
from airflow.exceptions import AirflowSkipException
from airflow.sdk import dag, task

from music_airflow.utils.polars_io_manager import JSONIOManager, PolarsParquetIOManager
from music_airflow.lastfm_client import LastFMClient
from music_airflow.utils.constants import LAST_FM_USERNAMES
from music_airflow.transformers import transform_scrobbles_raw_to_structured


@dag(
    schedule="@daily",
    start_date=dt.datetime(2025, 11, 1, tzinfo=dt.timezone.utc),
    catchup=True,  # backfill from start date
    max_active_runs=1,  # Prevent concurrent runs
    tags=["lastfm", "bronze"],
    doc_md=__doc__,
)
def lastfm_scrobbles():
    """
    Fetch and store Last.fm scrobble history for multiple users.

    This DAG incrementally builds a local copy of listening history for all
    configured users in LASTFM_USERNAMES list.
    With catchup=True, it backfills from start_date.
    On each run, it fetches scrobbles for the logical_date (the day being processed).
    """

    @task(multiple_outputs=False)
    def fetch_scrobbles(
        username: str,
        logical_date: dt.datetime,
    ) -> dict[str, Any]:
        """
        Fetch scrobbles from Last.fm API for the logical date.

        Fetches scrobbles for the full 24-hour period of logical_date.
        For a daily DAG running on day T, this fetches data from T 00:00:00 to T+1 00:00:00.

        Handles pagination automatically and returns raw track data.

        Args:
            username: Last.fm username to fetch data for
            logical_date: Logical date of the DAG run (the day being processed)

        Returns:
            Metadata dict with path, rows, format, medallion_layer, username, from/to datetimes
        """
        # Validate context parameters
        if logical_date is None:
            raise ValueError("logical_date must be provided")

        # Fetch data for the full logical_date (T 00:00:00 to T+1 00:00:00)
        # This gives us the complete 24-hour period for this date
        from_dt = logical_date
        to_dt = logical_date + dt.timedelta(days=1)

        # Convert to timestamps for API call
        from_ts = int(from_dt.timestamp())
        to_ts = int(to_dt.timestamp())

        # Initialize client and fetch tracks
        client = LastFMClient(username=username)
        tracks = client.get_recent_tracks(
            from_timestamp=from_ts, to_timestamp=to_ts, extended=True
        )

        # Save raw JSON for this interval (bronze layer)
        date_str = from_dt.strftime("%Y%m%d")
        filename = f"scrobbles/raw_scrobbles_{username}_{date_str}.json"

        # Check if empty (user not yet signed up or no activity this day)
        if not tracks:
            raise AirflowSkipException(
                f"No scrobbles found for {username} on {date_str} "
                f"(may not be signed up yet or no activity)"
            )

        # Write raw data using JSON IO manager
        io_manager = JSONIOManager(medallion_layer="bronze")
        write_metadata: dict[str, Any] = io_manager.write_json(tracks, filename)

        # Return io_manager metadata with additional context
        return {
            **write_metadata,
            "username": username,
            "from_datetime": from_dt.isoformat(),
            "to_datetime": to_dt.isoformat(),
        }

    @task(multiple_outputs=False)
    def transform_and_save(fetch_metadata: dict[str, Any]) -> dict[str, Any]:
        """
        Transform raw JSON scrobbles to structured Parquet.

        Extracts relevant fields, flattens nested structures, and converts
        to Polars LazyFrame for efficient processing.

        Args:
            fetch_metadata: Dict with raw_path, track_count

        Returns:
            Metadata dict with path, rows, schema, format, medallion_layer, username, from/to datetimes
        """
        # Read raw JSON using Polars
        io_manager = JSONIOManager(medallion_layer="bronze")
        raw_path = fetch_metadata["path"]
        # Extract just the filename relative to bronze directory
        # raw_path is absolute, get the part after 'bronze/'
        raw_path_obj = Path(raw_path)
        # Find 'bronze' in the path and get everything after it
        parts = raw_path_obj.parts
        bronze_idx = parts.index("bronze")
        filename = Path(*parts[bronze_idx + 1 :])
        tracks_df = io_manager.read_json(str(filename))
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
        df = transform_scrobbles_raw_to_structured(tracks_df, username)

        # Write using io_manager (silver layer)
        interval_start = dt.datetime.fromisoformat(fetch_metadata["from_datetime"])
        date_str = interval_start.strftime("%Y%m%d")
        filename = f"scrobbles/scrobbles_{username}_{date_str}.parquet"

        io_manager = PolarsParquetIOManager(medallion_layer="silver")
        write_metadata = io_manager.write_parquet(df, filename)

        # Return io_manager metadata with additional context
        return {
            **write_metadata,
            "username": username,
            "from_datetime": fetch_metadata["from_datetime"],
            "to_datetime": fetch_metadata["to_datetime"],
        }

    # Define task dependencies - process each user independently
    # Use expand() for dynamic task mapping across multiple users
    fetch_results = fetch_scrobbles.expand(username=LAST_FM_USERNAMES)
    transform_and_save.expand(fetch_metadata=fetch_results)


# Instantiate the DAG
lastfm_scrobbles()
