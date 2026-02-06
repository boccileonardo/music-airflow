"""
User Backfill DAG - Load historical play data for a new user.

This DAG is designed to be triggered manually with parameters to backfill
play history for a specific user. Use this when adding a new user to the
system who needs their historical Last.fm data loaded.

Usage:
    1. Trigger the DAG manually from Airflow UI or CLI
    2. Provide the 'username' parameter (Last.fm username)
    3. Optionally provide 'start_date' and 'end_date' (YYYY-MM-DD format)

The DAG will:
    1. Fetch play history from Last.fm API for the date range
    2. Write raw data to bronze layer
    3. Transform and save to silver layer
    4. Trigger downstream DAGs (candidates, dimensions) via asset updates

Note: This DAG does not use catchup - it processes all dates in a single run.
"""

import datetime as dt
from typing import Any

from airflow.sdk import Asset, dag, task
from airflow.sdk.definitions.param import Param

from music_airflow.utils.constants import DAG_START_DATE, LAST_FM_USERNAMES
from music_airflow.extract import extract_plays_to_bronze
from music_airflow.transform import transform_plays_to_silver

# Define the plays asset - same as extract_plays.py
plays_asset = Asset("delta://data/silver/plays")


@dag(
    schedule=None,  # Manual trigger only
    start_date=DAG_START_DATE,
    catchup=False,
    max_active_runs=1,
    tags=["lastfm", "backfill", "manual"],
    doc_md=__doc__,
    params={
        "username": Param(
            default=LAST_FM_USERNAMES[0] if LAST_FM_USERNAMES else "",
            description="Last.fm username to backfill",
            type="string",
            enum=LAST_FM_USERNAMES if LAST_FM_USERNAMES else None,
        ),
        "start_date": Param(
            default=DAG_START_DATE.strftime("%Y-%m-%d"),
            description="Start date for backfill (YYYY-MM-DD)",
            type="string",
        ),
        "end_date": Param(
            default=(
                dt.datetime.now(tz=dt.timezone.utc) - dt.timedelta(days=1)
            ).strftime("%Y-%m-%d"),
            description="End date for backfill (YYYY-MM-DD), defaults to yesterday",
            type="string",
        ),
    },
)
def backfill_user():
    """
    Backfill play history for a specific user over a date range.

    This DAG processes each day sequentially, fetching plays from Last.fm
    and writing to both bronze and silver layers. After completion, downstream
    asset-triggered DAGs will process the new data.
    """

    @task
    def generate_date_range() -> list[str]:
        """
        Generate list of dates to process based on DAG params.

        Returns:
            List of date strings (YYYY-MM-DD) from start_date to end_date inclusive
        """
        from airflow.sdk import get_current_context

        context = get_current_context()
        params = context["params"]

        start = dt.datetime.strptime(params["start_date"], "%Y-%m-%d").replace(
            tzinfo=dt.timezone.utc
        )
        end = dt.datetime.strptime(params["end_date"], "%Y-%m-%d").replace(
            tzinfo=dt.timezone.utc
        )

        dates = []
        current = start
        while current <= end:
            dates.append(current.strftime("%Y-%m-%d"))
            current += dt.timedelta(days=1)

        return dates

    @task
    def fetch_and_transform_day(date_str: str) -> dict[str, Any]:
        """
        Fetch plays for one day and transform to silver.

        Handles both extraction and transformation in one task to reduce
        task overhead for backfill operations.

        Args:
            date_str: Date string (YYYY-MM-DD)

        Returns:
            Metadata dict with results or skip info
        """
        import asyncio
        import logging
        from airflow.sdk import get_current_context

        logger = logging.getLogger(__name__)
        context = get_current_context()
        params = context["params"]
        username = params["username"]

        # Parse date
        date = dt.datetime.strptime(date_str, "%Y-%m-%d").replace(
            tzinfo=dt.timezone.utc
        )
        from_dt = date
        to_dt = date + dt.timedelta(days=1)

        logger.info(f"Backfilling {username} for {date_str}")

        # Extract to bronze
        fetch_metadata = asyncio.run(extract_plays_to_bronze(username, from_dt, to_dt))

        # Check if extraction returned skipped (no plays for this date)
        if fetch_metadata.get("skipped"):
            return {
                "date": date_str,
                "username": username,
                "skipped": True,
                "reason": fetch_metadata.get("reason", "No plays found"),
            }

        # Transform to silver
        transform_metadata = transform_plays_to_silver(fetch_metadata)

        return {
            "date": date_str,
            "username": username,
            "rows": transform_metadata.get("rows", 0),
            "skipped": False,
        }

    @task(outlets=[plays_asset])
    def summarize_backfill(results: list) -> dict[str, Any]:
        """
        Summarize backfill results and trigger downstream assets.

        This task produces the plays_asset to trigger candidate generation
        and dimension updates.

        Args:
            results: List of results from each day's processing

        Returns:
            Summary metadata
        """
        from airflow.sdk import get_current_context

        context = get_current_context()
        params = context["params"]
        username = params["username"]

        total_rows = sum(r.get("rows", 0) for r in results if not r.get("skipped"))
        days_processed = len([r for r in results if not r.get("skipped")])
        days_skipped = len([r for r in results if r.get("skipped")])

        return {
            "username": username,
            "total_rows": total_rows,
            "days_processed": days_processed,
            "days_skipped": days_skipped,
            "start_date": params["start_date"],
            "end_date": params["end_date"],
        }

    # Define workflow
    dates = generate_date_range()
    # Process dates sequentially (not parallel to respect API rate limits)
    results = fetch_and_transform_day.expand(date_str=dates)
    summarize_backfill(results)  # type: ignore[arg-type]


# Instantiate the DAG
backfill_user()
