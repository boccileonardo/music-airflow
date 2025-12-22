"""
Extract music play history to bronze layer.
"""

import datetime as dt
from typing import Any

from airflow.exceptions import AirflowSkipException

from music_airflow.lastfm_client import LastFMClient
from music_airflow.utils.polars_io_manager import JSONIOManager


def extract_plays_to_bronze(
    username: str,
    from_dt: dt.datetime,
    to_dt: dt.datetime,
) -> dict[str, Any]:
    """
    Extract plays from Last.fm API for a specific date range to bronze layer.

    Handles API interaction, pagination, and data persistence to bronze layer.
    Raises AirflowSkipException if no data is found (user not yet signed up or no activity).

    Args:
        username: Last.fm username to fetch data for
        from_dt: Start datetime (inclusive)
        to_dt: End datetime (exclusive)

    Returns:
        Metadata dict with path, filename, rows, format, medallion_layer, username, from/to datetimes

    Raises:
        AirflowSkipException: If no plays found for the date range
    """
    # Convert to timestamps for API call
    from_ts = int(from_dt.timestamp())
    to_ts = int(to_dt.timestamp())

    # Initialize client and fetch tracks
    client = LastFMClient(username=username)
    tracks = client.get_recent_tracks(
        from_timestamp=from_ts, to_timestamp=to_ts, extended=True
    )

    # Check if empty (user not yet signed up or no activity this day)
    date_str = from_dt.strftime("%Y%m%d")
    if not tracks:
        raise AirflowSkipException(
            f"No plays found for {username} on {date_str} "
            f"(may not be signed up yet or no activity)"
        )

    # Save raw JSON for this interval (bronze layer)
    filename = f"plays/{username}/{date_str}.json"

    # Write raw data using JSON IO manager
    io_manager = JSONIOManager(medallion_layer="bronze")
    write_metadata = io_manager.write_json(tracks, filename)

    # Return io_manager metadata with additional context
    return {
        **write_metadata,
        "username": username,
        "from_datetime": from_dt.isoformat(),
        "to_datetime": to_dt.isoformat(),
    }
