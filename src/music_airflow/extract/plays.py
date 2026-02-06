"""
Extract music play history to bronze layer.
"""

import datetime as dt
from typing import Any

from music_airflow.lastfm_client import LastFMClient
from music_airflow.utils.polars_io_manager import JSONIOManager


async def extract_plays_to_bronze(
    username: str,
    from_dt: dt.datetime,
    to_dt: dt.datetime,
) -> dict[str, Any]:
    """
    Extract plays from Last.fm API for a specific date range to bronze layer.

    Handles API interaction, pagination, and data persistence to bronze layer.
    Returns metadata with skipped=True if no data is found (allows downstream to continue).

    Args:
        username: Last.fm username to fetch data for
        from_dt: Start datetime (inclusive)
        to_dt: End datetime (exclusive)

    Returns:
        Metadata dict with path, filename, rows, format, medallion_layer, username, from/to datetimes
        Or dict with skipped=True if no plays found
    """
    # Convert to timestamps for API call
    from_ts = int(from_dt.timestamp())
    to_ts = int(to_dt.timestamp())

    # Initialize client and fetch tracks
    async with LastFMClient(username=username) as client:
        tracks = await client.get_recent_tracks(
            from_timestamp=from_ts, to_timestamp=to_ts, extended=True
        )

        # Check if empty (user not yet signed up or no activity this day)
        date_str = from_dt.strftime("%Y%m%d")
        if not tracks:
            return {
                "skipped": True,
                "reason": f"No plays found for {username} on {date_str}",
                "username": username,
                "from_datetime": from_dt.isoformat(),
                "to_datetime": to_dt.isoformat(),
            }

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
