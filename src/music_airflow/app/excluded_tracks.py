"""
Module for managing excluded track and artist recommendations.

Users can exclude tracks or entire artists from recommendations, and these
exclusions are stored in the gold layer for persistence across sessions.
"""
# todo: ensure these are also being applied to old favorites

import datetime as dt
import polars as pl
from music_airflow.utils.polars_io_manager import PolarsDeltaIOManager


def write_excluded_track(
    username: str, track_id: str, track_name: str, artist_name: str
) -> dict:
    """
    Write an excluded track to the gold layer excluded_recommendations table.

    Uses merge mode to avoid duplicates - if the same track is excluded again,
    the excluded_at timestamp is updated.

    Args:
        username: Username who excluded the track
        track_id: Track identifier
        track_name: Track name
        artist_name: Artist name

    Returns:
        Metadata dict from the write operation
    """
    gold_io = PolarsDeltaIOManager("gold")

    # Create DataFrame with excluded track
    excluded_df = pl.DataFrame(
        {
            "username": [username],
            "track_id": [track_id],
            "track_name": [track_name],
            "artist_name": [artist_name],
            "excluded_at": [dt.datetime.now(tz=dt.timezone.utc)],
        }
    )

    # Write to Delta table with merge to avoid duplicates
    return gold_io.write_delta(
        df=excluded_df,
        table_name="excluded_recommendations",
        mode="merge",
        predicate="s.username = t.username AND s.track_id = t.track_id",
    )


def read_excluded_tracks(username: str) -> pl.LazyFrame:
    """
    Read excluded tracks for a user from the gold layer.

    Args:
        username: Username to get exclusions for

    Returns:
        LazyFrame with columns: username, track_id, track_name, artist_name, excluded_at
        Empty LazyFrame if no exclusions exist or table doesn't exist
    """
    gold_io = PolarsDeltaIOManager("gold")

    try:
        return gold_io.read_delta("excluded_recommendations").filter(
            pl.col("username") == username
        )
    except Exception:
        # Table doesn't exist yet, return empty LazyFrame
        return pl.LazyFrame(
            schema={
                "username": pl.String,
                "track_id": pl.String,
                "track_name": pl.String,
                "artist_name": pl.String,
                "excluded_at": pl.Datetime(time_zone="UTC"),
            }
        )


def write_excluded_artist(username: str, artist_name: str) -> dict:
    """
    Write an excluded artist to the gold layer excluded_artists table.

    Uses merge mode to avoid duplicates - if the same artist is excluded again,
    the excluded_at timestamp is updated.

    Args:
        username: Username who excluded the artist
        artist_name: Artist name to exclude

    Returns:
        Metadata dict from the write operation
    """
    gold_io = PolarsDeltaIOManager("gold")

    # Create DataFrame with excluded artist
    excluded_df = pl.DataFrame(
        {
            "username": [username],
            "artist_name": [artist_name],
            "excluded_at": [dt.datetime.now(tz=dt.timezone.utc)],
        }
    )

    # Write to Delta table with merge to avoid duplicates
    return gold_io.write_delta(
        df=excluded_df,
        table_name="excluded_artists",
        mode="merge",
        predicate="s.username = t.username AND s.artist_name = t.artist_name",
    )


def read_excluded_artists(username: str) -> pl.LazyFrame:
    """
    Read excluded artists for a user from the gold layer.

    Args:
        username: Username to get artist exclusions for

    Returns:
        LazyFrame with columns: username, artist_name, excluded_at
        Empty LazyFrame if no exclusions exist or table doesn't exist
    """
    gold_io = PolarsDeltaIOManager("gold")

    try:
        return gold_io.read_delta("excluded_artists").filter(
            pl.col("username") == username
        )
    except Exception:
        # Table doesn't exist yet, return empty LazyFrame
        return pl.LazyFrame(
            schema={
                "username": pl.String,
                "artist_name": pl.String,
                "excluded_at": pl.Datetime(time_zone="UTC"),
            }
        )
