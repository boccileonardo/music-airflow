"""
Module for managing excluded track and artist recommendations.

Users can exclude tracks or entire artists from recommendations, and these
exclusions are stored in Firestore for persistence across sessions.
"""

import polars as pl
from music_airflow.utils.firestore_io_manager import FirestoreIOManager


def write_excluded_track(
    username: str, track_id: str, track_name: str, artist_name: str
) -> dict:
    """
    Write an excluded track to Firestore.

    Args:
        username: Username who excluded the track
        track_id: Track identifier
        track_name: Track name
        artist_name: Artist name

    Returns:
        Metadata dict from the write operation
    """
    firestore_io = FirestoreIOManager()
    return firestore_io.write_excluded_track(
        username, track_id, track_name, artist_name
    )


def read_excluded_tracks(username: str) -> pl.LazyFrame:
    """
    Read excluded tracks for a user from Firestore.

    Args:
        username: Username to get exclusions for

    Returns:
        LazyFrame with columns: username, track_id, track_name, artist_name, excluded_at
        Empty LazyFrame if no exclusions exist
    """
    firestore_io = FirestoreIOManager()
    df = firestore_io.read_excluded_tracks(username)
    return df.lazy()


def write_excluded_artist(username: str, artist_name: str) -> dict:
    """
    Write an excluded artist to Firestore.

    Args:
        username: Username who excluded the artist
        artist_name: Artist name to exclude

    Returns:
        Metadata dict from the write operation
    """
    firestore_io = FirestoreIOManager()
    return firestore_io.write_excluded_artist(username, artist_name)


def read_excluded_artists(username: str) -> pl.LazyFrame:
    """
    Read excluded artists for a user from Firestore.

    Args:
        username: Username to get artist exclusions for

    Returns:
        LazyFrame with columns: username, artist_name, excluded_at
        Empty LazyFrame if no exclusions exist
    """
    firestore_io = FirestoreIOManager()
    df = firestore_io.read_excluded_artists(username)
    return df.lazy()


def remove_excluded_track(
    username: str, track_id: str, track_name: str, artist_name: str
) -> dict:
    """
    Remove a track exclusion from Firestore.

    Args:
        username: Username who excluded the track
        track_id: Track identifier
        track_name: Track name (unused, kept for API compatibility)
        artist_name: Artist name (unused, kept for API compatibility)

    Returns:
        Metadata dict from the delete operation
    """
    firestore_io = FirestoreIOManager()
    return firestore_io.delete_excluded_track(username, track_id)


def remove_excluded_artist(username: str, artist_name: str) -> dict:
    """
    Remove an artist exclusion from Firestore.

    Args:
        username: Username who excluded the artist
        artist_name: Artist name to un-exclude

    Returns:
        Metadata dict from the delete operation
    """
    firestore_io = FirestoreIOManager()
    return firestore_io.delete_excluded_artist(username, artist_name)
