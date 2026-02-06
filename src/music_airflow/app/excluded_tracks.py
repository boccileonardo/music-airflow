"""
Module for managing excluded track and artist recommendations.

Users can exclude tracks or entire artists from recommendations, and these
exclusions are stored in Firestore for persistence across sessions.

Read operations use async Firestore for better UI responsiveness.
Write/delete operations use sync Firestore (acceptable for occasional user actions).
"""

import asyncio

import polars as pl

from music_airflow.utils.firestore_async import AsyncFirestoreReader
from music_airflow.utils.firestore_io_manager import FirestoreIOManager


def _run_async(coro):
    """Run an async coroutine from sync context.

    Creates a fresh AsyncFirestoreReader for each call to avoid event loop issues.
    """
    return asyncio.run(coro)


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
    Read excluded tracks for a user from Firestore asynchronously.

    Args:
        username: Username to get exclusions for

    Returns:
        LazyFrame with columns: username, track_id, track_name, artist_name, excluded_at
        Empty LazyFrame if no exclusions exist
    """

    async def _read():
        reader = AsyncFirestoreReader()
        return await reader.read_excluded_tracks(username)

    df = _run_async(_read())
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
    Read excluded artists for a user from Firestore asynchronously.

    Args:
        username: Username to get artist exclusions for

    Returns:
        LazyFrame with columns: username, artist_name, excluded_at
        Empty LazyFrame if no exclusions exist
    """

    async def _read():
        reader = AsyncFirestoreReader()
        return await reader.read_excluded_artists(username)

    df = _run_async(_read())
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
