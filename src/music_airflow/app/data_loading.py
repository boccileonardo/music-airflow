"""
Data loading functions for the Streamlit app.

Handles loading and caching of track candidates, user statistics,
and artist data from Firestore using async operations for better UI responsiveness.
"""

import asyncio

import polars as pl
import streamlit as st

from music_airflow.utils.constants import LAST_FM_USERNAMES
from music_airflow.utils.firestore_async import AsyncFirestoreReader

# Internal limit for caching - always compute up to this to reuse cache
INTERNAL_LIMIT = 500


def _run_async(coro):
    """Run an async coroutine from sync context.

    Uses asyncio.run() which works reliably with Streamlit's execution model.
    """
    return asyncio.run(coro)


@st.cache_data(ttl=300)  # 5 minutes
def load_track_candidates_cached(username: str) -> pl.DataFrame:
    """Load and cache track candidates from Firestore asynchronously."""

    async def _load():
        reader = AsyncFirestoreReader()
        return await reader.read_track_candidates(username, limit=INTERNAL_LIMIT)

    df = _run_async(_load())
    return df.sort("score", descending=True)


def prefetch_all_users_track_candidates() -> None:
    """Prefetch track candidates for all users to warm the cache.

    This ensures instant user switching by loading all users' data upfront.
    The cached data is reused by load_track_candidates_cached.
    """
    for username in LAST_FM_USERNAMES:
        load_track_candidates_cached(username)


def load_track_candidates(username: str) -> pl.LazyFrame:
    """Load track candidates from Firestore - presentation-ready data."""
    return load_track_candidates_cached(username).lazy()


@st.cache_data(ttl="1d")
def load_user_statistics(username: str) -> dict:
    """Load user play statistics from Firestore asynchronously."""

    async def _load():
        reader = AsyncFirestoreReader()
        stats = await reader.read_user_stats(username)

        if "total_artists_played" not in stats:
            artist_plays = await reader.read_artist_play_counts(username)
            stats["total_artists_played"] = len(artist_plays)

        return stats

    return _run_async(_load())


@st.cache_data(ttl="1d")
def load_top_artists(username: str, limit: int = 10) -> pl.DataFrame:
    """Load top artists by play count from Firestore asynchronously."""

    async def _load():
        reader = AsyncFirestoreReader()
        artist_plays = await reader.read_artist_play_counts(username, limit=limit)
        return artist_plays.filter(pl.col("artist_name").is_not_null())

    try:
        return _run_async(_load())
    except Exception:
        return pl.DataFrame(schema={"artist_name": pl.String, "play_count": pl.Int64})
