"""
Async IO Manager for Firestore read operations in Streamlit.

Provides async versions of the most frequently used read operations for the
gold layer, optimized for the Streamlit UI where responsiveness is key.

The async client avoids blocking the Streamlit event loop during Firestore
network calls, making the UI feel more responsive during data loading.

Authentication (in order of preference):
1. Streamlit Cloud: st.secrets["gcp_service_account"] (JSON key contents)
2. Local: GOOGLE_APPLICATION_CREDENTIALS env var or ADC
3. On GCP: Automatically uses service account credentials

Note: Write operations remain synchronous as they are typically:
1. Called from background tasks (Airflow DAGs)
2. Don't need UI responsiveness
"""

import os
import logging
from typing import Any

from dotenv import load_dotenv
import polars as pl
from google.cloud.firestore import AsyncClient, AsyncQuery
from google.oauth2 import service_account

load_dotenv()

logger = logging.getLogger(__name__)


def _get_streamlit_credentials() -> tuple[
    service_account.Credentials | None, str | None, str | None
]:
    """
    Try to get GCP credentials from Streamlit secrets.

    Returns:
        Tuple of (credentials, project_id, database_id) or (None, None, None)
    """
    try:
        import streamlit as st

        if "gcp_service_account" in st.secrets:
            creds_info = dict(st.secrets["gcp_service_account"])
            credentials = service_account.Credentials.from_service_account_info(
                creds_info
            )
            project_id = st.secrets.get("GOOGLE_CLOUD_PROJECT") or creds_info.get(
                "project_id"
            )
            database_id = st.secrets.get("FIRESTORE_DATABASE_ID")
            return credentials, project_id, database_id
    except Exception:
        pass
    return None, None, None


def get_async_firestore_client() -> AsyncClient:
    """
    Get async Firestore client with appropriate credentials.

    Supports:
    - Streamlit Cloud: Uses st.secrets["gcp_service_account"]
    - Local/Airflow: Uses environment variables and ADC

    Returns:
        Async Firestore client instance
    """
    # Try Streamlit secrets first (for Streamlit Cloud deployment)
    credentials, st_project_id, st_database_id = _get_streamlit_credentials()

    if credentials:
        project_id = st_project_id
        database_id = st_database_id
        if database_id:
            return AsyncClient(
                project=project_id, database=database_id, credentials=credentials
            )
        return AsyncClient(project=project_id, credentials=credentials)

    # Fall back to environment variables and ADC (local/Airflow)
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        raise ValueError(
            "GOOGLE_CLOUD_PROJECT environment variable is required for Firestore. "
            "Set it in .env or run `gcloud config set project <project-id>`"
        )

    database_id = os.getenv("FIRESTORE_DATABASE_ID")

    if database_id:
        return AsyncClient(project=project_id, database=database_id)
    return AsyncClient(project=project_id)


class AsyncFirestoreReader:
    """
    Async reader for Firestore gold layer data.

    Optimized for fast reads in Streamlit where UI responsiveness matters.
    Only implements read operations - writes should use the sync FirestoreIOManager.
    """

    def __init__(self):
        """Initialize async Firestore reader."""
        self._client: AsyncClient | None = None

    @property
    def client(self) -> AsyncClient:
        """Lazy initialization of async Firestore client."""
        if self._client is None:
            self._client = get_async_firestore_client()
        return self._client

    async def read_track_candidates(
        self,
        username: str,
        limit: int | None = None,
    ) -> pl.DataFrame:
        """
        Read track candidates for a user from Firestore asynchronously.

        Args:
            username: Target user
            limit: Optional limit on number of candidates

        Returns:
            DataFrame with track candidates, sorted by score descending
        """
        collection_ref = (
            self.client.collection("users")
            .document(username)
            .collection("track_candidates")
        )

        query = collection_ref.order_by("score", direction=AsyncQuery.DESCENDING)
        if limit:
            query = query.limit(limit)

        rows = []
        async for doc in query.stream():
            rows.append(doc.to_dict())

        if not rows:
            return pl.DataFrame(schema=self._get_candidates_schema())

        return pl.DataFrame(rows).select(self._get_candidates_schema().keys())

    async def read_user_stats(self, username: str) -> dict[str, Any]:
        """
        Read user statistics from Firestore asynchronously.

        Args:
            username: Target user

        Returns:
            Dict with user statistics, or empty defaults
        """
        doc_ref = (
            self.client.collection("users")
            .document(username)
            .collection("stats")
            .document("profile")
        )
        doc = await doc_ref.get()

        if doc.exists:
            return doc.to_dict() or {}

        return {
            "total_plays": 0,
            "total_tracks_played": 0,
            "total_artists_played": 0,
        }

    async def read_artist_play_counts(
        self,
        username: str,
        limit: int | None = None,
    ) -> pl.DataFrame:
        """
        Read artist play counts for a user asynchronously.

        Args:
            username: Target user
            limit: Optional limit

        Returns:
            DataFrame with artist play counts
        """
        collection_ref = (
            self.client.collection("aggregations")
            .document(username)
            .collection("artist_play_count")
        )

        query = collection_ref.order_by("play_count", direction=AsyncQuery.DESCENDING)
        if limit:
            query = query.limit(limit)

        rows = []
        async for doc in query.stream():
            rows.append(doc.to_dict())

        schema = {"artist_name": pl.String, "play_count": pl.Int64}

        if not rows:
            return pl.DataFrame(schema=schema)

        return pl.DataFrame(rows, schema=schema)

    async def read_excluded_tracks(self, username: str) -> pl.DataFrame:
        """
        Read excluded tracks for a user asynchronously.

        Args:
            username: Target user

        Returns:
            DataFrame with excluded tracks
        """
        collection_ref = (
            self.client.collection("users")
            .document(username)
            .collection("excluded_tracks")
        )

        rows = []
        async for doc in collection_ref.stream():
            rows.append(doc.to_dict())

        schema = {
            "username": pl.String,
            "track_id": pl.String,
            "track_name": pl.String,
            "artist_name": pl.String,
            "excluded_at": pl.Datetime(time_zone="UTC"),
        }

        if not rows:
            return pl.DataFrame(schema=schema)

        return pl.DataFrame(rows, schema=schema)

    async def read_excluded_artists(self, username: str) -> pl.DataFrame:
        """
        Read excluded artists for a user asynchronously.

        Args:
            username: Target user

        Returns:
            DataFrame with excluded artists
        """
        collection_ref = (
            self.client.collection("users")
            .document(username)
            .collection("excluded_artists")
        )

        rows = []
        async for doc in collection_ref.stream():
            rows.append(doc.to_dict())

        schema = {
            "username": pl.String,
            "artist_name": pl.String,
            "excluded_at": pl.Datetime(time_zone="UTC"),
        }

        if not rows:
            return pl.DataFrame(schema=schema)

        return pl.DataFrame(rows, schema=schema)

    def _get_candidates_schema(self) -> dict:
        """Get the expected schema for track candidates."""
        return {
            "username": pl.String,
            "track_id": pl.String,
            "track_name": pl.String,
            "artist_name": pl.String,
            "score": pl.Float64,
            "similar_artist": pl.Boolean,
            "similar_tag": pl.Boolean,
            "deep_cut_same_artist": pl.Boolean,
            "old_favorite": pl.Boolean,
            "why_similar_artist_name": pl.String,
            "why_similar_artist_pct": pl.Float64,
            "why_similar_tags": pl.String,
            "why_tag_match_count": pl.Int64,
            "why_deep_cut_artist": pl.String,
            "duration_ms": pl.Int64,
            "tags": pl.String,
            "youtube_url": pl.String,
            "spotify_url": pl.String,
        }
