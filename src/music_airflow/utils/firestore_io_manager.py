"""
IO Manager for handling Firestore read/write operations for the gold layer.

This module provides a centralized interface for Firestore operations optimized
for the low-latency serving layer. Designed for small document collections
(~300 items per user) with fast read access from Streamlit.

Firestore Configuration:
    Set GOOGLE_CLOUD_PROJECT in .env (or use GCP default project).
    Set FIRESTORE_DATABASE_ID in .env (optional, uses default database).

    Authentication (in order of preference):
    1. Streamlit Cloud: st.secrets["gcp_service_account"] (JSON key contents)
    2. Local: GOOGLE_APPLICATION_CREDENTIALS env var or ADC
    3. On GCP: Automatically uses service account credentials

Document Structure:
    users/{username}/
        track_candidates/{track_id} - recommendation candidates
        stats/profile - user statistics (plays, tracks, artists)
        excluded_tracks/{track_id} - user track exclusions
        excluded_artists/{artist_name} - user artist exclusions

    aggregations/{username}/
        artist_play_count/{artist_id} - artist play statistics
        track_play_count/{track_id} - track play statistics
"""

import os
import logging
from typing import Any

from dotenv import load_dotenv
import polars as pl
from google.cloud import firestore
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


def get_firestore_client() -> firestore.Client:
    """
    Get Firestore client with appropriate credentials.

    Supports:
    - Streamlit Cloud: Uses st.secrets["gcp_service_account"]
    - Local/Airflow: Uses environment variables and ADC

    Returns:
        Firestore client instance
    """
    # Try Streamlit secrets first (for Streamlit Cloud deployment)
    credentials, st_project_id, st_database_id = _get_streamlit_credentials()

    if credentials:
        project_id = st_project_id
        database_id = st_database_id
        if database_id:
            return firestore.Client(
                project=project_id, database=database_id, credentials=credentials
            )
        return firestore.Client(project=project_id, credentials=credentials)

    # Fall back to environment variables and ADC (local/Airflow)
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        raise ValueError(
            "GOOGLE_CLOUD_PROJECT environment variable is required for Firestore. "
            "Set it in .env or run `gcloud config set project <project-id>`"
        )

    database_id = os.getenv("FIRESTORE_DATABASE_ID")

    if database_id:
        return firestore.Client(project=project_id, database=database_id)
    return firestore.Client(project=project_id)


class FirestoreIOManager:
    """
    Manager for reading and writing to Firestore for gold layer serving.

    Optimized for:
    - Fast reads by username (Streamlit app)
    - Batch writes from Airflow DAGs
    - Small document collections (~300 items per user)

    Methods:
        write_track_candidates: Write track candidates for a user
        read_track_candidates: Read track candidates for a user
        write_user_stats: Write aggregated user statistics
        read_user_stats: Read user statistics
        write_play_counts: Write artist/track play counts
        read_play_counts: Read artist/track play counts
        write_exclusion: Write a track/artist exclusion
        read_exclusions: Read exclusions for a user
        delete_exclusion: Remove an exclusion
    """

    def __init__(self):
        """Initialize Firestore IO Manager."""
        self._client: firestore.Client | None = None

    @property
    def client(self) -> firestore.Client:
        """Lazy initialization of Firestore client."""
        if self._client is None:
            self._client = get_firestore_client()
        return self._client

    def write_track_candidates(
        self,
        username: str,
        candidates_df: pl.DataFrame,
    ) -> dict[str, Any]:
        """
        Write track candidates for a user to Firestore.

        Replaces all existing candidates for the user with new ones.
        Uses batch writes for efficiency.

        Args:
            username: Target user
            candidates_df: DataFrame with track candidates

        Returns:
            Metadata dict with rows written
        """
        collection_ref = (
            self.client.collection("users")
            .document(username)
            .collection("track_candidates")
        )

        # Delete existing candidates
        self._delete_collection(collection_ref)

        # Write new candidates in batches
        batch = self.client.batch()
        batch_count = 0
        total_written = 0

        for row in candidates_df.iter_rows(named=True):
            doc_ref = collection_ref.document(row["track_id"])
            # Convert row to dict, handling None values
            doc_data = {k: v for k, v in row.items() if v is not None}
            batch.set(doc_ref, doc_data)
            batch_count += 1
            total_written += 1

            # Firestore batch limit is 500
            if batch_count >= 400:
                batch.commit()
                batch = self.client.batch()
                batch_count = 0

        # Commit remaining
        if batch_count > 0:
            batch.commit()

        logger.info(f"Wrote {total_written} track candidates for {username}")
        return {
            "rows": total_written,
            "username": username,
            "collection": "track_candidates",
        }

    def read_track_candidates(
        self,
        username: str,
        limit: int | None = None,
    ) -> pl.DataFrame:
        """
        Read track candidates for a user from Firestore.

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

        query = collection_ref.order_by("score", direction=firestore.Query.DESCENDING)
        if limit:
            query = query.limit(limit)

        docs = query.stream()
        rows = [doc.to_dict() for doc in docs]

        if not rows:
            return pl.DataFrame(schema=self._get_candidates_schema())

        return pl.DataFrame(rows).select(self._get_candidates_schema().keys())

    def write_user_stats(
        self,
        username: str,
        stats: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Write aggregated user statistics to Firestore.

        Args:
            username: Target user
            stats: Dict with total_plays, total_tracks_played, total_artists_played

        Returns:
            Metadata dict
        """
        doc_ref = (
            self.client.collection("users")
            .document(username)
            .collection("stats")
            .document("profile")
        )
        doc_ref.set(stats)

        logger.info(f"Wrote user stats for {username}")
        return {"username": username, "collection": "stats"}

    def read_user_stats(self, username: str) -> dict[str, Any]:
        """
        Read user statistics from Firestore.

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
        doc = doc_ref.get()

        if doc.exists:
            return doc.to_dict()

        return {
            "total_plays": 0,
            "total_tracks_played": 0,
            "total_artists_played": 0,
        }

    def write_artist_play_counts(
        self,
        username: str,
        play_counts_df: pl.DataFrame,
    ) -> dict[str, Any]:
        """
        Write artist play counts for a user.

        Args:
            username: Target user
            play_counts_df: DataFrame with artist play counts

        Returns:
            Metadata dict with rows written
        """
        collection_ref = (
            self.client.collection("aggregations")
            .document(username)
            .collection("artist_play_count")
        )

        # Delete existing
        self._delete_collection(collection_ref)

        # Write new in batches
        batch = self.client.batch()
        batch_count = 0
        total_written = 0

        for row in play_counts_df.iter_rows(named=True):
            doc_id = row.get("artist_id") or row.get("artist_name", "unknown")
            doc_ref = collection_ref.document(doc_id)
            doc_data = self._serialize_row(row)
            batch.set(doc_ref, doc_data)
            batch_count += 1
            total_written += 1

            if batch_count >= 400:
                batch.commit()
                batch = self.client.batch()
                batch_count = 0

        if batch_count > 0:
            batch.commit()

        logger.info(f"Wrote {total_written} artist play counts for {username}")
        return {
            "rows": total_written,
            "username": username,
            "collection": "artist_play_count",
        }

    def read_artist_play_counts(
        self,
        username: str,
        limit: int | None = None,
    ) -> pl.DataFrame:
        """
        Read artist play counts for a user.

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

        query = collection_ref.order_by(
            "play_count", direction=firestore.Query.DESCENDING
        )
        if limit:
            query = query.limit(limit)

        docs = query.stream()
        rows = [doc.to_dict() for doc in docs]

        if not rows:
            return pl.DataFrame(
                schema={"artist_name": pl.String, "play_count": pl.Int64}
            )

        return pl.DataFrame(rows)

    def write_track_play_counts(
        self,
        username: str,
        play_counts_df: pl.DataFrame,
    ) -> dict[str, Any]:
        """
        Write track play counts for a user.

        Args:
            username: Target user
            play_counts_df: DataFrame with track play counts

        Returns:
            Metadata dict with rows written
        """
        collection_ref = (
            self.client.collection("aggregations")
            .document(username)
            .collection("track_play_count")
        )

        # Delete existing
        self._delete_collection(collection_ref)

        # Write new in batches
        batch = self.client.batch()
        batch_count = 0
        total_written = 0

        for row in play_counts_df.iter_rows(named=True):
            doc_id = row.get("track_id", "unknown")
            doc_ref = collection_ref.document(doc_id)
            doc_data = self._serialize_row(row)
            batch.set(doc_ref, doc_data)
            batch_count += 1
            total_written += 1

            if batch_count >= 400:
                batch.commit()
                batch = self.client.batch()
                batch_count = 0

        if batch_count > 0:
            batch.commit()

        logger.info(f"Wrote {total_written} track play counts for {username}")
        return {
            "rows": total_written,
            "username": username,
            "collection": "track_play_count",
        }

    def read_track_play_counts(
        self,
        username: str,
    ) -> pl.DataFrame:
        """
        Read track play counts for a user.

        Args:
            username: Target user

        Returns:
            DataFrame with track play counts
        """
        collection_ref = (
            self.client.collection("aggregations")
            .document(username)
            .collection("track_play_count")
        )

        docs = collection_ref.stream()
        rows = [doc.to_dict() for doc in docs]

        if not rows:
            return pl.DataFrame(schema={"track_id": pl.String, "play_count": pl.Int64})

        return pl.DataFrame(rows)

    def write_excluded_track(
        self,
        username: str,
        track_id: str,
        track_name: str,
        artist_name: str,
    ) -> dict[str, Any]:
        """
        Write a track exclusion for a user.

        Args:
            username: Target user
            track_id: Track identifier
            track_name: Track name
            artist_name: Artist name

        Returns:
            Metadata dict
        """
        import datetime as dt

        doc_ref = (
            self.client.collection("users")
            .document(username)
            .collection("excluded_tracks")
            .document(track_id)
        )
        doc_ref.set(
            {
                "username": username,
                "track_id": track_id,
                "track_name": track_name,
                "artist_name": artist_name,
                "excluded_at": dt.datetime.now(tz=dt.timezone.utc),
            }
        )

        return {"username": username, "track_id": track_id}

    def read_excluded_tracks(self, username: str) -> pl.DataFrame:
        """
        Read excluded tracks for a user.

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

        docs = collection_ref.stream()
        rows = [doc.to_dict() for doc in docs]

        if not rows:
            return pl.DataFrame(
                schema={
                    "username": pl.String,
                    "track_id": pl.String,
                    "track_name": pl.String,
                    "artist_name": pl.String,
                    "excluded_at": pl.Datetime(time_zone="UTC"),
                }
            )

        return pl.DataFrame(rows)

    def delete_excluded_track(
        self,
        username: str,
        track_id: str,
    ) -> dict[str, Any]:
        """
        Remove a track exclusion.

        Args:
            username: Target user
            track_id: Track identifier

        Returns:
            Metadata dict
        """
        doc_ref = (
            self.client.collection("users")
            .document(username)
            .collection("excluded_tracks")
            .document(track_id)
        )
        doc_ref.delete()

        return {"username": username, "track_id": track_id, "deleted": True}

    def write_excluded_artist(
        self,
        username: str,
        artist_name: str,
    ) -> dict[str, Any]:
        """
        Write an artist exclusion for a user.

        Args:
            username: Target user
            artist_name: Artist name

        Returns:
            Metadata dict
        """
        import datetime as dt

        # Use a sanitized version of artist_name as doc ID
        doc_id = artist_name.replace("/", "_").replace("\\", "_")
        doc_ref = (
            self.client.collection("users")
            .document(username)
            .collection("excluded_artists")
            .document(doc_id)
        )
        doc_ref.set(
            {
                "username": username,
                "artist_name": artist_name,
                "excluded_at": dt.datetime.now(tz=dt.timezone.utc),
            }
        )

        return {"username": username, "artist_name": artist_name}

    def read_excluded_artists(self, username: str) -> pl.DataFrame:
        """
        Read excluded artists for a user.

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

        docs = collection_ref.stream()
        rows = [doc.to_dict() for doc in docs]

        if not rows:
            return pl.DataFrame(
                schema={
                    "username": pl.String,
                    "artist_name": pl.String,
                    "excluded_at": pl.Datetime(time_zone="UTC"),
                }
            )

        return pl.DataFrame(rows)

    def delete_excluded_artist(
        self,
        username: str,
        artist_name: str,
    ) -> dict[str, Any]:
        """
        Remove an artist exclusion.

        Args:
            username: Target user
            artist_name: Artist name

        Returns:
            Metadata dict
        """
        doc_id = artist_name.replace("/", "_").replace("\\", "_")
        doc_ref = (
            self.client.collection("users")
            .document(username)
            .collection("excluded_artists")
            .document(doc_id)
        )
        doc_ref.delete()

        return {"username": username, "artist_name": artist_name, "deleted": True}

    def _delete_collection(self, collection_ref, batch_size: int = 400) -> int:
        """
        Delete all documents in a collection.

        Args:
            collection_ref: Firestore collection reference
            batch_size: Number of documents to delete per batch

        Returns:
            Number of documents deleted
        """
        deleted = 0
        docs = collection_ref.limit(batch_size).stream()
        doc_list = list(docs)

        while doc_list:
            batch = self.client.batch()
            for doc in doc_list:
                batch.delete(doc.reference)
                deleted += 1
            batch.commit()

            docs = collection_ref.limit(batch_size).stream()
            doc_list = list(docs)

        return deleted

    def _serialize_row(self, row: dict) -> dict:
        """
        Serialize a row for Firestore, converting polars types.

        Args:
            row: Dict from DataFrame.iter_rows(named=True)

        Returns:
            Firestore-compatible dict
        """
        result = {}
        for k, v in row.items():
            if v is None:
                continue
            # Convert numpy/polars types to Python native
            if hasattr(v, "item"):
                result[k] = v.item()
            else:
                result[k] = v
        return result

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
