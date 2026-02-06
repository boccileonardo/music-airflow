"""
OAuth token storage for Spotify and YouTube Music.

Stores user-specific OAuth tokens in Firestore, enabling multi-tenant
authentication where each user can connect their own accounts.

Document Structure:
    users/{username}/oauth_tokens/spotify - Spotify OAuth tokens
    users/{username}/oauth_tokens/youtube - YouTube OAuth tokens

Each token document contains:
    - access_token: Current access token
    - refresh_token: Token for refreshing access
    - token_type: Usually "Bearer"
    - expires_at: Timestamp when access_token expires (optional)
    - updated_at: When tokens were last updated
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from google.cloud.firestore import SERVER_TIMESTAMP

from music_airflow.utils.firestore_io_manager import get_firestore_client

logger = logging.getLogger(__name__)


class OAuthTokenStorage:
    """Storage for OAuth tokens in Firestore."""

    COLLECTION = "oauth_tokens"

    def __init__(self):
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = get_firestore_client()
        return self._client

    def _get_token_doc_ref(self, username: str, provider: str):
        """Get reference to token document."""
        return (
            self.client.collection("users")
            .document(username)
            .collection(self.COLLECTION)
            .document(provider)
        )

    def save_tokens(
        self,
        username: str,
        provider: str,
        access_token: str,
        refresh_token: str,
        expires_in: Optional[int] = None,
    ) -> bool:
        """
        Save OAuth tokens for a user.

        Args:
            username: The user to save tokens for
            provider: 'spotify' or 'youtube'
            access_token: The access token
            refresh_token: The refresh token
            expires_in: Optional seconds until token expires

        Returns:
            True if saved successfully
        """
        try:
            doc_ref = self._get_token_doc_ref(username, provider)

            data = {
                "access_token": access_token,
                "refresh_token": refresh_token,
                "token_type": "Bearer",
                "updated_at": SERVER_TIMESTAMP,
            }

            if expires_in:
                expires_at = datetime.now(timezone.utc).timestamp() + expires_in
                data["expires_at"] = expires_at

            doc_ref.set(data)
            logger.info(f"Saved {provider} tokens for user {username}")
            return True

        except Exception as e:
            logger.error(f"Failed to save {provider} tokens for {username}: {e}")
            return False

    def get_tokens(self, username: str, provider: str) -> Optional[dict]:
        """
        Get OAuth tokens for a user.

        Args:
            username: The user to get tokens for
            provider: 'spotify' or 'youtube'

        Returns:
            Token dict with access_token, refresh_token, etc. or None
        """
        try:
            doc_ref = self._get_token_doc_ref(username, provider)
            doc = doc_ref.get()

            if doc.exists:
                return doc.to_dict()
            return None

        except Exception as e:
            logger.error(f"Failed to get {provider} tokens for {username}: {e}")
            return None

    def update_access_token(
        self,
        username: str,
        provider: str,
        access_token: str,
        refresh_token: Optional[str] = None,
        expires_in: Optional[int] = None,
    ) -> bool:
        """
        Update access token after refresh.

        Args:
            username: The user to update tokens for
            provider: 'spotify' or 'youtube'
            access_token: New access token
            refresh_token: New refresh token (if provided by OAuth provider)
            expires_in: Seconds until token expires

        Returns:
            True if updated successfully
        """
        try:
            doc_ref = self._get_token_doc_ref(username, provider)

            data = {
                "access_token": access_token,
                "updated_at": SERVER_TIMESTAMP,
            }

            if refresh_token:
                data["refresh_token"] = refresh_token

            if expires_in:
                expires_at = datetime.now(timezone.utc).timestamp() + expires_in
                data["expires_at"] = expires_at

            doc_ref.update(data)
            logger.info(f"Updated {provider} access token for user {username}")
            return True

        except Exception as e:
            logger.error(f"Failed to update {provider} token for {username}: {e}")
            return False

    def delete_tokens(self, username: str, provider: str) -> bool:
        """
        Delete OAuth tokens for a user (disconnect).

        Args:
            username: The user to delete tokens for
            provider: 'spotify' or 'youtube'

        Returns:
            True if deleted successfully
        """
        try:
            doc_ref = self._get_token_doc_ref(username, provider)
            doc_ref.delete()
            logger.info(f"Deleted {provider} tokens for user {username}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete {provider} tokens for {username}: {e}")
            return False

    def has_tokens(self, username: str, provider: str) -> bool:
        """Check if user has tokens for a provider."""
        tokens = self.get_tokens(username, provider)
        return tokens is not None and bool(tokens.get("refresh_token"))


# Singleton instance
_storage: Optional[OAuthTokenStorage] = None


def get_oauth_storage() -> OAuthTokenStorage:
    """Get singleton OAuth storage instance."""
    global _storage
    if _storage is None:
        _storage = OAuthTokenStorage()
    return _storage
