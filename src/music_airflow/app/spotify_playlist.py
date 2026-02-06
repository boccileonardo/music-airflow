"""
Spotify playlist generation for music recommendations.

Creates Spotify playlists from track recommendations using Spotify Web API.
Uses spotipy library for API interactions.

Authentication:
--------------
Set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET in .env
Run OAuth flow from Streamlit UI to get user tokens
Tokens stored in SPOTIFY_ACCESS_TOKEN and SPOTIFY_REFRESH_TOKEN

Note: Creating playlists and adding tracks requires user authorization.
The client credentials flow (used for search) is not sufficient.
"""

import asyncio
import logging
import os
import re
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv
import httpx
import polars as pl
import streamlit as st
from spotipy import Spotify

logger = logging.getLogger(__name__)

SPOTIFY_SCOPES = [
    "playlist-modify-public",
    "playlist-modify-private",
    "playlist-read-private",
]
TOKEN_URI = "https://accounts.spotify.com/api/token"
AUTH_URI = "https://accounts.spotify.com/authorize"


@dataclass
class SpotifyOAuthCredentials:
    """OAuth credentials for Spotify API."""

    client_id: str
    client_secret: str
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None

    def has_client_creds(self) -> bool:
        return bool(self.client_id and self.client_secret)

    def has_tokens(self) -> bool:
        return bool(self.refresh_token)


def _get_secret(key: str) -> Optional[str]:
    """Get secret from .env file or Streamlit secrets."""
    load_dotenv()

    value = os.getenv(key)
    if value:
        return value

    try:
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass

    return None


def load_spotify_creds() -> Optional[SpotifyOAuthCredentials]:
    """Load Spotify API OAuth credentials."""
    client_id = _get_secret("SPOTIFY_CLIENT_ID")
    client_secret = _get_secret("SPOTIFY_CLIENT_SECRET")

    if not client_id or not client_secret:
        return None

    return SpotifyOAuthCredentials(
        client_id=client_id,
        client_secret=client_secret,
        access_token=_get_secret("SPOTIFY_ACCESS_TOKEN"),
        refresh_token=_get_secret("SPOTIFY_REFRESH_TOKEN"),
    )


def refresh_spotify_token(
    client_id: str, client_secret: str, refresh_token: str
) -> Optional[dict]:
    """
    Refresh Spotify access token using refresh token.

    Uses async httpx for non-blocking network calls.
    Returns token_info dict if successful, None otherwise.
    """

    async def _refresh() -> Optional[dict]:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    TOKEN_URI,
                    data={
                        "grant_type": "refresh_token",
                        "refresh_token": refresh_token,
                    },
                    auth=(client_id, client_secret),
                )

                if response.status_code == 200:
                    token_data = response.json()
                    return {
                        "access_token": token_data["access_token"],
                        "refresh_token": token_data.get("refresh_token", refresh_token),
                        "expires_in": token_data.get("expires_in"),
                        "token_type": "Bearer",
                    }
                else:
                    logger.error(f"Token refresh failed: {response.text}")
                    return None
        except Exception as e:
            logger.error(f"Token refresh error: {e}")
            return None

    return asyncio.run(_refresh())


def poll_device_token(
    client_id: str, client_secret: str, device_code: str
) -> Optional[dict]:
    """
    Poll once for device authorization token.

    Uses async httpx for non-blocking network calls.
    Returns token_info dict if authorized, None if still pending.
    Raises Exception on permanent errors.
    """

    async def _poll() -> Optional[dict]:
        async with httpx.AsyncClient() as client:
            token_response = await client.post(
                TOKEN_URI,
                data={
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "device_code": device_code,
                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                },
            )

            token_data = token_response.json()

            if "access_token" in token_data:
                return {
                    "access_token": token_data["access_token"],
                    "refresh_token": token_data.get("refresh_token"),
                    "expires_in": token_data.get("expires_in"),
                    "token_type": "Bearer",
                }

            error = token_data.get("error")
            if error == "authorization_pending":
                return None
            elif error == "slow_down":
                return None
            else:
                raise Exception(
                    f"OAuth error: {error} - {token_data.get('error_description', '')}"
                )

    return asyncio.run(_poll())


def run_spotify_oauth(client_id: str, client_secret: str) -> tuple[str, str]:
    """
    Generate Spotify OAuth authorization URL.

    Returns tuple of (auth_url, state).
    User must visit auth_url and authorize, then we get the code from redirect.
    """
    import secrets
    from urllib.parse import urlencode

    state = secrets.token_urlsafe(16)
    # Use loopback IP (not localhost) - must match exactly what's registered in Spotify Dashboard
    redirect_uri = "http://127.0.0.1:8501/"

    params = {
        "client_id": client_id,
        "response_type": "code",
        "redirect_uri": redirect_uri,
        "scope": " ".join(SPOTIFY_SCOPES),
        "state": state,
        "show_dialog": "true",
    }

    # Use urlencode to properly encode spaces and special characters
    auth_url = f"{AUTH_URI}?{urlencode(params)}"
    return auth_url, state


def exchange_code_for_token(
    client_id: str, client_secret: str, code: str
) -> Optional[dict]:
    """
    Exchange authorization code for access/refresh tokens.

    Uses async httpx for non-blocking network calls.
    Returns token_info dict if successful, None otherwise.
    """
    redirect_uri = "http://127.0.0.1:8501/"

    async def _exchange() -> Optional[dict]:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    TOKEN_URI,
                    data={
                        "grant_type": "authorization_code",
                        "code": code,
                        "redirect_uri": redirect_uri,
                    },
                    auth=(client_id, client_secret),
                )

                if response.status_code == 200:
                    token_data = response.json()
                    return {
                        "access_token": token_data["access_token"],
                        "refresh_token": token_data.get("refresh_token"),
                        "expires_in": token_data.get("expires_in"),
                        "token_type": "Bearer",
                    }
                else:
                    logger.error(f"Token exchange failed: {response.text}")
                    return None
        except Exception as e:
            logger.error(f"Token exchange error: {e}")
            return None

    return asyncio.run(_exchange())


class SpotifyPlaylistGenerator:
    """Generate Spotify playlists from track recommendations."""

    def __init__(self):
        self.spotify: Optional[Spotify] = None
        self.user_id: Optional[str] = None
        self.search_cache: dict[str, Optional[str]] = {}

    def authenticate(self) -> bool:
        """Authenticate with Spotify API using stored tokens."""
        creds = load_spotify_creds()
        if not creds or not creds.has_client_creds():
            logger.error("No Spotify client credentials found")
            return False

        if not creds.has_tokens():
            logger.error("No Spotify tokens found - authentication required")
            return False

        # Try to use stored tokens
        access_token = creds.access_token
        refresh_token = creds.refresh_token

        # Try to refresh the token
        if refresh_token:
            token_info = refresh_spotify_token(
                creds.client_id, creds.client_secret, refresh_token
            )
            if token_info:
                access_token = token_info["access_token"]
                self._store_refreshed_token(token_info)
                logger.info("Spotify token refreshed successfully")

        if not access_token:
            logger.error("No valid Spotify access token")
            return False

        try:
            self.spotify = Spotify(auth=access_token)
            # Verify authentication by getting user profile
            user = self.spotify.current_user()
            self.user_id = user["id"]
            logger.info(f"Spotify authenticated as user: {self.user_id}")
            return True
        except Exception as e:
            logger.error(f"Spotify authentication failed: {e}")
            return False

    def _store_refreshed_token(self, token_info: dict) -> None:
        """Store refreshed token in session state for UI display."""
        try:
            st.session_state.refreshed_spotify_token = {
                "access_token": token_info.get("access_token"),
                "refresh_token": token_info.get("refresh_token"),
            }
        except Exception:
            pass

    @staticmethod
    def get_auth_status() -> dict:
        """Check authentication status."""
        spotify = load_spotify_creds()

        return {
            "spotify": {
                "has_client": spotify.has_client_creds() if spotify else False,
                "has_tokens": spotify.has_tokens() if spotify else False,
            },
        }

    @staticmethod
    def needs_authentication() -> bool:
        """Check if authentication is needed."""
        status = SpotifyPlaylistGenerator.get_auth_status()
        return not status["spotify"]["has_tokens"]

    def search_track(self, track_name: str, artist_name: str) -> Optional[str]:
        """
        Search for a track and return track ID.

        Args:
            track_name: Track name to search
            artist_name: Artist name for context

        Returns:
            Track ID if found, None otherwise
        """
        if not self.spotify:
            return None

        query = f"track:{track_name} artist:{artist_name}"

        if query in self.search_cache:
            return self.search_cache[query]

        try:
            results = self.spotify.search(query, type="track", limit=5)
            tracks = results.get("tracks", {}).get("items", [])

            if tracks:
                track_id = tracks[0].get("id")
                self.search_cache[query] = track_id
                return track_id

            # Fallback: simpler search
            fallback_query = f"{track_name} {artist_name}"
            results = self.spotify.search(fallback_query, type="track", limit=5)
            tracks = results.get("tracks", {}).get("items", [])

            if tracks:
                track_id = tracks[0].get("id")
                self.search_cache[query] = track_id
                return track_id

        except Exception as e:
            logger.warning(f"Spotify search error for '{query}': {e}")

        self.search_cache[query] = None
        return None

    def find_playlist_by_title(self, title: str) -> Optional[str]:
        """Find playlist by title in user's playlists."""
        if not self.spotify or not self.user_id:
            return None

        try:
            offset = 0
            limit = 50
            while True:
                response = self.spotify.current_user_playlists(
                    limit=limit, offset=offset
                )
                items = response.get("items", [])
                if not items:
                    break

                for item in items:
                    if item["name"] == title and item["owner"]["id"] == self.user_id:
                        return item["id"]

                if len(items) < limit:
                    break
                offset += limit

        except Exception as e:
            logger.error(f"Find playlist error: {e}")
        return None

    def create_playlist(
        self, title: str, description: str = "", public: bool = True
    ) -> Optional[str]:
        """Create a new playlist."""
        if not self.spotify or not self.user_id:
            return None

        try:
            response = self.spotify.user_playlist_create(
                user=self.user_id,
                name=title,
                public=public,
                description=description,
            )
            return response["id"]
        except Exception as e:
            logger.error(f"Create playlist error: {e}")
        return None

    def delete_playlist(self, playlist_id: str) -> bool:
        """Unfollow (effectively delete) a playlist."""
        if not self.spotify:
            return False

        try:
            self.spotify.current_user_unfollow_playlist(playlist_id)
            return True
        except Exception as e:
            logger.error(f"Delete playlist error: {e}")
        return False

    def add_tracks_to_playlist(self, playlist_id: str, track_ids: list[str]) -> bool:
        """Add tracks to playlist (max 100 at a time)."""
        if not self.spotify or not track_ids:
            return False

        try:
            # Spotify accepts up to 100 tracks at a time
            track_uris = [f"spotify:track:{tid}" for tid in track_ids]
            for i in range(0, len(track_uris), 100):
                batch = track_uris[i : i + 100]
                self.spotify.playlist_add_items(playlist_id, batch)
            return True
        except Exception as e:
            logger.error(f"Add tracks error: {e}")
            return False

    def create_playlist_from_tracks(
        self,
        tracks_df: pl.DataFrame,
        playlist_title: str,
        playlist_description: str = "",
        public: bool = True,
        progress_bar=None,
        status_text=None,
    ) -> Optional[dict]:
        """Create playlist from DataFrame of tracks."""
        if not self.spotify:
            if not self.authenticate():
                return None

        has_urls = "spotify_url" in tracks_df.columns

        # Delete existing playlist if found
        if status_text:
            status_text.text("Checking for existing playlist...")

        existing = self.find_playlist_by_title(playlist_title)
        if existing:
            if status_text:
                status_text.text("Removing old playlist...")
            self.delete_playlist(existing)

        # Create new playlist
        if status_text:
            status_text.text("Creating playlist...")

        playlist_id = self.create_playlist(playlist_title, playlist_description, public)
        if not playlist_id:
            return None

        # Collect track IDs from stored URLs or search
        tracks_to_add: list[str] = []
        tracks_missing_url: list[str] = []
        tracks_not_found: list[str] = []

        total_tracks = len(tracks_df)
        for i, row in enumerate(tracks_df.iter_rows(named=True)):
            track_name = row.get("track_name", "Unknown")
            artist_name = row.get("artist_name", "Unknown")
            track_label = f"{track_name} - {artist_name}"

            if progress_bar:
                progress_bar.progress((i + 1) / total_tracks)
            if status_text:
                status_text.text(f"Processing {i + 1}/{total_tracks}: {track_label}")

            track_id = None

            # Try to get track ID from URL
            if has_urls and row.get("spotify_url"):
                track_id = self._extract_track_id(row["spotify_url"])

            # Fallback: search for the track
            if not track_id:
                tracks_missing_url.append(track_label)
                track_id = self.search_track(track_name, artist_name)

            if track_id:
                tracks_to_add.append(track_id)
            else:
                tracks_not_found.append(track_label)

        # Add all tracks at once (more efficient)
        if status_text:
            status_text.text(f"Adding {len(tracks_to_add)} tracks to playlist...")

        if tracks_to_add:
            success = self.add_tracks_to_playlist(playlist_id, tracks_to_add)
            if not success:
                logger.error("Failed to add tracks to playlist")

        result = {
            "playlist_id": playlist_id,
            "playlist_url": f"https://open.spotify.com/playlist/{playlist_id}",
            "tracks_added": len(tracks_to_add),
            "tracks_not_found": tracks_not_found,
            "tracks_missing_url": tracks_missing_url,
        }

        return result

    @staticmethod
    def _extract_track_id(url: str) -> Optional[str]:
        """Extract track ID from Spotify URL."""
        # Handle URLs like:
        # https://open.spotify.com/track/4uLU6hMCjMI75M1A2tKUQC
        # spotify:track:4uLU6hMCjMI75M1A2tKUQC
        patterns = [
            r"spotify\.com/track/([a-zA-Z0-9]{22})",
            r"spotify:track:([a-zA-Z0-9]{22})",
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None

    @staticmethod
    def get_playlist_url(playlist_id: str) -> str:
        return f"https://open.spotify.com/playlist/{playlist_id}"
