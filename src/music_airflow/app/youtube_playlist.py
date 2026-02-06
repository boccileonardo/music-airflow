"""
YouTube playlist generation for music recommendations.

Creates YouTube playlists from track recommendations using YouTube Data API v3.
Uses YouTube Music API (ytmusicapi) for searching to find audio versions
instead of music videos, avoiding quota usage for search operations.

Authentication:
--------------
Set YOUTUBE_CLIENT_ID and YOUTUBE_CLIENT_SECRET in .env or Streamlit secrets.
Users authenticate via Device Authorization Flow in the Streamlit app.
Tokens are stored per-user in Firestore for multi-tenant support.

Note: YouTube Data API has quota limits (10,000 units/day).
Playlist creation costs 50 units per insert.
Search now uses ytmusicapi (no quota cost) instead of YouTube Data API.
"""

import asyncio
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build, Resource
from googleapiclient.errors import HttpError
import httpx
import polars as pl
import streamlit as st
from ytmusicapi import YTMusic

from music_airflow.app.oauth_storage import get_oauth_storage
from music_airflow.utils.constants import (
    DEFAULT_USERNAME,
    YOUTUBE_MAX_TRACKS_DEFAULT,
    YOUTUBE_MAX_TRACKS_OWNER,
)

logger = logging.getLogger(__name__)

YOUTUBE_SCOPES = ["https://www.googleapis.com/auth/youtube.force-ssl"]
TOKEN_URI = "https://oauth2.googleapis.com/token"


@dataclass
class OAuthCredentials:
    """OAuth credentials for YouTube API (app-level only)."""

    client_id: str
    client_secret: str

    def has_client_creds(self) -> bool:
        return bool(self.client_id and self.client_secret)


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


def load_youtube_creds() -> Optional[OAuthCredentials]:
    """Load YouTube Data API OAuth credentials (app-level only)."""
    client_id = _get_secret("YOUTUBE_CLIENT_ID")
    client_secret = _get_secret("YOUTUBE_CLIENT_SECRET")

    if not client_id or not client_secret:
        return None

    return OAuthCredentials(
        client_id=client_id,
        client_secret=client_secret,
    )


AUTH_URI = "https://accounts.google.com/o/oauth2/v2/auth"


def get_youtube_redirect_uri() -> str:
    """Get the redirect URI for YouTube OAuth based on deployment context."""
    # Check for explicit override
    override = _get_secret("YOUTUBE_REDIRECT_URI")
    if override:
        return override

    # Use Streamlit's URL if available (works on Streamlit Cloud)
    try:
        if "streamlit_url" in st.session_state:
            return st.session_state["streamlit_url"]

        # Default to loopback IP for local development
        return "http://127.0.0.1:8501/"
    except Exception:
        return "http://127.0.0.1:8501/"


def run_youtube_oauth(
    client_id: str, client_secret: str, username: str
) -> tuple[str, str]:
    """
    Generate YouTube OAuth authorization URL.

    Returns tuple of (auth_url, state).
    State includes username to persist across redirect.
    User must visit auth_url and authorize, then we get the code from redirect.
    """
    import secrets
    from urllib.parse import urlencode

    # Encode username in state to survive page reload
    nonce = secrets.token_urlsafe(16)
    state = f"youtube:{username}:{nonce}"
    redirect_uri = get_youtube_redirect_uri()

    params = {
        "client_id": client_id,
        "response_type": "code",
        "redirect_uri": redirect_uri,
        "scope": " ".join(YOUTUBE_SCOPES),
        "state": state,
        "access_type": "offline",
        "prompt": "consent",
    }

    auth_url = f"{AUTH_URI}?{urlencode(params)}"
    return auth_url, state


def exchange_youtube_code_for_token(
    client_id: str, client_secret: str, code: str
) -> Optional[dict]:
    """
    Exchange authorization code for access/refresh tokens.

    Uses async httpx for non-blocking network calls.
    Returns token_info dict if successful, None otherwise.
    """
    redirect_uri = get_youtube_redirect_uri()

    async def _exchange() -> Optional[dict]:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    TOKEN_URI,
                    data={
                        "grant_type": "authorization_code",
                        "code": code,
                        "redirect_uri": redirect_uri,
                        "client_id": client_id,
                        "client_secret": client_secret,
                    },
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
                    logger.error(f"YouTube token exchange failed: {response.text}")
                    return None
        except Exception as e:
            logger.error(f"YouTube token exchange error: {e}")
            return None

    return asyncio.run(_exchange())


class YouTubePlaylistGenerator:
    """Generate YouTube playlists from track recommendations."""

    def __init__(self, username: str):
        """
        Initialize generator for a specific user.

        Args:
            username: The app username (for token storage)
        """
        self.username = username
        self.youtube: Optional[Resource] = None
        self.ytmusic: Optional[YTMusic] = None
        self.search_cache: dict[str, Optional[str]] = {}
        self._init_ytmusic()

    def _init_ytmusic(self) -> None:
        """Initialize YTMusic client for unauthenticated search."""
        try:
            self.ytmusic = YTMusic()
            logger.info("YTMusic client initialized for search")
        except Exception as e:
            logger.warning(f"Failed to initialize YTMusic: {e}")
            self.ytmusic = None

    def authenticate(self) -> bool:
        """Authenticate with YouTube Data API using tokens from Firestore."""
        creds = load_youtube_creds()
        if not creds or not creds.has_client_creds():
            logger.error("No YouTube client credentials found")
            return False

        # Get tokens from Firestore
        storage = get_oauth_storage()
        tokens = storage.get_tokens(self.username, "youtube")

        if not tokens or not tokens.get("refresh_token"):
            logger.error("No YouTube tokens found - authentication required")
            return False

        google_creds = None

        try:
            google_creds = Credentials(
                token=tokens.get("access_token"),
                refresh_token=tokens.get("refresh_token"),
                token_uri=TOKEN_URI,
                client_id=creds.client_id,
                client_secret=creds.client_secret,
                scopes=YOUTUBE_SCOPES,
            )

            if google_creds.expired and google_creds.refresh_token:
                logger.info("Refreshing expired YouTube token...")
                google_creds.refresh(Request())
                logger.info("Token refreshed successfully")
                # Update tokens in Firestore
                if google_creds.token:
                    storage.update_access_token(
                        self.username,
                        "youtube",
                        google_creds.token,
                        google_creds.refresh_token,
                    )

        except Exception as e:
            logger.warning(f"Token refresh failed: {e}")
            google_creds = None

        if not google_creds or not google_creds.valid:
            logger.error("YouTube credentials invalid - re-authentication needed")
            return False

        try:
            self.youtube = build("youtube", "v3", credentials=google_creds)
            logger.info("YouTube Data API authenticated successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to build YouTube service: {e}")
            return False

    @staticmethod
    def get_auth_status(username: str) -> dict:
        """Check authentication status for a user."""
        creds = load_youtube_creds()
        storage = get_oauth_storage()
        has_tokens = storage.has_tokens(username, "youtube")

        return {
            "youtube": {
                "has_client": creds.has_client_creds() if creds else False,
                "has_tokens": has_tokens,
            },
        }

    @staticmethod
    def needs_authentication(username: str) -> bool:
        """Check if authentication is needed for a user."""
        status = YouTubePlaylistGenerator.get_auth_status(username)
        return not status["youtube"]["has_tokens"]

    @staticmethod
    def disconnect(username: str) -> bool:
        """Disconnect YouTube (remove tokens) for a user."""
        storage = get_oauth_storage()
        return storage.delete_tokens(username, "youtube")

    def search_track_ytmusic(self, track_name: str, artist_name: str) -> Optional[str]:
        """
        Search for a track using YTMusic API (no quota cost).

        Prioritizes songs (audio-only) over videos to avoid music videos,
        live performances, and other non-audio content.

        Args:
            track_name: Track name to search
            artist_name: Artist name for context

        Returns:
            Video ID if found, None otherwise
        """
        if not self.ytmusic:
            return None

        query = f"{track_name} {artist_name}"

        try:
            # Search for songs (audio-only, not videos)
            results = self.ytmusic.search(query, filter="songs", limit=5)

            if results:
                # Return first song result - these are audio versions
                video_id = results[0].get("videoId")
                if video_id:
                    logger.debug(
                        f"YTMusic found song: {results[0].get('title')} by {results[0].get('artists', [{}])[0].get('name', 'Unknown')}"
                    )
                    return video_id

            # Fallback: search without filter but prefer audio
            results = self.ytmusic.search(query, limit=10)
            for result in results:
                result_type = result.get("resultType", "")
                # Prefer songs over videos
                if result_type == "song":
                    video_id = result.get("videoId")
                    if video_id:
                        return video_id

            # Last resort: any result with a video ID
            for result in results:
                video_id = result.get("videoId")
                if video_id:
                    return video_id

        except Exception as e:
            logger.warning(f"YTMusic search error for '{query}': {e}")

        return None

    def search_track(
        self, track_name: str, artist_name: str, max_results: int = 5
    ) -> Optional[str]:
        """
        Search for a track and return video ID.

        Uses YTMusic API first (no quota cost, prefers audio versions),
        falls back to YouTube Data API if needed.
        """
        query = f"{track_name} {artist_name}"

        if query in self.search_cache:
            return self.search_cache[query]

        # Try YTMusic first (no quota, prefers audio)
        video_id = self.search_track_ytmusic(track_name, artist_name)
        if video_id:
            self.search_cache[query] = video_id
            return video_id

        # Fallback to YouTube Data API (uses quota)
        if self.youtube:
            video_id = self._search_track_youtube_api(
                track_name, artist_name, max_results
            )
            self.search_cache[query] = video_id
            return video_id

        self.search_cache[query] = None
        return None

    def _search_track_youtube_api(
        self, track_name: str, artist_name: str, max_results: int = 5
    ) -> Optional[str]:
        """
        Fallback search using YouTube Data API (costs 100 quota units).

        Only used if YTMusic search fails.
        """
        query = f"{track_name} {artist_name}"

        music_video_keywords = [
            "official video",
            "music video",
            " mv ",
            "[mv]",
            "(mv)",
            "official music video",
            "videoclip",
            "live",
            "concert",
            "performance",
        ]
        audio_keywords = ["audio", "lyric", "lyrics", "topic"]

        try:
            response = (
                self.youtube.search()  # type: ignore[union-attr]
                .list(
                    part="snippet",
                    q=query,
                    type="video",
                    videoCategoryId="10",
                    maxResults=max_results,
                )
                .execute()
            )

            items = response.get("items", [])
            if not items:
                return None

            # Prefer Topic channels (auto-generated audio-only)
            for item in items:
                if "- Topic" in item["snippet"]["channelTitle"]:
                    return item["id"]["videoId"]

            # Prefer audio versions
            for item in items:
                title = item["snippet"]["title"].lower()
                if any(kw in title for kw in music_video_keywords):
                    continue
                if any(kw in title for kw in audio_keywords):
                    return item["id"]["videoId"]

            # First non-music-video
            for item in items:
                title = item["snippet"]["title"].lower()
                if not any(kw in title for kw in music_video_keywords):
                    return item["id"]["videoId"]

            # Fallback to first result
            return items[0]["id"]["videoId"]

        except HttpError as e:
            if e.resp.status == 403 and "quotaExceeded" in str(e):
                logger.error("YouTube API quota exceeded")
            else:
                logger.error(f"Search error: {e}")
            return None
        except Exception as e:
            logger.error(f"Search error: {e}")
            return None

    def find_playlist_by_title(self, title: str) -> Optional[str]:
        """Find playlist by title."""
        if not self.youtube:
            return None

        try:
            response = (
                self.youtube.playlists()  # type: ignore[union-attr]
                .list(part="snippet", mine=True, maxResults=50)
                .execute()
            )
            for item in response.get("items", []):
                if item["snippet"]["title"] == title:
                    return item["id"]
        except Exception as e:
            logger.error(f"Find playlist error: {e}")
        return None

    def create_playlist(
        self, title: str, description: str = "", privacy_status: str = "public"
    ) -> Optional[str]:
        """Create a new playlist."""
        if not self.youtube:
            return None

        try:
            response = (
                self.youtube.playlists()  # type: ignore[union-attr]
                .insert(
                    part="snippet,status",
                    body={
                        "snippet": {"title": title, "description": description},
                        "status": {"privacyStatus": privacy_status},
                    },
                )
                .execute()
            )
            return response["id"]
        except Exception as e:
            logger.error(f"Create playlist error: {e}")
        return None

    def delete_playlist(self, playlist_id: str) -> bool:
        """Delete a playlist."""
        if not self.youtube:
            return False

        try:
            self.youtube.playlists().delete(id=playlist_id).execute()  # type: ignore[union-attr]
            return True
        except Exception as e:
            logger.error(f"Delete playlist error: {e}")
        return False

    def add_video_to_playlist(self, playlist_id: str, video_id: str) -> bool:
        """Add video to playlist."""
        if not self.youtube:
            return False

        for attempt in range(2):
            try:
                self.youtube.playlistItems().insert(  # type: ignore[union-attr]
                    part="snippet",
                    body={
                        "snippet": {
                            "playlistId": playlist_id,
                            "resourceId": {
                                "kind": "youtube#video",
                                "videoId": video_id,
                            },
                        }
                    },
                ).execute()
                return True
            except HttpError as e:
                if e.resp.status == 403 and "quotaExceeded" in str(e.content):
                    logger.error("Quota exceeded")
                    return False
                if e.resp.status in (409, 429, 503) and attempt == 0:
                    time.sleep(2)
                    continue
                logger.error(f"Add video error: {e}")
                return False
            except Exception as e:
                logger.error(f"Add video error: {e}")
                return False
        return False

    def create_playlist_from_tracks(
        self,
        tracks_df: pl.DataFrame,
        playlist_title: str,
        playlist_description: str = "",
        privacy_status: str = "unlisted",
        progress_bar=None,
        status_text=None,
    ) -> Optional[dict]:
        """Create playlist from DataFrame of tracks."""
        if not self.youtube:
            if not self.authenticate():
                return None

        # Apply track limit based on user (API quota is per-app)
        max_tracks = (
            YOUTUBE_MAX_TRACKS_OWNER
            if self.username == DEFAULT_USERNAME
            else YOUTUBE_MAX_TRACKS_DEFAULT
        )
        if len(tracks_df) > max_tracks:
            logger.info(
                f"Limiting YouTube playlist to {max_tracks} tracks for user {self.username}"
            )
            tracks_df = tracks_df.head(max_tracks)

        has_urls = "youtube_url" in tracks_df.columns

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

        playlist_id = self.create_playlist(
            playlist_title, playlist_description, privacy_status
        )
        if not playlist_id:
            return None

        # Collect tracks with video IDs from stored URLs
        # URLs should be populated during dimension extraction (including YTMusic fallback)
        tracks_to_add = []
        tracks_missing_url = []
        for row in tracks_df.iter_rows(named=True):
            track_name = row.get("track_name", "Unknown")
            artist_name = row.get("artist_name", "Unknown")
            track_label = f"{track_name} - {artist_name}"

            video_id = None
            if has_urls and row.get("youtube_url"):
                video_id = self._extract_video_id(row["youtube_url"])

            if video_id:
                tracks_to_add.append((video_id, track_label))
            else:
                tracks_missing_url.append(track_label)

        if tracks_missing_url:
            logger.warning(
                f"{len(tracks_missing_url)} tracks missing YouTube URLs - "
                "re-run dimension extraction to populate"
            )

        # Add tracks
        tracks_added = 0
        tracks_not_found = []
        quota_exceeded = False

        for i, (video_id, track_label) in enumerate(tracks_to_add):
            if quota_exceeded:
                tracks_not_found.extend([t[1] for t in tracks_to_add[i:]])
                break

            if progress_bar:
                progress_bar.progress((i + 1) / len(tracks_to_add))
            if status_text:
                status_text.text(f"Adding {i + 1}/{len(tracks_to_add)}: {track_label}")

            if self.add_video_to_playlist(playlist_id, video_id):
                tracks_added += 1
            else:
                tracks_not_found.append(track_label)

            # Rate limiting
            time.sleep(2.0)

        result = {
            "playlist_id": playlist_id,
            "playlist_url": f"https://music.youtube.com/playlist?list={playlist_id}",
            "tracks_added": tracks_added,
            "tracks_not_found": tracks_not_found,
            "tracks_missing_url": tracks_missing_url,
        }

        if quota_exceeded:
            result["quota_exceeded"] = True

        return result

    @staticmethod
    def _extract_video_id(url: str) -> Optional[str]:
        """Extract video ID from YouTube URL."""
        patterns = [
            r"(?:v=|/)([0-9A-Za-z_-]{11})",
            r"youtu\.be/([0-9A-Za-z_-]{11})",
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None

    @staticmethod
    def get_playlist_url(playlist_id: str) -> str:
        return f"https://music.youtube.com/playlist?list={playlist_id}"
