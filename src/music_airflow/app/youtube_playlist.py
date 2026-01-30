"""
YouTube playlist generation for music recommendations.

Creates YouTube playlists from track recommendations using YouTube Data API v3.

Authentication:
--------------
Set YOUTUBE_CLIENT_ID and YOUTUBE_CLIENT_SECRET in .env
Run OAuth flow from Streamlit UI to get tokens
Tokens stored in YOUTUBE_ACCESS_TOKEN and YOUTUBE_REFRESH_TOKEN

Note: YouTube Data API has quota limits (10,000 units/day).
Each search costs 100 units, playlist creation costs 50 units.
"""

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
import polars as pl
import requests
import streamlit as st

logger = logging.getLogger(__name__)

YOUTUBE_SCOPES = ["https://www.googleapis.com/auth/youtube.force-ssl"]
TOKEN_URI = "https://oauth2.googleapis.com/token"


@dataclass
class OAuthCredentials:
    """OAuth credentials for YouTube API."""

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


def load_youtube_creds() -> Optional[OAuthCredentials]:
    """Load YouTube Data API OAuth credentials."""
    client_id = _get_secret("YOUTUBE_CLIENT_ID")
    client_secret = _get_secret("YOUTUBE_CLIENT_SECRET")

    if not client_id or not client_secret:
        return None

    return OAuthCredentials(
        client_id=client_id,
        client_secret=client_secret,
        access_token=_get_secret("YOUTUBE_ACCESS_TOKEN"),
        refresh_token=_get_secret("YOUTUBE_REFRESH_TOKEN"),
    )


def poll_device_token(
    client_id: str, client_secret: str, device_code: str
) -> Optional[dict]:
    """
    Poll once for device authorization token.

    Returns token_info dict if authorized, None if still pending.
    Raises Exception on permanent errors.
    """
    token_response = requests.post(
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


def run_youtube_oauth(
    client_id: str, client_secret: str
) -> tuple[Optional[dict], Optional[dict]]:
    """
    Run YouTube Data API OAuth flow using device authorization.

    Returns tuple of (device_info, None). Use poll_device_token() to get tokens.
    """
    device_response = requests.post(
        "https://oauth2.googleapis.com/device/code",
        data={
            "client_id": client_id,
            "scope": " ".join(YOUTUBE_SCOPES),
        },
    )

    if device_response.status_code != 200:
        logger.error(f"Device code request failed: {device_response.text}")
        return None, None

    device_data = device_response.json()

    device_info = {
        "device_code": device_data["device_code"],
        "verification_url": device_data["verification_url"],
        "user_code": device_data["user_code"],
        "interval": device_data.get("interval", 5),
        "expires_in": device_data.get("expires_in", 1800),
    }

    return device_info, None


class YouTubePlaylistGenerator:
    """Generate YouTube playlists from track recommendations."""

    def __init__(self):
        self.youtube: Optional[Resource] = None
        self.search_cache: dict[str, Optional[str]] = {}

    def authenticate(self) -> bool:
        """Authenticate with YouTube Data API."""
        creds = load_youtube_creds()
        if not creds or not creds.has_client_creds():
            logger.error("No YouTube client credentials found")
            return False

        google_creds = None

        if creds.has_tokens():
            try:
                google_creds = Credentials(
                    token=creds.access_token,
                    refresh_token=creds.refresh_token,
                    token_uri=TOKEN_URI,
                    client_id=creds.client_id,
                    client_secret=creds.client_secret,
                    scopes=YOUTUBE_SCOPES,
                )

                if google_creds.expired and google_creds.refresh_token:
                    logger.info("Refreshing expired YouTube token...")
                    google_creds.refresh(Request())
                    logger.info("Token refreshed successfully")
                    self._store_refreshed_token(google_creds)

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

    def _store_refreshed_token(self, creds: Credentials) -> None:
        """Store refreshed token in session state for UI display."""
        try:
            st.session_state.refreshed_youtube_token = {
                "access_token": creds.token,
                "refresh_token": creds.refresh_token,
            }
        except Exception:
            pass

    @staticmethod
    def get_auth_status() -> dict:
        """Check authentication status."""
        youtube = load_youtube_creds()

        return {
            "youtube": {
                "has_client": youtube.has_client_creds() if youtube else False,
                "has_tokens": youtube.has_tokens() if youtube else False,
            },
        }

    @staticmethod
    def needs_authentication() -> bool:
        """Check if authentication is needed."""
        status = YouTubePlaylistGenerator.get_auth_status()
        return not status["youtube"]["has_tokens"]

    def search_track(
        self, track_name: str, artist_name: str, max_results: int = 5
    ) -> Optional[str]:
        """Search for a track and return video ID."""
        query = f"{track_name} {artist_name}"

        if query in self.search_cache:
            return self.search_cache[query]

        music_video_keywords = [
            "official video",
            "music video",
            " mv ",
            "[mv]",
            "(mv)",
            "official music video",
            "videoclip",
        ]
        audio_keywords = ["audio", "lyric", "lyrics"]

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
                self.search_cache[query] = None
                return None

            # Prefer Topic channels
            for item in items:
                if "- Topic" in item["snippet"]["channelTitle"]:
                    video_id = item["id"]["videoId"]
                    self.search_cache[query] = video_id
                    return video_id

            # Prefer audio versions
            for item in items:
                title = item["snippet"]["title"].lower()
                if any(kw in title for kw in music_video_keywords):
                    continue
                if any(kw in title for kw in audio_keywords):
                    video_id = item["id"]["videoId"]
                    self.search_cache[query] = video_id
                    return video_id

            # First non-music-video
            for item in items:
                title = item["snippet"]["title"].lower()
                if not any(kw in title for kw in music_video_keywords):
                    video_id = item["id"]["videoId"]
                    self.search_cache[query] = video_id
                    return video_id

            # Fallback to first result
            video_id = items[0]["id"]["videoId"]
            self.search_cache[query] = video_id
            return video_id

        except HttpError as e:
            if e.resp.status == 403 and "quotaExceeded" in str(e):
                logger.error("YouTube API quota exceeded")
            else:
                logger.error(f"Search error: {e}")
            self.search_cache[query] = None
            return None
        except Exception as e:
            logger.error(f"Search error: {e}")
            self.search_cache[query] = None
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

        # Collect tracks with video IDs
        tracks_to_add = []
        for row in tracks_df.iter_rows(named=True):
            track_name = row.get("track_name", "Unknown")
            artist_name = row.get("artist_name", "Unknown")
            track_label = f"{track_name} - {artist_name}"

            video_id = None
            if has_urls and row.get("youtube_url"):
                video_id = self._extract_video_id(row["youtube_url"])

            if not video_id and self.youtube:
                video_id = self.search_track(track_name, artist_name)

            if video_id:
                tracks_to_add.append((video_id, track_label))

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
