"""
YouTube playlist generation for music recommendations.

Creates YouTube playlists from track recommendations using YouTube Data API v3.
Playlists can be opened in YouTube Music or compatible players like Metrolist.
"""

import json
import logging
import os
import re
import time
from typing import Optional

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build, Resource
from googleapiclient.errors import HttpError
import polars as pl
import streamlit as st

logger = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/youtube.force-ssl"]
TOKEN_FILE = ".youtube_token.json"


class YouTubePlaylistGenerator:
    """Generate YouTube playlists from track recommendations."""

    def __init__(self):
        """Initialize YouTube API client."""
        self.youtube: Optional[Resource] = None
        self.search_cache: dict[str, Optional[str]] = {}

    def authenticate(self) -> bool:
        """
        Authenticate with YouTube API.

        Priority:
        1. Streamlit secrets (cloud deployment)
        2. Local token file (local after first auth)
        3. OAuth flow with secrets credentials (local first time)

        Returns:
            True if authentication successful, False otherwise
        """
        creds = None
        token_info = None

        # Try Streamlit secrets first (for cloud deployment)
        try:
            if "youtube_token" in st.secrets and st.secrets["youtube_token"].get(
                "token"
            ):
                token_info = dict(st.secrets["youtube_token"])
                logger.info("Loaded credentials from Streamlit secrets")
        except Exception as e:
            logger.debug(f"No credentials in Streamlit secrets: {e}")

        # Try local token file (local development after first auth)
        if not token_info and os.path.exists(TOKEN_FILE):
            try:
                with open(TOKEN_FILE) as f:
                    token_info = json.load(f)
                logger.info("Loaded credentials from local token file")
            except Exception as e:
                logger.warning(f"Failed to load token file: {e}")

        # Create credentials from token info
        if token_info:
            try:
                creds = Credentials(
                    token=token_info.get("token"),
                    refresh_token=token_info.get("refresh_token"),
                    token_uri=token_info.get(
                        "token_uri", "https://oauth2.googleapis.com/token"
                    ),
                    client_id=token_info.get("client_id"),
                    client_secret=token_info.get("client_secret"),
                    scopes=SCOPES,
                )
            except Exception as e:
                logger.warning(f"Failed to create credentials from token: {e}")
                creds = None

        # If no valid credentials, do OAuth flow (local first time only)
        if not creds or not creds.valid:
            try:
                if "youtube" not in st.secrets:
                    logger.error("YouTube credentials not found in Streamlit secrets")
                    return False

                client_config = {
                    "installed": {
                        "client_id": st.secrets["youtube"]["client_id"],
                        "client_secret": st.secrets["youtube"]["client_secret"],
                        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                        "token_uri": "https://oauth2.googleapis.com/token",
                        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                        "redirect_uris": ["http://localhost"],
                    }
                }
                flow = InstalledAppFlow.from_client_config(client_config, SCOPES)
                creds = flow.run_local_server(port=8090)
                logger.info("Completed OAuth flow")

                # Save token locally
                token_info = {
                    "token": creds.token,
                    "refresh_token": creds.refresh_token,
                    "token_uri": creds.token_uri,
                    "client_id": creds.client_id,
                    "client_secret": creds.client_secret,
                    "scopes": creds.scopes,
                }
                with open(TOKEN_FILE, "w") as token:
                    json.dump(token_info, token)
                logger.info(f"Saved token to {TOKEN_FILE}")

            except Exception as e:
                logger.error(f"Authentication failed: {e}")
                return False

        # Build YouTube service
        try:
            self.youtube = build("youtube", "v3", credentials=creds)
            return True
        except Exception as e:
            logger.error(f"Failed to build YouTube service: {e}")
            return False

    def search_track(
        self,
        track_name: str,
        artist_name: str,
        max_results: int = 5,
    ) -> Optional[str]:
        """
        Search for a track on YouTube and return the video ID.
        Prioritizes Topic channels and filters out music videos.

        Args:
            track_name: Name of the track
            artist_name: Name of the artist
            max_results: Maximum number of results to consider

        Returns:
            Video ID if found, None otherwise
        """
        query = f"{track_name} {artist_name}"

        # Check cache first
        if query in self.search_cache:
            logger.debug(f"Cache hit for '{query}'")
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
                    videoCategoryId="10",  # Music category
                    maxResults=max_results,
                )
                .execute()
            )

            items = response.get("items", [])
            if not items:
                logger.warning(f"No results found for '{query}'")
                self.search_cache[query] = None
                return None

            # First pass: Look for Topic channels (auto-generated, usually audio-only)
            for item in items:
                channel = item["snippet"]["channelTitle"]
                if "- Topic" in channel:
                    video_id = item["id"]["videoId"]
                    logger.info(f"Found Topic channel video for '{query}': {video_id}")
                    self.search_cache[query] = video_id
                    return video_id

            # Second pass: Prefer audio/lyric versions, skip music videos
            for item in items:
                title = item["snippet"]["title"].lower()

                # Skip obvious music videos
                if any(keyword in title for keyword in music_video_keywords):
                    continue

                # Prefer audio versions
                if any(keyword in title for keyword in audio_keywords):
                    video_id = item["id"]["videoId"]
                    logger.info(f"Found audio version for '{query}': {video_id}")
                    self.search_cache[query] = video_id
                    return video_id

            # Third pass: Return first non-music-video result
            for item in items:
                title = item["snippet"]["title"].lower()
                if not any(keyword in title for keyword in music_video_keywords):
                    video_id = item["id"]["videoId"]
                    logger.info(f"Found video for '{query}': {video_id}")
                    self.search_cache[query] = video_id
                    return video_id

            # Fallback: Return first result even if it's a music video
            video_id = items[0]["id"]["videoId"]
            logger.info(f"Fallback to first result for '{query}': {video_id}")
            self.search_cache[query] = video_id
            return video_id

        except HttpError as e:
            if e.resp.status == 403 and "quotaExceeded" in str(e):
                logger.error(
                    f"YouTube API quota exceeded while searching for '{query}'. "
                    f"Quota resets at midnight Pacific Time."
                )
            else:
                logger.error(f"HTTP error searching for '{query}': {e}")
            self.search_cache[query] = None
            return None
        except Exception as e:
            logger.error(f"Error searching for '{query}': {e}")
            self.search_cache[query] = None
            return None

    def find_playlist_by_title(self, title: str) -> Optional[str]:
        """
        Find a playlist by its exact title.

        Args:
            title: Exact playlist title to search for

        Returns:
            Playlist ID if found, None otherwise
        """
        try:
            response = (
                self.youtube.playlists()  # type: ignore[union-attr]
                .list(
                    part="snippet",
                    mine=True,
                    maxResults=50,
                )
                .execute()
            )

            for item in response.get("items", []):
                if item["snippet"]["title"] == title:
                    playlist_id = item["id"]
                    logger.info(f"Found existing playlist '{title}': {playlist_id}")
                    return playlist_id

            return None

        except Exception as e:
            logger.error(f"Error searching for playlist: {e}")
            return None

    def create_playlist(
        self,
        title: str,
        description: str = "",
        privacy_status: str = "public",
    ) -> Optional[str]:
        """
        Create a new YouTube playlist.

        Args:
            title: Playlist title
            description: Playlist description
            privacy_status: 'public', 'private', or 'unlisted'

        Returns:
            Playlist ID if successful, None otherwise
        """
        try:
            response = (
                self.youtube.playlists()  # type: ignore[union-attr]
                .insert(
                    part="snippet,status",
                    body={
                        "snippet": {
                            "title": title,
                            "description": description,
                        },
                        "status": {
                            "privacyStatus": privacy_status,
                        },
                    },
                )
                .execute()
            )

            playlist_id = response["id"]
            logger.info(f"Created playlist '{title}': {playlist_id}")
            return playlist_id

        except Exception as e:
            logger.error(f"Error creating playlist: {e}")
            return None

    def delete_playlist(self, playlist_id: str) -> bool:
        """
        Delete a playlist.

        Args:
            playlist_id: ID of the playlist to delete

        Returns:
            True if successful, False otherwise
        """
        try:
            self.youtube.playlists().delete(id=playlist_id).execute()  # type: ignore[union-attr]
            logger.info(f"Deleted playlist {playlist_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting playlist: {e}")
            return False

    def add_video_to_playlist(self, playlist_id: str, video_id: str) -> bool:
        """
        Add a video to a playlist with retry logic.

        Args:
            playlist_id: ID of the playlist
            video_id: ID of the video to add

        Returns:
            True if successful, False otherwise
        """
        max_retries = 2
        retry_delay = 1  # Start with 1 second

        for attempt in range(max_retries):
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

                logger.debug(
                    f"Successfully added video {video_id} to playlist {playlist_id}"
                )
                return True

            except HttpError as e:
                error_content = e.content.decode() if e.content else str(e)

                # Quota exceeded - don't retry
                if e.resp.status == 403 and "quotaExceeded" in error_content:
                    logger.error(
                        f"YouTube API quota exceeded. Cannot add video {video_id}. "
                        f"Quota resets at midnight Pacific Time."
                    )
                    return False

                # Rate limit or service unavailable - retry with backoff
                if e.resp.status in (409, 429, 503):
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"Retryable error (status {e.resp.status}) adding video {video_id}, "
                            f"attempt {attempt + 1}/{max_retries}. Retrying in {retry_delay}s..."
                        )
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                        continue
                    else:
                        logger.error(
                            f"Failed to add video {video_id} after {max_retries} attempts: {e}"
                        )
                        return False

                # Other HTTP errors - don't retry
                logger.error(f"HTTP error adding video {video_id}: {e}")
                return False

            except Exception as e:
                logger.error(f"Error adding video {video_id} to playlist: {e}")
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
        """
        Create a YouTube playlist from a DataFrame of tracks.

        If tracks_df has 'youtube_url' column, extracts video IDs directly (efficient).
        Otherwise falls back to searching for each track (uses more API quota).

        If a playlist with the same title exists, replaces its tracks.

        Args:
            tracks_df: DataFrame with either:
                - 'youtube_url' column (preferred), or
                - 'track_name' and 'artist_name' columns (fallback to search)
            playlist_title: Title for the playlist
            playlist_description: Description for the playlist
            privacy_status: 'public', 'private', or 'unlisted'
            progress_bar: Streamlit progress bar object (optional)
            status_text: Streamlit text object for status updates (optional)

        Returns:
            Dict with playlist_id, playlist_url, tracks_added, tracks_not_found
            or None if failed
        """
        if not self.youtube:
            if not self.authenticate():
                return None

        # Check if we have direct YouTube URLs (preferred method)
        has_youtube_urls = "youtube_url" in tracks_df.columns

        if has_youtube_urls:
            logger.info("Using direct YouTube URLs from gold table (efficient mode)")
            return self._create_playlist_from_urls(
                tracks_df,
                playlist_title,
                playlist_description,
                privacy_status,
                progress_bar,
                status_text,
            )
        else:
            logger.info("Falling back to YouTube search (uses more API quota)")
            return self._create_playlist_from_tracks_search(
                tracks_df,
                playlist_title,
                playlist_description,
                privacy_status,
                progress_bar,
                status_text,
            )

    def _create_playlist_from_urls(
        self,
        tracks_df: pl.DataFrame,
        playlist_title: str,
        playlist_description: str,
        privacy_status: str,
        progress_bar,
        status_text,
    ) -> Optional[dict]:
        """
        Create playlist directly from YouTube URLs (no search needed).
        Much more efficient than searching - saves ~95% of API quota.
        """
        # Check if playlist already exists
        if status_text:
            status_text.text("Checking for existing playlist...")

        playlist_id = self.find_playlist_by_title(playlist_title)

        if playlist_id:
            # Delete and recreate (more efficient than clearing for API quota)
            logger.info(f"Deleting existing playlist to recreate: {playlist_title}")
            if status_text:
                status_text.text("Deleting existing playlist...")
            self.delete_playlist(playlist_id)
            playlist_id = None

        # Create new playlist
        logger.info(f"Creating new playlist: {playlist_title}")
        if status_text:
            status_text.text("Creating new playlist...")

        playlist_id = self.create_playlist(
            title=playlist_title,
            description=playlist_description,
            privacy_status=privacy_status,
        )

        if not playlist_id:
            return None

        # Extract video IDs from YouTube URLs
        total_input_tracks = len(tracks_df)
        logger.info(f"Processing {total_input_tracks} tracks for playlist creation")

        tracks_with_urls = []
        skipped_no_url = 0
        skipped_bad_url = 0
        skipped_wrong_type = 0

        for row in tracks_df.iter_rows(named=True):
            youtube_url = row.get("youtube_url")
            track_name = row.get("track_name", "Unknown")
            artist_name = row.get("artist_name", "Unknown")
            track_id = f"{track_name} - {artist_name}"

            if not youtube_url:
                skipped_no_url += 1
                logger.debug(f"Skipping {track_id}: No youtube_url")
                continue

            if not isinstance(youtube_url, str):
                skipped_wrong_type += 1
                logger.warning(
                    f"Skipping {track_id}: youtube_url is {type(youtube_url)}, not string"
                )
                continue

            # Extract video ID from URL
            video_id = self._extract_video_id(youtube_url)
            if video_id:
                tracks_with_urls.append((video_id, track_id))
            else:
                skipped_bad_url += 1
                logger.warning(
                    f"Could not extract video ID from URL for {track_id}: {youtube_url}"
                )

        logger.info(
            f"Track filtering results: {total_input_tracks} input -> {len(tracks_with_urls)} with valid URLs "
            f"(skipped: {skipped_no_url} no URL, {skipped_wrong_type} wrong type, {skipped_bad_url} bad URL format)"
        )

        # Add videos to playlist sequentially with progress updates
        tracks_added = 0
        tracks_not_found = []
        quota_exceeded = False

        for i, (video_id, track_id) in enumerate(tracks_with_urls):
            if quota_exceeded:
                logger.info(
                    f"Stopping due to quota exceeded. Added {tracks_added}/{len(tracks_with_urls)} tracks."
                )
                tracks_not_found.extend([t[1] for t in tracks_with_urls[i:]])
                break

            # Update progress
            if progress_bar:
                progress_bar.progress((i + 1) / len(tracks_with_urls))
            if status_text:
                status_text.text(
                    f"Adding track {i + 1}/{len(tracks_with_urls)}: {track_id}"
                )

            # Add video with delay between requests
            success = self.add_video_to_playlist(playlist_id, video_id)

            if success:
                tracks_added += 1
            else:
                tracks_not_found.append(track_id)
                # Check if it was a quota error
                if "quotaExceeded" in str(success):
                    quota_exceeded = True

            # Rate limiting: 2 seconds between requests to avoid 409 errors
            time.sleep(2.0)

        playlist_url = f"https://music.youtube.com/playlist?list={playlist_id}"

        result = {
            "playlist_id": playlist_id,
            "playlist_url": playlist_url,
            "tracks_added": tracks_added,
            "tracks_not_found": tracks_not_found,
        }

        if quota_exceeded:
            result["quota_exceeded"] = True
            logger.error(
                f"YouTube API quota exceeded. Successfully added {tracks_added}/{len(tracks_with_urls)} tracks. "
                f"Quota resets at midnight Pacific Time."
            )

        return result

    def _create_playlist_from_tracks_search(
        self,
        tracks_df: pl.DataFrame,
        playlist_title: str,
        playlist_description: str,
        privacy_status: str,
        progress_bar,
        status_text,
    ) -> Optional[dict]:
        """
        Create playlist by searching for each track (uses more API quota).
        """
        # Check if playlist already exists
        if status_text:
            status_text.text("Checking for existing playlist...")

        playlist_id = self.find_playlist_by_title(playlist_title)

        if playlist_id:
            logger.info(f"Deleting existing playlist to recreate: {playlist_title}")
            if status_text:
                status_text.text("Deleting existing playlist...")
            self.delete_playlist(playlist_id)
            playlist_id = None

        # Create new playlist
        logger.info(f"Creating new playlist: {playlist_title}")
        if status_text:
            status_text.text("Creating new playlist...")

        playlist_id = self.create_playlist(
            title=playlist_title,
            description=playlist_description,
            privacy_status=privacy_status,
        )

        if not playlist_id:
            return None

        # Prepare tracks
        tracks = [
            (row.get("track_name", ""), row.get("artist_name", ""))
            for row in tracks_df.iter_rows(named=True)
            if row.get("track_name") and row.get("artist_name")
        ]

        logger.info(f"Processing {len(tracks)} tracks for playlist {playlist_id}")

        # Process tracks sequentially with progress updates
        tracks_added = 0
        tracks_not_found = []
        quota_exceeded = False

        for i, (track_name, artist_name) in enumerate(tracks):
            if quota_exceeded:
                logger.info(
                    f"Stopping due to quota exceeded. Added {tracks_added}/{len(tracks)} tracks."
                )
                tracks_not_found.extend([f"{t[0]} - {t[1]}" for t in tracks[i:]])
                break

            track_id = f"{track_name} - {artist_name}"

            # Update progress
            if progress_bar:
                progress_bar.progress((i + 1) / len(tracks))
            if status_text:
                status_text.text(
                    f"Searching & adding {i + 1}/{len(tracks)}: {track_id}"
                )

            # Search for track
            video_id = self.search_track(track_name, artist_name)
            if not video_id:
                tracks_not_found.append(track_id)
                continue

            # Add to playlist
            time.sleep(1.0)  # Rate limiting between search and add
            success = self.add_video_to_playlist(playlist_id, video_id)

            if success:
                tracks_added += 1
            else:
                tracks_not_found.append(track_id)
                if "quotaExceeded" in str(success):
                    quota_exceeded = True

            # Rate limiting between tracks
            time.sleep(1.0)

        playlist_url = f"https://music.youtube.com/playlist?list={playlist_id}"

        result = {
            "playlist_id": playlist_id,
            "playlist_url": playlist_url,
            "tracks_added": tracks_added,
            "tracks_not_found": tracks_not_found,
        }

        if quota_exceeded:
            result["quota_exceeded"] = True
            logger.error(
                f"YouTube API quota exceeded. Successfully added {tracks_added}/{len(tracks)} tracks. "
                f"Quota resets at midnight Pacific Time."
            )

        return result

    @staticmethod
    def _extract_video_id(youtube_url: str) -> Optional[str]:
        """Extract video ID from various YouTube URL formats."""
        # Handle various YouTube URL formats:
        # - https://www.youtube.com/watch?v=VIDEO_ID
        # - https://youtu.be/VIDEO_ID
        # - https://music.youtube.com/watch?v=VIDEO_ID
        patterns = [
            r"(?:v=|/)([0-9A-Za-z_-]{11}).*",
            r"youtu\.be/([0-9A-Za-z_-]{11})",
            r"youtube\.com/embed/([0-9A-Za-z_-]{11})",
        ]

        for pattern in patterns:
            match = re.search(pattern, youtube_url)
            if match:
                return match.group(1)

        return None

    @staticmethod
    def get_playlist_url(playlist_id: str) -> str:
        """Get YouTube Music URL for a playlist."""
        return f"https://music.youtube.com/playlist?list={playlist_id}"
