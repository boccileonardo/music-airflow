"""
Last.fm API Client for fetching user scrobble data.

Handles API requests, pagination, rate limiting, and error handling.
Credentials are loaded from environment variables.
"""

import os
import time
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)


class LastFMClient:
    """Client for interacting with the Last.fm API."""

    BASE_URL = "http://ws.audioscrobbler.com/2.0/"

    def __init__(self, api_key: str | None = None, username: str | None = None):
        """
        Initialize Last.fm API client.

        Args:
            api_key: Last.fm API key (if None, loads from .env)
            username: Last.fm username (if None, must be provided in method calls)
        """
        # Load .env file from project root
        env_path = Path(__file__).parent.parent.parent / ".env"
        load_dotenv(env_path)

        self.api_key = api_key or os.getenv("api_key")
        self.username = username

        if not self.api_key:
            raise ValueError(
                "Last.fm API key not found. Set 'api_key' in .env or pass to constructor."
            )

    def get_recent_tracks(
        self,
        username: str | None = None,
        from_timestamp: int | None = None,
        to_timestamp: int | None = None,
        limit: int = 200,
        extended: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Fetch all recent tracks for a user with pagination.

        Last.fm API documentation:
        - No explicit retention limit mentioned - fetches all historical scrobbles
        - Maximum 200 results per page
        - Supports from/to timestamp filtering for incremental loads
        - Returns tracks in reverse chronological order (newest first)

        Args:
            username: Last.fm username (uses instance username if None)
            from_timestamp: Start timestamp (Unix timestamp, UTC)
            to_timestamp: End timestamp (Unix timestamp, UTC)
            limit: Results per page (max 200)
            extended: Include extended metadata (artist, album details)

        Returns:
            List of track dictionaries containing scrobble data

        Raises:
            ValueError: If no username provided
            requests.HTTPError: If API request fails after retries
        """
        username = username or self.username
        if not username:
            raise ValueError("Username must be provided")

        all_tracks = []
        page = 1
        total_pages = None

        while True:
            params = {
                "method": "user.getrecenttracks",
                "user": username,
                "api_key": self.api_key,
                "format": "json",
                "limit": limit,
                "page": page,
                "extended": 1 if extended else 0,
            }

            if from_timestamp is not None:
                params["from"] = from_timestamp
            if to_timestamp is not None:
                params["to"] = to_timestamp

            # Make request with retry logic
            response_data = self._make_request(params)

            # Extract tracks from response
            if "recenttracks" not in response_data:
                break

            recent_tracks = response_data["recenttracks"]
            tracks = recent_tracks.get("track", [])

            # Handle single track response (not a list)
            if isinstance(tracks, dict):
                tracks = [tracks]

            # Filter out "now playing" tracks (no timestamp)
            tracks = [t for t in tracks if "date" in t]

            all_tracks.extend(tracks)

            # Check pagination
            if total_pages is None:
                total_pages = int(recent_tracks.get("@attr", {}).get("totalPages", 1))

            if page >= total_pages or not tracks:
                break

            page += 1

            # Small delay to avoid rate limiting
            time.sleep(0.2)

        return all_tracks

    @retry(
        retry=retry_if_exception_type(requests.exceptions.RequestException),
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(multiplier=2, min=2, max=60),
        reraise=True,
    )
    def _make_request(self, params: dict[str, Any]) -> dict[str, Any]:
        """
        Make API request with retry logic using tenacity.

        Retries up to 3 times with exponential backoff (2s, 4s, 8s, ...).
        Only retries on requests.exceptions.RequestException.

        Args:
            params: API request parameters

        Returns:
            JSON response as dictionary

        Raises:
            requests.HTTPError: If request fails after all retries
            ValueError: If Last.fm API returns an error
        """
        response = requests.get(self.BASE_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        # Check for Last.fm API errors
        if "error" in data:
            error_code = data.get("error")
            error_msg = data.get("message", "Unknown error")
            raise ValueError(f"Last.fm API error {error_code}: {error_msg}")

        return data

    def get_user_info(self, username: str | None = None) -> dict[str, Any]:
        """
        Get basic user information.

        Args:
            username: Last.fm username (uses instance username if None)

        Returns:
            User information dictionary

        Raises:
            ValueError: If no username provided
        """
        username = username or self.username
        if not username:
            raise ValueError("Username must be provided")

        params = {
            "method": "user.getinfo",
            "user": username,
            "api_key": self.api_key,
            "format": "json",
        }

        response_data = self._make_request(params)
        return response_data.get("user", {})

    def get_track_info(
        self, track: str, artist: str, mbid: str | None = None, autocorrect: bool = True
    ) -> dict[str, Any]:
        """
        Get detailed information about a track.

        Args:
            track: Track name
            artist: Artist name
            mbid: MusicBrainz ID (optional, used if provided)
            autocorrect: Transform misspelled names into correct versions (default: True)

        Returns:
            Track information dictionary including tags, listeners, playcount

        Raises:
            ValueError: If track not found or API error
        """
        params = {
            "method": "track.getinfo",
            "api_key": self.api_key,
            "format": "json",
            "autocorrect": 1 if autocorrect else 0,
        }

        if mbid:
            params["mbid"] = mbid
        else:
            params["track"] = track
            params["artist"] = artist

        response_data = self._make_request(params)
        return response_data.get("track", {})

    def get_artist_info(
        self, artist: str, mbid: str | None = None, autocorrect: bool = True
    ) -> dict[str, Any]:
        """
        Get detailed information about an artist.

        Args:
            artist: Artist name
            mbid: MusicBrainz ID (optional, used if provided)
            autocorrect: Transform misspelled names into correct versions (default: True)

        Returns:
            Artist information dictionary including tags, listeners, playcount, bio

        Raises:
            ValueError: If artist not found or API error
        """
        params = {
            "method": "artist.getinfo",
            "api_key": self.api_key,
            "format": "json",
            "autocorrect": 1 if autocorrect else 0,
        }

        if mbid:
            params["mbid"] = mbid
        else:
            params["artist"] = artist

        response_data = self._make_request(params)
        return response_data.get("artist", {})

    def get_similar_artists(
        self,
        artist: str,
        mbid: str | None = None,
        limit: int = 100,
        autocorrect: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Get similar artists to a given artist.

        Args:
            artist: Artist name
            mbid: MusicBrainz ID (optional, used if provided)
            limit: Number of results to return (max 100)
            autocorrect: Transform misspelled names into correct versions (default: True)

        Returns:
            List of similar artists with match scores and metadata

        Raises:
            ValueError: If artist not found or API error
        """
        params = {
            "method": "artist.getSimilar",
            "api_key": self.api_key,
            "format": "json",
            "limit": min(limit, 100),
            "autocorrect": 1 if autocorrect else 0,
        }

        if mbid:
            params["mbid"] = mbid
        else:
            params["artist"] = artist

        response_data = self._make_request(params)
        similar_artists = response_data.get("similarartists", {})
        artists = similar_artists.get("artist", [])

        # Handle single artist response (not a list)
        if isinstance(artists, dict):
            artists = [artists]

        return artists

    def get_artist_top_tracks(
        self,
        artist: str,
        mbid: str | None = None,
        limit: int = 50,
        autocorrect: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Get top tracks for an artist.

        Args:
            artist: Artist name
            mbid: MusicBrainz ID (optional, used if provided)
            limit: Number of results to return (default 50)
            autocorrect: Transform misspelled names into correct versions (default: True)

        Returns:
            List of top tracks with playcount, listeners, and metadata

        Raises:
            ValueError: If artist not found or API error
        """
        params = {
            "method": "artist.getTopTracks",
            "api_key": self.api_key,
            "format": "json",
            "limit": limit,
            "autocorrect": 1 if autocorrect else 0,
        }

        if mbid:
            params["mbid"] = mbid
        else:
            params["artist"] = artist

        response_data = self._make_request(params)
        top_tracks = response_data.get("toptracks", {})
        tracks = top_tracks.get("track", [])

        # Handle single track response (not a list)
        if isinstance(tracks, dict):
            tracks = [tracks]

        return tracks

    def get_artist_top_albums(
        self,
        artist: str,
        mbid: str | None = None,
        limit: int = 50,
        autocorrect: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Get top albums for an artist.

        Args:
            artist: Artist name
            mbid: MusicBrainz ID (optional, used if provided)
            limit: Number of results to return (default 50)
            autocorrect: Transform misspelled names into correct versions (default: True)

        Returns:
            List of top albums with playcount and metadata

        Raises:
            ValueError: If artist not found or API error
        """
        params = {
            "method": "artist.getTopAlbums",
            "api_key": self.api_key,
            "format": "json",
            "limit": limit,
            "autocorrect": 1 if autocorrect else 0,
        }

        if mbid:
            params["mbid"] = mbid
        else:
            params["artist"] = artist

        response_data = self._make_request(params)
        top_albums = response_data.get("topalbums", {})
        albums = top_albums.get("album", [])

        # Handle single album response (not a list)
        if isinstance(albums, dict):
            albums = [albums]

        return albums

    def get_similar_tags(self, tag: str) -> list[dict[str, Any]]:
        """
        Get similar tags to a given tag.

        Args:
            tag: Tag name

        Returns:
            List of similar tags

        Raises:
            ValueError: If tag not found or API error
        """
        params = {
            "method": "tag.getSimilar",
            "tag": tag,
            "api_key": self.api_key,
            "format": "json",
        }

        response_data = self._make_request(params)
        similar_tags = response_data.get("similartags", {})
        tags = similar_tags.get("tag", [])

        # Handle single tag response (not a list)
        if isinstance(tags, dict):
            tags = [tags]

        return tags

    def get_tag_top_tracks(self, tag: str, limit: int = 50) -> list[dict[str, Any]]:
        """
        Get top tracks for a tag.

        Args:
            tag: Tag name
            limit: Number of results to return (default 50)

        Returns:
            List of top tracks with playcount, listeners, and metadata

        Raises:
            ValueError: If tag not found or API error
        """
        params = {
            "method": "tag.getTopTracks",
            "tag": tag,
            "api_key": self.api_key,
            "format": "json",
            "limit": limit,
        }

        response_data = self._make_request(params)
        top_tracks = response_data.get("tracks", {})
        tracks = top_tracks.get("track", [])

        # Handle single track response (not a list)
        if isinstance(tracks, dict):
            tracks = [tracks]

        return tracks

    def get_album_info(
        self,
        album: str,
        artist: str,
        mbid: str | None = None,
        autocorrect: bool = True,
    ) -> dict[str, Any]:
        """
        Get detailed information about an album including its tracks.

        Args:
            album: Album name
            artist: Artist name
            mbid: MusicBrainz ID (optional, used if provided)
            autocorrect: Transform misspelled names into correct versions (default: True)

        Returns:
            Album information dictionary including tracks, listeners, playcount

        Raises:
            ValueError: If album not found or API error
        """
        params = {
            "method": "album.getinfo",
            "api_key": self.api_key,
            "format": "json",
            "autocorrect": 1 if autocorrect else 0,
        }

        if mbid:
            params["mbid"] = mbid
        else:
            params["album"] = album
            params["artist"] = artist

        response_data = self._make_request(params)
        return response_data.get("album", {})
