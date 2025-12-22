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
