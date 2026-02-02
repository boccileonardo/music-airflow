"""
Spotify search utility for finding tracks.

Uses spotipy with client credentials flow (no user auth required for search).
Includes retry logic with exponential backoff for rate limiting.
"""

import logging
import os
from typing import Optional

from dotenv import load_dotenv
from spotipy import Spotify
from spotipy.exceptions import SpotifyException
from spotipy.oauth2 import SpotifyClientCredentials
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

# Global Spotify instance (lazy initialized)
_spotify: Optional[Spotify] = None


def _get_spotify_credentials() -> tuple[Optional[str], Optional[str]]:
    """Get Spotify credentials from environment."""
    load_dotenv()
    client_id = os.getenv("SPOTIFY_CLIENT_ID")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
    return client_id, client_secret


def _get_spotify(force_new: bool = False) -> Optional[Spotify]:
    """Get or initialize Spotify client using client credentials flow."""
    global _spotify
    if _spotify is None or force_new:
        client_id, client_secret = _get_spotify_credentials()
        if not client_id or not client_secret:
            logger.warning(
                "Spotify credentials not found. Set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET in .env"
            )
            return None
        try:
            auth_manager = SpotifyClientCredentials(
                client_id=client_id, client_secret=client_secret
            )
            _spotify = Spotify(auth_manager=auth_manager)
            logger.info("Spotify client initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Spotify: {e}")
            return None
    return _spotify


def _log_retry_with_query(retry_state: RetryCallState) -> None:
    """Log retry attempt including the search query."""
    query = retry_state.args[1] if len(retry_state.args) > 1 else "unknown"
    wait = retry_state.next_action.sleep if retry_state.next_action else 0
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    logger.warning(
        f"Spotify search failed for '{query}', retrying in {wait:.1f}s: {exc}"
    )


@retry(
    retry=retry_if_exception_type(SpotifyException),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    before_sleep=_log_retry_with_query,
    reraise=True,
)
def _search_with_retry(spotify: Spotify, query: str, **kwargs) -> dict:
    """Execute Spotify search with retry on SpotifyException."""
    return spotify.search(query, **kwargs)


def search_spotify_url(track_name: str, artist_name: str) -> Optional[str]:
    """
    Search for a track on Spotify and return the track URL.

    Args:
        track_name: Track name to search
        artist_name: Artist name for context

    Returns:
        Spotify URL (https://open.spotify.com/track/TRACK_ID) if found, None otherwise
    """
    track_id = search_spotify_track_id(track_name, artist_name)
    if track_id:
        return f"https://open.spotify.com/track/{track_id}"
    return None


def search_spotify_track_id(track_name: str, artist_name: str) -> Optional[str]:
    """
    Search for a track on Spotify and return the track ID.

    Args:
        track_name: Track name to search
        artist_name: Artist name for context

    Returns:
        Track ID if found, None otherwise
    """
    spotify = _get_spotify()
    if not spotify:
        return None

    query = f"track:{track_name} artist:{artist_name}"

    try:
        results = _search_with_retry(spotify, query, type="track", limit=5)
        tracks = results.get("tracks", {}).get("items", [])

        if tracks:
            # Return first track result
            track = tracks[0]
            track_id = track.get("id")
            if track_id:
                logger.debug(
                    f"Spotify found track: {track.get('name')} by {', '.join(a['name'] for a in track.get('artists', []))}"
                )
                return track_id

        # Fallback: simpler search without field specifiers
        fallback_query = f"{track_name} {artist_name}"
        results = _search_with_retry(spotify, fallback_query, type="track", limit=5)
        tracks = results.get("tracks", {}).get("items", [])

        if tracks:
            return tracks[0].get("id")

    except SpotifyException as e:
        logger.warning(f"Spotify search error for '{query}': {e}")
    except Exception as e:
        logger.warning(f"Unexpected error searching Spotify for '{query}': {e}")

    return None


def is_spotify_configured() -> bool:
    """Check if Spotify credentials are configured."""
    client_id, client_secret = _get_spotify_credentials()
    return bool(client_id and client_secret)
