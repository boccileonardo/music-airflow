"""
YouTube Music search utility for finding audio versions of tracks.

Uses ytmusicapi for unauthenticated search (no quota cost).
Prioritizes songs (audio-only) over music videos.
Includes retry logic with exponential backoff for rate limiting.
"""

import logging
from json import JSONDecodeError
from typing import Optional

from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from ytmusicapi import YTMusic

logger = logging.getLogger(__name__)

# Global YTMusic instance (lazy initialized)
_ytmusic: Optional[YTMusic] = None


def _get_ytmusic(force_new: bool = False) -> Optional[YTMusic]:
    """Get or initialize YTMusic client."""
    global _ytmusic
    if _ytmusic is None or force_new:
        try:
            _ytmusic = YTMusic()
            logger.info("YTMusic client initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize YTMusic: {e}")
            return None
    return _ytmusic


def _log_retry_with_query(retry_state: RetryCallState) -> None:
    """Log retry attempt including the search query."""
    query = retry_state.args[1] if len(retry_state.args) > 1 else "unknown"
    wait = retry_state.next_action.sleep if retry_state.next_action else 0
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    logger.warning(
        f"YTMusic search failed for '{query}', retrying in {wait:.1f}s: {exc}"
    )


@retry(
    retry=retry_if_exception_type(JSONDecodeError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    before_sleep=_log_retry_with_query,
    reraise=True,
)
def _search_with_retry(ytmusic: YTMusic, query: str, **kwargs) -> list:
    """Execute YTMusic search with retry on JSONDecodeError (empty response)."""
    return ytmusic.search(query, **kwargs)


def search_youtube_url(track_name: str, artist_name: str) -> Optional[str]:
    """
    Search for a track on YouTube Music and return the video URL.

    Prioritizes songs (audio-only) over videos to avoid music videos,
    live performances, and other non-audio content.

    Args:
        track_name: Track name to search
        artist_name: Artist name for context

    Returns:
        YouTube URL (https://www.youtube.com/watch?v=VIDEO_ID) if found, None otherwise
    """
    video_id = search_youtube_video_id(track_name, artist_name)
    if video_id:
        return f"https://www.youtube.com/watch?v={video_id}"
    return None


def search_youtube_video_id(track_name: str, artist_name: str) -> Optional[str]:
    """
    Search for a track on YouTube Music and return the video ID.

    Prioritizes songs (audio-only) over videos to avoid music videos,
    live performances, and other non-audio content.

    Args:
        track_name: Track name to search
        artist_name: Artist name for context

    Returns:
        Video ID if found, None otherwise
    """
    ytmusic = _get_ytmusic()
    if not ytmusic:
        return None

    query = f"{track_name} {artist_name}"

    try:
        # Search for songs (audio-only, not videos)
        results = _search_with_retry(ytmusic, query, filter="songs", limit=5)

        if results:
            video_id = results[0].get("videoId")
            if video_id:
                logger.debug(
                    f"YTMusic found song: {results[0].get('title')} by "
                    f"{results[0].get('artists', [{}])[0].get('name', 'Unknown')}"
                )
                return video_id

        # Fallback: search without filter but prefer audio
        results = _search_with_retry(ytmusic, query, limit=10)
        for result in results:
            result_type = result.get("resultType", "")
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
