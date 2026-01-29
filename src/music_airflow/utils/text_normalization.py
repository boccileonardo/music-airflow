"""
Text normalization utilities for fuzzy track/artist matching.

Provides functions to normalize track and artist names for consistent
identification across different recordings and versions.
"""

import re
from typing import Pattern


# Compiled regex patterns for performance
_REMASTER_PATTERN: Pattern = re.compile(
    r"\s*[\(\[]"
    r"(remaster(ed)?|re-master(ed)?|"
    r"\d{4}\s+remaster(ed)?|"
    r"remaster(ed)?\s+\d{4})"
    r"[\)\]]",
    re.IGNORECASE,
)

_LIVE_PATTERN: Pattern = re.compile(
    r"\s*[\(\[]"
    r"(live|live\s+at|live\s+from|live\s+in|live\s+on|"
    r"live\s+\d{4}|live\s+version|live\s+recording)"
    r".*?[\)\]]",
    re.IGNORECASE,
)

_VERSION_PATTERN: Pattern = re.compile(
    r"\s*[\(\[]"
    r"(.*?\s+version|.*?\s+mix|.*?\s+edit|.*?\s+remix|"
    r".*?\s+take|single\s+version|album\s+version|radio\s+edit|"
    r"extended\s+(version|mix|edit)?|remix)"
    r"[\)\]]",
    re.IGNORECASE,
)

_EXPLICIT_PATTERN: Pattern = re.compile(
    r"\s*[\(\[]"
    r"(explicit|clean|censored)"
    r"[\)\]]",
    re.IGNORECASE,
)

_FEAT_PATTERN: Pattern = re.compile(
    r"\s*[\(\[]?"
    r"(feat\.?|ft\.?|featuring|with|vs\.?|versus)"
    r".*?[\)\]]?$",
    re.IGNORECASE,
)

_MUSIC_VIDEO_PATTERN: Pattern = re.compile(
    r"(music\s+video|official\s+video|video\s+clip|"
    r"visuali[sz]er|lyric\s+video|official\s+audio)",
    re.IGNORECASE,
)

_YEAR_PATTERN: Pattern = re.compile(r"\s*[\(\[]?\d{4}[\)\]]?\s*")

_PUNCTUATION_PATTERN: Pattern = re.compile(r"[^\w\s-]")

_WHITESPACE_PATTERN: Pattern = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """
    Normalize text for fuzzy matching by removing variations.

    Removes common suffixes like "(Remastered)", "(Live)", version indicators,
    featured artists, years, punctuation, and normalizes whitespace.

    Args:
        text: Track or artist name to normalize

    Returns:
        Normalized lowercase text with variations removed

    Examples:
        >>> normalize_text("Highway Star (Remastered 2012)")
        'highway star'
        >>> normalize_text("Bohemian Rhapsody (Live)")
        'bohemian rhapsody'
        >>> normalize_text("Song - Radio Edit")
        'song radio edit'
        >>> normalize_text("Track (feat. Artist)")
        'track'
    """
    if not text:
        return ""

    # Lowercase and strip
    text = text.lower().strip()

    # Remove patterns in order of specificity
    text = _REMASTER_PATTERN.sub("", text)
    text = _LIVE_PATTERN.sub("", text)
    text = _VERSION_PATTERN.sub("", text)
    text = _EXPLICIT_PATTERN.sub("", text)
    text = _FEAT_PATTERN.sub("", text)
    text = _YEAR_PATTERN.sub("", text)

    # Remove " - " followed by version indicators (not in parentheses)
    text = re.sub(
        r"\s+-\s+(remaster(ed)?|radio\s+edit|live|remix|extended|single)",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Remove punctuation except hyphens (keep "hip-hop", etc.)
    text = _PUNCTUATION_PATTERN.sub("", text)

    # Normalize whitespace
    text = _WHITESPACE_PATTERN.sub(" ", text)

    return text.strip()


def is_music_video(track_name: str) -> bool:
    """
    Detect if a track name indicates a music video version.

    Args:
        track_name: Track name to check

    Returns:
        True if track name contains music video indicators

    Examples:
        >>> is_music_video("Song Name (Official Video)")
        True
        >>> is_music_video("Song Name (Music Video)")
        True
        >>> is_music_video("Song Name")
        False
    """
    if not track_name:
        return False

    return bool(_MUSIC_VIDEO_PATTERN.search(track_name))


def generate_canonical_track_id(track_name: str, artist_name: str) -> str:
    """
    Generate canonical track ID from normalized track and artist names.

    Uses pipe separator to combine normalized names into stable ID.

    Args:
        track_name: Track name (will be normalized)
        artist_name: Artist name (will be normalized)

    Returns:
        Canonical track ID in format "normalized_track|normalized_artist"

    Examples:
        >>> generate_canonical_track_id("Highway Star (Live)", "Deep Purple")
        'highway star|deep purple'
        >>> generate_canonical_track_id("Song (Remastered)", "Artist")
        'song|artist'
    """
    normalized_track = normalize_text(track_name)
    normalized_artist = normalize_text(artist_name)

    # Handle edge case where normalization removes everything
    if not normalized_track or not normalized_artist:
        # Fallback to original text if normalization produces empty string
        normalized_track = normalized_track or track_name.lower().strip()
        normalized_artist = normalized_artist or artist_name.lower().strip()

    return f"{normalized_track}|{normalized_artist}"


def generate_canonical_artist_id(artist_name: str) -> str:
    """
    Generate canonical artist ID from normalized artist name.

    Args:
        artist_name: Artist name (will be normalized)

    Returns:
        Normalized artist name as canonical ID

    Examples:
        >>> generate_canonical_artist_id("Deep Purple")
        'deep purple'
        >>> generate_canonical_artist_id("The Beatles")
        'the beatles'
    """
    normalized = normalize_text(artist_name)

    # Handle edge case where normalization removes everything
    if not normalized:
        normalized = artist_name.lower().strip()

    return normalized
