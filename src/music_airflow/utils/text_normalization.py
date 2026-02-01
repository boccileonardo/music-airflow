"""
Text normalization utilities for fuzzy track/artist matching.

Provides functions to normalize track and artist names for consistent
identification across different recordings and versions.
"""

import re
from typing import Pattern


# Compiled regex patterns for performance

# Parenthetical patterns - match content inside () or []
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

_AUDIO_FORMAT_PATTERN: Pattern = re.compile(
    r"\s*[\(\[]"
    r"(stereo|mono|stereo\s+mix|mono\s+mix|"
    r"stereo\s+version|mono\s+version|"
    r"original\s+stereo|original\s+mono|"
    r"true\s+stereo|simulated\s+stereo)"
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

# Parenthetical patterns for demo/take/video/audio indicators
# These are inside () or [] brackets
_DEMO_TAKE_PATTERN: Pattern = re.compile(
    r"\s*[\(\[]"
    r"("
    r"demo|early\s+demo|"
    r"take\s+\d+|early\s+take|"
    r"instrumental|acoustic|"
    # Video patterns - may have prefix like "Agent Elvis - "
    r"(.*\s*-\s*)?(official\s+)?(music\s+)?video|"
    r"(.*\s*-\s*)?(official\s+)?(hd|4k|animated\s+music|animated)?\s*video|"
    r"(.*\s*-\s*)?(official\s+)?lyric\s+video|"
    r"(.*\s*-\s*)?visuali[sz]er|"
    r"(.*\s*-\s*)?official\s+audio|"
    r"outtakes?|sessions?|"
    r"alternate|"
    r"dub"
    r")"
    r"[\)\]]",
    re.IGNORECASE,
)

# Dash-separated suffix patterns (e.g., "Song - Early Take", "Song - Demo")
# These appear after " - " and indicate alternate versions
# Match everything after " - " that looks like a version indicator
_DASH_SUFFIX_PATTERN: Pattern = re.compile(
    r"\s+-\s+("
    # Year prefix patterns like "2004 Remastered Edition"
    r"\d{4}\s+(remaster(ed)?(\s+edition)?|mix|version)|"
    # Remasters and versions
    r"remaster(ed)?(\s+\d{4}|\s+edition)?|re-master(ed)?|"
    r"radio\s+edit|remix|extended|single|"
    # Demo/take patterns
    r"(early\s+)?demo|early\s+take|take\s+\d+|"
    r"(\w+\s+)*(instrumental|demo|take|rehearsal|rough)|"
    r"(studio\s+)?(guide\s+vocal\s+)?(instrumental\s+)?rough|"
    # Recording session material
    r"sessions?(\s+\w+)*\s*((&|and)\s*)?outtakes?|outtakes?|"
    r"alternate\s+mix|"
    # Mix patterns (Sunset Sound Mix, etc.)
    r"(\w+\s+)+mix|"
    # Live recordings
    r"live(\s+at\s+\w+.*)?|at\s+\w+.*|"
    # Audio formats
    r"mono(\s*/\s*remastered.*)?|stereo|"
    # Instrumental/acoustic
    r"instrumental|acoustic(\s+\w+)*|"
    # Video indicators
    r"official\s+(music\s+)?video|music\s+video|"
    r"official\s+(hd|4k|animated)?\s*video|"
    r"(official\s+)?lyric\s+video|"
    r"(hd|4k)\s+video|visuali[sz]er|official\s+audio|"
    # Other
    r"shortened\s+edit|vocal\s+version|"
    r"from\s+.+"  # From "Movie" etc
    r").*$",
    re.IGNORECASE,
)

# Trailing patterns without dash (e.g., "Song Name demo", "Song Name official video")
# These appear at the end of the track name without separator
_TRAILING_SUFFIX_PATTERN: Pattern = re.compile(
    r"\s+("
    # Demo/take at end
    r"demo|"
    r"take\s+\d+|"
    r"instrumental|"
    # Video/audio indicators at end
    r"official\s+(music\s+)?video|"
    r"(official\s+)?(hd|4k|animated)\s+video|"
    r"(official\s+)?lyric\s+video|"
    r"(official\s+)?visuali[sz]er|"
    r"official\s+audio|"
    # Live location at end (e.g., "at wembley")
    r"at\s+\w+(\s+\w+)*|"
    # Audio format at end
    r"mono|stereo|"
    # Excerpt numbering
    r"excerpt\s+\d+|"
    # Dub
    r"dub"
    r")$",
    re.IGNORECASE,
)


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
        'song'
        >>> normalize_text("Track (feat. Artist)")
        'track'
        >>> normalize_text("Song Name demo")
        'song name'
        >>> normalize_text("Song Name official video")
        'song name'
    """
    if not text:
        return ""

    # Lowercase and strip
    text = text.lower().strip()

    # Remove dash-separated suffixes FIRST (before year removal breaks them)
    # e.g., "Song - 2004 Remastered Edition" should be caught here
    text = _DASH_SUFFIX_PATTERN.sub("", text)

    # Remove parenthetical patterns
    text = _REMASTER_PATTERN.sub("", text)
    text = _LIVE_PATTERN.sub("", text)
    text = _VERSION_PATTERN.sub("", text)
    text = _EXPLICIT_PATTERN.sub("", text)
    text = _AUDIO_FORMAT_PATTERN.sub("", text)
    text = _DEMO_TAKE_PATTERN.sub("", text)
    text = _FEAT_PATTERN.sub("", text)
    text = _YEAR_PATTERN.sub("", text)

    # Remove trailing suffixes without dash (e.g., "Song Name demo")
    text = _TRAILING_SUFFIX_PATTERN.sub("", text)

    # Remove punctuation except hyphens (keep "hip-hop", etc.)
    text = _PUNCTUATION_PATTERN.sub("", text)

    # After punctuation removal, try dash suffix again for malformed cases
    # e.g., "Song - (Instrumental" becomes "Song - Instrumental" after punct removal
    text = _DASH_SUFFIX_PATTERN.sub("", text)

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
