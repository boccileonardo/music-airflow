"""
Text normalization utilities for fuzzy track/artist matching.

Provides functions to normalize track and artist names for consistent
identification across different recordings and versions.
"""

import re
from typing import Pattern

import polars as pl


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


# =============================================================================
# Native Polars expressions for text normalization (faster than map_elements)
# =============================================================================

# All patterns use (?i) for case-insensitivity (Rust regex inline flag)

# Dash-separated suffixes (e.g., "Song - Remastered 2012", "Song - Live at Wembley")
_POLARS_DASH_SUFFIX = (
    r"(?i)\s+-\s+("
    r"\d{4}\s+(remaster(ed)?(\s+edition)?|mix|version)|"
    r"remaster(ed)?(\s+\d{4}|\s+edition)?|re-master(ed)?|"
    r"radio\s+edit|remix|extended|single|"
    r"(early\s+)?demo|early\s+take|take\s+\d+|"
    r"(\w+\s+)*(instrumental|demo|take|rehearsal|rough)|"
    r"(studio\s+)?(guide\s+vocal\s+)?(instrumental\s+)?rough|"
    r"sessions?(\s+\w+)*\s*((&|and)\s*)?outtakes?|outtakes?|"
    r"alternate\s+mix|"
    r"(\w+\s+)+mix|"
    r"live(\s+at\s+\w+.*)?|at\s+\w+.*|"
    r"mono(\s*/\s*remastered.*)?|stereo|"
    r"instrumental|acoustic(\s+\w+)*|"
    r"official\s+(music\s+)?video|music\s+video|"
    r"official\s+(hd|4k|animated)?\s*video|"
    r"(official\s+)?lyric\s+video|"
    r"(hd|4k)\s+video|visuali[sz]er|official\s+audio|"
    r"shortened\s+edit|vocal\s+version|"
    r"from\s+.+"
    r").*$"
)

# Parenthetical patterns - content in () or []
_POLARS_REMASTER = r"(?i)\s*[\(\[](remaster(ed)?|re-master(ed)?|\d{4}\s+remaster(ed)?|remaster(ed)?\s+\d{4})[\)\]]"
_POLARS_LIVE = r"(?i)\s*[\(\[]((live|live\s+(at|from|in|on)|live\s+\d{4}|live\s+(version|recording)).*?)[\)\]]"
_POLARS_VERSION = r"(?i)\s*[\(\[](.*?\s+(version|mix|edit|remix|take)|single\s+version|album\s+version|radio\s+edit|extended(\s+(version|mix|edit))?|remix)[\)\]]"
_POLARS_EXPLICIT = r"(?i)\s*[\(\[](explicit|clean|censored)[\)\]]"
_POLARS_AUDIO_FORMAT = r"(?i)\s*[\(\[]((stereo|mono)(\s+(mix|version))?|original\s+(stereo|mono)|true\s+stereo|simulated\s+stereo)[\)\]]"
_POLARS_DEMO_TAKE = (
    r"(?i)\s*[\(\[]("
    r"demo|early\s+demo|"
    r"take\s+\d+|early\s+take|"
    r"instrumental|acoustic|"
    r"(.*\s*-\s*)?(official\s+)?(music\s+)?video|"
    r"(.*\s*-\s*)?(official\s+)?(hd|4k|animated(\s+music)?|animated)?\s*video|"
    r"(.*\s*-\s*)?(official\s+)?lyric\s+video|"
    r"(.*\s*-\s*)?visuali[sz]er|"
    r"(.*\s*-\s*)?official\s+audio|"
    r"outtakes?|sessions?|"
    r"alternate|"
    r"dub"
    r")[\)\]]"
)
_POLARS_FEAT = r"(?i)\s*[\(\[]?(feat\.?|ft\.?|featuring|with|vs\.?|versus).*?[\)\]]?$"
_POLARS_YEAR = r"\s*[\(\[]?\d{4}[\)\]]?\s*"

# Trailing suffixes without dash
_POLARS_TRAILING = (
    r"(?i)\s+("
    r"demo|"
    r"take\s+\d+|"
    r"instrumental|"
    r"official\s+(music\s+)?video|"
    r"(official\s+)?(hd|4k|animated)\s+video|"
    r"(official\s+)?lyric\s+video|"
    r"(official\s+)?visuali[sz]er|"
    r"official\s+audio|"
    r"at\s+\w+(\s+\w+)*|"
    r"mono|stereo|"
    r"excerpt\s+\d+|"
    r"dub"
    r")$"
)

# Punctuation (keep hyphens for "hip-hop" etc)
_POLARS_PUNCTUATION = r"[^\w\s-]"
_POLARS_WHITESPACE = r"\s+"

# Music video detection pattern
_POLARS_MUSIC_VIDEO = r"(?i)(music\s+video|official\s+video|video\s+clip|visuali[sz]er|lyric\s+video|official\s+audio)"


def normalize_text_expr(col: str | pl.Expr) -> pl.Expr:
    """
    Create a Polars expression that normalizes text for fuzzy matching.

    This is the native Polars version of normalize_text() - much faster
    than using map_elements with the Python function.

    Args:
        col: Column name or expression to normalize

    Returns:
        Polars expression that performs normalization

    Examples:
        >>> df.with_columns(normalized=normalize_text_expr("track_name"))
        >>> df.with_columns(normalized=normalize_text_expr(pl.col("track_name")))
    """
    if isinstance(col, str):
        expr = pl.col(col)
    else:
        expr = col

    return (
        expr.str.to_lowercase()
        .str.strip_chars()
        # Remove dash-separated suffixes first
        .str.replace_all(_POLARS_DASH_SUFFIX, "")
        # Remove parenthetical patterns
        .str.replace_all(_POLARS_REMASTER, "")
        .str.replace_all(_POLARS_LIVE, "")
        .str.replace_all(_POLARS_VERSION, "")
        .str.replace_all(_POLARS_EXPLICIT, "")
        .str.replace_all(_POLARS_AUDIO_FORMAT, "")
        .str.replace_all(_POLARS_DEMO_TAKE, "")
        .str.replace_all(_POLARS_FEAT, "")
        .str.replace_all(_POLARS_YEAR, "")
        # Remove trailing suffixes
        .str.replace_all(_POLARS_TRAILING, "")
        # Remove punctuation except hyphens
        .str.replace_all(_POLARS_PUNCTUATION, "")
        # Try dash suffix again for malformed cases
        .str.replace_all(_POLARS_DASH_SUFFIX, "")
        # Normalize whitespace
        .str.replace_all(_POLARS_WHITESPACE, " ")
        .str.strip_chars()
    )


def generate_canonical_track_id_expr(
    track_col: str | pl.Expr, artist_col: str | pl.Expr
) -> pl.Expr:
    """
    Create a Polars expression that generates canonical track IDs.

    This is the native Polars version of generate_canonical_track_id() -
    much faster than using map_elements with the Python function.

    Args:
        track_col: Column name or expression for track name
        artist_col: Column name or expression for artist name

    Returns:
        Polars expression that generates "normalized_track|normalized_artist" IDs

    Examples:
        >>> df.with_columns(track_id=generate_canonical_track_id_expr("track_name", "artist_name"))
    """
    normalized_track = normalize_text_expr(track_col)
    normalized_artist = normalize_text_expr(artist_col)

    return pl.concat_str([normalized_track, pl.lit("|"), normalized_artist])


def generate_canonical_artist_id_expr(artist_col: str | pl.Expr) -> pl.Expr:
    """
    Create a Polars expression that generates canonical artist IDs.

    This is the native Polars version of generate_canonical_artist_id() -
    much faster than using map_elements with the Python function.

    Args:
        artist_col: Column name or expression for artist name

    Returns:
        Polars expression that generates normalized artist ID

    Examples:
        >>> df.with_columns(artist_id=generate_canonical_artist_id_expr("artist_name"))
    """
    return normalize_text_expr(artist_col)


def is_music_video_expr(col: str | pl.Expr) -> pl.Expr:
    """
    Create a Polars expression that detects music video versions.

    This is the native Polars version of is_music_video() -
    much faster than using map_elements with the Python function.

    Args:
        col: Column name or expression for track name

    Returns:
        Polars boolean expression (True if track appears to be a music video)

    Examples:
        >>> df.with_columns(is_video=is_music_video_expr("track_name"))
    """
    if isinstance(col, str):
        expr = pl.col(col)
    else:
        expr = col

    return expr.str.contains(_POLARS_MUSIC_VIDEO)


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
