"""
Filtering and recommendation logic for the Streamlit app.

Handles filtering track candidates based on user preferences,
exclusions, and discovery settings.
"""

import polars as pl


def filter_candidates(
    candidates: pl.LazyFrame,
    use_similar_tags: bool,
    use_similar_artists: bool,
    use_deep_cuts: bool,
    discovery_weight: float,
    excluded_tracks: pl.LazyFrame | None = None,
    excluded_artists: pl.LazyFrame | None = None,
) -> pl.LazyFrame:
    """Filter candidates based on selected systems and discovery balance.

    Args:
        candidates: LazyFrame of track candidates
        use_similar_tags: Include tracks with similar tags
        use_similar_artists: Include tracks from similar artists
        use_deep_cuts: Include deep cuts from loved artists
        discovery_weight: 0 = old favorites, 1 = new discoveries
        excluded_tracks: Tracks to exclude from results
        excluded_artists: Artists to exclude from results

    Returns:
        Filtered LazyFrame with weighted_score column added
    """
    # Build filter for candidate types
    type_filters = []
    if use_similar_tags:
        type_filters.append(pl.col("similar_tag"))
    if use_similar_artists:
        type_filters.append(pl.col("similar_artist"))
    if use_deep_cuts:
        type_filters.append(pl.col("deep_cut_same_artist"))

    if not type_filters:
        return candidates.filter(pl.lit(False))

    # Combine filters with OR
    type_filter = type_filters[0]
    for f in type_filters[1:]:
        type_filter = type_filter | f

    candidates = candidates.filter(type_filter)

    # Filter out excluded artists
    if excluded_artists is not None:
        excluded_artist_names = excluded_artists.select("artist_name")
        candidates = candidates.join(
            excluded_artist_names,
            on="artist_name",
            how="anti",
        )

    # Filter out excluded tracks by track_name + artist_name
    if excluded_tracks is not None:
        excluded_track_keys = excluded_tracks.select(["track_name", "artist_name"])
        candidates = candidates.join(
            excluded_track_keys,
            on=["track_name", "artist_name"],
            how="anti",
        )

    # Apply discovery weighting
    candidates = candidates.with_columns(
        (
            pl.when(pl.col("old_favorite"))
            .then(pl.col("score") * (1.1 - discovery_weight))
            .otherwise(pl.col("score") * (0.1 + discovery_weight))
        ).alias("weighted_score")
    )

    return candidates


def apply_artist_limit(
    candidates: pl.LazyFrame,
    max_songs_per_artist: int,
) -> pl.LazyFrame:
    """Limit tracks from the same artist for variety.

    Args:
        candidates: LazyFrame sorted by weighted_score descending
        max_songs_per_artist: Maximum tracks per artist

    Returns:
        Filtered LazyFrame with at most max_songs_per_artist per artist
    """
    return (
        candidates.sort("weighted_score", descending=True)
        .with_columns(artist_rank=pl.int_range(pl.len()).over("artist_name"))
        .filter(pl.col("artist_rank") < max_songs_per_artist)
        .drop("artist_rank")
    )


def load_recommendation_reasons(track_row: dict) -> dict:
    """Extract 'why' data from baked-in columns in gold table.

    The gold table includes pre-computed "why" columns:
    - why_similar_artist_name, why_similar_artist_pct
    - why_similar_tags, why_tag_match_count
    - why_deep_cut_artist

    Args:
        track_row: Dictionary containing track data with why columns

    Returns:
        Dict with keys: similar_artist, similar_tag, deep_cut
    """
    reasons = {}

    if track_row.get("why_similar_artist_name"):
        reasons["similar_artist"] = {
            "source_artist": track_row["why_similar_artist_name"],
            "similarity": track_row.get("why_similar_artist_pct", 0),
        }

    if track_row.get("why_similar_tags"):
        source_tags = track_row["why_similar_tags"]
        tags = list(dict.fromkeys(t.strip() for t in source_tags.split(",")))[:5]
        reasons["similar_tag"] = {
            "tags": tags,
            "match_count": track_row.get("why_tag_match_count", 0),
        }

    if track_row.get("why_deep_cut_artist"):
        reasons["deep_cut"] = {
            "source_artist": track_row["why_deep_cut_artist"],
        }

    return reasons


# Columns to select for candidate pool (for replacements)
CANDIDATE_POOL_COLUMNS = [
    "track_id",
    "track_name",
    "artist_name",
    "score",
    "similar_artist",
    "similar_tag",
    "deep_cut_same_artist",
    "old_favorite",
    "youtube_url",
    "spotify_url",
    "tags",
    "duration_ms",
    "username",
    "weighted_score",
]
