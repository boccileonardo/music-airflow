import polars as pl
import streamlit as st
from music_airflow.utils.polars_io_manager import PolarsDeltaIOManager
from music_airflow.utils.constants import LAST_FM_USERNAMES


@st.cache_data
def load_track_candidates(username: str) -> pl.LazyFrame:
    """Load track candidates from Delta table and join with track metadata."""
    # Load candidates
    gold_io = PolarsDeltaIOManager("gold")
    silver_io = PolarsDeltaIOManager("silver")

    candidates = gold_io.read_delta("track_candidates").filter(
        pl.col("username") == username
    )
    track_info = silver_io.read_delta("candidate_enriched_tracks")

    candidates = (
        candidates.with_columns(
            pl.col("track_id").str.split_exact("|", 1).struct.unnest()
        )
        .rename(mapping={"field_0": "track_name", "field_1": "artist_name"})
        .select(
            "track_name",
            "artist_name",
            "score",
            "similar_artist",
            "similar_tag",
            "deep_cut_same_artist",
            "old_favorite",
        )
        .join(
            track_info,
            on=["artist_name", "track_name"],
        )
        .sort("score", "recommended_at", descending=True)
        .group_by(
            "track_id", "track_name", "artist_name"
        )  # todo: figure out why duplicate tracks with diff id
        .first()
    )
    return candidates


def filter_candidates(
    candidates: pl.LazyFrame,
    use_similar_tags: bool,
    use_similar_artists: bool,
    use_deep_cuts: bool,
    discovery_weight: float,
) -> pl.LazyFrame:
    """Filter candidates based on selected systems and discovery balance."""
    # Build filter for candidate types
    type_filters = []
    if use_similar_tags:
        type_filters.append(pl.col("similar_tag"))
    if use_similar_artists:
        type_filters.append(pl.col("similar_artist"))
    if use_deep_cuts:
        type_filters.append(pl.col("deep_cut_same_artist"))

    if not type_filters:
        # No systems selected, return empty
        return candidates.filter(pl.lit(False))

    # Combine filters with OR
    type_filter = type_filters[0]
    for f in type_filters[1:]:
        type_filter = type_filter | f

    candidates = candidates.filter(type_filter)

    # discovery_weight: 0 = all old favorites, 1 = all new discoveries
    candidates = candidates.with_columns(
        pl.when(pl.col("old_favorite"))
        .then(pl.col("score") * (1 - discovery_weight))
        .otherwise(pl.col("score") * discovery_weight)
        .alias("weighted_score")
    )

    return candidates


def main():
    st.title("🎵 Music Recommendation System")

    username = st.selectbox("username", options=LAST_FM_USERNAMES)

    st.header("Recommendation Settings")

    # Discovery vs Replay slider
    discovery_weight = st.slider(
        "Replay Old Songs (0) or Discover New Songs (1).",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help="0 = Only old favorites, 1 = Only new discoveries",
    )

    max_songs_per_artist = int(
        st.number_input("Max songs per artist", min_value=1, step=1)
    )

    # System selection checkboxes
    st.subheader("Candidate Generation Systems")
    col1, col2, col3 = st.columns(3)

    with col1:
        use_tags = st.checkbox("Similar Tags", value=True)
    with col2:
        use_artists = st.checkbox("Similar Artists", value=True)
    with col3:
        use_deep_cuts = st.checkbox("Deep Cuts", value=True)

    # Number of recommendations slider
    n_recommendations = st.slider(
        "Number of Recommendations",
        min_value=10,
        max_value=100,
        value=30,
        step=5,
    )

    # Generate button
    if st.button("Generate Recommendations", type="primary"):
        with st.spinner("Loading recommendations..."):
            try:
                # Load candidates
                candidates = load_track_candidates(username)

                # Filter based on settings
                candidates = filter_candidates(
                    candidates,
                    use_similar_tags=use_tags,
                    use_similar_artists=use_artists,
                    use_deep_cuts=use_deep_cuts,
                    discovery_weight=discovery_weight,
                )

                # Filter out tracks from the same artist exceeding max allowed per artist
                candidates = (
                    candidates.sort("score", descending=True)
                    .with_columns(rank=pl.row_index().over("artist_name"))
                    .filter(pl.col("rank") < max_songs_per_artist)
                )

                # Sample recommendations
                recommendations = (
                    candidates.sort("weighted_score", descending=True)
                    .limit(n_recommendations)
                    .collect()
                )

                if len(recommendations) == 0:
                    st.warning(
                        "No recommendations found. Try selecting at least one system."
                    )
                else:
                    st.success(f"Generated {len(recommendations)} recommendations!")

                    # Display results
                    st.header("Your Recommendations")

                    # Show statistics
                    n_old_favorites = recommendations["old_favorite"].sum()
                    n_new = len(recommendations) - n_old_favorites

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Tracks", len(recommendations))
                    col2.metric("New Discoveries", n_new)
                    col3.metric("Old Favorites", n_old_favorites)

                    # Use display names from joined data
                    recommendations = (
                        recommendations.group_by("track_name", "artist_name")
                        .first()
                        .with_columns(
                            pct_score=pl.col("weighted_score")
                            / (pl.col("weighted_score").max())
                        )
                        .select(
                            [
                                pl.col("track_name").alias("Track"),
                                pl.col("artist_name").alias("Artist"),
                                pl.col("pct_score").alias("Score").round(1),
                                pl.col("old_favorite").alias("Old Favorite"),
                                pl.col("similar_artist").alias("From Similar Artist"),
                                pl.col("similar_tag").alias("From Similar Tag"),
                                pl.col("deep_cut_same_artist").alias(
                                    "Deep Cut from Loved Artist"
                                ),
                            ]
                        )
                        .sort("Score", descending=True)
                    )

                    # Display as dataframe
                    st.dataframe(
                        recommendations,
                    )

            except Exception as e:
                st.error(f"Error generating recommendations: {e}")


if __name__ == "__main__":
    main()
