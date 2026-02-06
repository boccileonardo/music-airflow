"""
AirStream.FM - Music Recommendation Streamlit App.

Main entry point for the Streamlit application.
"""

import logging

import polars as pl
import streamlit as st

from music_airflow.app.data_loading import (
    load_top_artists,
    load_track_candidates,
    load_user_statistics,
    prefetch_all_users_track_candidates,
)
from music_airflow.app.exclusions_ui import (
    get_cached_excluded_artists,
    get_cached_excluded_tracks,
    render_exclusions_expander,
)
from music_airflow.app.filtering import (
    CANDIDATE_POOL_COLUMNS,
    apply_artist_limit,
    filter_candidates,
    load_recommendation_reasons,
)
from music_airflow.app.playlist_export_ui import render_playlist_export_section
from music_airflow.utils.constants import LAST_FM_USERNAMES

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def main():
    st.set_page_config(
        page_title="AirStream.FM",
        page_icon="🎵",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Prefetch track candidates for all users to enable instant switching
    prefetch_all_users_track_candidates()

    # Render sidebar and get settings
    settings = _render_sidebar()
    username = settings["username"]

    # Render main content
    _render_user_profile(username)
    recommendations = _generate_recommendations(username, settings)

    if recommendations is not None:
        _render_recommendations(recommendations)
        _render_why_recommended_expander(recommendations)
        render_exclusions_expander(username, recommendations)

    render_playlist_export_section()


def _render_sidebar() -> dict:
    """Render sidebar and return settings."""
    with st.sidebar:
        st.title("🎵 AirStream.FM")
        st.caption("Music Recommendation System")

        st.divider()

        username = st.selectbox("👤 User", options=LAST_FM_USERNAMES)

        st.divider()
        st.subheader("⚙️ Recommendation Settings")

        discovery_mode = st.radio(
            "Discovery Mode",
            options=["🔍 Discover New", "⚖️ Balanced", "🔄 Old Favorites"],
            index=1,
            horizontal=True,
            help="Choose how to balance new discoveries vs familiar tracks",
        )

        discovery_weight = {
            "🔍 Discover New": 1.0,
            "⚖️ Balanced": 0.5,
            "🔄 Old Favorites": 0.0,
        }[discovery_mode]

        n_recommendations = st.slider(
            "Number of Tracks",
            min_value=10,
            max_value=100,
            value=30,
            step=5,
        )

        max_songs_per_artist = st.number_input(
            "Max per Artist",
            min_value=1,
            max_value=10,
            value=3,
            step=1,
            help="Limit tracks from the same artist for variety",
        )

        st.divider()
        st.subheader("🎯 Sources")
        st.caption("Include tracks from:")

        use_tags = st.checkbox(
            "Similar Tags",
            value=True,
            help="Tracks with similar genre/mood tags",
        )
        use_artists = st.checkbox(
            "Similar Artists",
            value=True,
            help="Tracks from artists similar to your favorites",
        )
        use_deep_cuts = st.checkbox(
            "Deep Cuts",
            value=True,
            help="Lesser-known tracks from artists you love",
        )

    return {
        "username": username,
        "discovery_weight": discovery_weight,
        "n_recommendations": n_recommendations,
        "max_songs_per_artist": max_songs_per_artist,
        "use_tags": use_tags,
        "use_artists": use_artists,
        "use_deep_cuts": use_deep_cuts,
    }


def _render_user_profile(username: str) -> None:
    """Render user profile section with stats."""
    stats = load_user_statistics(username)

    st.header(f"📊 {username}'s Music Profile")

    col1, col2, col3 = st.columns(3)
    col1.metric("🎧 Plays", f"{stats['total_plays']:,}")
    col2.metric("🎵 Tracks", f"{stats['total_tracks_played']:,}")
    col3.metric("🎤 Artists", f"{stats['total_artists_played']:,}")

    with st.expander("🏆 Top Artists", expanded=False):
        top_artists = load_top_artists(username, limit=10)

        if len(top_artists) > 0:
            display_df = (
                top_artists.with_columns(pl.col("play_count").alias("Plays"))
                .select(["artist_name", "Plays"])
                .rename({"artist_name": "Artist"})
            )
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        else:
            st.info("No artist play data available yet.")

    st.divider()


def _generate_recommendations(username: str, settings: dict) -> pl.DataFrame | None:
    """Generate recommendations based on settings."""
    try:
        candidates = load_track_candidates(username)

        excluded_tracks = get_cached_excluded_tracks(username).lazy()
        excluded_artists = get_cached_excluded_artists(username).lazy()

        candidates = filter_candidates(
            candidates,
            use_similar_tags=settings["use_tags"],
            use_similar_artists=settings["use_artists"],
            use_deep_cuts=settings["use_deep_cuts"],
            discovery_weight=settings["discovery_weight"],
            excluded_tracks=excluded_tracks,
            excluded_artists=excluded_artists,
        )

        candidates = apply_artist_limit(candidates, settings["max_songs_per_artist"])

        all_recommendations = candidates.sort("weighted_score", descending=True)
        recommendations = all_recommendations.head(
            settings["n_recommendations"]
        ).collect()

        if len(recommendations) == 0:
            st.warning("No recommendations found. Try selecting at least one source.")
            st.session_state.recommendations = None
            return None

        # Store in session state
        st.session_state.recommendations = recommendations
        st.session_state.username = username
        st.session_state.playlist_settings = {
            "discovery_weight": settings["discovery_weight"],
            "use_tags": settings["use_tags"],
            "use_artists": settings["use_artists"],
            "use_deep_cuts": settings["use_deep_cuts"],
        }

        # Store candidate pool for replacements
        available_columns = [
            c
            for c in CANDIDATE_POOL_COLUMNS
            if c in candidates.collect_schema().names()
        ]
        st.session_state.candidate_pool = (
            candidates.select(available_columns)
            .sort("weighted_score", descending=True)
            .limit(settings["n_recommendations"] * 3)
            .collect()
        )

        return recommendations

    except Exception as e:
        st.error(f"Error generating recommendations: {e}")
        st.session_state.recommendations = None
        return None


def _render_recommendations(recommendations: pl.DataFrame) -> None:
    """Render the recommendations dataframe."""
    n_old_favorites = recommendations["old_favorite"].sum()
    n_new = len(recommendations) - n_old_favorites

    st.header("🎵 Your Recommendations")
    st.caption(
        f"**{len(recommendations)}** tracks · "
        f"**{n_new}** new discoveries · "
        f"**{n_old_favorites}** old favorites"
    )

    display_recommendations = (
        recommendations.sort("weighted_score", descending=True)
        .with_columns(
            ((pl.col("weighted_score") / pl.col("weighted_score").max()) * 100)
            .round(1)
            .alias("normalized_score")
        )
        .group_by("track_name", "artist_name")
        .first()
        .select(
            [
                pl.col("track_name").alias("Track"),
                pl.col("artist_name").alias("Artist"),
                pl.col("normalized_score").alias("Score"),
                pl.col("old_favorite").alias("🔄"),
                pl.col("similar_artist").alias("👥"),
                pl.col("similar_tag").alias("🏷️"),
                pl.col("deep_cut_same_artist").alias("💎"),
            ]
        )
        .sort("Score", descending=True)
    )

    st.dataframe(
        display_recommendations,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Score": st.column_config.ProgressColumn(
                "Score",
                min_value=0,
                max_value=100,
                format="%.0f",
            ),
            "🔄": st.column_config.CheckboxColumn("Old Fav", help="Previously played"),
            "👥": st.column_config.CheckboxColumn(
                "Similar Artist", help="From similar artist"
            ),
            "🏷️": st.column_config.CheckboxColumn(
                "Similar Tag", help="From similar tags"
            ),
            "💎": st.column_config.CheckboxColumn(
                "Deep Cut", help="Deep cut from loved artist"
            ),
        },
    )


def _render_why_recommended_expander(recommendations: pl.DataFrame) -> None:
    """Render the 'Why was this recommended?' expander."""
    with st.expander("❓ Why was this recommended?", expanded=False):
        why_columns = [
            "track_id",
            "track_name",
            "artist_name",
            "similar_artist",
            "similar_tag",
            "deep_cut_same_artist",
            "old_favorite",
            "why_similar_artist_name",
            "why_similar_artist_pct",
            "why_similar_tags",
            "why_tag_match_count",
            "why_deep_cut_artist",
        ]
        available_columns = [c for c in why_columns if c in recommendations.columns]
        track_options = {
            f"{row['track_name']} - {row['artist_name']}": row
            for row in recommendations.select(available_columns).to_dicts()
        }

        selected_track_display = st.selectbox(
            "Select a track to see why it was recommended",
            options=list(track_options.keys()),
            key="track_detail_select",
            label_visibility="collapsed",
        )

        if selected_track_display:
            track_info = track_options[selected_track_display]

            st.markdown(f"### {track_info['track_name']}")
            st.caption(f"by {track_info['artist_name']}")

            reasons = load_recommendation_reasons(track_info)
            has_reasons = False

            if track_info.get("old_favorite"):
                has_reasons = True
                st.info(
                    "🔄 **Old Favorite** — You've played this before but not recently. Time to revisit!"
                )

            if track_info.get("similar_artist") and "similar_artist" in reasons:
                has_reasons = True
                r = reasons["similar_artist"]
                st.success(
                    f"👥 **Similar Artist** — Recommended because you listen to "
                    f"**{r['source_artist']}** ({r['similarity']}% similarity)"
                )

            if track_info.get("similar_tag") and "similar_tag" in reasons:
                has_reasons = True
                r = reasons["similar_tag"]
                tags_str = ", ".join(f"*{t}*" for t in r["tags"])
                st.success(
                    f"🏷️ **Similar Tags** — Matches {r['match_count']} of your favorite tags: {tags_str}"
                )

            if track_info.get("deep_cut_same_artist") and "deep_cut" in reasons:
                has_reasons = True
                r = reasons["deep_cut"]
                st.success(
                    f"💎 **Deep Cut** — A lesser-known track from **{r['source_artist']}**, "
                    f"one of your favorites"
                )

            if not has_reasons:
                st.caption("Recommendation details not available for this track.")


if __name__ == "__main__":
    main()
