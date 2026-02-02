import logging
import traceback

import polars as pl
import streamlit as st

from music_airflow.utils.polars_io_manager import PolarsDeltaIOManager
from music_airflow.utils.constants import LAST_FM_USERNAMES
from music_airflow.app.youtube_playlist import (
    YouTubePlaylistGenerator,
    load_youtube_creds,
    poll_device_token,
    run_youtube_oauth,
)
from music_airflow.app.spotify_playlist import (
    SpotifyPlaylistGenerator,
    load_spotify_creds,
    run_spotify_oauth,
    exchange_code_for_token,
)
from music_airflow.app.excluded_tracks import (
    write_excluded_track,
    read_excluded_tracks,
    write_excluded_artist,
    read_excluded_artists,
    remove_excluded_track,
    remove_excluded_artist,
)

# Configure logging to show in Streamlit
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Internal limit for caching - always compute up to 100 to reuse cache
INTERNAL_LIMIT = 300


@st.cache_data(ttl=300)  # 5 minutes
def load_track_candidates_cached(username: str) -> pl.DataFrame:
    """Load and cache track candidates from gold table."""
    gold_io = PolarsDeltaIOManager("gold")
    return (
        gold_io.read_delta("track_candidates")
        .filter(pl.col("username") == username)
        .sort("score", descending=True)
        .limit(INTERNAL_LIMIT)
        .collect()
    )


def load_track_candidates(username: str) -> pl.LazyFrame:
    """Load track candidates from gold table - presentation-ready data."""
    # Use cached DataFrame and convert back to LazyFrame for filtering
    return load_track_candidates_cached(username).lazy()


@st.cache_data(ttl="1d")
def load_user_statistics(username: str) -> dict:
    """Load user play statistics from gold aggregation tables."""
    gold_io = PolarsDeltaIOManager("gold")

    stats = {}

    # Load track play counts
    try:
        track_plays = (
            gold_io.read_delta("track_play_count")
            .filter(pl.col("username") == username)
            .collect()
        )
        stats["total_tracks_played"] = len(track_plays)
        stats["total_plays"] = track_plays["play_count"].sum()
    except Exception:
        stats["total_tracks_played"] = 0
        stats["total_plays"] = 0

    # Load artist play counts
    try:
        artist_plays = (
            gold_io.read_delta("artist_play_count")
            .filter(pl.col("username") == username)
            .collect()
        )
        stats["total_artists_played"] = len(artist_plays)
    except Exception:
        stats["total_artists_played"] = 0

    return stats


@st.cache_data(ttl="1d")
def load_top_artists(username: str, limit: int = 10) -> pl.DataFrame:
    """Load top artists by play count."""
    gold_io = PolarsDeltaIOManager("gold")

    try:
        artist_plays = gold_io.read_delta("artist_play_count").filter(
            (pl.col("username") == username) & pl.col("artist_name").is_not_null()
        )
        return artist_plays.sort("play_count", descending=True).head(limit).collect()
    except Exception:
        return pl.DataFrame(schema={"artist_name": pl.String, "play_count": pl.Int64})


def load_recommendation_reasons(username: str, track_id: str) -> dict:
    """Load the 'why' data for a recommendation from silver candidate tables.

    Returns a dict with keys: similar_artist, similar_tag, deep_cut, each containing
    the source information if available.
    """
    silver_io = PolarsDeltaIOManager("silver")
    reasons = {}

    # Check similar artist source
    try:
        similar_artist_df = (
            silver_io.read_delta("candidate_similar_artist")
            .filter((pl.col("username") == username) & (pl.col("track_id") == track_id))
            .collect()
        )
        if len(similar_artist_df) > 0:
            source_artist_id = similar_artist_df["source_artist_id"][0]
            similarity = similar_artist_df["similarity"][0]
            # Look up artist name from the artists dimension
            try:
                artists_df = (
                    silver_io.read_delta("artists")
                    .filter(pl.col("artist_id") == source_artist_id)
                    .select("artist_name")
                    .collect()
                )
                if len(artists_df) > 0:
                    source_artist_name = artists_df["artist_name"][0]
                else:
                    source_artist_name = source_artist_id.replace("_", " ").title()
            except Exception:
                source_artist_name = source_artist_id.replace("_", " ").title()

            reasons["similar_artist"] = {
                "source_artist": source_artist_name,
                "similarity": round(similarity * 100, 1),
            }
    except Exception:
        pass

    # Check similar tag source
    try:
        similar_tag_df = (
            silver_io.read_delta("candidate_similar_tag")
            .filter((pl.col("username") == username) & (pl.col("track_id") == track_id))
            .collect()
        )
        if len(similar_tag_df) > 0:
            source_tags = similar_tag_df["source_tags"][0]
            tag_match_count = similar_tag_df["tag_match_count"][0]
            # Parse and dedupe tags
            tags = list(dict.fromkeys(t.strip() for t in source_tags.split(",")))[:5]
            reasons["similar_tag"] = {
                "tags": tags,
                "match_count": tag_match_count,
            }
    except Exception:
        pass

    # Check deep cut source
    try:
        deep_cut_df = (
            silver_io.read_delta("candidate_deep_cut")
            .filter((pl.col("username") == username) & (pl.col("track_id") == track_id))
            .collect()
        )
        if len(deep_cut_df) > 0:
            source_artist_id = deep_cut_df["source_artist_id"][0]
            # Look up artist name
            try:
                artists_df = (
                    silver_io.read_delta("artists")
                    .filter(pl.col("artist_id") == source_artist_id)
                    .select("artist_name")
                    .collect()
                )
                if len(artists_df) > 0:
                    source_artist_name = artists_df["artist_name"][0]
                else:
                    source_artist_name = source_artist_id.replace("_", " ").title()
            except Exception:
                source_artist_name = source_artist_id.replace("_", " ").title()

            reasons["deep_cut"] = {
                "source_artist": source_artist_name,
            }
    except Exception:
        pass

    return reasons


def filter_candidates(
    candidates: pl.LazyFrame,
    use_similar_tags: bool,
    use_similar_artists: bool,
    use_deep_cuts: bool,
    discovery_weight: float,
    excluded_tracks: pl.LazyFrame | None = None,
    excluded_artists: pl.LazyFrame | None = None,
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

    # Filter out excluded artists
    if excluded_artists is not None:
        excluded_artist_names = excluded_artists.select("artist_name")
        candidates = candidates.join(
            excluded_artist_names,
            on="artist_name",
            how="anti",  # Keep only artists not in excluded list
        )

    # Filter out excluded tracks by track_name + artist_name
    # (more stable than track_id which may vary due to normalization)
    if excluded_tracks is not None:
        excluded_track_keys = excluded_tracks.select(["track_name", "artist_name"])
        candidates = candidates.join(
            excluded_track_keys,
            on=["track_name", "artist_name"],
            how="anti",  # Keep only tracks not in excluded list
        )

    # discovery_weight: 0 = all old favorites, 1 = all new discoveries
    # Apply blending: weight both categories and keep the base score component
    candidates = candidates.with_columns(
        (
            pl.when(pl.col("old_favorite"))
            .then(
                pl.col("score") * (1.1 - discovery_weight)
            )  # Old favorites: 1.1x at weight=0, 0.1x at weight=1
            .otherwise(
                pl.col("score") * (0.1 + discovery_weight)
            )  # New discoveries: 0.1x at weight=0, 1.1x at weight=1
        ).alias("weighted_score")
    )

    return candidates


def main():
    st.set_page_config(
        page_title="KainosFM",
        page_icon="🎵",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # --- Sidebar: All Settings ---
    with st.sidebar:
        st.title("🎵 KainosFM")
        st.caption("Music Recommendation System")

        st.divider()

        username = st.selectbox("👤 User", options=LAST_FM_USERNAMES)

        st.divider()
        st.subheader("⚙️ Recommendation Settings")

        discovery_weight = st.slider(
            "Discovery Balance",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="0 = Replay old favorites · 1 = Discover new music",
        )

        # Visual indicator for the slider
        if discovery_weight < 0.3:
            st.caption("🔄 *Focusing on familiar favorites*")
        elif discovery_weight > 0.7:
            st.caption("🔍 *Exploring new music*")
        else:
            st.caption("⚖️ *Balanced mix*")

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

    # --- Main Content Area ---
    # Load stats (needed for display)
    with st.spinner("Loading statistics..."):
        stats = load_user_statistics(username)

    # Header with stats
    st.header(f"📊 {username}'s Music Profile")

    # Stats in a more compact row
    col1, col2, col3 = st.columns(3)
    col1.metric("🎧 Plays", f"{stats['total_plays']:,}")
    col2.metric("🎵 Tracks", f"{stats['total_tracks_played']:,}")
    col3.metric("🎤 Artists", f"{stats['total_artists_played']:,}")

    # Top Artists in expander (secondary info)
    with st.expander("🏆 Top Artists", expanded=False):
        with st.spinner("Loading..."):
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

    # Generate recommendations automatically (reactive to settings)
    with st.spinner("Loading recommendations..."):
        try:
            # Load candidates
            candidates = load_track_candidates(username)

            # Load exclusions for this user
            excluded_tracks = read_excluded_tracks(username)
            excluded_artists = read_excluded_artists(username)

            # Filter based on settings
            candidates = filter_candidates(
                candidates,
                use_similar_tags=use_tags,
                use_similar_artists=use_artists,
                use_deep_cuts=use_deep_cuts,
                discovery_weight=discovery_weight,
                excluded_tracks=excluded_tracks,
                excluded_artists=excluded_artists,
            )

            # Filter out tracks from the same artist exceeding max allowed per artist
            # Use row_number within each artist group, ordered by weighted_score descending
            candidates = (
                candidates.sort("weighted_score", descending=True)
                .with_columns(artist_rank=pl.int_range(pl.len()).over("artist_name"))
                .filter(pl.col("artist_rank") < max_songs_per_artist)
                .drop("artist_rank")  # Remove the temporary ranking column
            )
            # Finally, sort by weighted score to get the best recommendations
            all_recommendations = candidates.sort("weighted_score", descending=True)
            # Then slice to requested amount
            recommendations = all_recommendations.head(n_recommendations).collect()

            if len(recommendations) == 0:
                st.warning(
                    "No recommendations found. Try selecting at least one source."
                )
                st.session_state.recommendations = None
            else:
                # Store recommendations in session state for playlist creation
                st.session_state.recommendations = recommendations
                st.session_state.username = username
                st.session_state.playlist_settings = {
                    "discovery_weight": discovery_weight,
                    "use_tags": use_tags,
                    "use_artists": use_artists,
                    "use_deep_cuts": use_deep_cuts,
                }
                # Store the available pool of candidates for replacements
                st.session_state.candidate_pool = (
                    candidates.select(
                        [
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
                    )
                    .sort("weighted_score", descending=True)
                    .limit(n_recommendations * 3)  # Get 3x for replacement pool
                    .collect()
                )

        except Exception as e:
            st.error(f"Error generating recommendations: {e}")
            st.session_state.recommendations = None

    # Display recommendations if they exist in session state
    if (
        "recommendations" in st.session_state
        and st.session_state.recommendations is not None
    ):
        recommendations = st.session_state.recommendations

        # Show statistics
        n_old_favorites = recommendations["old_favorite"].sum()
        n_new = len(recommendations) - n_old_favorites

        # Header with inline stats
        st.header("🎵 Your Recommendations")
        st.caption(
            f"**{len(recommendations)}** tracks · "
            f"**{n_new}** new discoveries · "
            f"**{n_old_favorites}** old favorites"
        )

        # Use display names from joined data
        display_recommendations = (
            recommendations.sort("weighted_score", descending=True)
            .with_columns(
                # Normalize score: best song = 100%, rest scale down
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

        # Display as dataframe with column config
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
                "🔄": st.column_config.CheckboxColumn(
                    "Old Fav", help="Previously played"
                ),
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

        # Why this recommendation - track details
        with st.expander("❓ Why was this recommended?", expanded=False):
            with st.spinner("Loading recommendation details..."):
                # Build track options for dropdown
                track_options = {
                    f"{row['track_name']} - {row['artist_name']}": row
                    for row in recommendations.select(
                        [
                            "track_id",
                            "track_name",
                            "artist_name",
                            "similar_artist",
                            "similar_tag",
                            "deep_cut_same_artist",
                            "old_favorite",
                        ]
                    ).to_dicts()
                }

                selected_track_display = st.selectbox(
                    "Select a track to see why it was recommended",
                    options=list(track_options.keys()),
                    key="track_detail_select",
                    label_visibility="collapsed",
                )

                if selected_track_display:
                    track_info = track_options[selected_track_display]
                    track_id = track_info["track_id"]

                    st.markdown(f"### {track_info['track_name']}")
                    st.caption(f"by {track_info['artist_name']}")

                    # Load reasons from silver tables
                    reasons = load_recommendation_reasons(username, track_id)

                    # Display reasons based on which flags are true
                    has_reasons = False

                    if track_info["old_favorite"]:
                        has_reasons = True
                        st.info(
                            "🔄 **Old Favorite** — You've played this before but not recently. Time to revisit!"
                        )

                    if track_info["similar_artist"] and "similar_artist" in reasons:
                        has_reasons = True
                        r = reasons["similar_artist"]
                        st.success(
                            f"👥 **Similar Artist** — Recommended because you listen to **{r['source_artist']}** "
                            f"({r['similarity']}% similarity)"
                        )

                    if track_info["similar_tag"] and "similar_tag" in reasons:
                        has_reasons = True
                        r = reasons["similar_tag"]
                        tags_str = ", ".join(f"*{t}*" for t in r["tags"])
                        st.success(
                            f"🏷️ **Similar Tags** — Matches {r['match_count']} of your favorite tags: {tags_str}"
                        )

                    if track_info["deep_cut_same_artist"] and "deep_cut" in reasons:
                        has_reasons = True
                        r = reasons["deep_cut"]
                        st.success(
                            f"💎 **Deep Cut** — A lesser-known track from **{r['source_artist']}**, one of your favorites"
                        )

                    if not has_reasons:
                        st.caption(
                            "Recommendation details not available for this track."
                        )

        # Exclusion Management - Simplified with tabs
        with st.expander("🚫 Manage Exclusions", expanded=False):
            with st.spinner("Loading exclusions..."):
                # Load current exclusions
                excluded_tracks_df = read_excluded_tracks(username)
                excluded_artists_df = read_excluded_artists(username)

                try:
                    excluded_tracks_collected = excluded_tracks_df.collect()
                    n_excluded_tracks = len(excluded_tracks_collected)
                except Exception:
                    excluded_tracks_collected = pl.DataFrame()
                    n_excluded_tracks = 0

                try:
                    excluded_artists_collected = excluded_artists_df.collect()
                    n_excluded_artists = len(excluded_artists_collected)
                except Exception:
                    excluded_artists_collected = pl.DataFrame()
                    n_excluded_artists = 0

                tab_tracks, tab_artists = st.tabs(
                    [
                        f"🎵 Tracks ({n_excluded_tracks})",
                        f"🎤 Artists ({n_excluded_artists})",
                    ]
                )

                with tab_tracks:
                    # Get current track IDs for dropdown
                    current_tracks = recommendations.select(
                        ["track_id", "track_name", "artist_name"]
                    ).to_dicts()
                    track_options = {
                        f"{t['track_name']} - {t['artist_name']}": t
                        for t in current_tracks
                    }

                    if len(track_options) > 0:
                        selected_track_display = st.selectbox(
                            "Exclude a track from recommendations",
                            options=list(track_options.keys()),
                            key="track_to_remove",
                        )

                        if st.button(
                            "🗑️ Exclude Track", type="secondary", key="exclude_track_btn"
                        ):
                            selected_track = track_options[selected_track_display]
                            track_id = selected_track["track_id"]

                            try:
                                write_excluded_track(
                                    username=st.session_state.username,
                                    track_id=track_id,
                                    track_name=selected_track["track_name"],
                                    artist_name=selected_track["artist_name"],
                                )

                                excluded_key = (
                                    selected_track["track_name"],
                                    selected_track["artist_name"],
                                )
                                if "excluded_in_session" not in st.session_state:
                                    st.session_state.excluded_in_session = set()
                                st.session_state.excluded_in_session.add(excluded_key)

                                st.session_state.recommendations = (
                                    recommendations.filter(
                                        (
                                            pl.col("track_name")
                                            != selected_track["track_name"]
                                        )
                                        | (
                                            pl.col("artist_name")
                                            != selected_track["artist_name"]
                                        )
                                    )
                                )

                                # Find replacement from candidate pool
                                if "candidate_pool" in st.session_state:
                                    displayed_tracks = set(
                                        zip(
                                            recommendations.select("track_name")
                                            .to_series()
                                            .to_list(),
                                            recommendations.select("artist_name")
                                            .to_series()
                                            .to_list(),
                                        )
                                    )
                                    displayed_tracks.update(
                                        st.session_state.excluded_in_session
                                    )

                                    pool = st.session_state.candidate_pool
                                    replacement = None
                                    for row in pool.iter_rows(named=True):
                                        track_key = (
                                            row["track_name"],
                                            row["artist_name"],
                                        )
                                        if track_key not in displayed_tracks:
                                            replacement = pool.filter(
                                                (
                                                    pl.col("track_name")
                                                    == row["track_name"]
                                                )
                                                & (
                                                    pl.col("artist_name")
                                                    == row["artist_name"]
                                                )
                                            ).limit(1)
                                            break

                                    if replacement is not None and len(replacement) > 0:
                                        replacement = replacement.select(
                                            st.session_state.recommendations.columns
                                        )
                                        st.session_state.recommendations = pl.concat(
                                            [
                                                st.session_state.recommendations,
                                                replacement,
                                            ]
                                        )
                                        st.toast(
                                            f"Excluded and replaced with '{replacement['track_name'][0]}'"
                                        )
                                    else:
                                        st.toast(
                                            f"Excluded '{selected_track['track_name']}'"
                                        )
                                else:
                                    st.toast(
                                        f"Excluded '{selected_track['track_name']}'"
                                    )

                                st.rerun()

                            except Exception as e:
                                st.error(f"Error excluding track: {e}")

                    # Restore excluded tracks
                    if n_excluded_tracks > 0:
                        st.caption("Previously excluded:")
                        display_excluded_tracks = excluded_tracks_collected.select(
                            [
                                pl.col("track_name").alias("Track"),
                                pl.col("artist_name").alias("Artist"),
                            ]
                        )
                        st.dataframe(
                            display_excluded_tracks,
                            use_container_width=True,
                            hide_index=True,
                        )

                        track_to_revert_options = {
                            f"{row['track_name']} - {row['artist_name']}": row
                            for row in excluded_tracks_collected.to_dicts()
                        }

                        selected_track_to_revert = st.selectbox(
                            "Restore a track",
                            options=list(track_to_revert_options.keys()),
                            key="track_to_revert",
                        )

                        if st.button(
                            "✅ Restore", type="secondary", key="restore_track_btn"
                        ):
                            try:
                                track_info = track_to_revert_options[
                                    selected_track_to_revert
                                ]
                                remove_excluded_track(
                                    username=username,
                                    track_id=track_info["track_id"],
                                    track_name=track_info["track_name"],
                                    artist_name=track_info["artist_name"],
                                )
                                st.toast(f"Restored '{track_info['track_name']}'")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error restoring track: {e}")

                with tab_artists:
                    # Get unique artists from current recommendations
                    current_artists = (
                        recommendations.select("artist_name")
                        .unique()
                        .sort("artist_name")
                        .to_series()
                        .to_list()
                    )

                    if len(current_artists) > 0:
                        selected_artist = st.selectbox(
                            "Block an artist (removes all their tracks)",
                            options=current_artists,
                            key="artist_to_block",
                        )

                        if st.button(
                            "🚫 Block Artist", type="secondary", key="block_artist_btn"
                        ):
                            try:
                                write_excluded_artist(
                                    username=st.session_state.username,
                                    artist_name=selected_artist,
                                )

                                if (
                                    "excluded_artists_in_session"
                                    not in st.session_state
                                ):
                                    st.session_state.excluded_artists_in_session = set()
                                st.session_state.excluded_artists_in_session.add(
                                    selected_artist
                                )

                                tracks_removed = len(
                                    recommendations.filter(
                                        pl.col("artist_name") == selected_artist
                                    )
                                )

                                st.session_state.recommendations = (
                                    recommendations.filter(
                                        pl.col("artist_name") != selected_artist
                                    )
                                )

                                # Try to replace removed tracks
                                if (
                                    "candidate_pool" in st.session_state
                                    and tracks_removed > 0
                                ):
                                    pool = st.session_state.candidate_pool
                                    updated_recommendations = (
                                        st.session_state.recommendations
                                    )
                                    displayed_tracks = set(
                                        zip(
                                            updated_recommendations.select("track_name")
                                            .to_series()
                                            .to_list(),
                                            updated_recommendations.select(
                                                "artist_name"
                                            )
                                            .to_series()
                                            .to_list(),
                                        )
                                    )
                                    if "excluded_in_session" in st.session_state:
                                        displayed_tracks.update(
                                            st.session_state.excluded_in_session
                                        )

                                    excluded_artists_set = (
                                        st.session_state.excluded_artists_in_session
                                    )

                                    replacements = []
                                    for row in pool.iter_rows(named=True):
                                        if len(replacements) >= tracks_removed:
                                            break
                                        track_key = (
                                            row["track_name"],
                                            row["artist_name"],
                                        )
                                        if (
                                            track_key not in displayed_tracks
                                            and row["artist_name"]
                                            not in excluded_artists_set
                                        ):
                                            replacements.append(row)
                                            displayed_tracks.add(track_key)

                                    if replacements:
                                        replacement_df = pl.DataFrame(
                                            replacements, schema=pool.schema
                                        ).select(
                                            st.session_state.recommendations.columns
                                        )
                                        st.session_state.recommendations = pl.concat(
                                            [
                                                st.session_state.recommendations,
                                                replacement_df,
                                            ]
                                        )
                                        st.toast(
                                            f"Blocked '{selected_artist}' and replaced {len(replacements)} tracks"
                                        )
                                    else:
                                        st.toast(
                                            f"Blocked '{selected_artist}' ({tracks_removed} tracks removed)"
                                        )
                                else:
                                    st.toast(f"Blocked '{selected_artist}'")

                                st.rerun()

                            except Exception as e:
                                st.error(f"Error blocking artist: {e}")

                    # Restore blocked artists
                    if n_excluded_artists > 0:
                        st.caption("Currently blocked:")
                        display_excluded_artists = excluded_artists_collected.select(
                            pl.col("artist_name").alias("Artist")
                        )
                        st.dataframe(
                            display_excluded_artists,
                            use_container_width=True,
                            hide_index=True,
                        )

                        artist_to_revert_options = (
                            excluded_artists_collected.select("artist_name")
                            .to_series()
                            .to_list()
                        )

                        selected_artist_to_revert = st.selectbox(
                            "Restore an artist",
                            options=artist_to_revert_options,
                            key="artist_to_revert",
                        )

                        if st.button(
                            "✅ Restore", type="secondary", key="restore_artist_btn"
                        ):
                            try:
                                remove_excluded_artist(
                                    username=username,
                                    artist_name=selected_artist_to_revert,
                                )
                                st.toast(f"Restored '{selected_artist_to_revert}'")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error restoring artist: {e}")

    with st.spinner("Setting up playlist export..."):
        # Playlist creation section
        if (
            "recommendations" in st.session_state
            and st.session_state.recommendations is not None
        ):
            st.divider()
            st.header("� Export to Playlist")

            # Tabs for different services
            youtube_tab, spotify_tab = st.tabs(["🎬 YouTube Music", "🎧 Spotify"])

            # Common playlist settings
            col1, col2 = st.columns([3, 1])
            with col1:
                playlist_name = st.text_input(
                    "Playlist Name",
                    value=f"{st.session_state.get('username', 'User')} - KainosFM",
                    key="playlist_name_input",
                    label_visibility="collapsed",
                    placeholder="Playlist name...",
                )
            with col2:
                privacy = st.selectbox(
                    "Privacy",
                    options=["public", "private", "unlisted"],
                    index=0,
                    key="privacy_selector",
                    label_visibility="collapsed",
                )

            # YouTube Music tab
            with youtube_tab:
                needs_yt_auth = YouTubePlaylistGenerator.needs_authentication()

                if needs_yt_auth:
                    with st.container(border=True):
                        st.warning("⚠️ YouTube authentication required")
                        st.caption(
                            "Configure credentials in .env, then run OAuth flow below."
                        )

                        youtube_creds = load_youtube_creds()
                        if youtube_creds and youtube_creds.has_client_creds():
                            if st.button(
                                "🔐 Authenticate with YouTube", type="primary"
                            ):
                                device_info, _ = run_youtube_oauth(
                                    youtube_creds.client_id, youtube_creds.client_secret
                                )
                                if device_info:
                                    st.session_state["youtube_device_info"] = (
                                        device_info
                                    )
                                    st.session_state["youtube_client_id"] = (
                                        youtube_creds.client_id
                                    )
                                    st.session_state["youtube_client_secret"] = (
                                        youtube_creds.client_secret
                                    )
                                    st.rerun()
                                else:
                                    st.error("Failed to start OAuth flow")
                        else:
                            st.info(
                                "Set YOUTUBE_CLIENT_ID and YOUTUBE_CLIENT_SECRET in .env"
                            )

                        # Show pending OAuth flow
                        if "youtube_device_info" in st.session_state:
                            device_info = st.session_state["youtube_device_info"]
                            st.markdown("---")
                            st.markdown(
                                f"**1.** Go to: `{device_info['verification_url']}`"
                            )
                            st.markdown(
                                f"**2.** Enter code: `{device_info['user_code']}`"
                            )
                            st.markdown("**3.** Click below after authorizing:")

                            if st.button("✅ I've authorized", key="poll_youtube"):
                                try:
                                    token_info = poll_device_token(
                                        st.session_state["youtube_client_id"],
                                        st.session_state["youtube_client_secret"],
                                        device_info["device_code"],
                                    )
                                    if token_info:
                                        st.success("✅ Authenticated! Add to .env:")
                                        st.code(
                                            f"YOUTUBE_ACCESS_TOKEN={token_info.get('access_token', '')}\n"
                                            f"YOUTUBE_REFRESH_TOKEN={token_info.get('refresh_token', '')}",
                                            language="bash",
                                        )
                                        del st.session_state["youtube_device_info"]
                                        del st.session_state["youtube_client_id"]
                                        del st.session_state["youtube_client_secret"]
                                    else:
                                        st.warning(
                                            "Still pending. Complete the flow and try again."
                                        )
                                except Exception as e:
                                    st.error(f"OAuth failed: {e}")
                                    del st.session_state["youtube_device_info"]

                if st.button(
                    "🎬 Create YouTube Playlist",
                    type="primary" if not needs_yt_auth else "secondary",
                    disabled=needs_yt_auth,
                    use_container_width=True,
                    key="create_youtube_playlist",
                ):
                    _create_youtube_playlist(playlist_name, privacy)

            # Spotify tab
            with spotify_tab:
                needs_spotify_auth = SpotifyPlaylistGenerator.needs_authentication()

                if needs_spotify_auth:
                    with st.container(border=True):
                        st.warning("⚠️ Spotify authentication required")
                        st.caption(
                            "Configure credentials in .env, then run OAuth flow below."
                        )

                        spotify_creds = load_spotify_creds()
                        if spotify_creds and spotify_creds.has_client_creds():
                            # Check if we already started the OAuth flow
                            if "spotify_auth_url" not in st.session_state:
                                if st.button(
                                    "🔐 Authenticate with Spotify", type="primary"
                                ):
                                    auth_url, state = run_spotify_oauth(
                                        spotify_creds.client_id,
                                        spotify_creds.client_secret,
                                    )
                                    st.session_state["spotify_auth_url"] = auth_url
                                    st.session_state["spotify_auth_state"] = state
                                    st.session_state["spotify_client_id"] = (
                                        spotify_creds.client_id
                                    )
                                    st.session_state["spotify_client_secret"] = (
                                        spotify_creds.client_secret
                                    )
                                    st.rerun()

                            # Show OAuth instructions if flow started
                            if "spotify_auth_url" in st.session_state:
                                auth_url = st.session_state["spotify_auth_url"]
                                st.markdown("---")
                                st.info(
                                    "**Important:** Add `http://localhost:8501/` as a Redirect URI in your "
                                    "[Spotify Developer Dashboard](https://developer.spotify.com/dashboard) app settings."
                                )
                                st.markdown(
                                    "**Step 1:** Click the button below to authorize:"
                                )
                                st.link_button(
                                    "🔗 Open Spotify Authorization",
                                    auth_url,
                                    type="primary",
                                )
                                st.markdown(
                                    "**Step 2:** After authorizing, you'll be redirected back here."
                                )
                                st.markdown(
                                    "**Step 3:** Copy the `code` from the URL bar and paste below:"
                                )
                                st.caption(
                                    "The URL will look like: `http://localhost:8501/?code=AQB...xyz&state=...`"
                                )
                                st.caption(
                                    "Copy **only** the part after `code=` and before `&state=`"
                                )

                                auth_code = st.text_input(
                                    "Authorization Code",
                                    placeholder="Paste the code from the redirect URL...",
                                    key="spotify_auth_code_input",
                                )

                                col1, col2 = st.columns(2)
                                with col1:
                                    if st.button(
                                        "✅ Complete Authentication",
                                        key="complete_spotify_auth",
                                        type="primary",
                                    ):
                                        if auth_code:
                                            token_info = exchange_code_for_token(
                                                st.session_state["spotify_client_id"],
                                                st.session_state[
                                                    "spotify_client_secret"
                                                ],
                                                auth_code,
                                            )
                                            if token_info:
                                                st.success(
                                                    "✅ Authenticated! Add to .env:"
                                                )
                                                st.code(
                                                    f"SPOTIFY_ACCESS_TOKEN={token_info.get('access_token', '')}\n"
                                                    f"SPOTIFY_REFRESH_TOKEN={token_info.get('refresh_token', '')}",
                                                    language="bash",
                                                )
                                                # Clean up session state
                                                for key in [
                                                    "spotify_auth_url",
                                                    "spotify_auth_state",
                                                    "spotify_client_id",
                                                    "spotify_client_secret",
                                                ]:
                                                    st.session_state.pop(key, None)
                                            else:
                                                st.error(
                                                    "Failed to exchange code for token. The code may have expired - try again."
                                                )
                                        else:
                                            st.warning(
                                                "Please paste the authorization code"
                                            )
                                with col2:
                                    if st.button(
                                        "❌ Cancel", key="cancel_spotify_auth"
                                    ):
                                        for key in [
                                            "spotify_auth_url",
                                            "spotify_auth_state",
                                            "spotify_client_id",
                                            "spotify_client_secret",
                                        ]:
                                            st.session_state.pop(key, None)
                                        st.rerun()
                        else:
                            st.info(
                                "Set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET in .env"
                            )

                if st.button(
                    "🎧 Create Spotify Playlist",
                    type="primary" if not needs_spotify_auth else "secondary",
                    disabled=needs_spotify_auth,
                    use_container_width=True,
                    key="create_spotify_playlist",
                ):
                    _create_spotify_playlist(playlist_name, privacy == "public")


def _create_youtube_playlist(playlist_name: str, privacy: str) -> None:
    """Create a YouTube playlist from recommendations."""
    try:
        recommendations = st.session_state.recommendations
        logging.info(f"Starting YouTube playlist creation: {playlist_name}")
        logging.info(f"Number of tracks in recommendations: {len(recommendations)}")
        logging.info(f"Columns in recommendations: {recommendations.columns}")
        logging.info(
            f"Tracks with youtube_url: {recommendations['youtube_url'].is_not_null().sum()}"
        )

        playlist_generator = YouTubePlaylistGenerator()

        if not playlist_generator.authenticate():
            st.error(
                "❌ YouTube authentication failed.\n\n"
                "Your credentials may have expired. Please re-authenticate using the button above."
            )
            st.stop()

        progress_bar = st.progress(0)
        status_text = st.empty()

        settings = st.session_state.playlist_settings
        result = playlist_generator.create_playlist_from_tracks(
            tracks_df=st.session_state.recommendations,
            playlist_title=playlist_name,
            playlist_description=f"Generated by Music Recommendation System\nDiscovery weight: {settings['discovery_weight']}\nSystems: {'Tags' if settings['use_tags'] else ''} {'Artists' if settings['use_artists'] else ''} {'Deep Cuts' if settings['use_deep_cuts'] else ''}",
            privacy_status=privacy,
            progress_bar=progress_bar,
            status_text=status_text,
        )

        progress_bar.empty()
        status_text.empty()

        if result:
            total_tracks = len(st.session_state.recommendations)
            success_rate = (
                (result["tracks_added"] / total_tracks * 100) if total_tracks > 0 else 0
            )

            if result.get("quota_exceeded"):
                st.error(
                    f"⚠️ YouTube API quota exceeded! Added {result['tracks_added']}/{total_tracks} tracks before hitting the limit.\n\n"
                    f"**What happened:** YouTube limits the number of API requests per day.\n\n"
                    f"**Solution:** Quota resets at midnight Pacific Time. Try again tomorrow, or reduce the number of recommendations."
                )
            elif success_rate > 90:
                st.success(
                    f"✅ Playlist ready! Added {result['tracks_added']}/{total_tracks} tracks."
                )
            elif success_rate > 50:
                st.warning(
                    f"⚠️ Playlist created with issues. Added {result['tracks_added']}/{total_tracks} tracks."
                )
            else:
                st.error(
                    f"❌ Playlist created but most tracks failed. Only {result['tracks_added']}/{total_tracks} tracks added."
                )

            st.markdown(
                f"**Open in YouTube Music:** [{playlist_name}]({result['playlist_url']})"
            )

            if result["tracks_not_found"]:
                with st.expander(
                    f"⚠️ {len(result['tracks_not_found'])} tracks had issues"
                ):
                    st.info("Check the terminal/logs for detailed error messages.")
                    for track in result["tracks_not_found"]:
                        st.text(f"• {track}")
        else:
            st.error(
                "❌ Failed to create playlist. Possible causes:\n\n"
                "• YouTube authentication failed\n"
                "• API quota exceeded\n"
                "• Network issues\n\n"
                "Check the terminal logs for detailed error messages."
            )
            st.info(
                "💡 Check your terminal where Streamlit is running for detailed error logs."
            )

    except Exception as e:
        st.error(f"❌ Error creating playlist: {type(e).__name__}")

        error_str = str(e)
        if "invalid_grant" in error_str:
            st.warning(
                "**Authentication expired!**\n\n"
                "Your refresh token is no longer valid. "
                "Please re-authenticate using the 'Authenticate with YouTube' button above."
            )
        elif "quotaExceeded" in error_str or "403" in error_str:
            st.warning(
                "**YouTube API quota exceeded!**\n\n"
                "Quota resets at midnight Pacific Time. "
                "Try again tomorrow with fewer tracks."
            )
        elif "401" in error_str or "Unauthorized" in error_str:
            st.warning(
                "**Authentication issue.**\n\n"
                "Your YouTube credentials may have expired. "
                "Please re-authenticate using the button above."
            )

        with st.expander("Show technical details"):
            st.code(traceback.format_exc())


def _create_spotify_playlist(playlist_name: str, public: bool) -> None:
    """Create a Spotify playlist from recommendations."""
    try:
        recommendations = st.session_state.recommendations
        logging.info(f"Starting Spotify playlist creation: {playlist_name}")
        logging.info(f"Number of tracks in recommendations: {len(recommendations)}")
        logging.info(
            f"Tracks with spotify_url: {recommendations['spotify_url'].is_not_null().sum()}"
        )

        playlist_generator = SpotifyPlaylistGenerator()

        if not playlist_generator.authenticate():
            st.error(
                "❌ Spotify authentication failed.\n\n"
                "Your credentials may have expired. Please re-authenticate using the button above."
            )
            st.stop()

        progress_bar = st.progress(0)
        status_text = st.empty()

        settings = st.session_state.playlist_settings
        result = playlist_generator.create_playlist_from_tracks(
            tracks_df=st.session_state.recommendations,
            playlist_title=playlist_name,
            playlist_description=f"Generated by Music Recommendation System | Discovery weight: {settings['discovery_weight']} | Systems: {'Tags' if settings['use_tags'] else ''} {'Artists' if settings['use_artists'] else ''} {'Deep Cuts' if settings['use_deep_cuts'] else ''}",
            public=public,
            progress_bar=progress_bar,
            status_text=status_text,
        )

        progress_bar.empty()
        status_text.empty()

        if result:
            total_tracks = len(st.session_state.recommendations)
            success_rate = (
                (result["tracks_added"] / total_tracks * 100) if total_tracks > 0 else 0
            )

            if success_rate > 90:
                st.success(
                    f"✅ Playlist ready! Added {result['tracks_added']}/{total_tracks} tracks."
                )
            elif success_rate > 50:
                st.warning(
                    f"⚠️ Playlist created with issues. Added {result['tracks_added']}/{total_tracks} tracks."
                )
            else:
                st.error(
                    f"❌ Playlist created but most tracks failed. Only {result['tracks_added']}/{total_tracks} tracks added."
                )

            st.markdown(
                f"**Open in Spotify:** [{playlist_name}]({result['playlist_url']})"
            )

            if result["tracks_not_found"]:
                with st.expander(
                    f"⚠️ {len(result['tracks_not_found'])} tracks not found"
                ):
                    st.info("These tracks couldn't be found on Spotify.")
                    for track in result["tracks_not_found"]:
                        st.text(f"• {track}")
        else:
            st.error(
                "❌ Failed to create playlist. Possible causes:\n\n"
                "• Spotify authentication failed\n"
                "• Rate limit exceeded\n"
                "• Network issues\n\n"
                "Check the terminal logs for detailed error messages."
            )
            st.info(
                "💡 Check your terminal where Streamlit is running for detailed error logs."
            )

    except Exception as e:
        st.error(f"❌ Error creating playlist: {type(e).__name__}")

        error_str = str(e)
        if "invalid_grant" in error_str or "401" in error_str:
            st.warning(
                "**Authentication expired!**\n\n"
                "Your refresh token is no longer valid. "
                "Please re-authenticate using the 'Authenticate with Spotify' button above."
            )
        elif "rate" in error_str.lower() or "429" in error_str:
            st.warning(
                "**Spotify rate limit exceeded!**\n\n"
                "Please wait a few minutes and try again."
            )

        with st.expander("Show technical details"):
            st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
