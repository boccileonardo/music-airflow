import logging
import polars as pl
import streamlit as st
from music_airflow.utils.polars_io_manager import PolarsDeltaIOManager
from music_airflow.utils.constants import LAST_FM_USERNAMES
from music_airflow.app.youtube_playlist import YouTubePlaylistGenerator
from music_airflow.app.excluded_tracks import (
    write_excluded_track,
    read_excluded_tracks,
    write_excluded_artist,
    read_excluded_artists,
)

# Configure logging to show in Streamlit
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


@st.cache_data
def load_track_candidates(username: str) -> pl.LazyFrame:
    """Load track candidates from gold table - presentation-ready data."""
    gold_io = PolarsDeltaIOManager("gold")
    return gold_io.read_delta("track_candidates").filter(pl.col("username") == username)


@st.cache_data
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


@st.cache_data
def load_top_artists(username: str, limit: int = 10) -> pl.DataFrame:
    """Load top artists by play count."""
    gold_io = PolarsDeltaIOManager("gold")

    try:
        artist_plays = (
            gold_io.read_delta("artist_play_count")
            .filter(
                (pl.col("username") == username) & pl.col("artist_name").is_not_null()
            )
            .collect()
        )
        return artist_plays.sort("play_count", descending=True).head(limit)
    except Exception:
        return pl.DataFrame(schema={"artist_name": pl.String, "play_count": pl.Int64})


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
    st.title("🎵 Music Recommendation System")

    username = st.selectbox("username", options=LAST_FM_USERNAMES)

    # Display user statistics
    with st.spinner("Loading statistics..."):
        stats = load_user_statistics(username)

    # Statistics in columns
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Plays", f"{stats['total_plays']:,}")
    with col2:
        st.metric("Unique Tracks", f"{stats['total_tracks_played']:,}")
    with col3:
        st.metric("Unique Artists", f"{stats['total_artists_played']:,}")

    # Top Artists section
    st.header("🎤 Top Artists")
    with st.spinner("Loading top artists..."):
        top_artists = load_top_artists(username, limit=10)

    if len(top_artists) > 0:
        # Display as a table
        display_df = (
            top_artists.with_columns(pl.col("play_count").alias("Plays"))
            .select(["artist_name", "Plays"])
            .rename({"artist_name": "Artist"})
        )
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.info("No artist play data available yet.")

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
                    .with_columns(
                        artist_rank=pl.int_range(pl.len()).over("artist_name")
                    )
                    .filter(pl.col("artist_rank") < max_songs_per_artist)
                    .drop("artist_rank")  # Remove the temporary ranking column
                )

                # Take exactly the requested number of recommendations
                # Gold table is already deduplicated by youtube_url, so we can trust it
                recommendations = (
                    candidates.sort("weighted_score", descending=True)
                    .limit(n_recommendations)
                    .collect()
                )

                if len(recommendations) == 0:
                    st.warning(
                        "No recommendations found. Try selecting at least one system."
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
                    # Gold table is already deduplicated, so just fetch more tracks
                    # Note: candidates still has artist_rank column at this point, need to drop it
                    st.session_state.candidate_pool = (
                        candidates.select(
                            [
                                "track_id",
                                "track_name",
                                "artist_name",
                                "album_name",
                                "score",
                                "similar_artist",
                                "similar_tag",
                                "deep_cut_same_artist",
                                "old_favorite",
                                "youtube_url",
                                "spotify_url",
                                "tags",
                                "duration_ms",
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
                    pl.col("old_favorite").alias("Old Favorite"),
                    pl.col("similar_artist").alias("From Similar Artist"),
                    pl.col("similar_tag").alias("From Similar Tag"),
                    pl.col("deep_cut_same_artist").alias("Deep Cut from Loved Artist"),
                ]
            )
            .sort("Score", descending=True)
        )

        # Display as dataframe
        st.dataframe(
            display_recommendations,
        )

        # Track exclusion UI
        st.subheader("Remove Unwanted Tracks")
        st.write(
            "Don't like a recommendation? Remove it and we'll replace it with a new one."
        )

        # Get current track IDs for dropdown
        current_tracks = recommendations.select(
            ["track_id", "track_name", "artist_name"]
        ).to_dicts()
        track_options = {
            f"{t['track_name']} - {t['artist_name']}": t for t in current_tracks
        }

        if len(track_options) > 0:
            selected_track_display = st.selectbox(
                "Select track to remove",
                options=list(track_options.keys()),
                key="track_to_remove",
            )

            if st.button("🗑️ Remove and Replace Track", type="secondary"):
                selected_track = track_options[selected_track_display]
                track_id = selected_track["track_id"]

                # Write to excluded recommendations table
                try:
                    write_excluded_track(
                        username=st.session_state.username,
                        track_id=track_id,
                        track_name=selected_track["track_name"],
                        artist_name=selected_track["artist_name"],
                    )

                    # Track exclusion in session using track_name + artist_name (stable keys)
                    excluded_key = (
                        selected_track["track_name"],
                        selected_track["artist_name"],
                    )
                    if "excluded_in_session" not in st.session_state:
                        st.session_state.excluded_in_session = set()
                    st.session_state.excluded_in_session.add(excluded_key)

                    # Remove from current recommendations
                    st.session_state.recommendations = recommendations.filter(
                        (pl.col("track_name") != selected_track["track_name"])
                        | (pl.col("artist_name") != selected_track["artist_name"])
                    )

                    # Find replacement from candidate pool
                    if "candidate_pool" in st.session_state:
                        # Get already displayed tracks (track_name, artist_name) pairs
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
                        # Add newly excluded tracks
                        displayed_tracks.update(st.session_state.excluded_in_session)

                        # Find first track not in displayed or excluded
                        pool = st.session_state.candidate_pool
                        replacement = None
                        for row in pool.iter_rows(named=True):
                            track_key = (row["track_name"], row["artist_name"])
                            if track_key not in displayed_tracks:
                                replacement = pool.filter(
                                    (pl.col("track_name") == row["track_name"])
                                    & (pl.col("artist_name") == row["artist_name"])
                                ).limit(1)
                                break

                        if replacement is not None and len(replacement) > 0:
                            # Add replacement to recommendations
                            st.session_state.recommendations = pl.concat(
                                [
                                    st.session_state.recommendations,
                                    replacement,
                                ]
                            )
                            st.success(
                                f"✅ Removed '{selected_track['track_name']}' and replaced with '{replacement['track_name'][0]}'"
                            )
                        else:
                            st.warning(
                                f"⚠️ Removed '{selected_track['track_name']}' but no replacement available from current pool."
                            )
                    else:
                        st.success(
                            f"✅ Removed '{selected_track['track_name']}' from recommendations."
                        )

                    # Force rerun to update display
                    st.rerun()

                except Exception as e:
                    st.error(f"Error excluding track: {e}")
        else:
            st.info("No tracks to remove.")

        # Artist exclusion UI
        st.divider()
        st.subheader("Block Artists")
        st.write("Don't want to see any tracks from an artist? Block them entirely.")

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
                "Select artist to block",
                options=current_artists,
                key="artist_to_block",
            )

            if st.button("🚫 Block Artist", type="secondary"):
                try:
                    write_excluded_artist(
                        username=st.session_state.username,
                        artist_name=selected_artist,
                    )

                    # Track artist exclusion in session
                    if "excluded_artists_in_session" not in st.session_state:
                        st.session_state.excluded_artists_in_session = set()
                    st.session_state.excluded_artists_in_session.add(selected_artist)

                    # Remove all tracks from this artist
                    tracks_removed = len(
                        recommendations.filter(pl.col("artist_name") == selected_artist)
                    )

                    st.session_state.recommendations = recommendations.filter(
                        pl.col("artist_name") != selected_artist
                    )

                    # Try to replace removed tracks
                    if "candidate_pool" in st.session_state and tracks_removed > 0:
                        pool = st.session_state.candidate_pool

                        # Get already displayed and excluded info from the UPDATED recommendations
                        updated_recommendations = st.session_state.recommendations
                        displayed_tracks = set(
                            zip(
                                updated_recommendations.select("track_name")
                                .to_series()
                                .to_list(),
                                updated_recommendations.select("artist_name")
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

                        # Find replacement tracks
                        replacements = []
                        for row in pool.iter_rows(named=True):
                            if len(replacements) >= tracks_removed:
                                break
                            track_key = (row["track_name"], row["artist_name"])
                            if (
                                track_key not in displayed_tracks
                                and row["artist_name"] not in excluded_artists_set
                            ):
                                replacements.append(row)
                                displayed_tracks.add(track_key)

                        if replacements:
                            # Create DataFrame from replacements using the same schema as recommendations
                            replacement_df = pl.DataFrame(
                                replacements,
                                schema=pool.schema,
                            )
                            st.session_state.recommendations = pl.concat(
                                [
                                    st.session_state.recommendations,
                                    replacement_df,
                                ]
                            )
                            st.success(
                                f"✅ Blocked '{selected_artist}' and replaced {len(replacements)}/{tracks_removed} tracks"
                            )
                        else:
                            st.warning(
                                f"⚠️ Blocked '{selected_artist}' ({tracks_removed} tracks removed) but no replacements available."
                            )
                    else:
                        st.success(
                            f"✅ Blocked '{selected_artist}' from future recommendations."
                        )

                    # Force rerun to update display
                    st.rerun()

                except Exception as e:
                    st.error(f"Error blocking artist: {e}")
        else:
            st.info("No artists to block.")

    # Playlist creation section
    if (
        "recommendations" in st.session_state
        and st.session_state.recommendations is not None
    ):
        st.divider()
        st.subheader("🎵 Create Playlist")

        col1, col2 = st.columns([3, 1])
        with col1:
            playlist_name = st.text_input(
                "Playlist Name",
                value=f"{username} - KainosFM",
                key="playlist_name_input",
            )

        with col2:
            privacy = st.selectbox(
                "Privacy",
                options=["unlisted", "private", "public"],
                index=0,
                key="privacy_selector",
            )

        if st.button("🎬 Create YouTube Music Playlist", type="secondary"):
            try:
                # Log playlist creation attempt with column info
                recommendations = st.session_state.recommendations
                logging.info(f"Starting playlist creation: {playlist_name}")
                logging.info(
                    f"Number of tracks in recommendations: {len(recommendations)}"
                )
                logging.info(f"Columns in recommendations: {recommendations.columns}")
                logging.info(
                    f"Tracks with youtube_url: {recommendations['youtube_url'].is_not_null().sum()}"
                )

                generator = YouTubePlaylistGenerator()

                # Authenticate silently (uses stored credentials)
                if not generator.authenticate():
                    st.error(
                        "YouTube authentication not configured. "
                        "Please contact the app administrator."
                    )
                    st.stop()

                # Create progress bar and status text
                progress_bar = st.progress(0)
                status_text = st.empty()

                settings = st.session_state.playlist_settings
                result = generator.create_playlist_from_tracks(
                    tracks_df=st.session_state.recommendations,
                    playlist_title=playlist_name,
                    playlist_description=f"Generated by Music Recommendation System\nDiscovery weight: {settings['discovery_weight']}\nSystems: {'Tags' if settings['use_tags'] else ''} {'Artists' if settings['use_artists'] else ''} {'Deep Cuts' if settings['use_deep_cuts'] else ''}",
                    privacy_status=privacy,
                    progress_bar=progress_bar,
                    status_text=status_text,
                )

                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()

                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()

                if result:
                    total_tracks = len(st.session_state.recommendations)
                    success_rate = (
                        (result["tracks_added"] / total_tracks * 100)
                        if total_tracks > 0
                        else 0
                    )

                    # Check if quota was exceeded
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

                    # Show clickable link
                    st.markdown(
                        f"**Open in YouTube Music:** [{playlist_name}]({result['playlist_url']})"
                    )

                    # Show tracks not found
                    if result["tracks_not_found"]:
                        with st.expander(
                            f"⚠️ {len(result['tracks_not_found'])} tracks had issues"
                        ):
                            st.info(
                                "Check the terminal/logs for detailed error messages."
                            )
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
                    # Show a sample of logs if possible
                    st.info(
                        "💡 Check your terminal where Streamlit is running for detailed error logs."
                    )

            except Exception as e:
                st.error(f"❌ Error creating playlist: {type(e).__name__}")

                # Show user-friendly error message based on error type
                error_str = str(e)
                if "quotaExceeded" in error_str or "403" in error_str:
                    st.warning(
                        "**YouTube API quota exceeded!**\n\n"
                        "Quota resets at midnight Pacific Time. "
                        "Try again tomorrow with fewer tracks."
                    )
                elif "401" in error_str or "Unauthorized" in error_str:
                    st.warning(
                        "**Authentication issue.**\n\n"
                        "Your YouTube credentials may have expired. "
                        "Contact the administrator."
                    )

                # Show full traceback in expander
                with st.expander("Show technical details"):
                    import traceback

                    st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
