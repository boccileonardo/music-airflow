"""
Exclusion management UI components for the Streamlit app.

Handles the UI and logic for excluding/restoring tracks and artists
from recommendations.
"""

import datetime as dt

import polars as pl
import streamlit as st

from music_airflow.app.excluded_tracks import (
    read_excluded_artists,
    read_excluded_tracks,
    remove_excluded_artist,
    remove_excluded_track,
    write_excluded_artist,
    write_excluded_track,
)


def get_cached_excluded_tracks(username: str) -> pl.DataFrame:
    """Get excluded tracks from session_state cache or load from storage.

    Uses local-first approach: changes are reflected instantly in session_state
    and persisted to storage in background.
    """
    cache_key = f"excluded_tracks_{username}"
    if cache_key not in st.session_state:
        try:
            st.session_state[cache_key] = read_excluded_tracks(username).collect()
        except Exception:
            st.session_state[cache_key] = pl.DataFrame(
                schema={
                    "username": pl.String,
                    "track_id": pl.String,
                    "track_name": pl.String,
                    "artist_name": pl.String,
                    "excluded_at": pl.Datetime(time_zone="UTC"),
                }
            )
    return st.session_state[cache_key]


def get_cached_excluded_artists(username: str) -> pl.DataFrame:
    """Get excluded artists from session_state cache or load from storage."""
    cache_key = f"excluded_artists_{username}"
    if cache_key not in st.session_state:
        try:
            st.session_state[cache_key] = read_excluded_artists(username).collect()
        except Exception:
            st.session_state[cache_key] = pl.DataFrame(
                schema={
                    "username": pl.String,
                    "artist_name": pl.String,
                    "excluded_at": pl.Datetime(time_zone="UTC"),
                }
            )
    return st.session_state[cache_key]


def add_excluded_track_local(
    username: str, track_id: str, track_name: str, artist_name: str
) -> None:
    """Add an excluded track to local cache and persist to storage."""
    cache_key = f"excluded_tracks_{username}"
    current = get_cached_excluded_tracks(username)

    new_row = pl.DataFrame(
        {
            "username": [username],
            "track_id": [track_id],
            "track_name": [track_name],
            "artist_name": [artist_name],
            "excluded_at": [dt.datetime.now(tz=dt.timezone.utc)],
        }
    )
    st.session_state[cache_key] = pl.concat([current, new_row]).unique(
        subset=["username", "track_id"]
    )

    write_excluded_track(username, track_id, track_name, artist_name)


def remove_excluded_track_local(
    username: str, track_id: str, track_name: str, artist_name: str
) -> None:
    """Remove an excluded track from local cache and storage."""
    cache_key = f"excluded_tracks_{username}"
    current = get_cached_excluded_tracks(username)

    st.session_state[cache_key] = current.filter(
        ~((pl.col("username") == username) & (pl.col("track_id") == track_id))
    )

    remove_excluded_track(username, track_id, track_name, artist_name)


def add_excluded_artist_local(username: str, artist_name: str) -> None:
    """Add an excluded artist to local cache and persist to storage."""
    cache_key = f"excluded_artists_{username}"
    current = get_cached_excluded_artists(username)

    new_row = pl.DataFrame(
        {
            "username": [username],
            "artist_name": [artist_name],
            "excluded_at": [dt.datetime.now(tz=dt.timezone.utc)],
        }
    )
    st.session_state[cache_key] = pl.concat([current, new_row]).unique(
        subset=["username", "artist_name"]
    )

    write_excluded_artist(username, artist_name)


def remove_excluded_artist_local(username: str, artist_name: str) -> None:
    """Remove an excluded artist from local cache and storage."""
    cache_key = f"excluded_artists_{username}"
    current = get_cached_excluded_artists(username)

    st.session_state[cache_key] = current.filter(
        ~((pl.col("username") == username) & (pl.col("artist_name") == artist_name))
    )

    remove_excluded_artist(username, artist_name)


def _find_replacement_track(
    recommendations: pl.DataFrame,
    excluded_in_session: set,
) -> pl.DataFrame | None:
    """Find a replacement track from the candidate pool."""
    if "candidate_pool" not in st.session_state:
        return None

    displayed_tracks = set(
        zip(
            recommendations.select("track_name").to_series().to_list(),
            recommendations.select("artist_name").to_series().to_list(),
        )
    )
    displayed_tracks.update(excluded_in_session)

    pool = st.session_state.candidate_pool
    for row in pool.iter_rows(named=True):
        track_key = (row["track_name"], row["artist_name"])
        if track_key not in displayed_tracks:
            return pool.filter(
                (pl.col("track_name") == row["track_name"])
                & (pl.col("artist_name") == row["artist_name"])
            ).limit(1)

    return None


def _find_replacement_tracks_for_artist(
    recommendations: pl.DataFrame,
    excluded_in_session: set,
    excluded_artists_in_session: set,
    count: int,
) -> list[dict]:
    """Find replacement tracks when blocking an artist."""
    if "candidate_pool" not in st.session_state:
        return []

    displayed_tracks = set(
        zip(
            recommendations.select("track_name").to_series().to_list(),
            recommendations.select("artist_name").to_series().to_list(),
        )
    )
    displayed_tracks.update(excluded_in_session)

    pool = st.session_state.candidate_pool
    replacements = []
    for row in pool.iter_rows(named=True):
        if len(replacements) >= count:
            break
        track_key = (row["track_name"], row["artist_name"])
        if (
            track_key not in displayed_tracks
            and row["artist_name"] not in excluded_artists_in_session
        ):
            replacements.append(row)
            displayed_tracks.add(track_key)

    return replacements


def render_exclusions_expander(username: str, recommendations: pl.DataFrame) -> None:
    """Render the exclusions management expander."""
    with st.expander("🚫 Manage Exclusions", expanded=False):
        excluded_tracks_collected = get_cached_excluded_tracks(username)
        excluded_artists_collected = get_cached_excluded_artists(username)
        n_excluded_tracks = len(excluded_tracks_collected)
        n_excluded_artists = len(excluded_artists_collected)

        tab_tracks, tab_artists = st.tabs(
            [
                f"🎵 Tracks ({n_excluded_tracks})",
                f"🎤 Artists ({n_excluded_artists})",
            ]
        )

        with tab_tracks:
            _render_track_exclusions(
                username, recommendations, excluded_tracks_collected, n_excluded_tracks
            )

        with tab_artists:
            _render_artist_exclusions(
                username,
                recommendations,
                excluded_artists_collected,
                n_excluded_artists,
            )


def _render_track_exclusions(
    username: str,
    recommendations: pl.DataFrame,
    excluded_tracks_collected: pl.DataFrame,
    n_excluded_tracks: int,
) -> None:
    """Render track exclusion UI."""
    current_tracks = recommendations.select(
        ["track_id", "track_name", "artist_name"]
    ).to_dicts()
    track_options = {
        f"{t['track_name']} - {t['artist_name']}": t for t in current_tracks
    }

    if len(track_options) > 0:
        selected_track_display = st.selectbox(
            "Exclude a track from recommendations",
            options=list(track_options.keys()),
            key="track_to_remove",
        )

        if st.button("🗑️ Exclude Track", type="secondary", key="exclude_track_btn"):
            _handle_exclude_track(
                username, track_options[selected_track_display], recommendations
            )

    if n_excluded_tracks > 0:
        st.caption("Previously excluded:")
        display_excluded_tracks = excluded_tracks_collected.select(
            [
                pl.col("track_name").alias("Track"),
                pl.col("artist_name").alias("Artist"),
            ]
        )
        st.dataframe(display_excluded_tracks, use_container_width=True, hide_index=True)

        track_to_revert_options = {
            f"{row['track_name']} - {row['artist_name']}": row
            for row in excluded_tracks_collected.to_dicts()
        }

        selected_track_to_revert = st.selectbox(
            "Restore a track",
            options=list(track_to_revert_options.keys()),
            key="track_to_revert",
        )

        if st.button("✅ Restore", type="secondary", key="restore_track_btn"):
            _handle_restore_track(
                username, track_to_revert_options[selected_track_to_revert]
            )


def _handle_exclude_track(
    username: str, selected_track: dict, recommendations: pl.DataFrame
) -> None:
    """Handle excluding a track."""
    try:
        add_excluded_track_local(
            username=username,
            track_id=selected_track["track_id"],
            track_name=selected_track["track_name"],
            artist_name=selected_track["artist_name"],
        )

        excluded_key = (selected_track["track_name"], selected_track["artist_name"])
        if "excluded_in_session" not in st.session_state:
            st.session_state.excluded_in_session = set()
        st.session_state.excluded_in_session.add(excluded_key)

        st.session_state.recommendations = recommendations.filter(
            (pl.col("track_name") != selected_track["track_name"])
            | (pl.col("artist_name") != selected_track["artist_name"])
        )

        replacement = _find_replacement_track(
            recommendations, st.session_state.excluded_in_session
        )

        if replacement is not None and len(replacement) > 0:
            pool = st.session_state.candidate_pool
            common_columns = [
                c for c in st.session_state.recommendations.columns if c in pool.columns
            ]
            replacement = replacement.select(common_columns)
            st.session_state.recommendations = pl.concat(
                [st.session_state.recommendations.select(common_columns), replacement]
            )
            st.toast(f"Excluded and replaced with '{replacement['track_name'][0]}'")
        else:
            st.toast(f"Excluded '{selected_track['track_name']}'")

        st.rerun()

    except Exception as e:
        st.error(f"Error excluding track: {e}")


def _handle_restore_track(username: str, track_info: dict) -> None:
    """Handle restoring an excluded track."""
    try:
        remove_excluded_track_local(
            username=username,
            track_id=track_info["track_id"],
            track_name=track_info["track_name"],
            artist_name=track_info["artist_name"],
        )
        st.toast(f"Restored '{track_info['track_name']}'")
        st.rerun()
    except Exception as e:
        st.error(f"Error restoring track: {e}")


def _render_artist_exclusions(
    username: str,
    recommendations: pl.DataFrame,
    excluded_artists_collected: pl.DataFrame,
    n_excluded_artists: int,
) -> None:
    """Render artist exclusion UI."""
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

        if st.button("🚫 Block Artist", type="secondary", key="block_artist_btn"):
            _handle_block_artist(username, selected_artist, recommendations)

    if n_excluded_artists > 0:
        st.caption("Currently blocked:")
        display_excluded_artists = excluded_artists_collected.select(
            pl.col("artist_name").alias("Artist")
        )
        st.dataframe(
            display_excluded_artists, use_container_width=True, hide_index=True
        )

        artist_to_revert_options = (
            excluded_artists_collected.select("artist_name").to_series().to_list()
        )

        selected_artist_to_revert = st.selectbox(
            "Restore an artist",
            options=artist_to_revert_options,
            key="artist_to_revert",
        )

        if st.button("✅ Restore", type="secondary", key="restore_artist_btn"):
            _handle_restore_artist(username, selected_artist_to_revert)


def _handle_block_artist(
    username: str, selected_artist: str, recommendations: pl.DataFrame
) -> None:
    """Handle blocking an artist."""
    try:
        add_excluded_artist_local(username=username, artist_name=selected_artist)

        if "excluded_artists_in_session" not in st.session_state:
            st.session_state.excluded_artists_in_session = set()
        st.session_state.excluded_artists_in_session.add(selected_artist)

        tracks_removed = len(
            recommendations.filter(pl.col("artist_name") == selected_artist)
        )

        st.session_state.recommendations = recommendations.filter(
            pl.col("artist_name") != selected_artist
        )

        if "candidate_pool" in st.session_state and tracks_removed > 0:
            excluded_in_session = st.session_state.get("excluded_in_session", set())
            replacements = _find_replacement_tracks_for_artist(
                st.session_state.recommendations,
                excluded_in_session,
                st.session_state.excluded_artists_in_session,
                tracks_removed,
            )

            if replacements:
                pool = st.session_state.candidate_pool
                common_columns = [
                    c
                    for c in st.session_state.recommendations.columns
                    if c in pool.columns
                ]
                replacement_df = pl.DataFrame(replacements, schema=pool.schema).select(
                    common_columns
                )
                st.session_state.recommendations = pl.concat(
                    [
                        st.session_state.recommendations.select(common_columns),
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


def _handle_restore_artist(username: str, artist_name: str) -> None:
    """Handle restoring a blocked artist."""
    try:
        remove_excluded_artist_local(username=username, artist_name=artist_name)
        st.toast(f"Restored '{artist_name}'")
        st.rerun()
    except Exception as e:
        st.error(f"Error restoring artist: {e}")
