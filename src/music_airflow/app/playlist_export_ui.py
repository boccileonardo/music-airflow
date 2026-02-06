"""
Playlist export UI components for the Streamlit app.

Handles the UI for exporting recommendations to YouTube Music and Spotify playlists.
"""

import logging
import traceback

import streamlit as st

from music_airflow.app.spotify_playlist import (
    SpotifyPlaylistGenerator,
    exchange_code_for_token,
    load_spotify_creds,
    run_spotify_oauth,
)
from music_airflow.app.youtube_playlist import (
    YouTubePlaylistGenerator,
    load_youtube_creds,
    poll_device_token,
    run_youtube_oauth,
)


def render_playlist_export_section() -> None:
    """Render the playlist export section."""
    if (
        "recommendations" not in st.session_state
        or st.session_state.recommendations is None
    ):
        return

    st.divider()
    st.header("📤 Export to Playlist")

    youtube_tab, spotify_tab = st.tabs(["🎬 YouTube Music", "🎧 Spotify"])

    # Common playlist settings
    col1, col2 = st.columns([3, 1])
    username = st.session_state.get("username", "User")
    with col1:
        playlist_name = st.text_input(
            "Playlist Name",
            value=f"{username} - AirStream.FM",
            key=f"playlist_name_input_{username}",
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

    with youtube_tab:
        _render_youtube_tab(playlist_name, privacy)

    with spotify_tab:
        _render_spotify_tab(playlist_name, privacy == "public")


def _render_youtube_tab(playlist_name: str, privacy: str) -> None:
    """Render YouTube Music export tab."""
    needs_auth = YouTubePlaylistGenerator.needs_authentication()

    if needs_auth:
        _render_youtube_auth_flow()

    if st.button(
        "🎬 Create YouTube Playlist",
        type="primary" if not needs_auth else "secondary",
        disabled=needs_auth,
        use_container_width=True,
        key="create_youtube_playlist",
    ):
        _create_youtube_playlist(playlist_name, privacy)


def _render_youtube_auth_flow() -> None:
    """Render YouTube OAuth flow UI."""
    with st.container(border=True):
        st.warning("⚠️ YouTube authentication required")
        st.caption("Configure credentials in .env, then run OAuth flow below.")

        youtube_creds = load_youtube_creds()
        if youtube_creds and youtube_creds.has_client_creds():
            if st.button("🔐 Authenticate with YouTube", type="primary"):
                device_info, _ = run_youtube_oauth(
                    youtube_creds.client_id, youtube_creds.client_secret
                )
                if device_info:
                    st.session_state["youtube_device_info"] = device_info
                    st.session_state["youtube_client_id"] = youtube_creds.client_id
                    st.session_state["youtube_client_secret"] = (
                        youtube_creds.client_secret
                    )
                    st.rerun()
                else:
                    st.error("Failed to start OAuth flow")
        else:
            st.info("Set YOUTUBE_CLIENT_ID and YOUTUBE_CLIENT_SECRET in .env")

        if "youtube_device_info" in st.session_state:
            device_info = st.session_state["youtube_device_info"]
            st.markdown("---")
            st.markdown(f"**1.** Go to: `{device_info['verification_url']}`")
            st.markdown(f"**2.** Enter code: `{device_info['user_code']}`")
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
                        st.warning("Still pending. Complete the flow and try again.")
                except Exception as e:
                    st.error(f"OAuth failed: {e}")
                    del st.session_state["youtube_device_info"]


def _render_spotify_tab(playlist_name: str, public: bool) -> None:
    """Render Spotify export tab."""
    needs_auth = SpotifyPlaylistGenerator.needs_authentication()

    if needs_auth:
        _render_spotify_auth_flow()

    if st.button(
        "🎧 Create Spotify Playlist",
        type="primary" if not needs_auth else "secondary",
        disabled=needs_auth,
        use_container_width=True,
        key="create_spotify_playlist",
    ):
        _create_spotify_playlist(playlist_name, public)


def _render_spotify_auth_flow() -> None:
    """Render Spotify OAuth flow UI."""
    with st.container(border=True):
        st.warning("⚠️ Spotify authentication required")
        st.caption("Configure credentials in .env, then run OAuth flow below.")

        spotify_creds = load_spotify_creds()
        if spotify_creds and spotify_creds.has_client_creds():
            if "spotify_auth_url" not in st.session_state:
                if st.button("🔐 Authenticate with Spotify", type="primary"):
                    auth_url, state = run_spotify_oauth(
                        spotify_creds.client_id,
                        spotify_creds.client_secret,
                    )
                    st.session_state["spotify_auth_url"] = auth_url
                    st.session_state["spotify_auth_state"] = state
                    st.session_state["spotify_client_id"] = spotify_creds.client_id
                    st.session_state["spotify_client_secret"] = (
                        spotify_creds.client_secret
                    )
                    st.rerun()

            if "spotify_auth_url" in st.session_state:
                _render_spotify_code_entry()
        else:
            st.info("Set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET in .env")


def _render_spotify_code_entry() -> None:
    """Render Spotify authorization code entry UI."""
    auth_url = st.session_state["spotify_auth_url"]
    st.markdown("---")
    st.info(
        "**Important:** Add `http://localhost:8501/` as a Redirect URI in your "
        "[Spotify Developer Dashboard](https://developer.spotify.com/dashboard) app settings."
    )
    st.markdown("**Step 1:** Click the button below to authorize:")
    st.link_button("🔗 Open Spotify Authorization", auth_url, type="primary")
    st.markdown("**Step 2:** After authorizing, you'll be redirected back here.")
    st.markdown("**Step 3:** Copy the `code` from the URL bar and paste below:")
    st.caption(
        "The URL will look like: `http://localhost:8501/?code=AQB...xyz&state=...`"
    )
    st.caption("Copy **only** the part after `code=` and before `&state=`")

    auth_code = st.text_input(
        "Authorization Code",
        placeholder="Paste the code from the redirect URL...",
        key="spotify_auth_code_input",
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button(
            "✅ Complete Authentication", key="complete_spotify_auth", type="primary"
        ):
            if auth_code:
                token_info = exchange_code_for_token(
                    st.session_state["spotify_client_id"],
                    st.session_state["spotify_client_secret"],
                    auth_code,
                )
                if token_info:
                    st.success("✅ Authenticated! Add to .env:")
                    st.code(
                        f"SPOTIFY_ACCESS_TOKEN={token_info.get('access_token', '')}\n"
                        f"SPOTIFY_REFRESH_TOKEN={token_info.get('refresh_token', '')}",
                        language="bash",
                    )
                    _clear_spotify_session_state()
                else:
                    st.error(
                        "Failed to exchange code for token. The code may have expired - try again."
                    )
            else:
                st.warning("Please paste the authorization code")
    with col2:
        if st.button("❌ Cancel", key="cancel_spotify_auth"):
            _clear_spotify_session_state()
            st.rerun()


def _clear_spotify_session_state() -> None:
    """Clear Spotify OAuth session state."""
    for key in [
        "spotify_auth_url",
        "spotify_auth_state",
        "spotify_client_id",
        "spotify_client_secret",
    ]:
        st.session_state.pop(key, None)


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
            playlist_description=(
                f"Generated by Music Recommendation System\n"
                f"Discovery weight: {settings['discovery_weight']}\n"
                f"Systems: {'Tags' if settings['use_tags'] else ''} "
                f"{'Artists' if settings['use_artists'] else ''} "
                f"{'Deep Cuts' if settings['use_deep_cuts'] else ''}"
            ),
            privacy_status=privacy,
            progress_bar=progress_bar,
            status_text=status_text,
        )

        progress_bar.empty()
        status_text.empty()

        _display_youtube_result(result, playlist_name)

    except Exception as e:
        _handle_youtube_error(e)


def _display_youtube_result(result: dict | None, playlist_name: str) -> None:
    """Display YouTube playlist creation result."""
    if not result:
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
        return

    total_tracks = len(st.session_state.recommendations)
    success_rate = (
        (result["tracks_added"] / total_tracks * 100) if total_tracks > 0 else 0
    )

    if result.get("quota_exceeded"):
        st.error(
            f"⚠️ YouTube API quota exceeded! Added {result['tracks_added']}/{total_tracks} tracks "
            f"before hitting the limit.\n\n"
            f"**What happened:** YouTube limits the number of API requests per day.\n\n"
            f"**Solution:** Quota resets at midnight Pacific Time. Try again tomorrow, "
            f"or reduce the number of recommendations."
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
            f"❌ Playlist created but most tracks failed. "
            f"Only {result['tracks_added']}/{total_tracks} tracks added."
        )

    st.markdown(
        f"**Open in YouTube Music:** [{playlist_name}]({result['playlist_url']})"
    )

    if result["tracks_not_found"]:
        with st.expander(f"⚠️ {len(result['tracks_not_found'])} tracks had issues"):
            st.info("Check the terminal/logs for detailed error messages.")
            for track in result["tracks_not_found"]:
                st.text(f"• {track}")


def _handle_youtube_error(e: Exception) -> None:
    """Handle YouTube playlist creation error."""
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
            playlist_description=(
                f"Generated by Music Recommendation System | "
                f"Discovery weight: {settings['discovery_weight']} | "
                f"Systems: {'Tags' if settings['use_tags'] else ''} "
                f"{'Artists' if settings['use_artists'] else ''} "
                f"{'Deep Cuts' if settings['use_deep_cuts'] else ''}"
            ),
            public=public,
            progress_bar=progress_bar,
            status_text=status_text,
        )

        progress_bar.empty()
        status_text.empty()

        _display_spotify_result(result, playlist_name)

    except Exception as e:
        _handle_spotify_error(e)


def _display_spotify_result(result: dict | None, playlist_name: str) -> None:
    """Display Spotify playlist creation result."""
    if not result:
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
        return

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
            f"❌ Playlist created but most tracks failed. "
            f"Only {result['tracks_added']}/{total_tracks} tracks added."
        )

    st.markdown(f"**Open in Spotify:** [{playlist_name}]({result['playlist_url']})")

    if result["tracks_not_found"]:
        with st.expander(f"⚠️ {len(result['tracks_not_found'])} tracks not found"):
            st.info("These tracks couldn't be found on Spotify.")
            for track in result["tracks_not_found"]:
                st.text(f"• {track}")


def _handle_spotify_error(e: Exception) -> None:
    """Handle Spotify playlist creation error."""
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
