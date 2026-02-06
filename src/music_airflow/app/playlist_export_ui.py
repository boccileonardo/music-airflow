"""
Playlist export UI components for the Streamlit app.

Handles the UI for exporting recommendations to YouTube Music and Spotify playlists.
Supports per-user OAuth authentication with tokens stored in Firestore.

Security: When app authentication is enabled, OAuth tokens are scoped to the
authenticated user. Users can only connect and use their own OAuth tokens.
"""

import logging
import traceback

import streamlit as st

from music_airflow.app.auth import get_authenticated_username, is_auth_configured
from music_airflow.app.oauth_storage import get_oauth_storage
from music_airflow.app.spotify_playlist import (
    SpotifyPlaylistGenerator,
    exchange_code_for_token,
    get_spotify_redirect_uri,
    load_spotify_creds,
    run_spotify_oauth,
)
from music_airflow.app.youtube_playlist import (
    YouTubePlaylistGenerator,
    exchange_youtube_code_for_token,
    load_youtube_creds,
    run_youtube_oauth,
)


def handle_oauth_callback() -> None:
    """
    Handle OAuth callback from URL query parameters.

    This should be called early in the app to process redirects from OAuth providers.
    The state parameter contains the provider and username: "provider:username:nonce"

    Security: When authentication is enabled, verifies that the OAuth callback's
    username matches the authenticated user before storing tokens.
    """
    params = st.query_params

    # Handle OAuth callback (Spotify or YouTube)
    if "code" in params and "state" in params:
        code = params.get("code")
        state = params.get("state")

        if not code or not state:
            return

        # Parse state to get provider and username
        # Format: "provider:username:nonce"
        parts = state.split(":", 2)
        if len(parts) < 3:
            logging.warning(f"Invalid OAuth state format: {state}")
            return

        provider, username, _nonce = parts

        # Security check: verify username matches authenticated user
        if not _verify_oauth_user(username):
            logging.warning(
                f"OAuth callback rejected: username mismatch for {provider}"
            )
            st.query_params.clear()
            st.error(
                "Security error: OAuth callback does not match your account. "
                "Please try connecting again."
            )
            return

        if provider == "spotify":
            _process_spotify_callback(code, username)
            st.query_params.clear()
            st.rerun()
        elif provider == "youtube":
            _process_youtube_callback(code, username)
            st.query_params.clear()
            st.rerun()


def _verify_oauth_user(callback_username: str) -> bool:
    """
    Verify that the OAuth callback username matches the authenticated user.

    When authentication is not configured, allows all usernames (dev mode).
    When authentication is configured, only allows the authenticated user's username.

    Returns:
        True if the username is allowed, False otherwise.
    """
    if not is_auth_configured():
        return True

    auth_username = get_authenticated_username()
    if auth_username is None:
        # User not authenticated - reject
        return False

    return callback_username == auth_username


def _process_spotify_callback(code: str, username: str) -> None:
    """Process Spotify OAuth callback and store tokens."""
    spotify_creds = load_spotify_creds()
    if not spotify_creds:
        st.error("Spotify credentials not configured")
        return

    token_info = exchange_code_for_token(
        spotify_creds.client_id,
        spotify_creds.client_secret,
        code,
    )

    if token_info:
        storage = get_oauth_storage()
        storage.save_tokens(
            username,
            "spotify",
            token_info["access_token"],
            token_info["refresh_token"],
            token_info.get("expires_in"),
        )
        st.session_state["spotify_connected"] = True
        st.session_state["username"] = username  # Restore username in session
        st.toast("✅ Spotify connected successfully!", icon="🎵")
    else:
        st.error("Failed to connect Spotify. Please try again.")


def _process_youtube_callback(code: str, username: str) -> None:
    """Process YouTube OAuth callback and store tokens."""
    youtube_creds = load_youtube_creds()
    if not youtube_creds:
        st.error("YouTube credentials not configured")
        return

    token_info = exchange_youtube_code_for_token(
        youtube_creds.client_id,
        youtube_creds.client_secret,
        code,
    )

    if token_info:
        storage = get_oauth_storage()
        storage.save_tokens(
            username,
            "youtube",
            token_info["access_token"],
            token_info["refresh_token"],
            token_info.get("expires_in"),
        )
        st.session_state["youtube_connected"] = True
        st.session_state["username"] = username  # Restore username in session
        st.toast("✅ YouTube connected successfully!", icon="🎬")
    else:
        st.error("Failed to connect YouTube. Please try again.")


def render_playlist_export_section() -> None:
    """Render the playlist export section."""
    if (
        "recommendations" not in st.session_state
        or st.session_state.recommendations is None
    ):
        return

    st.divider()
    st.header("📤 Export to Playlist")

    username = st.session_state.get("username", "User")
    playlist_name = st.text_input(
        "Playlist Name",
        value=f"{username} - AirStream.FM",
        key=f"playlist_name_input_{username}",
        label_visibility="collapsed",
        placeholder="Playlist name...",
    )

    youtube_tab, spotify_tab = st.tabs(["🎬 YouTube Music", "🎧 Spotify"])

    with youtube_tab:
        youtube_privacy = st.selectbox(
            "Visibility",
            options=["public", "private", "unlisted"],
            index=0,
            key="youtube_privacy_selector",
        )
        _render_youtube_tab(username, playlist_name, youtube_privacy)

    with spotify_tab:
        spotify_public = st.selectbox(
            "Visibility",
            options=["public", "private"],
            index=0,
            key="spotify_privacy_selector",
        )
        _render_spotify_tab(username, playlist_name, spotify_public == "public")


def _render_youtube_tab(username: str, playlist_name: str, privacy: str) -> None:
    """Render YouTube Music export tab."""
    needs_auth = YouTubePlaylistGenerator.needs_authentication(username)

    if needs_auth:
        _render_youtube_auth_flow(username)
    else:
        _render_youtube_connected_status(username)

    if st.button(
        "🎬 Create YouTube Playlist",
        type="primary" if not needs_auth else "secondary",
        disabled=needs_auth,
        width="stretch",
        key="create_youtube_playlist",
    ):
        _create_youtube_playlist(username, playlist_name, privacy)


def _render_youtube_connected_status(username: str) -> None:
    """Render connected status with disconnect option."""
    col1, col2 = st.columns([3, 1])
    with col1:
        st.success("✅ YouTube connected")
    with col2:
        if st.button("Disconnect", key="disconnect_youtube", type="secondary"):
            YouTubePlaylistGenerator.disconnect(username)
            st.rerun()


def _render_youtube_auth_flow(username: str) -> None:
    """Render YouTube OAuth flow UI."""
    with st.container(border=True):
        st.warning("⚠️ Connect your YouTube account to create playlists")

        youtube_creds = load_youtube_creds()
        if youtube_creds and youtube_creds.has_client_creds():
            auth_url, _state = run_youtube_oauth(
                youtube_creds.client_id,
                youtube_creds.client_secret,
                username,
            )
            st.link_button(
                "🔐 Connect YouTube",
                auth_url,
                type="primary",
                use_container_width=True,
            )
        else:
            st.info("YouTube API not configured. Contact the app administrator.")


def _render_spotify_tab(username: str, playlist_name: str, public: bool) -> None:
    """Render Spotify export tab."""
    needs_auth = SpotifyPlaylistGenerator.needs_authentication(username)

    if needs_auth:
        _render_spotify_auth_flow(username)
    else:
        _render_spotify_connected_status(username)

    if st.button(
        "🎧 Create Spotify Playlist",
        type="primary" if not needs_auth else "secondary",
        disabled=needs_auth,
        width="stretch",
        key="create_spotify_playlist",
    ):
        _create_spotify_playlist(username, playlist_name, public)


def _render_spotify_connected_status(username: str) -> None:
    """Render connected status with disconnect option."""
    col1, col2 = st.columns([3, 1])
    with col1:
        st.success("✅ Spotify connected")
    with col2:
        if st.button("Disconnect", key="disconnect_spotify", type="secondary"):
            SpotifyPlaylistGenerator.disconnect(username)
            st.rerun()


def _render_spotify_auth_flow(username: str) -> None:
    """Render Spotify OAuth flow UI."""
    with st.container(border=True):
        st.warning("⚠️ Connect your Spotify account to create playlists")

        spotify_creds = load_spotify_creds()
        if spotify_creds and spotify_creds.has_client_creds():
            redirect_uri = get_spotify_redirect_uri()
            st.caption(f"Redirect URI: `{redirect_uri}`")

            auth_url, _state = run_spotify_oauth(
                spotify_creds.client_id,
                spotify_creds.client_secret,
                username,
            )
            st.link_button(
                "🔐 Connect Spotify",
                auth_url,
                type="primary",
                use_container_width=True,
            )
        else:
            st.info("Spotify API not configured. Contact the app administrator.")


def _create_youtube_playlist(username: str, playlist_name: str, privacy: str) -> None:
    """Create a YouTube playlist from recommendations."""
    try:
        recommendations = st.session_state.recommendations
        logging.info(f"Starting YouTube playlist creation: {playlist_name}")
        logging.info(f"Number of tracks in recommendations: {len(recommendations)}")
        logging.info(f"Columns in recommendations: {recommendations.columns}")
        logging.info(
            f"Tracks with youtube_url: {recommendations['youtube_url'].is_not_null().sum()}"
        )

        playlist_generator = YouTubePlaylistGenerator(username)

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


def _create_spotify_playlist(username: str, playlist_name: str, public: bool) -> None:
    """Create a Spotify playlist from recommendations."""
    try:
        recommendations = st.session_state.recommendations
        logging.info(f"Starting Spotify playlist creation: {playlist_name}")
        logging.info(f"Number of tracks in recommendations: {len(recommendations)}")
        logging.info(
            f"Tracks with spotify_url: {recommendations['spotify_url'].is_not_null().sum()}"
        )

        playlist_generator = SpotifyPlaylistGenerator(username)

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
