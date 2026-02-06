"""
Authentication module for the Streamlit app.

Uses Streamlit's built-in OIDC authentication to verify user identity.
Maps authenticated emails to Last.fm usernames for data access.
"""

import logging
from typing import Optional

import streamlit as st

logger = logging.getLogger(__name__)

# Mapping from email to Last.fm username
# Users can only access data for their mapped username
EMAIL_TO_USERNAME: dict[str, str] = {
    "lelopolel@gmail.com": "lelopolel",
    "mzkzjz@gmail.com": "Martazie",
}


def is_auth_configured() -> bool:
    """Check if authentication is configured in secrets."""
    try:
        return "auth" in st.secrets and "client_id" in st.secrets.get("auth", {})
    except Exception:
        return False


def get_authenticated_user() -> Optional[dict]:
    """
    Get the currently authenticated user's info.

    Returns:
        User info dict with 'email', 'name', etc. if logged in, None otherwise.
    """
    if not is_auth_configured():
        return None

    if st.user.is_logged_in:
        return st.user.to_dict()
    return None


def get_authenticated_username() -> Optional[str]:
    """
    Get the Last.fm username for the authenticated user.

    Returns:
        Last.fm username if the user is authenticated and mapped, None otherwise.
    """
    user = get_authenticated_user()
    if not user:
        return None

    email = user.get("email", "").lower()
    return EMAIL_TO_USERNAME.get(email)


def require_authentication() -> Optional[str]:
    """
    Require authentication and return the username.

    Shows login UI if not authenticated.
    Shows error if authenticated but not authorized.

    Returns:
        Last.fm username if authorized, None if login UI shown.
    """
    if not is_auth_configured():
        # Auth not configured - allow all users (development mode)
        logger.warning("Authentication not configured - running in development mode")
        return None

    user = get_authenticated_user()

    if not user:
        _render_login_page()
        return None

    email = user.get("email", "").lower()
    username = EMAIL_TO_USERNAME.get(email)

    if not username:
        _render_unauthorized_page(email)
        return None

    return username


def _render_login_page() -> None:
    """Render the login page."""
    st.set_page_config(
        page_title="AirStream.FM - Login",
        page_icon="🎵",
        layout="centered",
    )

    st.title("🎵 AirStream.FM")
    st.subheader("Music Recommendation System")

    st.divider()

    st.markdown(
        """
        Welcome to AirStream.FM! This app provides personalized music
        recommendations based on your Last.fm listening history.

        Please sign in to access your recommendations.
        """
    )

    if st.button("🔐 Sign in with Google", type="primary", use_container_width=True):
        st.login()

    st.stop()


def _render_unauthorized_page(email: str) -> None:
    """Render the unauthorized page for users not in the allowed list."""
    st.set_page_config(
        page_title="AirStream.FM - Unauthorized",
        page_icon="🎵",
        layout="centered",
    )

    st.title("🎵 AirStream.FM")

    st.error(
        f"**Access Denied**\n\n"
        f"The email `{email}` is not authorized to use this app.\n\n"
        f"If you believe this is an error, please contact the administrator."
    )

    if st.button("🚪 Sign out", type="secondary"):
        st.logout()

    st.stop()


def render_user_menu() -> None:
    """Render user menu in sidebar showing logged-in user with logout option."""
    if not is_auth_configured():
        return

    user = get_authenticated_user()
    if not user:
        return

    with st.sidebar:
        st.divider()
        col1, col2 = st.columns([3, 1])
        with col1:
            st.caption(f"👤 {user.get('name', user.get('email', 'User'))}")
        with col2:
            if st.button("Logout", key="logout_btn", type="secondary"):
                st.logout()
