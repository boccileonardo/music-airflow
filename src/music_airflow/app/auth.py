"""
Authentication module for the Streamlit app.

Uses Streamlit's built-in OIDC authentication to verify user identity.
Maps authenticated emails to Last.fm usernames for data access.

Supports two modes:
- Full access: Authenticated user with mapped Last.fm username
- Demo mode: Unauthenticated or unmapped users can view default user's data (read-only)
"""

import logging
from dataclasses import dataclass
from typing import Optional

import streamlit as st

from music_airflow.utils.constants import DEFAULT_USERNAME

logger = logging.getLogger(__name__)

# Mapping from email to Last.fm username
# Users can only access data for their mapped username
EMAIL_TO_USERNAME: dict[str, str] = {
    "lelopolel@gmail.com": "lelopolel",
    "mzkzjz@gmail.com": "Martazie",
}


@dataclass
class AuthState:
    """Authentication state for the current session."""

    username: str
    is_demo_mode: bool
    is_dev_mode: bool = False  # True when auth is not configured (local development)
    user_email: Optional[str] = None
    user_name: Optional[str] = None


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


def get_auth_state() -> AuthState:
    """
    Get the authentication state for the current session.

    Returns:
        AuthState with username and demo_mode flag.
        - Dev mode: auth not configured (local development), full access with user selector
        - Full access: authenticated user with mapped username
        - Demo mode: unauthenticated or unmapped user viewing default user's data (read-only)
    """
    if not is_auth_configured():
        # Auth not configured - development mode, full access with user selector
        logger.warning("Authentication not configured - running in development mode")
        return AuthState(
            username=DEFAULT_USERNAME, is_demo_mode=False, is_dev_mode=True
        )

    user = get_authenticated_user()

    if not user:
        # Not logged in - demo mode
        return AuthState(username=DEFAULT_USERNAME, is_demo_mode=True)

    email = user.get("email", "").lower()
    username = EMAIL_TO_USERNAME.get(email)

    if not username:
        # Logged in but not mapped - demo mode
        return AuthState(
            username=DEFAULT_USERNAME,
            is_demo_mode=True,
            user_email=email,
            user_name=user.get("name"),
        )

    # Fully authenticated and mapped
    return AuthState(
        username=username,
        is_demo_mode=False,
        user_email=email,
        user_name=user.get("name"),
    )


def render_user_menu(is_demo_mode: bool = False) -> None:
    """Render user menu in sidebar showing logged-in user with logout option."""
    if not is_auth_configured():
        return

    user = get_authenticated_user()

    with st.sidebar:
        st.divider()
        if user:
            col1, col2 = st.columns([3, 1])
            with col1:
                display_name = user.get("name", user.get("email", "User"))
                if is_demo_mode:
                    st.caption(f"👤 {display_name} (Demo)")
                else:
                    st.caption(f"👤 {display_name}")
            with col2:
                if st.button("Logout", key="logout_btn", type="secondary"):
                    st.logout()
        else:
            # Show login button for demo mode users
            st.caption("👁️ Demo Mode")
            if st.button("🔐 Sign in", key="login_btn", type="primary"):
                st.login()
