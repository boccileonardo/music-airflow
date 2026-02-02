"""Tests for Spotify search utility."""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_spotify_credentials():
    """Mock Spotify credentials."""
    with patch(
        "music_airflow.utils.spotify_search._get_spotify_credentials"
    ) as mock_creds:
        mock_creds.return_value = ("test_client_id", "test_client_secret")
        yield mock_creds


@pytest.fixture
def mock_spotify_client():
    """Mock Spotify client."""
    with patch("music_airflow.utils.spotify_search.Spotify") as mock_spotify:
        client = MagicMock()
        mock_spotify.return_value = client
        yield client


class TestSpotifySearch:
    """Tests for Spotify search functions."""

    def test_search_spotify_url_returns_url_on_success(
        self, mock_spotify_credentials, mock_spotify_client
    ):
        """Test that search_spotify_url returns a URL when track is found."""
        # Import after mocking
        from music_airflow.utils.spotify_search import search_spotify_url

        # Reset the global client
        import music_airflow.utils.spotify_search as spotify_module

        spotify_module._spotify = None

        mock_spotify_client.search.return_value = {
            "tracks": {
                "items": [
                    {
                        "id": "test_track_id_123",
                        "name": "Test Track",
                        "artists": [{"name": "Test Artist"}],
                    }
                ]
            }
        }

        # Need to mock the SpotifyClientCredentials and Spotify initialization
        with patch(
            "music_airflow.utils.spotify_search.SpotifyClientCredentials"
        ) as mock_auth:
            mock_auth.return_value = MagicMock()
            with patch(
                "music_airflow.utils.spotify_search.Spotify"
            ) as mock_spotify_class:
                mock_spotify_class.return_value = mock_spotify_client

                result = search_spotify_url("Test Track", "Test Artist")

        assert result == "https://open.spotify.com/track/test_track_id_123"

    def test_search_spotify_url_returns_none_when_no_credentials(self):
        """Test that search_spotify_url returns None when credentials are missing."""
        from music_airflow.utils.spotify_search import search_spotify_url

        # Reset the global client
        import music_airflow.utils.spotify_search as spotify_module

        spotify_module._spotify = None

        with patch(
            "music_airflow.utils.spotify_search._get_spotify_credentials"
        ) as mock_creds:
            mock_creds.return_value = (None, None)

            result = search_spotify_url("Test Track", "Test Artist")

        assert result is None

    def test_search_spotify_track_id_returns_none_on_empty_results(
        self, mock_spotify_credentials
    ):
        """Test that search returns None when no tracks found."""
        from music_airflow.utils.spotify_search import search_spotify_track_id

        # Reset the global client
        import music_airflow.utils.spotify_search as spotify_module

        spotify_module._spotify = None

        mock_client = MagicMock()
        mock_client.search.return_value = {"tracks": {"items": []}}

        with patch(
            "music_airflow.utils.spotify_search.SpotifyClientCredentials"
        ) as mock_auth:
            mock_auth.return_value = MagicMock()
            with patch(
                "music_airflow.utils.spotify_search.Spotify"
            ) as mock_spotify_class:
                mock_spotify_class.return_value = mock_client

                result = search_spotify_track_id("Unknown Track", "Unknown Artist")

        assert result is None

    def test_is_spotify_configured_returns_true_with_credentials(self):
        """Test is_spotify_configured returns True when credentials are set."""
        from music_airflow.utils.spotify_search import is_spotify_configured

        with patch(
            "music_airflow.utils.spotify_search._get_spotify_credentials"
        ) as mock_creds:
            mock_creds.return_value = ("client_id", "client_secret")
            assert is_spotify_configured() is True

    def test_is_spotify_configured_returns_false_without_credentials(self):
        """Test is_spotify_configured returns False when credentials are missing."""
        from music_airflow.utils.spotify_search import is_spotify_configured

        with patch(
            "music_airflow.utils.spotify_search._get_spotify_credentials"
        ) as mock_creds:
            mock_creds.return_value = (None, None)
            assert is_spotify_configured() is False


class TestSpotifyPlaylistGenerator:
    """Tests for Spotify playlist generator."""

    def test_extract_track_id_from_url(self):
        """Test track ID extraction from various URL formats."""
        from music_airflow.app.spotify_playlist import SpotifyPlaylistGenerator

        # Standard URL
        url1 = "https://open.spotify.com/track/4uLU6hMCjMI75M1A2tKUQC"
        assert (
            SpotifyPlaylistGenerator._extract_track_id(url1) == "4uLU6hMCjMI75M1A2tKUQC"
        )

        # URI format
        url2 = "spotify:track:4uLU6hMCjMI75M1A2tKUQC"
        assert (
            SpotifyPlaylistGenerator._extract_track_id(url2) == "4uLU6hMCjMI75M1A2tKUQC"
        )

        # Invalid URL
        url3 = "https://example.com/track/abc"
        assert SpotifyPlaylistGenerator._extract_track_id(url3) is None

    def test_get_playlist_url(self):
        """Test playlist URL generation."""
        from music_airflow.app.spotify_playlist import SpotifyPlaylistGenerator

        url = SpotifyPlaylistGenerator.get_playlist_url("test_playlist_id")
        assert url == "https://open.spotify.com/playlist/test_playlist_id"

    def test_needs_authentication_without_tokens(self):
        """Test needs_authentication returns True when no tokens."""
        from music_airflow.app.spotify_playlist import SpotifyPlaylistGenerator

        with patch(
            "music_airflow.app.spotify_playlist.load_spotify_creds"
        ) as mock_load:
            mock_creds = MagicMock()
            mock_creds.has_client_creds.return_value = True
            mock_creds.has_tokens.return_value = False
            mock_load.return_value = mock_creds

            assert SpotifyPlaylistGenerator.needs_authentication() is True

    def test_needs_authentication_with_tokens(self):
        """Test needs_authentication returns False when tokens exist."""
        from music_airflow.app.spotify_playlist import SpotifyPlaylistGenerator

        with patch(
            "music_airflow.app.spotify_playlist.load_spotify_creds"
        ) as mock_load:
            mock_creds = MagicMock()
            mock_creds.has_client_creds.return_value = True
            mock_creds.has_tokens.return_value = True
            mock_load.return_value = mock_creds

            assert SpotifyPlaylistGenerator.needs_authentication() is False
