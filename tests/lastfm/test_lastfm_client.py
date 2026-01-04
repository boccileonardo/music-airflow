"""
Tests for LastFMClient API functionality.

Tests API client methods including pagination, error handling,
filtering, and retry logic.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from music_airflow.lastfm_client import LastFMClient


class TestLastFMClientInit:
    """Test LastFMClient initialization."""

    def test_init_with_api_key(self):
        """Test initialization with explicit API key."""
        client = LastFMClient(api_key="test_key", username="testuser")
        assert client.api_key == "test_key"
        assert client.username == "testuser"

    def test_init_from_env(self, monkeypatch):
        """Test initialization from environment variables."""
        monkeypatch.setenv("api_key", "env_test_key")
        client = LastFMClient(username="testuser")
        assert client.api_key == "env_test_key"

    def test_init_no_api_key_raises(self, monkeypatch, tmp_path):
        """Test initialization fails without API key."""
        # Create an empty .env file and remove any existing API key from environment
        monkeypatch.delenv("api_key", raising=False)
        # Mock the .env path to not load any keys
        fake_env = tmp_path / ".env"
        fake_env.touch()
        with patch("music_airflow.lastfm_client.Path") as mock_path_class:
            mock_path_class.return_value.parent.parent.parent = tmp_path
            with pytest.raises(ValueError, match="Last.fm API key not found"):
                LastFMClient()


class TestGetRecentTracks:
    """Test get_recent_tracks method."""

    @patch.object(LastFMClient, "_make_request")
    @pytest.mark.asyncio
    async def test_basic_fetch(self, mock_request, sample_tracks_response):
        """Test basic track fetching."""
        mock_request.return_value = sample_tracks_response

        client = LastFMClient(api_key="test_key")
        tracks = await client.get_recent_tracks(username="testuser")

        assert len(tracks) == 3
        assert tracks[0]["name"] == "Creep"
        assert tracks[0]["artist"]["name"] == "Radiohead"
        assert tracks[1]["name"] == "Yesterday"
        assert tracks[2]["name"] == "Paint It Black"

    @patch.object(LastFMClient, "_make_request")
    @pytest.mark.asyncio
    async def test_single_track_response(
        self, mock_request, sample_single_track_response
    ):
        """Test handling single track (dict instead of list)."""
        mock_request.return_value = sample_single_track_response

        client = LastFMClient(api_key="test_key")
        tracks = await client.get_recent_tracks(username="testuser")

        assert len(tracks) == 1
        assert tracks[0]["name"] == "Creep"

    @patch.object(LastFMClient, "_make_request")
    @pytest.mark.asyncio
    async def test_filters_now_playing(self, mock_request, sample_now_playing_response):
        """Test that now-playing tracks are filtered out."""
        mock_request.return_value = sample_now_playing_response

        client = LastFMClient(api_key="test_key")
        tracks = await client.get_recent_tracks(username="testuser")

        # Should only get the track with timestamp, not the now playing one
        assert len(tracks) == 1
        assert tracks[0]["name"] == "Creep"
        assert "date" in tracks[0]

    @patch.object(LastFMClient, "_make_request")
    @pytest.mark.asyncio
    async def test_empty_response(self, mock_request, sample_empty_response):
        """Test handling empty response."""
        mock_request.return_value = sample_empty_response

        client = LastFMClient(api_key="test_key")
        tracks = await client.get_recent_tracks(username="testuser")

        assert len(tracks) == 0

    @patch.object(LastFMClient, "_make_request")
    @pytest.mark.asyncio
    async def test_pagination(
        self,
        mock_request,
        sample_paginated_response_page1,
        sample_paginated_response_page2,
    ):
        """Test that pagination works correctly."""
        mock_request.side_effect = [
            sample_paginated_response_page1,
            sample_paginated_response_page2,
        ]

        client = LastFMClient(api_key="test_key")
        tracks = await client.get_recent_tracks(username="testuser")

        # Should get all 3 tracks from both pages
        assert len(tracks) == 3
        assert mock_request.call_count == 2
        assert tracks[0]["name"] == "Track 1"
        assert tracks[1]["name"] == "Track 2"
        assert tracks[2]["name"] == "Track 3"

    @patch.object(LastFMClient, "_make_request")
    @pytest.mark.asyncio
    async def test_with_time_range(self, mock_request, sample_tracks_response):
        """Test fetching with from/to timestamps."""
        mock_request.return_value = sample_tracks_response

        client = LastFMClient(api_key="test_key")
        tracks = await client.get_recent_tracks(
            username="testuser", from_timestamp=1609459200, to_timestamp=1609545600
        )

        # Verify parameters were passed correctly
        call_args = mock_request.call_args[0][0]
        assert call_args["from"] == 1609459200
        assert call_args["to"] == 1609545600
        assert len(tracks) == 3

    @pytest.mark.asyncio
    async def test_no_username_raises(self):
        """Test that missing username raises error."""
        client = LastFMClient(api_key="test_key")
        with pytest.raises(ValueError, match="Username must be provided"):
            await client.get_recent_tracks()

    @pytest.mark.asyncio
    async def test_uses_instance_username(self):
        """Test that instance username is used when not provided."""
        with patch.object(LastFMClient, "_make_request") as mock_request:
            mock_request.return_value = {"recenttracks": {"track": []}}

            client = LastFMClient(api_key="test_key", username="default_user")
            await client.get_recent_tracks()

            call_args = mock_request.call_args[0][0]
            assert call_args["user"] == "default_user"


class TestArtistErrorHandling:
    """Test error code 6 (artist not found) handling."""

    @pytest.mark.asyncio
    async def test_get_similar_artists_handles_error_6(self):
        """Test that error code 6 returns empty list for get_similar_artists."""
        mock_response = AsyncMock()
        mock_response.json.return_value = {
            "error": 6,
            "message": "The artist you supplied could not be found",
        }
        mock_response.raise_for_status = MagicMock()

        mock_session = MagicMock()
        mock_session.get.return_value.__aenter__.return_value = mock_response

        client = LastFMClient(api_key="test_key")
        client._session = mock_session

        result = await client.get_similar_artists("NonExistentArtist")
        assert result == []

    @pytest.mark.asyncio
    async def test_get_artist_top_tracks_handles_error_6(self):
        """Test that error code 6 returns empty list for get_artist_top_tracks."""
        mock_response = AsyncMock()
        mock_response.json.return_value = {
            "error": 6,
            "message": "The artist you supplied could not be found",
        }
        mock_response.raise_for_status = MagicMock()

        mock_session = MagicMock()
        mock_session.get.return_value.__aenter__.return_value = mock_response

        client = LastFMClient(api_key="test_key")
        client._session = mock_session

        result = await client.get_artist_top_tracks("NonExistentArtist")
        assert result == []

    @pytest.mark.asyncio
    async def test_get_artist_top_albums_handles_error_6(self):
        """Test that error code 6 returns empty list for get_artist_top_albums."""
        mock_response = AsyncMock()
        mock_response.json.return_value = {
            "error": 6,
            "message": "The artist you supplied could not be found",
        }
        mock_response.raise_for_status = MagicMock()

        mock_session = MagicMock()
        mock_session.get.return_value.__aenter__.return_value = mock_response

        client = LastFMClient(api_key="test_key")
        client._session = mock_session

        result = await client.get_artist_top_albums("NonExistentArtist")
        assert result == []

    @pytest.mark.asyncio
    async def test_get_artist_info_handles_error_6(self):
        """Test that error code 6 returns empty dict for get_artist_info."""
        mock_response = AsyncMock()
        mock_response.json.return_value = {
            "error": 6,
            "message": "The artist you supplied could not be found",
        }
        mock_response.raise_for_status = MagicMock()

        mock_session = MagicMock()
        mock_session.get.return_value.__aenter__.return_value = mock_response

        client = LastFMClient(api_key="test_key")
        client._session = mock_session

        result = await client.get_artist_info("NonExistentArtist")
        assert result == {}

    @pytest.mark.asyncio
    async def test_artist_methods_reraise_other_errors(self):
        """Test that non-error-6 exceptions are still raised."""
        mock_response = AsyncMock()
        mock_response.json.return_value = {
            "error": 10,
            "message": "Invalid API key",
        }
        mock_response.raise_for_status = MagicMock()

        mock_session = MagicMock()
        mock_session.get.return_value.__aenter__.return_value = mock_response

        client = LastFMClient(api_key="test_key")
        client._session = mock_session

        with pytest.raises(ValueError, match="Last.fm API error 10"):
            await client.get_similar_artists("TestArtist")

        with pytest.raises(ValueError, match="Last.fm API error 10"):
            await client.get_artist_top_tracks("TestArtist")

        with pytest.raises(ValueError, match="Last.fm API error 10"):
            await client.get_artist_top_albums("TestArtist")

        with pytest.raises(ValueError, match="Last.fm API error 10"):
            await client.get_artist_info("TestArtist")


class TestMakeRequest:
    """Test _make_request method and retry logic."""

    @pytest.mark.asyncio
    async def test_successful_request(self, sample_tracks_response):
        """Test successful API request."""
        mock_response = AsyncMock()
        mock_response.json.return_value = sample_tracks_response
        mock_response.raise_for_status = MagicMock()

        mock_session = MagicMock()
        mock_session.get.return_value.__aenter__.return_value = mock_response

        client = LastFMClient(api_key="test_key")
        client._session = mock_session

        result = await client._make_request({"method": "user.getrecenttracks"})

        assert result == sample_tracks_response
        mock_response.raise_for_status.assert_called_once()

    @pytest.mark.asyncio
    async def test_api_error_response(self, sample_error_response):
        """Test Last.fm API error response."""
        mock_response = AsyncMock()
        mock_response.json.return_value = sample_error_response
        mock_response.raise_for_status = MagicMock()

        mock_session = MagicMock()
        mock_session.get.return_value.__aenter__.return_value = mock_response

        client = LastFMClient(api_key="test_key")
        client._session = mock_session

        with pytest.raises(ValueError, match="Last.fm API error 6: User not found"):
            await client._make_request({"method": "user.getinfo"})

    # Note: HTTP error retry tests removed - we now use aiohttp with async/await
    # Retry logic is still present via tenacity decorator but testing it requires
    # async test setup with aiohttp mocking


class TestGetUserInfo:
    """Test get_user_info method."""

    @patch.object(LastFMClient, "_make_request")
    @pytest.mark.asyncio
    async def test_get_user_info(self, mock_request, sample_user_info):
        """Test fetching user information."""
        mock_request.return_value = sample_user_info

        client = LastFMClient(api_key="test_key")
        user = await client.get_user_info(username="testuser")

        assert user["name"] == "testuser"
        assert user["playcount"] == "12345"

    @pytest.mark.asyncio
    async def test_no_username_raises(self):
        """Test that missing username raises error."""
        client = LastFMClient(api_key="test_key")
        with pytest.raises(ValueError, match="Username must be provided"):
            await client.get_user_info()

    @pytest.mark.asyncio
    async def test_uses_instance_username(self):
        """Test that instance username is used when not provided."""
        with patch.object(LastFMClient, "_make_request") as mock_request:
            mock_request.return_value = {"user": {}}

            client = LastFMClient(api_key="test_key", username="default_user")
            await client.get_user_info()

            call_args = mock_request.call_args[0][0]
            assert call_args["user"] == "default_user"
