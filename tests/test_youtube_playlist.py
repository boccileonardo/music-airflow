"""Tests for YouTube playlist generation."""

import polars as pl
import pytest
from unittest.mock import MagicMock, patch
from googleapiclient.errors import HttpError
from music_airflow.app.youtube_playlist import YouTubePlaylistGenerator


@pytest.fixture
def sample_tracks():
    """Sample tracks DataFrame with correct column names."""
    return pl.DataFrame(
        {
            "track_name": [
                "Bohemian Rhapsody",
                "Hotel California",
                "Stairway to Heaven",
            ],
            "artist_name": ["Queen", "Eagles", "Led Zeppelin"],
            "score": [1.0, 0.9, 0.8],
        }
    )


@pytest.fixture
def sample_tracks_with_urls():
    """Sample tracks DataFrame with YouTube URLs."""
    return pl.DataFrame(
        {
            "track_name": [
                "Bohemian Rhapsody",
                "Hotel California",
                "Stairway to Heaven",
            ],
            "artist_name": ["Queen", "Eagles", "Led Zeppelin"],
            "score": [1.0, 0.9, 0.8],
            "youtube_url": [
                "https://www.youtube.com/watch?v=fJ9rUzIMcZQ",
                "https://www.youtube.com/watch?v=09839DpTctU",
                "https://www.youtube.com/watch?v=QkF3oxziUI4",
            ],
        }
    )


def test_search_track_success():
    """Test successful track search."""
    generator = YouTubePlaylistGenerator()

    # Mock the YouTube API service
    mock_youtube = MagicMock()
    mock_search = MagicMock()
    mock_list = MagicMock()

    mock_youtube.search.return_value = mock_search
    mock_search.list.return_value = mock_list
    mock_list.execute.return_value = {
        "items": [
            {
                "id": {"videoId": "fJ9rUzIMcZQ"},
                "snippet": {
                    "title": "Bohemian Rhapsody",
                    "channelTitle": "Queen - Topic",
                },
            }
        ]
    }

    generator.youtube = mock_youtube

    video_id = generator.search_track("Bohemian Rhapsody", "Queen")

    assert video_id == "fJ9rUzIMcZQ"
    mock_search.list.assert_called_once()


def test_search_track_prioritizes_topic_channel():
    """Test that Topic channels are prioritized."""
    generator = YouTubePlaylistGenerator()

    mock_youtube = MagicMock()
    mock_search = MagicMock()
    mock_list = MagicMock()

    mock_youtube.search.return_value = mock_search
    mock_search.list.return_value = mock_list
    mock_list.execute.return_value = {
        "items": [
            {
                "id": {"videoId": "music_video_id"},
                "snippet": {
                    "title": "Song Name Official Music Video",
                    "channelTitle": "Artist VEVO",
                },
            },
            {
                "id": {"videoId": "topic_id"},
                "snippet": {"title": "Song Name", "channelTitle": "Artist - Topic"},
            },
        ]
    }

    generator.youtube = mock_youtube

    video_id = generator.search_track("Song Name", "Artist")

    # Should return Topic channel video, not VEVO music video
    assert video_id == "topic_id"


def test_search_track_filters_music_videos():
    """Test that music videos are filtered out when possible."""
    generator = YouTubePlaylistGenerator()

    mock_youtube = MagicMock()
    mock_search = MagicMock()
    mock_list = MagicMock()

    mock_youtube.search.return_value = mock_search
    mock_search.list.return_value = mock_list
    mock_list.execute.return_value = {
        "items": [
            {
                "id": {"videoId": "music_video_id"},
                "snippet": {
                    "title": "Song Name Official Music Video",
                    "channelTitle": "Artist",
                },
            },
            {
                "id": {"videoId": "audio_id"},
                "snippet": {"title": "Song Name (Audio)", "channelTitle": "Artist"},
            },
        ]
    }

    generator.youtube = mock_youtube

    video_id = generator.search_track("Song Name", "Artist")

    # Should return audio version, not music video
    assert video_id == "audio_id"


def test_search_track_no_results():
    """Test track search with no results."""
    generator = YouTubePlaylistGenerator()

    mock_youtube = MagicMock()
    mock_search = MagicMock()
    mock_list = MagicMock()

    mock_youtube.search.return_value = mock_search
    mock_search.list.return_value = mock_list
    mock_list.execute.return_value = {"items": []}

    generator.youtube = mock_youtube

    video_id = generator.search_track("NonexistentTrack", "NonexistentArtist")

    assert video_id is None


def test_search_track_uses_cache():
    """Test that search results are cached."""
    generator = YouTubePlaylistGenerator()

    mock_youtube = MagicMock()
    mock_search = MagicMock()
    mock_list = MagicMock()

    mock_youtube.search.return_value = mock_search
    mock_search.list.return_value = mock_list
    mock_list.execute.return_value = {
        "items": [
            {
                "id": {"videoId": "test_id"},
                "snippet": {
                    "title": "Test Track",
                    "channelTitle": "Test Artist - Topic",
                },
            }
        ]
    }

    generator.youtube = mock_youtube

    # First call - should hit API
    video_id1 = generator.search_track("Test Track", "Test Artist")
    assert video_id1 == "test_id"

    # Second call - should use cache
    video_id2 = generator.search_track("Test Track", "Test Artist")
    assert video_id2 == "test_id"

    # API should only be called once
    mock_list.execute.assert_called_once()


def test_create_playlist_success():
    """Test successful playlist creation."""
    generator = YouTubePlaylistGenerator()

    mock_youtube = MagicMock()
    mock_playlists = MagicMock()
    mock_insert = MagicMock()

    mock_youtube.playlists.return_value = mock_playlists
    mock_playlists.insert.return_value = mock_insert
    mock_insert.execute.return_value = {"id": "PLtest123"}

    generator.youtube = mock_youtube

    playlist_id = generator.create_playlist(
        title="Test Playlist",
        description="Test description",
        privacy_status="unlisted",
    )

    assert playlist_id == "PLtest123"
    mock_playlists.insert.assert_called_once()


def test_delete_playlist_success():
    """Test successful playlist deletion."""
    generator = YouTubePlaylistGenerator()

    mock_youtube = MagicMock()
    mock_playlists = MagicMock()
    mock_delete = MagicMock()

    mock_youtube.playlists.return_value = mock_playlists
    mock_playlists.delete.return_value = mock_delete
    mock_delete.execute.return_value = {}

    generator.youtube = mock_youtube

    result = generator.delete_playlist("PLtest123")

    assert result is True
    mock_playlists.delete.assert_called_once_with(id="PLtest123")


def test_add_video_to_playlist_success():
    """Test successfully adding video to playlist."""
    generator = YouTubePlaylistGenerator()

    mock_youtube = MagicMock()
    mock_playlist_items = MagicMock()
    mock_insert = MagicMock()

    mock_youtube.playlistItems.return_value = mock_playlist_items
    mock_playlist_items.insert.return_value = mock_insert
    mock_insert.execute.return_value = {}

    generator.youtube = mock_youtube

    result = generator.add_video_to_playlist("PLtest123", "fJ9rUzIMcZQ")

    assert result is True
    mock_playlist_items.insert.assert_called_once()


def test_add_video_to_playlist_retries_on_409():
    """Test that 409 errors trigger retry logic."""
    generator = YouTubePlaylistGenerator()

    mock_youtube = MagicMock()
    mock_playlist_items = MagicMock()
    mock_insert = MagicMock()

    mock_youtube.playlistItems.return_value = mock_playlist_items
    mock_playlist_items.insert.return_value = mock_insert

    generator.youtube = mock_youtube

    # Mock HttpError with 409 status
    mock_resp = MagicMock()
    mock_resp.status = 409
    error = HttpError(mock_resp, b'{"error": {"code": 409, "message": "Conflict"}}')

    # First attempt fails with 409, second succeeds (max_retries=2 means 2 total attempts)
    with patch("time.sleep"):  # Mock sleep to speed up test
        mock_insert.execute.side_effect = [error, {}]

        result = generator.add_video_to_playlist("PLtest123", "fJ9rUzIMcZQ")

        assert result is True
        assert mock_insert.execute.call_count == 2


def test_add_video_to_playlist_stops_on_quota_exceeded():
    """Test that quota exceeded errors don't trigger retries."""
    generator = YouTubePlaylistGenerator()

    mock_youtube = MagicMock()
    mock_playlist_items = MagicMock()
    mock_insert = MagicMock()

    mock_youtube.playlistItems.return_value = mock_playlist_items
    mock_playlist_items.insert.return_value = mock_insert

    generator.youtube = mock_youtube

    # Mock HttpError with 403 quota exceeded
    mock_resp = MagicMock()
    mock_resp.status = 403
    error = HttpError(
        mock_resp, b'{"error": {"code": 403, "message": "quotaExceeded"}}'
    )

    mock_insert.execute.side_effect = error

    result = generator.add_video_to_playlist("PLtest123", "fJ9rUzIMcZQ")

    assert result is False
    # Should only be called once - no retries for quota exceeded
    assert mock_insert.execute.call_count == 1


def test_find_playlist_by_title_found():
    """Test finding an existing playlist by title."""
    generator = YouTubePlaylistGenerator()

    mock_youtube = MagicMock()
    mock_playlists = MagicMock()
    mock_list = MagicMock()

    mock_youtube.playlists.return_value = mock_playlists
    mock_playlists.list.return_value = mock_list
    mock_list.execute.return_value = {
        "items": [
            {"id": "PL123", "snippet": {"title": "My Playlist"}},
            {"id": "PL456", "snippet": {"title": "Another Playlist"}},
        ]
    }

    generator.youtube = mock_youtube

    playlist_id = generator.find_playlist_by_title("My Playlist")

    assert playlist_id == "PL123"


def test_find_playlist_by_title_not_found():
    """Test when playlist is not found."""
    generator = YouTubePlaylistGenerator()

    mock_youtube = MagicMock()
    mock_playlists = MagicMock()
    mock_list = MagicMock()

    mock_youtube.playlists.return_value = mock_playlists
    mock_playlists.list.return_value = mock_list
    mock_list.execute.return_value = {
        "items": [
            {"id": "PL123", "snippet": {"title": "Different Playlist"}},
        ]
    }

    generator.youtube = mock_youtube

    playlist_id = generator.find_playlist_by_title("My Playlist")

    assert playlist_id is None


def test_extract_video_id_standard_url():
    """Test extracting video ID from standard YouTube URL."""
    video_id = YouTubePlaylistGenerator._extract_video_id(
        "https://www.youtube.com/watch?v=fJ9rUzIMcZQ"
    )
    assert video_id == "fJ9rUzIMcZQ"


def test_extract_video_id_short_url():
    """Test extracting video ID from short YouTube URL."""
    video_id = YouTubePlaylistGenerator._extract_video_id(
        "https://youtu.be/fJ9rUzIMcZQ"
    )
    assert video_id == "fJ9rUzIMcZQ"


def test_extract_video_id_music_url():
    """Test extracting video ID from YouTube Music URL."""
    video_id = YouTubePlaylistGenerator._extract_video_id(
        "https://music.youtube.com/watch?v=fJ9rUzIMcZQ"
    )
    assert video_id == "fJ9rUzIMcZQ"


def test_get_playlist_url():
    """Test playlist URL generation."""
    url = YouTubePlaylistGenerator.get_playlist_url("PLtest123")
    assert url == "https://music.youtube.com/playlist?list=PLtest123"


@patch("time.sleep")  # Mock sleep to speed up tests
def test_create_playlist_from_urls(mock_sleep, sample_tracks_with_urls):
    """Test creating playlist directly from YouTube URLs."""
    generator = YouTubePlaylistGenerator()

    # Mock YouTube API
    mock_youtube = MagicMock()
    generator.youtube = mock_youtube

    # Mock find_playlist_by_title - returns None (no existing playlist)
    mock_playlists_list = MagicMock()
    mock_playlists_list.execute.return_value = {"items": []}
    mock_youtube.playlists.return_value.list.return_value = mock_playlists_list

    # Mock create_playlist
    mock_playlists_insert = MagicMock()
    mock_playlists_insert.execute.return_value = {"id": "PLnew123"}
    mock_youtube.playlists.return_value.insert.return_value = mock_playlists_insert

    # Mock add_video_to_playlist
    mock_playlist_items_insert = MagicMock()
    mock_playlist_items_insert.execute.return_value = {}
    mock_youtube.playlistItems.return_value.insert.return_value = (
        mock_playlist_items_insert
    )

    result = generator.create_playlist_from_tracks(
        tracks_df=sample_tracks_with_urls,
        playlist_title="Test Playlist",
        playlist_description="Test",
        privacy_status="unlisted",
    )

    assert result is not None
    assert result["playlist_id"] == "PLnew123"
    assert result["tracks_added"] == 3
    assert len(result["tracks_not_found"]) == 0

    # Should call insert 3 times (once per track)
    assert mock_playlist_items_insert.execute.call_count == 3


@patch("time.sleep")  # Mock sleep to speed up tests
def test_create_playlist_from_search(mock_sleep, sample_tracks):
    """Test creating playlist by searching for tracks."""
    generator = YouTubePlaylistGenerator()

    # Mock YouTube API
    mock_youtube = MagicMock()
    generator.youtube = mock_youtube

    # Mock find_playlist_by_title - returns None (no existing playlist)
    mock_playlists_list = MagicMock()
    mock_playlists_list.execute.return_value = {"items": []}
    mock_youtube.playlists.return_value.list.return_value = mock_playlists_list

    # Mock create_playlist
    mock_playlists_insert = MagicMock()
    mock_playlists_insert.execute.return_value = {"id": "PLnew123"}
    mock_youtube.playlists.return_value.insert.return_value = mock_playlists_insert

    # Mock search_track
    mock_search_list = MagicMock()
    mock_search_list.execute.return_value = {
        "items": [
            {
                "id": {"videoId": "test_vid"},
                "snippet": {"title": "Track", "channelTitle": "Artist - Topic"},
            }
        ]
    }
    mock_youtube.search.return_value.list.return_value = mock_search_list

    # Mock add_video_to_playlist
    mock_playlist_items_insert = MagicMock()
    mock_playlist_items_insert.execute.return_value = {}
    mock_youtube.playlistItems.return_value.insert.return_value = (
        mock_playlist_items_insert
    )

    result = generator.create_playlist_from_tracks(
        tracks_df=sample_tracks,
        playlist_title="Test Playlist",
        playlist_description="Test",
        privacy_status="unlisted",
    )

    assert result is not None
    assert result["playlist_id"] == "PLnew123"
    assert result["tracks_added"] == 3
    assert len(result["tracks_not_found"]) == 0

    # Should search 3 times and insert 3 times
    assert mock_search_list.execute.call_count == 3
    assert mock_playlist_items_insert.execute.call_count == 3


@patch("time.sleep")  # Mock sleep to speed up tests
def test_create_playlist_updates_existing(mock_sleep, sample_tracks_with_urls):
    """Test that existing playlist is deleted and recreated."""
    generator = YouTubePlaylistGenerator()

    # Mock YouTube API
    mock_youtube = MagicMock()
    generator.youtube = mock_youtube

    # Mock find_playlist_by_title - returns existing playlist
    mock_playlists_list = MagicMock()
    mock_playlists_list.execute.return_value = {
        "items": [{"id": "PLexisting", "snippet": {"title": "Test Playlist"}}]
    }
    mock_youtube.playlists.return_value.list.return_value = mock_playlists_list

    # Mock delete_playlist
    mock_playlists_delete = MagicMock()
    mock_playlists_delete.execute.return_value = {}
    mock_youtube.playlists.return_value.delete.return_value = mock_playlists_delete

    # Mock create_playlist
    mock_playlists_insert = MagicMock()
    mock_playlists_insert.execute.return_value = {"id": "PLnew123"}
    mock_youtube.playlists.return_value.insert.return_value = mock_playlists_insert

    # Mock add_video_to_playlist
    mock_playlist_items_insert = MagicMock()
    mock_playlist_items_insert.execute.return_value = {}
    mock_youtube.playlistItems.return_value.insert.return_value = (
        mock_playlist_items_insert
    )

    result = generator.create_playlist_from_tracks(
        tracks_df=sample_tracks_with_urls,
        playlist_title="Test Playlist",
        playlist_description="Test",
        privacy_status="unlisted",
    )

    assert result is not None
    # Should delete existing and create new one
    mock_playlists_delete.execute.assert_called_once()
    assert result["playlist_id"] == "PLnew123"
