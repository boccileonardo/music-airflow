"""
Pytest configuration and fixtures for Last.fm testing.

Provides mock API responses and common test data.
"""

import datetime as dt
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def sample_tracks_response() -> dict[str, Any]:
    """Sample Last.fm API response for user.getrecenttracks."""
    return {
        "recenttracks": {
            "@attr": {
                "user": "testuser",
                "totalPages": "1",
                "page": "1",
                "perPage": "200",
                "total": "3",
            },
            "track": [
                {
                    "artist": {
                        "mbid": "a74b1b7f-71a5-4011-9441-d0b5e4122711",
                        "name": "Radiohead",
                    },
                    "mbid": "6b9a509f-6907-4a6e-9345-2f12da09ba4b",
                    "name": "Creep",
                    "album": {
                        "mbid": "b2c9e8b0-5a8c-4e76-8b1f-3f9b0c8f1b0e",
                        "#text": "Pablo Honey",
                    },
                    "url": "https://www.last.fm/music/Radiohead/_/Creep",
                    "date": {
                        "uts": "1609459200",  # 2021-01-01 00:00:00 UTC
                        "#text": "01 Jan 2021, 00:00",
                    },
                },
                {
                    "artist": {
                        "mbid": "",
                        "name": "The Beatles",
                    },
                    "mbid": "",
                    "name": "Yesterday",
                    "album": {
                        "mbid": "",
                        "#text": "Help!",
                    },
                    "url": "https://www.last.fm/music/The+Beatles/_/Yesterday",
                    "date": {
                        "uts": "1609462800",  # 2021-01-01 01:00:00 UTC
                        "#text": "01 Jan 2021, 01:00",
                    },
                },
                {
                    "artist": {
                        "mbid": "ba0d6274-db14-4ef5-b28d-657ebde1a396",
                        "name": "The Rolling Stones",
                    },
                    "mbid": "e4feb630-fe7c-4f8d-8a72-05b65e2e51b0",
                    "name": "Paint It Black",
                    "album": {
                        "mbid": "c1b0e8b0-5a8c-4e76-8b1f-3f9b0c8f1b0e",
                        "#text": "Aftermath",
                    },
                    "url": "https://www.last.fm/music/The+Rolling+Stones/_/Paint+It+Black",
                    "date": {
                        "uts": "1609466400",  # 2021-01-01 02:00:00 UTC
                        "#text": "01 Jan 2021, 02:00",
                    },
                },
            ],
        }
    }


@pytest.fixture
def sample_single_track_response() -> dict[str, Any]:
    """Sample Last.fm API response with a single track (dict, not list)."""
    return {
        "recenttracks": {
            "@attr": {
                "user": "testuser",
                "totalPages": "1",
                "page": "1",
                "perPage": "200",
                "total": "1",
            },
            "track": {
                "artist": {
                    "mbid": "a74b1b7f-71a5-4011-9441-d0b5e4122711",
                    "name": "Radiohead",
                },
                "mbid": "6b9a509f-6907-4a6e-9345-2f12da09ba4b",
                "name": "Creep",
                "album": {
                    "mbid": "b2c9e8b0-5a8c-4e76-8b1f-3f9b0c8f1b0e",
                    "#text": "Pablo Honey",
                },
                "url": "https://www.last.fm/music/Radiohead/_/Creep",
                "date": {
                    "uts": "1609459200",
                    "#text": "01 Jan 2021, 00:00",
                },
            },
        }
    }


@pytest.fixture
def sample_now_playing_response() -> dict[str, Any]:
    """Sample Last.fm API response with a 'now playing' track (should be filtered)."""
    return {
        "recenttracks": {
            "@attr": {
                "user": "testuser",
                "totalPages": "1",
                "page": "1",
                "perPage": "200",
                "total": "2",
            },
            "track": [
                {
                    "artist": {
                        "mbid": "a74b1b7f-71a5-4011-9441-d0b5e4122711",
                        "name": "Radiohead",
                    },
                    "mbid": "6b9a509f-6907-4a6e-9345-2f12da09ba4b",
                    "name": "No Surprises",
                    "album": {
                        "mbid": "b2c9e8b0-5a8c-4e76-8b1f-3f9b0c8f1b0e",
                        "#text": "OK Computer",
                    },
                    "url": "https://www.last.fm/music/Radiohead/_/No+Surprises",
                    "@attr": {"nowplaying": "true"},
                    # No "date" field for now playing tracks
                },
                {
                    "artist": {
                        "mbid": "a74b1b7f-71a5-4011-9441-d0b5e4122711",
                        "name": "Radiohead",
                    },
                    "mbid": "6b9a509f-6907-4a6e-9345-2f12da09ba4b",
                    "name": "Creep",
                    "album": {
                        "mbid": "b2c9e8b0-5a8c-4e76-8b1f-3f9b0c8f1b0e",
                        "#text": "Pablo Honey",
                    },
                    "url": "https://www.last.fm/music/Radiohead/_/Creep",
                    "date": {
                        "uts": "1609459200",
                        "#text": "01 Jan 2021, 00:00",
                    },
                },
            ],
        }
    }


@pytest.fixture
def sample_empty_response() -> dict[str, Any]:
    """Sample Last.fm API response with no tracks."""
    return {
        "recenttracks": {
            "@attr": {
                "user": "testuser",
                "totalPages": "0",
                "page": "1",
                "perPage": "200",
                "total": "0",
            },
            "track": [],
        }
    }


@pytest.fixture
def sample_paginated_response_page1() -> dict[str, Any]:
    """Sample Last.fm API response for pagination test - page 1."""
    return {
        "recenttracks": {
            "@attr": {
                "user": "testuser",
                "totalPages": "2",
                "page": "1",
                "perPage": "2",
                "total": "3",
            },
            "track": [
                {
                    "artist": {"mbid": "artist1", "name": "Artist 1"},
                    "mbid": "track1",
                    "name": "Track 1",
                    "album": {"mbid": "album1", "#text": "Album 1"},
                    "url": "https://www.last.fm/music/Artist+1/_/Track+1",
                    "date": {"uts": "1609459200", "#text": "01 Jan 2021, 00:00"},
                },
                {
                    "artist": {"mbid": "artist2", "name": "Artist 2"},
                    "mbid": "track2",
                    "name": "Track 2",
                    "album": {"mbid": "album2", "#text": "Album 2"},
                    "url": "https://www.last.fm/music/Artist+2/_/Track+2",
                    "date": {"uts": "1609462800", "#text": "01 Jan 2021, 01:00"},
                },
            ],
        }
    }


@pytest.fixture
def sample_paginated_response_page2() -> dict[str, Any]:
    """Sample Last.fm API response for pagination test - page 2."""
    return {
        "recenttracks": {
            "@attr": {
                "user": "testuser",
                "totalPages": "2",
                "page": "2",
                "perPage": "2",
                "total": "3",
            },
            "track": [
                {
                    "artist": {"mbid": "artist3", "name": "Artist 3"},
                    "mbid": "track3",
                    "name": "Track 3",
                    "album": {"mbid": "album3", "#text": "Album 3"},
                    "url": "https://www.last.fm/music/Artist+3/_/Track+3",
                    "date": {"uts": "1609466400", "#text": "01 Jan 2021, 02:00"},
                },
            ],
        }
    }


@pytest.fixture
def sample_error_response() -> dict[str, Any]:
    """Sample Last.fm API error response."""
    return {
        "error": 6,
        "message": "User not found",
    }


@pytest.fixture
def sample_user_info() -> dict[str, Any]:
    """Sample Last.fm API response for user.getinfo."""
    return {
        "user": {
            "name": "testuser",
            "realname": "Test User",
            "url": "https://www.last.fm/user/testuser",
            "playcount": "12345",
            "registered": {"unixtime": "1234567890"},
        }
    }


@pytest.fixture
def test_data_dir(tmp_path: Path) -> Path:
    """Create a temporary data directory for testing."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "bronze").mkdir()
    (data_dir / "silver").mkdir()
    (data_dir / "gold").mkdir()
    return data_dir


@pytest.fixture
def sample_timestamp() -> dt.datetime:
    """Sample timestamp for testing."""
    return dt.datetime(2021, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc)
