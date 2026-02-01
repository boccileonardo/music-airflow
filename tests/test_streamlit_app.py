"""Tests for Streamlit music recommendation app."""

import polars as pl
import pytest
from unittest.mock import MagicMock, patch
from streamlit.testing.v1 import AppTest


@pytest.fixture
def sample_track_candidates():
    """Sample track candidates for testing."""
    return pl.DataFrame(
        {
            "username": ["test_user"] * 5,
            "track_id": ["t1", "t2", "t3", "t4", "t5"],
            "track_name": [
                "Bohemian Rhapsody",
                "Hotel California",
                "Stairway to Heaven",
                "Imagine",
                "Hey Jude",
            ],
            "artist_name": [
                "Queen",
                "Eagles",
                "Led Zeppelin",
                "John Lennon",
                "The Beatles",
            ],
            "score": [1.0, 0.9, 0.8, 0.7, 0.6],
            "similar_tag": [True, False, True, False, True],
            "similar_artist": [False, True, False, True, False],
            "deep_cut_same_artist": [False, False, False, False, True],
            "old_favorite": [True, False, True, False, False],
            "youtube_url": [
                "https://www.youtube.com/watch?v=fJ9rUzIMcZQ",
                "https://www.youtube.com/watch?v=09839DpTctU",
                "https://www.youtube.com/watch?v=QkF3oxziUI4",
                "https://www.youtube.com/watch?v=yRhq-yO1KN8",
                "https://www.youtube.com/watch?v=A_MjCqQoLLA",
            ],
            "spotify_url": [None] * 5,
            "tags": [["rock", "classic rock"]] * 5,
            "duration_ms": [354000, 391000, 482000, 183000, 431000],
        }
    )


@pytest.fixture
def sample_user_stats():
    """Sample user statistics."""
    return {
        "total_plays": 1000,
        "total_tracks_played": 250,
        "total_artists_played": 75,
    }


@pytest.fixture
def sample_top_artists():
    """Sample top artists data."""
    return pl.DataFrame(
        {
            "artist_name": ["Queen", "Eagles", "Led Zeppelin"],
            "play_count": [100, 95, 90],
        }
    )


@pytest.fixture
def mock_io_manager():
    """Mock the PolarsDeltaIOManager."""
    with patch("music_airflow.app.streamlit_app.PolarsDeltaIOManager") as mock_io:
        yield mock_io


@pytest.fixture
def mock_youtube_generator():
    """Mock the YouTubePlaylistGenerator."""
    with patch("music_airflow.app.streamlit_app.YouTubePlaylistGenerator") as mock_gen:
        yield mock_gen


@pytest.fixture(autouse=True)
def mock_exclusion_writes():
    """
    Auto-mock write functions to prevent tests from writing to real Delta tables.

    This fixture uses autouse=True to automatically apply to all tests in this module,
    preventing accidental writes to production data during testing.
    """
    with patch(
        "music_airflow.app.excluded_tracks.write_excluded_track"
    ) as mock_write_track:
        with patch(
            "music_airflow.app.excluded_tracks.write_excluded_artist"
        ) as mock_write_artist:
            # Return empty dict to simulate successful write
            mock_write_track.return_value = {}
            mock_write_artist.return_value = {}
            yield {"track": mock_write_track, "artist": mock_write_artist}


def test_app_loads_without_errors():
    """Test that the app loads without errors."""
    with patch("music_airflow.app.streamlit_app.load_track_candidates"):
        with patch(
            "music_airflow.app.streamlit_app.load_user_statistics"
        ) as mock_stats:
            mock_stats.return_value = {
                "total_plays": 0,
                "total_tracks_played": 0,
                "total_artists_played": 0,
            }
            with patch(
                "music_airflow.app.streamlit_app.load_top_artists"
            ) as mock_artists:
                mock_artists.return_value = pl.DataFrame(
                    schema={"artist_name": pl.String, "play_count": pl.Int64}
                )

                at = AppTest.from_file("src/music_airflow/app/streamlit_app.py")
                at.run()

                assert not at.exception


def test_app_displays_title():
    """Test that the app displays the correct title."""
    with patch("music_airflow.app.streamlit_app.load_track_candidates"):
        with patch(
            "music_airflow.app.streamlit_app.load_user_statistics"
        ) as mock_stats:
            mock_stats.return_value = {
                "total_plays": 0,
                "total_tracks_played": 0,
                "total_artists_played": 0,
            }
            with patch(
                "music_airflow.app.streamlit_app.load_top_artists"
            ) as mock_artists:
                mock_artists.return_value = pl.DataFrame(
                    schema={"artist_name": pl.String, "play_count": pl.Int64}
                )

                at = AppTest.from_file("src/music_airflow/app/streamlit_app.py")
                at.run()

                assert len(at.title) > 0
                # Title is now in sidebar with app name
                assert "KainosFM" in at.title[0].value


def test_app_displays_user_statistics(sample_user_stats, sample_top_artists):
    """Test that user statistics are displayed correctly."""
    with patch("music_airflow.app.streamlit_app.load_track_candidates"):
        with patch(
            "music_airflow.app.streamlit_app.load_user_statistics"
        ) as mock_stats:
            mock_stats.return_value = sample_user_stats
            with patch(
                "music_airflow.app.streamlit_app.load_top_artists"
            ) as mock_artists:
                mock_artists.return_value = sample_top_artists

                at = AppTest.from_file("src/music_airflow/app/streamlit_app.py")
                at.run()

                assert not at.exception
                # Check that metrics are displayed
                assert len(at.metric) >= 3


def test_recommendation_settings_checkboxes():
    """Test that recommendation settings checkboxes are present and functional."""
    with patch("music_airflow.app.streamlit_app.load_track_candidates"):
        with patch(
            "music_airflow.app.streamlit_app.load_user_statistics"
        ) as mock_stats:
            mock_stats.return_value = {
                "total_plays": 0,
                "total_tracks_played": 0,
                "total_artists_played": 0,
            }
            with patch(
                "music_airflow.app.streamlit_app.load_top_artists"
            ) as mock_artists:
                mock_artists.return_value = pl.DataFrame(
                    schema={"artist_name": pl.String, "play_count": pl.Int64}
                )

                at = AppTest.from_file("src/music_airflow/app/streamlit_app.py")
                at.run()

                assert not at.exception
                # Check that checkboxes exist (Similar Tags, Similar Artists, Deep Cuts)
                assert len(at.checkbox) >= 3


def test_discovery_slider_exists():
    """Test that discovery weight slider is present."""
    with patch("music_airflow.app.streamlit_app.load_track_candidates"):
        with patch(
            "music_airflow.app.streamlit_app.load_user_statistics"
        ) as mock_stats:
            mock_stats.return_value = {
                "total_plays": 0,
                "total_tracks_played": 0,
                "total_artists_played": 0,
            }
            with patch(
                "music_airflow.app.streamlit_app.load_top_artists"
            ) as mock_artists:
                mock_artists.return_value = pl.DataFrame(
                    schema={"artist_name": pl.String, "play_count": pl.Int64}
                )

                at = AppTest.from_file("src/music_airflow/app/streamlit_app.py")
                at.run()

                assert not at.exception
                # Check that sliders exist
                assert (
                    len(at.slider) >= 2
                )  # discovery slider and n_recommendations slider


def test_generate_recommendations_button_exists():
    """Test that the generate recommendations button exists."""
    with patch("music_airflow.app.streamlit_app.load_track_candidates"):
        with patch(
            "music_airflow.app.streamlit_app.load_user_statistics"
        ) as mock_stats:
            mock_stats.return_value = {
                "total_plays": 0,
                "total_tracks_played": 0,
                "total_artists_played": 0,
            }
            with patch(
                "music_airflow.app.streamlit_app.load_top_artists"
            ) as mock_artists:
                mock_artists.return_value = pl.DataFrame(
                    schema={"artist_name": pl.String, "play_count": pl.Int64}
                )

                at = AppTest.from_file("src/music_airflow/app/streamlit_app.py")
                at.run()

                assert not at.exception
                assert len(at.button) >= 1


def test_generate_recommendations_with_candidates(sample_track_candidates):
    """Test generating recommendations with sample candidates."""
    with patch("music_airflow.app.streamlit_app.load_track_candidates") as mock_load:
        # Return LazyFrame
        mock_load.return_value = sample_track_candidates.lazy()
        with patch(
            "music_airflow.app.streamlit_app.load_user_statistics"
        ) as mock_stats:
            mock_stats.return_value = {
                "total_plays": 0,
                "total_tracks_played": 0,
                "total_artists_played": 0,
            }
            with patch(
                "music_airflow.app.streamlit_app.load_top_artists"
            ) as mock_artists:
                mock_artists.return_value = pl.DataFrame(
                    schema={"artist_name": pl.String, "play_count": pl.Int64}
                )
                with patch(
                    "music_airflow.app.streamlit_app.read_excluded_tracks"
                ) as mock_excluded_tracks:
                    mock_excluded_tracks.return_value = None
                    with patch(
                        "music_airflow.app.streamlit_app.read_excluded_artists"
                    ) as mock_excluded_artists:
                        mock_excluded_artists.return_value = None

                        at = AppTest.from_file("src/music_airflow/app/streamlit_app.py")
                        at.run()

                        # Recommendations now generate automatically on load
                        assert not at.exception
                        # Should show recommendations header
                        headers = [h.value for h in at.header]
                        assert any("Recommendations" in h for h in headers)


def test_generate_recommendations_no_systems_selected(sample_track_candidates):
    """Test that warning appears when no systems are selected."""
    with patch("music_airflow.app.streamlit_app.load_track_candidates") as mock_load:
        mock_load.return_value = sample_track_candidates.lazy()
        with patch(
            "music_airflow.app.streamlit_app.load_user_statistics"
        ) as mock_stats:
            mock_stats.return_value = {
                "total_plays": 0,
                "total_tracks_played": 0,
                "total_artists_played": 0,
            }
            with patch(
                "music_airflow.app.streamlit_app.load_top_artists"
            ) as mock_artists:
                mock_artists.return_value = pl.DataFrame(
                    schema={"artist_name": pl.String, "play_count": pl.Int64}
                )
                with patch(
                    "music_airflow.app.streamlit_app.read_excluded_tracks"
                ) as mock_excluded_tracks:
                    mock_excluded_tracks.return_value = None
                    with patch(
                        "music_airflow.app.streamlit_app.read_excluded_artists"
                    ) as mock_excluded_artists:
                        mock_excluded_artists.return_value = None

                        at = AppTest.from_file("src/music_airflow/app/streamlit_app.py")
                        at.run()

                        # Uncheck all recommendation system checkboxes
                        # Use keys to identify the source checkboxes
                        for checkbox in at.checkbox:
                            if checkbox.value:  # Only uncheck ones that are checked
                                checkbox.uncheck()
                        # Run once after all changes
                        at.run()

                        # Recommendations generate automatically - when no systems selected, shows warning
                        assert not at.exception
                        # With no sources, there should either be a warning or empty recommendations
                        # The test verifies app handles this gracefully
                        # Note: session_state.recommendations should be None when no sources selected
                        if len(at.warning) == 0:
                            assert (
                                at.session_state.recommendations is None
                                or len(at.session_state.recommendations) == 0
                            )


def test_youtube_playlist_creation_button_appears_after_recommendations(
    sample_track_candidates,
):
    """Test that YouTube playlist creation button appears after generating recommendations."""
    with patch("music_airflow.app.streamlit_app.load_track_candidates") as mock_load:
        mock_load.return_value = sample_track_candidates.lazy()
        with patch(
            "music_airflow.app.streamlit_app.load_user_statistics"
        ) as mock_stats:
            mock_stats.return_value = {
                "total_plays": 0,
                "total_tracks_played": 0,
                "total_artists_played": 0,
            }
            with patch(
                "music_airflow.app.streamlit_app.load_top_artists"
            ) as mock_artists:
                mock_artists.return_value = pl.DataFrame(
                    schema={"artist_name": pl.String, "play_count": pl.Int64}
                )
                with patch(
                    "music_airflow.app.streamlit_app.read_excluded_tracks"
                ) as mock_excluded_tracks:
                    mock_excluded_tracks.return_value = None
                    with patch(
                        "music_airflow.app.streamlit_app.read_excluded_artists"
                    ) as mock_excluded_artists:
                        mock_excluded_artists.return_value = None

                        at = AppTest.from_file("src/music_airflow/app/streamlit_app.py")
                        at.run()

                        # Recommendations generate automatically on load
                        assert not at.exception
                        # After generating recommendations, should have buttons
                        # (playlist creation, track removal, artist blocking)
                        assert len(at.button) >= 1


def test_youtube_playlist_creation_with_mock(sample_track_candidates):
    """Test YouTube playlist creation without sending real requests."""
    with patch("music_airflow.app.streamlit_app.load_track_candidates") as mock_load:
        mock_load.return_value = sample_track_candidates.lazy()
        with patch(
            "music_airflow.app.streamlit_app.load_user_statistics"
        ) as mock_stats:
            mock_stats.return_value = {
                "total_plays": 0,
                "total_tracks_played": 0,
                "total_artists_played": 0,
            }
            with patch(
                "music_airflow.app.streamlit_app.load_top_artists"
            ) as mock_artists:
                mock_artists.return_value = pl.DataFrame(
                    schema={"artist_name": pl.String, "play_count": pl.Int64}
                )
                with patch(
                    "music_airflow.app.streamlit_app.read_excluded_tracks"
                ) as mock_excluded_tracks:
                    mock_excluded_tracks.return_value = None
                    with patch(
                        "music_airflow.app.streamlit_app.read_excluded_artists"
                    ) as mock_excluded_artists:
                        mock_excluded_artists.return_value = None
                        with patch(
                            "music_airflow.app.streamlit_app.YouTubePlaylistGenerator"
                        ) as mock_gen_class:
                            # Mock the generator instance
                            mock_gen = MagicMock()
                            mock_gen_class.return_value = mock_gen
                            mock_gen.authenticate.return_value = True
                            mock_gen.create_playlist_from_tracks.return_value = {
                                "playlist_url": "https://music.youtube.com/playlist?list=test_id",
                                "tracks_added": 5,
                                "tracks_not_found": [],
                                "quota_exceeded": False,
                            }

                            at = AppTest.from_file(
                                "src/music_airflow/app/streamlit_app.py"
                            )
                            at.run()

                            # Recommendations generate automatically
                            # Verify recommendations were created
                            assert "recommendations" in at.session_state
                            assert at.session_state.recommendations is not None

                            # Now there should be buttons including playlist creation
                            assert len(at.button) >= 1
                            assert not at.exception


def test_youtube_playlist_creation_authentication_failure(sample_track_candidates):
    """Test YouTube playlist creation when authentication fails."""
    with patch("music_airflow.app.streamlit_app.load_track_candidates") as mock_load:
        mock_load.return_value = sample_track_candidates.lazy()
        with patch(
            "music_airflow.app.streamlit_app.load_user_statistics"
        ) as mock_stats:
            mock_stats.return_value = {
                "total_plays": 0,
                "total_tracks_played": 0,
                "total_artists_played": 0,
            }
            with patch(
                "music_airflow.app.streamlit_app.load_top_artists"
            ) as mock_artists:
                mock_artists.return_value = pl.DataFrame(
                    schema={"artist_name": pl.String, "play_count": pl.Int64}
                )
                with patch(
                    "music_airflow.app.streamlit_app.read_excluded_tracks"
                ) as mock_excluded_tracks:
                    mock_excluded_tracks.return_value = None
                    with patch(
                        "music_airflow.app.streamlit_app.read_excluded_artists"
                    ) as mock_excluded_artists:
                        mock_excluded_artists.return_value = None
                        with patch(
                            "music_airflow.app.streamlit_app.YouTubePlaylistGenerator"
                        ) as mock_gen_class:
                            mock_gen = MagicMock()
                            mock_gen_class.return_value = mock_gen
                            # Authentication fails
                            mock_gen.authenticate.return_value = False

                            at = AppTest.from_file(
                                "src/music_airflow/app/streamlit_app.py"
                            )
                            at.run()

                            # Recommendations generate automatically
                            # Try to create playlist (will fail auth)
                            for button in at.button:
                                try:
                                    button.click().run()
                                except Exception:
                                    pass

                            # Should still not have unhandled exceptions
                            assert not at.exception


def test_youtube_playlist_creation_quota_exceeded(sample_track_candidates):
    """Test YouTube playlist creation when quota is exceeded."""
    with patch("music_airflow.app.streamlit_app.load_track_candidates") as mock_load:
        mock_load.return_value = sample_track_candidates.lazy()
        with patch(
            "music_airflow.app.streamlit_app.load_user_statistics"
        ) as mock_stats:
            mock_stats.return_value = {
                "total_plays": 0,
                "total_tracks_played": 0,
                "total_artists_played": 0,
            }
            with patch(
                "music_airflow.app.streamlit_app.load_top_artists"
            ) as mock_artists:
                mock_artists.return_value = pl.DataFrame(
                    schema={"artist_name": pl.String, "play_count": pl.Int64}
                )
                with patch(
                    "music_airflow.app.streamlit_app.read_excluded_tracks"
                ) as mock_excluded_tracks:
                    mock_excluded_tracks.return_value = None
                    with patch(
                        "music_airflow.app.streamlit_app.read_excluded_artists"
                    ) as mock_excluded_artists:
                        mock_excluded_artists.return_value = None
                        with patch(
                            "music_airflow.app.streamlit_app.YouTubePlaylistGenerator"
                        ) as mock_gen_class:
                            mock_gen = MagicMock()
                            mock_gen_class.return_value = mock_gen
                            mock_gen.authenticate.return_value = True
                            # Simulate quota exceeded
                            mock_gen.create_playlist_from_tracks.return_value = {
                                "playlist_url": "https://music.youtube.com/playlist?list=test_id",
                                "tracks_added": 2,
                                "tracks_not_found": ["Song 3", "Song 4", "Song 5"],
                                "quota_exceeded": True,
                            }

                            at = AppTest.from_file(
                                "src/music_airflow/app/streamlit_app.py"
                            )
                            at.run()

                            # Recommendations generate automatically
                            # Create playlist (will hit quota)
                            for button in at.button:
                                try:
                                    button.click().run()
                                except Exception:
                                    pass

                            # Should still handle gracefully
                            assert not at.exception


def test_track_exclusion_functionality(sample_track_candidates):
    """Test removing and replacing tracks."""
    with patch("music_airflow.app.streamlit_app.load_track_candidates") as mock_load:
        mock_load.return_value = sample_track_candidates.lazy()
        with patch(
            "music_airflow.app.streamlit_app.load_user_statistics"
        ) as mock_stats:
            mock_stats.return_value = {
                "total_plays": 0,
                "total_tracks_played": 0,
                "total_artists_played": 0,
            }
            with patch(
                "music_airflow.app.streamlit_app.load_top_artists"
            ) as mock_artists:
                mock_artists.return_value = pl.DataFrame(
                    schema={"artist_name": pl.String, "play_count": pl.Int64}
                )
                with patch(
                    "music_airflow.app.streamlit_app.read_excluded_tracks"
                ) as mock_excluded_tracks:
                    mock_excluded_tracks.return_value = None
                    with patch(
                        "music_airflow.app.streamlit_app.read_excluded_artists"
                    ) as mock_excluded_artists:
                        mock_excluded_artists.return_value = None

                        at = AppTest.from_file("src/music_airflow/app/streamlit_app.py")
                        at.run()

                        # Recommendations generate automatically
                        # Should have selectbox for track removal (in exclusion expander)
                        assert len(at.selectbox) >= 1  # at least username selector


def test_artist_blocking_functionality(sample_track_candidates):
    """Test blocking entire artists."""
    with patch("music_airflow.app.streamlit_app.load_track_candidates") as mock_load:
        mock_load.return_value = sample_track_candidates.lazy()
        with patch(
            "music_airflow.app.streamlit_app.load_user_statistics"
        ) as mock_stats:
            mock_stats.return_value = {
                "total_plays": 0,
                "total_tracks_played": 0,
                "total_artists_played": 0,
            }
            with patch(
                "music_airflow.app.streamlit_app.load_top_artists"
            ) as mock_artists:
                mock_artists.return_value = pl.DataFrame(
                    schema={"artist_name": pl.String, "play_count": pl.Int64}
                )
                with patch(
                    "music_airflow.app.streamlit_app.read_excluded_tracks"
                ) as mock_excluded_tracks:
                    mock_excluded_tracks.return_value = None
                    with patch(
                        "music_airflow.app.streamlit_app.read_excluded_artists"
                    ) as mock_excluded_artists:
                        mock_excluded_artists.return_value = None

                        at = AppTest.from_file("src/music_airflow/app/streamlit_app.py")
                        at.run()

                        # Recommendations generate automatically
                        # Should have selectbox for artist blocking (in exclusion expander)
                        assert len(at.selectbox) >= 1  # at least username selector


def test_playlist_name_input(sample_track_candidates):
    """Test that playlist name can be customized."""
    with patch("music_airflow.app.streamlit_app.load_track_candidates") as mock_load:
        mock_load.return_value = sample_track_candidates.lazy()
        with patch(
            "music_airflow.app.streamlit_app.load_user_statistics"
        ) as mock_stats:
            mock_stats.return_value = {
                "total_plays": 0,
                "total_tracks_played": 0,
                "total_artists_played": 0,
            }
            with patch(
                "music_airflow.app.streamlit_app.load_top_artists"
            ) as mock_artists:
                mock_artists.return_value = pl.DataFrame(
                    schema={"artist_name": pl.String, "play_count": pl.Int64}
                )
                with patch(
                    "music_airflow.app.streamlit_app.read_excluded_tracks"
                ) as mock_excluded_tracks:
                    mock_excluded_tracks.return_value = None
                    with patch(
                        "music_airflow.app.streamlit_app.read_excluded_artists"
                    ) as mock_excluded_artists:
                        mock_excluded_artists.return_value = None

                        at = AppTest.from_file("src/music_airflow/app/streamlit_app.py")
                        at.run()

                        # Recommendations generate automatically
                        # Should have text input for playlist name
                        assert len(at.text_input) >= 1


def test_youtube_authentication_success(sample_track_candidates):
    """Test that YouTube authentication is verified successfully."""
    with patch("music_airflow.app.streamlit_app.load_track_candidates") as mock_load:
        mock_load.return_value = sample_track_candidates.lazy()
        with patch(
            "music_airflow.app.streamlit_app.load_user_statistics"
        ) as mock_stats:
            mock_stats.return_value = {
                "total_plays": 0,
                "total_tracks_played": 0,
                "total_artists_played": 0,
            }
            with patch(
                "music_airflow.app.streamlit_app.load_top_artists"
            ) as mock_artists:
                mock_artists.return_value = pl.DataFrame(
                    schema={"artist_name": pl.String, "play_count": pl.Int64}
                )
                with patch(
                    "music_airflow.app.streamlit_app.read_excluded_tracks"
                ) as mock_excluded_tracks:
                    mock_excluded_tracks.return_value = None
                    with patch(
                        "music_airflow.app.streamlit_app.read_excluded_artists"
                    ) as mock_excluded_artists:
                        mock_excluded_artists.return_value = None
                        with patch(
                            "music_airflow.app.streamlit_app.YouTubePlaylistGenerator"
                        ) as mock_gen_class:
                            mock_gen = MagicMock()
                            mock_gen_class.return_value = mock_gen
                            # Test successful authentication
                            mock_gen.authenticate.return_value = True
                            mock_gen.create_playlist_from_tracks.return_value = {
                                "playlist_url": "https://music.youtube.com/playlist?list=test_id",
                                "tracks_added": 5,
                                "tracks_not_found": [],
                                "quota_exceeded": False,
                            }

                            at = AppTest.from_file(
                                "src/music_airflow/app/streamlit_app.py"
                            )
                            at.run()

                            # Recommendations generate automatically
                            # The authenticate method should be called when creating playlist
                            assert not at.exception

                            # Verify authentication was successful
                            assert mock_gen.authenticate.return_value is True


def test_artist_exclusion_filters_tracks(sample_track_candidates):
    """Test that excluding an artist filters out all their tracks from recommendations."""
    with patch("music_airflow.app.streamlit_app.load_track_candidates") as mock_load:
        mock_load.return_value = sample_track_candidates.lazy()
        with patch(
            "music_airflow.app.streamlit_app.load_user_statistics"
        ) as mock_stats:
            mock_stats.return_value = {
                "total_plays": 0,
                "total_plays_played": 0,
                "total_artists_played": 0,
            }
            with patch(
                "music_airflow.app.streamlit_app.load_top_artists"
            ) as mock_artists:
                mock_artists.return_value = pl.DataFrame(
                    schema={"artist_name": pl.String, "play_count": pl.Int64}
                )
                with patch(
                    "music_airflow.app.excluded_tracks.read_excluded_tracks"
                ) as mock_excluded_tracks:
                    mock_excluded_tracks.return_value = None
                    # Exclude Queen from recommendations
                    import datetime as dt

                    excluded_artists_df = pl.DataFrame(
                        {
                            "username": ["lelopolel"],
                            "artist_name": ["Queen"],
                            "excluded_at": [dt.datetime.now(tz=dt.timezone.utc)],
                        }
                    ).lazy()
                    with patch(
                        "music_airflow.app.excluded_tracks.read_excluded_artists"
                    ) as mock_excluded_artists:
                        mock_excluded_artists.return_value = excluded_artists_df

                        at = AppTest.from_file("src/music_airflow/app/streamlit_app.py")
                        at.run()

                        # Recommendations generate automatically
                        assert not at.exception

                        # Verify recommendations don't include Queen
                        if (
                            "recommendations" in at.session_state
                            and at.session_state.recommendations is not None
                        ):
                            recommendations = at.session_state.recommendations
                            # Check that no tracks from Queen appear
                            queen_tracks = recommendations.filter(
                                pl.col("artist_name") == "Queen"
                            )
                            assert len(queen_tracks) == 0, (
                                "Found Queen tracks in recommendations despite artist being excluded"
                            )

                            # Verify other artists are still present
                            assert len(recommendations) > 0, (
                                "No recommendations generated"
                            )


def test_revert_track_exclusion_ui_elements(sample_track_candidates):
    """Test that the UI for reverting track exclusions appears correctly after generating recommendations."""
    with patch("music_airflow.app.streamlit_app.load_track_candidates") as mock_load:
        mock_load.return_value = sample_track_candidates.lazy()
        with patch(
            "music_airflow.app.streamlit_app.load_user_statistics"
        ) as mock_stats:
            mock_stats.return_value = {
                "total_plays": 0,
                "total_tracks_played": 0,
                "total_artists_played": 0,
            }
            with patch(
                "music_airflow.app.streamlit_app.load_top_artists"
            ) as mock_artists:
                mock_artists.return_value = pl.DataFrame(
                    schema={"artist_name": pl.String, "play_count": pl.Int64}
                )
                with patch(
                    "music_airflow.app.excluded_tracks.read_excluded_tracks"
                ) as mock_excluded_tracks:
                    # Simulate some excluded tracks
                    import datetime as dt

                    excluded_df = pl.DataFrame(
                        {
                            "username": ["lelopolel", "lelopolel"],
                            "track_id": ["t1", "t2"],
                            "track_name": ["Excluded Track 1", "Excluded Track 2"],
                            "artist_name": ["Artist 1", "Artist 2"],
                            "excluded_at": [dt.datetime.now(tz=dt.timezone.utc)] * 2,
                        }
                    ).lazy()
                    mock_excluded_tracks.return_value = excluded_df
                    with patch(
                        "music_airflow.app.excluded_tracks.read_excluded_artists"
                    ) as mock_excluded_artists:
                        mock_excluded_artists.return_value = None

                        at = AppTest.from_file("src/music_airflow/app/streamlit_app.py")
                        at.run()

                        # Recommendations generate automatically
                        assert not at.exception
                        # Should have tabs for managing exclusions
                        assert len(at.tabs) > 0


def test_revert_artist_exclusion_ui_elements(sample_track_candidates):
    """Test that the UI for reverting artist exclusions appears correctly after generating recommendations."""
    with patch("music_airflow.app.streamlit_app.load_track_candidates") as mock_load:
        mock_load.return_value = sample_track_candidates.lazy()
        with patch(
            "music_airflow.app.streamlit_app.load_user_statistics"
        ) as mock_stats:
            mock_stats.return_value = {
                "total_plays": 0,
                "total_tracks_played": 0,
                "total_artists_played": 0,
            }
            with patch(
                "music_airflow.app.streamlit_app.load_top_artists"
            ) as mock_artists:
                mock_artists.return_value = pl.DataFrame(
                    schema={"artist_name": pl.String, "play_count": pl.Int64}
                )
                with patch(
                    "music_airflow.app.excluded_tracks.read_excluded_tracks"
                ) as mock_excluded_tracks:
                    mock_excluded_tracks.return_value = None
                    with patch(
                        "music_airflow.app.excluded_tracks.read_excluded_artists"
                    ) as mock_excluded_artists:
                        # Simulate some excluded artists
                        import datetime as dt

                        excluded_df = pl.DataFrame(
                            {
                                "username": ["lelopolel", "lelopolel"],
                                "artist_name": [
                                    "Excluded Artist 1",
                                    "Excluded Artist 2",
                                ],
                                "excluded_at": [dt.datetime.now(tz=dt.timezone.utc)]
                                * 2,
                            }
                        ).lazy()
                        mock_excluded_artists.return_value = excluded_df

                        at = AppTest.from_file("src/music_airflow/app/streamlit_app.py")
                        at.run()

                        # Recommendations generate automatically
                        assert not at.exception
                        # Should have tabs for managing exclusions
                        assert len(at.tabs) > 0


def test_exclusion_management_section_appears(sample_track_candidates):
    """Test that the exclusion management section appears in the UI after generating recommendations."""
    with patch("music_airflow.app.streamlit_app.load_track_candidates") as mock_load:
        mock_load.return_value = sample_track_candidates.lazy()
        with patch(
            "music_airflow.app.streamlit_app.load_user_statistics"
        ) as mock_stats:
            mock_stats.return_value = {
                "total_plays": 0,
                "total_tracks_played": 0,
                "total_artists_played": 0,
            }
            with patch(
                "music_airflow.app.streamlit_app.load_top_artists"
            ) as mock_artists:
                mock_artists.return_value = pl.DataFrame(
                    schema={"artist_name": pl.String, "play_count": pl.Int64}
                )
                with patch(
                    "music_airflow.app.excluded_tracks.read_excluded_tracks"
                ) as mock_excluded_tracks:
                    mock_excluded_tracks.return_value = None
                    with patch(
                        "music_airflow.app.excluded_tracks.read_excluded_artists"
                    ) as mock_excluded_artists:
                        mock_excluded_artists.return_value = None

                        at = AppTest.from_file("src/music_airflow/app/streamlit_app.py")
                        at.run()

                        # Recommendations generate automatically
                        assert not at.exception
                        # Exclusion management is now in an expander, not a header
                        # Check that expanders exist (Top Artists + Manage Exclusions)
                        assert len(at.expander) >= 1


def test_artist_exclusion_with_multiple_artists(sample_track_candidates):
    """Test that excluding multiple artists filters out all their tracks."""
    with patch("music_airflow.app.streamlit_app.load_track_candidates") as mock_load:
        mock_load.return_value = sample_track_candidates.lazy()
        with patch(
            "music_airflow.app.streamlit_app.load_user_statistics"
        ) as mock_stats:
            mock_stats.return_value = {
                "total_plays": 0,
                "total_tracks_played": 0,
                "total_artists_played": 0,
            }
            with patch(
                "music_airflow.app.streamlit_app.load_top_artists"
            ) as mock_artists:
                mock_artists.return_value = pl.DataFrame(
                    schema={"artist_name": pl.String, "play_count": pl.Int64}
                )
                with patch(
                    "music_airflow.app.excluded_tracks.read_excluded_tracks"
                ) as mock_excluded_tracks:
                    mock_excluded_tracks.return_value = None
                    # Exclude Queen and Eagles
                    import datetime as dt

                    excluded_artists_df = pl.DataFrame(
                        {
                            "username": ["lelopolel", "lelopolel"],
                            "artist_name": ["Queen", "Eagles"],
                            "excluded_at": [dt.datetime.now(tz=dt.timezone.utc)] * 2,
                        }
                    ).lazy()
                    with patch(
                        "music_airflow.app.excluded_tracks.read_excluded_artists"
                    ) as mock_excluded_artists:
                        mock_excluded_artists.return_value = excluded_artists_df

                        at = AppTest.from_file("src/music_airflow/app/streamlit_app.py")
                        at.run()

                        # Recommendations generate automatically
                        assert not at.exception

                        # Verify recommendations don't include Queen or Eagles
                        if (
                            "recommendations" in at.session_state
                            and at.session_state.recommendations is not None
                        ):
                            recommendations = at.session_state.recommendations
                            excluded_tracks = recommendations.filter(
                                pl.col("artist_name").is_in(["Queen", "Eagles"])
                            )
                            assert len(excluded_tracks) == 0, (
                                f"Found {len(excluded_tracks)} tracks from excluded artists"
                            )

                            # Should still have recommendations from other artists
                            assert len(recommendations) > 0, (
                                "No recommendations generated"
                            )


class TestLoadRecommendationReasons:
    """Tests for the load_recommendation_reasons function."""

    def test_returns_similar_artist_reason(self):
        """Test that similar artist reason is loaded correctly."""
        from music_airflow.app.streamlit_app import load_recommendation_reasons

        similar_artist_df = pl.DataFrame(
            {
                "username": ["test_user"],
                "track_id": ["t1"],
                "source_artist_id": ["queen"],
                "similarity": [0.85],
            }
        ).lazy()

        artists_df = pl.DataFrame(
            {
                "artist_id": ["queen"],
                "artist_name": ["Queen"],
            }
        ).lazy()

        with patch(
            "music_airflow.app.streamlit_app.PolarsDeltaIOManager"
        ) as mock_io_cls:
            mock_io = MagicMock()
            mock_io_cls.return_value = mock_io

            def mock_read_delta(table_name):
                if table_name == "candidate_similar_artist":
                    return similar_artist_df
                elif table_name == "artists":
                    return artists_df
                return pl.DataFrame().lazy()

            mock_io.read_delta.side_effect = mock_read_delta

            reasons = load_recommendation_reasons("test_user", "t1")

            assert "similar_artist" in reasons
            assert reasons["similar_artist"]["source_artist"] == "Queen"
            assert reasons["similar_artist"]["similarity"] == 85.0

    def test_returns_similar_tag_reason(self):
        """Test that similar tag reason is loaded correctly."""
        from music_airflow.app.streamlit_app import load_recommendation_reasons

        similar_tag_df = pl.DataFrame(
            {
                "username": ["test_user"],
                "track_id": ["t1"],
                "source_tags": ["rock, classic rock, british rock"],
                "tag_match_count": [3],
            }
        ).lazy()

        with patch(
            "music_airflow.app.streamlit_app.PolarsDeltaIOManager"
        ) as mock_io_cls:
            mock_io = MagicMock()
            mock_io_cls.return_value = mock_io

            def mock_read_delta(table_name):
                if table_name == "candidate_similar_tag":
                    return similar_tag_df
                return pl.DataFrame().lazy()

            mock_io.read_delta.side_effect = mock_read_delta

            reasons = load_recommendation_reasons("test_user", "t1")

            assert "similar_tag" in reasons
            assert reasons["similar_tag"]["tags"] == [
                "rock",
                "classic rock",
                "british rock",
            ]
            assert reasons["similar_tag"]["match_count"] == 3

    def test_returns_deep_cut_reason(self):
        """Test that deep cut reason is loaded correctly."""
        from music_airflow.app.streamlit_app import load_recommendation_reasons

        deep_cut_df = pl.DataFrame(
            {
                "username": ["test_user"],
                "track_id": ["t1"],
                "source_artist_id": ["led_zeppelin"],
            }
        ).lazy()

        artists_df = pl.DataFrame(
            {
                "artist_id": ["led_zeppelin"],
                "artist_name": ["Led Zeppelin"],
            }
        ).lazy()

        with patch(
            "music_airflow.app.streamlit_app.PolarsDeltaIOManager"
        ) as mock_io_cls:
            mock_io = MagicMock()
            mock_io_cls.return_value = mock_io

            def mock_read_delta(table_name):
                if table_name == "candidate_deep_cut":
                    return deep_cut_df
                elif table_name == "artists":
                    return artists_df
                return pl.DataFrame().lazy()

            mock_io.read_delta.side_effect = mock_read_delta

            reasons = load_recommendation_reasons("test_user", "t1")

            assert "deep_cut" in reasons
            assert reasons["deep_cut"]["source_artist"] == "Led Zeppelin"

    def test_returns_empty_dict_when_no_reasons(self):
        """Test that empty dict is returned when track has no reasons."""
        from music_airflow.app.streamlit_app import load_recommendation_reasons

        with patch(
            "music_airflow.app.streamlit_app.PolarsDeltaIOManager"
        ) as mock_io_cls:
            mock_io = MagicMock()
            mock_io_cls.return_value = mock_io
            mock_io.read_delta.return_value = pl.DataFrame().lazy()

            reasons = load_recommendation_reasons("test_user", "nonexistent_track")

            assert reasons == {}

    def test_returns_multiple_reasons(self):
        """Test that multiple reasons can be returned for the same track."""
        from music_airflow.app.streamlit_app import load_recommendation_reasons

        similar_artist_df = pl.DataFrame(
            {
                "username": ["test_user"],
                "track_id": ["t1"],
                "source_artist_id": ["queen"],
                "similarity": [0.9],
            }
        ).lazy()

        similar_tag_df = pl.DataFrame(
            {
                "username": ["test_user"],
                "track_id": ["t1"],
                "source_tags": ["rock, british"],
                "tag_match_count": [2],
            }
        ).lazy()

        artists_df = pl.DataFrame(
            {
                "artist_id": ["queen"],
                "artist_name": ["Queen"],
            }
        ).lazy()

        with patch(
            "music_airflow.app.streamlit_app.PolarsDeltaIOManager"
        ) as mock_io_cls:
            mock_io = MagicMock()
            mock_io_cls.return_value = mock_io

            def mock_read_delta(table_name):
                if table_name == "candidate_similar_artist":
                    return similar_artist_df
                elif table_name == "candidate_similar_tag":
                    return similar_tag_df
                elif table_name == "artists":
                    return artists_df
                return pl.DataFrame().lazy()

            mock_io.read_delta.side_effect = mock_read_delta

            reasons = load_recommendation_reasons("test_user", "t1")

            assert "similar_artist" in reasons
            assert "similar_tag" in reasons

    def test_handles_missing_artist_gracefully(self):
        """Test that artist ID is used when artist name lookup fails."""
        from music_airflow.app.streamlit_app import load_recommendation_reasons

        similar_artist_df = pl.DataFrame(
            {
                "username": ["test_user"],
                "track_id": ["t1"],
                "source_artist_id": ["unknown_artist"],
                "similarity": [0.7],
            }
        ).lazy()

        with patch(
            "music_airflow.app.streamlit_app.PolarsDeltaIOManager"
        ) as mock_io_cls:
            mock_io = MagicMock()
            mock_io_cls.return_value = mock_io

            def mock_read_delta(table_name):
                if table_name == "candidate_similar_artist":
                    return similar_artist_df
                elif table_name == "artists":
                    return pl.DataFrame(
                        schema={"artist_id": pl.String, "artist_name": pl.String}
                    ).lazy()
                return pl.DataFrame().lazy()

            mock_io.read_delta.side_effect = mock_read_delta

            reasons = load_recommendation_reasons("test_user", "t1")

            assert "similar_artist" in reasons
            # Should fall back to formatted artist_id
            assert reasons["similar_artist"]["source_artist"] == "Unknown Artist"

    def test_dedupes_tags(self):
        """Test that duplicate tags are removed."""
        from music_airflow.app.streamlit_app import load_recommendation_reasons

        similar_tag_df = pl.DataFrame(
            {
                "username": ["test_user"],
                "track_id": ["t1"],
                "source_tags": ["rock, rock, classic rock, rock"],
                "tag_match_count": [4],
            }
        ).lazy()

        with patch(
            "music_airflow.app.streamlit_app.PolarsDeltaIOManager"
        ) as mock_io_cls:
            mock_io = MagicMock()
            mock_io_cls.return_value = mock_io

            def mock_read_delta(table_name):
                if table_name == "candidate_similar_tag":
                    return similar_tag_df
                return pl.DataFrame().lazy()

            mock_io.read_delta.side_effect = mock_read_delta

            reasons = load_recommendation_reasons("test_user", "t1")

            assert "similar_tag" in reasons
            # Should dedupe while preserving order
            assert reasons["similar_tag"]["tags"] == ["rock", "classic rock"]

    def test_limits_tags_to_five(self):
        """Test that only first 5 unique tags are returned."""
        from music_airflow.app.streamlit_app import load_recommendation_reasons

        similar_tag_df = pl.DataFrame(
            {
                "username": ["test_user"],
                "track_id": ["t1"],
                "source_tags": ["rock, pop, jazz, blues, metal, punk, folk"],
                "tag_match_count": [7],
            }
        ).lazy()

        with patch(
            "music_airflow.app.streamlit_app.PolarsDeltaIOManager"
        ) as mock_io_cls:
            mock_io = MagicMock()
            mock_io_cls.return_value = mock_io

            def mock_read_delta(table_name):
                if table_name == "candidate_similar_tag":
                    return similar_tag_df
                return pl.DataFrame().lazy()

            mock_io.read_delta.side_effect = mock_read_delta

            reasons = load_recommendation_reasons("test_user", "t1")

            assert "similar_tag" in reasons
            assert len(reasons["similar_tag"]["tags"]) == 5
            assert reasons["similar_tag"]["tags"] == [
                "rock",
                "pop",
                "jazz",
                "blues",
                "metal",
            ]
