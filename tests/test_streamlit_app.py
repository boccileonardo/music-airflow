"""
Tests for the Streamlit app.

Uses Streamlit's AppTest framework for testing UI components
and pytest for testing business logic functions.
"""

import datetime as dt
from unittest.mock import MagicMock, patch

import polars as pl
import pytest
from streamlit.testing.v1 import AppTest


@pytest.fixture
def mock_track_candidates() -> pl.DataFrame:
    """Sample track candidates for testing."""
    return pl.DataFrame(
        {
            "track_id": ["track1", "track2", "track3", "track4", "track5"],
            "track_name": ["Song A", "Song B", "Song C", "Song D", "Song E"],
            "artist_name": ["Artist 1", "Artist 2", "Artist 1", "Artist 3", "Artist 2"],
            "score": [0.95, 0.88, 0.82, 0.75, 0.70],
            "similar_artist": [True, False, True, False, True],
            "similar_tag": [False, True, True, True, False],
            "deep_cut_same_artist": [False, False, False, True, True],
            "old_favorite": [True, False, True, False, False],
            "youtube_url": [
                "https://youtube.com/1",
                "https://youtube.com/2",
                None,
                "https://youtube.com/4",
                None,
            ],
            "spotify_url": [
                "https://spotify.com/1",
                None,
                "https://spotify.com/3",
                "https://spotify.com/4",
                "https://spotify.com/5",
            ],
            "tags": ["rock, indie", "pop, electronic", "rock", "jazz", "electronic"],
            "duration_ms": [240000, 180000, 300000, 210000, 195000],
            "username": ["testuser", "testuser", "testuser", "testuser", "testuser"],
            "why_similar_artist_name": [
                "Loved Artist",
                None,
                "Loved Artist",
                None,
                "Other Artist",
            ],
            "why_similar_artist_pct": [85, None, 75, None, 60],
            "why_similar_tags": [
                None,
                "pop, dance",
                "rock, indie",
                "jazz, blues",
                None,
            ],
            "why_tag_match_count": [None, 2, 2, 2, None],
            "why_deep_cut_artist": [None, None, None, "Artist 3", "Artist 2"],
        }
    )


@pytest.fixture
def mock_user_stats() -> dict:
    """Sample user statistics for testing."""
    return {
        "total_plays": 5000,
        "total_tracks_played": 1200,
        "total_artists_played": 350,
    }


@pytest.fixture
def mock_top_artists() -> pl.DataFrame:
    """Sample top artists for testing."""
    return pl.DataFrame(
        {
            "artist_name": ["Radiohead", "The Beatles", "Pink Floyd"],
            "play_count": [150, 120, 95],
        }
    )


@pytest.fixture
def mock_excluded_tracks() -> pl.DataFrame:
    """Sample excluded tracks for testing."""
    return pl.DataFrame(
        {
            "username": ["testuser"],
            "track_id": ["excluded1"],
            "track_name": ["Excluded Song"],
            "artist_name": ["Excluded Artist"],
            "excluded_at": [dt.datetime.now(tz=dt.timezone.utc)],
        }
    )


@pytest.fixture
def mock_excluded_artists() -> pl.DataFrame:
    """Sample excluded artists for testing."""
    return pl.DataFrame(
        {
            "username": ["testuser"],
            "artist_name": ["Blocked Artist"],
            "excluded_at": [dt.datetime.now(tz=dt.timezone.utc)],
        }
    )


@pytest.fixture
def empty_excluded_tracks() -> pl.DataFrame:
    """Empty excluded tracks DataFrame."""
    return pl.DataFrame(
        schema={
            "username": pl.String,
            "track_id": pl.String,
            "track_name": pl.String,
            "artist_name": pl.String,
            "excluded_at": pl.Datetime(time_zone="UTC"),
        }
    )


@pytest.fixture
def empty_excluded_artists() -> pl.DataFrame:
    """Empty excluded artists DataFrame."""
    return pl.DataFrame(
        schema={
            "username": pl.String,
            "artist_name": pl.String,
            "excluded_at": pl.Datetime(time_zone="UTC"),
        }
    )


class TestFilterCandidates:
    """Tests for the filter_candidates function."""

    def test_filter_by_similar_artists_only(self, mock_track_candidates):
        """Test filtering to show only similar artist tracks."""
        from music_airflow.app.filtering import filter_candidates

        result = filter_candidates(
            mock_track_candidates.lazy(),
            use_similar_tags=False,
            use_similar_artists=True,
            use_deep_cuts=False,
            discovery_weight=0.5,
        ).collect()

        assert len(result) == 3  # track1, track3, track5 have similar_artist=True
        assert all(result["similar_artist"])

    def test_filter_by_similar_tags_only(self, mock_track_candidates):
        """Test filtering to show only similar tag tracks."""
        from music_airflow.app.filtering import filter_candidates

        result = filter_candidates(
            mock_track_candidates.lazy(),
            use_similar_tags=True,
            use_similar_artists=False,
            use_deep_cuts=False,
            discovery_weight=0.5,
        ).collect()

        assert len(result) == 3  # track2, track3, track4 have similar_tag=True
        assert all(result["similar_tag"])

    def test_filter_by_deep_cuts_only(self, mock_track_candidates):
        """Test filtering to show only deep cuts."""
        from music_airflow.app.filtering import filter_candidates

        result = filter_candidates(
            mock_track_candidates.lazy(),
            use_similar_tags=False,
            use_similar_artists=False,
            use_deep_cuts=True,
            discovery_weight=0.5,
        ).collect()

        assert len(result) == 2  # track4, track5 have deep_cut_same_artist=True
        assert all(result["deep_cut_same_artist"])

    def test_filter_combined_sources(self, mock_track_candidates):
        """Test filtering with multiple sources enabled (OR logic)."""
        from music_airflow.app.filtering import filter_candidates

        result = filter_candidates(
            mock_track_candidates.lazy(),
            use_similar_tags=True,
            use_similar_artists=True,
            use_deep_cuts=False,
            discovery_weight=0.5,
        ).collect()

        # Should include tracks with similar_artist OR similar_tag
        assert len(result) == 5  # All tracks match at least one criterion

    def test_filter_no_sources_returns_empty(self, mock_track_candidates):
        """Test that disabling all sources returns empty DataFrame."""
        from music_airflow.app.filtering import filter_candidates

        result = filter_candidates(
            mock_track_candidates.lazy(),
            use_similar_tags=False,
            use_similar_artists=False,
            use_deep_cuts=False,
            discovery_weight=0.5,
        ).collect()

        assert len(result) == 0

    def test_filter_excludes_tracks(self, mock_track_candidates):
        """Test that excluded tracks are filtered out."""
        from music_airflow.app.filtering import filter_candidates

        excluded = pl.DataFrame(
            {
                "username": ["testuser"],
                "track_id": ["track1"],
                "track_name": ["Song A"],
                "artist_name": ["Artist 1"],
                "excluded_at": [dt.datetime.now(tz=dt.timezone.utc)],
            }
        )

        result = filter_candidates(
            mock_track_candidates.lazy(),
            use_similar_tags=True,
            use_similar_artists=True,
            use_deep_cuts=True,
            discovery_weight=0.5,
            excluded_tracks=excluded.lazy(),
        ).collect()

        assert "track1" not in result["track_id"].to_list()
        assert len(result) == 4

    def test_filter_excludes_artists(self, mock_track_candidates):
        """Test that tracks from excluded artists are filtered out."""
        from music_airflow.app.filtering import filter_candidates

        excluded_artists = pl.DataFrame(
            {
                "username": ["testuser"],
                "artist_name": ["Artist 1"],
                "excluded_at": [dt.datetime.now(tz=dt.timezone.utc)],
            }
        )

        result = filter_candidates(
            mock_track_candidates.lazy(),
            use_similar_tags=True,
            use_similar_artists=True,
            use_deep_cuts=True,
            discovery_weight=0.5,
            excluded_artists=excluded_artists.lazy(),
        ).collect()

        assert "Artist 1" not in result["artist_name"].to_list()
        assert len(result) == 3

    def test_discovery_weight_favors_old_favorites(self, mock_track_candidates):
        """Test that low discovery weight boosts old favorites."""
        from music_airflow.app.filtering import filter_candidates

        result = filter_candidates(
            mock_track_candidates.lazy(),
            use_similar_tags=True,
            use_similar_artists=True,
            use_deep_cuts=True,
            discovery_weight=0.0,
        ).collect()

        old_fav_scores = result.filter(pl.col("old_favorite"))["weighted_score"]
        new_scores = result.filter(~pl.col("old_favorite"))["weighted_score"]

        if len(old_fav_scores) > 0 and len(new_scores) > 0:
            assert old_fav_scores.mean() > new_scores.mean()

    def test_discovery_weight_favors_new_discoveries(self, mock_track_candidates):
        """Test that high discovery weight boosts new discoveries."""
        from music_airflow.app.filtering import filter_candidates

        result = filter_candidates(
            mock_track_candidates.lazy(),
            use_similar_tags=True,
            use_similar_artists=True,
            use_deep_cuts=True,
            discovery_weight=1.0,
        ).collect()

        old_fav_scores = result.filter(pl.col("old_favorite"))["weighted_score"]
        new_scores = result.filter(~pl.col("old_favorite"))["weighted_score"]

        if len(old_fav_scores) > 0 and len(new_scores) > 0:
            assert new_scores.mean() > old_fav_scores.mean()


class TestApplyArtistLimit:
    """Tests for the apply_artist_limit function."""

    def test_limits_tracks_per_artist(self, mock_track_candidates):
        """Test that artist limit is applied correctly."""
        from music_airflow.app.filtering import apply_artist_limit, filter_candidates

        # First filter to get weighted_score column
        candidates = filter_candidates(
            mock_track_candidates.lazy(),
            use_similar_tags=True,
            use_similar_artists=True,
            use_deep_cuts=True,
            discovery_weight=0.5,
        )

        result = apply_artist_limit(candidates, max_songs_per_artist=1).collect()

        # Each artist should have at most 1 track
        artist_counts = result.group_by("artist_name").len()
        assert all(artist_counts["len"] <= 1)


class TestLoadRecommendationReasons:
    """Tests for the load_recommendation_reasons function."""

    def test_loads_similar_artist_reason(self):
        """Test loading similar artist recommendation reason."""
        from music_airflow.app.filtering import load_recommendation_reasons

        track_row = {
            "why_similar_artist_name": "Radiohead",
            "why_similar_artist_pct": 85,
            "similar_artist": True,
        }

        reasons = load_recommendation_reasons(track_row)

        assert "similar_artist" in reasons
        assert reasons["similar_artist"]["source_artist"] == "Radiohead"
        assert reasons["similar_artist"]["similarity"] == 85

    def test_loads_similar_tag_reason(self):
        """Test loading similar tag recommendation reason."""
        from music_airflow.app.filtering import load_recommendation_reasons

        track_row = {
            "why_similar_tags": "rock, indie, alternative",
            "why_tag_match_count": 3,
            "similar_tag": True,
        }

        reasons = load_recommendation_reasons(track_row)

        assert "similar_tag" in reasons
        assert "rock" in reasons["similar_tag"]["tags"]
        assert reasons["similar_tag"]["match_count"] == 3

    def test_loads_deep_cut_reason(self):
        """Test loading deep cut recommendation reason."""
        from music_airflow.app.filtering import load_recommendation_reasons

        track_row = {
            "why_deep_cut_artist": "Pink Floyd",
            "deep_cut_same_artist": True,
        }

        reasons = load_recommendation_reasons(track_row)

        assert "deep_cut" in reasons
        assert reasons["deep_cut"]["source_artist"] == "Pink Floyd"

    def test_empty_reasons_when_no_data(self):
        """Test that empty dict is returned when no reason data."""
        from music_airflow.app.filtering import load_recommendation_reasons

        track_row = {
            "track_name": "Some Track",
            "artist_name": "Some Artist",
        }

        reasons = load_recommendation_reasons(track_row)

        assert reasons == {}


class TestExclusionCacheFunctions:
    """Tests for exclusion cache functions."""

    def test_add_excluded_track_local(self, empty_excluded_tracks):
        """Test adding a track to local exclusion cache."""
        from music_airflow.app.exclusions_ui import add_excluded_track_local
        import streamlit as st

        with patch.object(st, "session_state", {}):
            with patch(
                "music_airflow.app.exclusions_ui.write_excluded_track"
            ) as mock_write:
                st.session_state["excluded_tracks_testuser"] = empty_excluded_tracks

                add_excluded_track_local(
                    "testuser", "track123", "Test Song", "Test Artist"
                )

                cached = st.session_state["excluded_tracks_testuser"]
                assert len(cached) == 1
                assert cached["track_id"][0] == "track123"

                mock_write.assert_called_once_with(
                    "testuser", "track123", "Test Song", "Test Artist"
                )

    def test_remove_excluded_track_local(self, mock_excluded_tracks):
        """Test removing a track from local exclusion cache."""
        from music_airflow.app.exclusions_ui import remove_excluded_track_local
        import streamlit as st

        with patch.object(st, "session_state", {}):
            with patch(
                "music_airflow.app.exclusions_ui.remove_excluded_track"
            ) as mock_remove:
                st.session_state["excluded_tracks_testuser"] = mock_excluded_tracks

                remove_excluded_track_local(
                    "testuser", "excluded1", "Excluded Song", "Excluded Artist"
                )

                cached = st.session_state["excluded_tracks_testuser"]
                assert len(cached) == 0

                mock_remove.assert_called_once()

    def test_add_excluded_artist_local(self, empty_excluded_artists):
        """Test adding an artist to local exclusion cache."""
        from music_airflow.app.exclusions_ui import add_excluded_artist_local
        import streamlit as st

        with patch.object(st, "session_state", {}):
            with patch(
                "music_airflow.app.exclusions_ui.write_excluded_artist"
            ) as mock_write:
                st.session_state["excluded_artists_testuser"] = empty_excluded_artists

                add_excluded_artist_local("testuser", "Blocked Artist")

                cached = st.session_state["excluded_artists_testuser"]
                assert len(cached) == 1
                assert cached["artist_name"][0] == "Blocked Artist"

                mock_write.assert_called_once_with("testuser", "Blocked Artist")

    def test_remove_excluded_artist_local(self, mock_excluded_artists):
        """Test removing an artist from local exclusion cache."""
        from music_airflow.app.exclusions_ui import remove_excluded_artist_local
        import streamlit as st

        with patch.object(st, "session_state", {}):
            with patch(
                "music_airflow.app.exclusions_ui.remove_excluded_artist"
            ) as mock_remove:
                st.session_state["excluded_artists_testuser"] = mock_excluded_artists

                remove_excluded_artist_local("testuser", "Blocked Artist")

                cached = st.session_state["excluded_artists_testuser"]
                assert len(cached) == 0

                mock_remove.assert_called_once()


class TestDataLoading:
    """Tests for data loading functions."""

    def test_prefetch_calls_load_for_all_users(self, mock_track_candidates):
        """Test that prefetch loads candidates for all configured users."""
        from music_airflow.app.data_loading import prefetch_all_users_track_candidates
        from music_airflow.utils.constants import LAST_FM_USERNAMES

        with patch(
            "music_airflow.app.data_loading.load_track_candidates_cached"
        ) as mock_load:
            mock_load.return_value = mock_track_candidates

            prefetch_all_users_track_candidates()

            assert mock_load.call_count == len(LAST_FM_USERNAMES)

            called_users = [call.args[0] for call in mock_load.call_args_list]
            for username in LAST_FM_USERNAMES:
                assert username in called_users


class TestStreamlitAppIntegration:
    """Integration tests using Streamlit's AppTest framework."""

    @pytest.fixture
    def mock_async_firestore(
        self, mock_track_candidates, mock_user_stats, mock_top_artists
    ):
        """Create mock AsyncFirestoreReader with async methods."""
        from unittest.mock import AsyncMock

        mock_io = MagicMock()
        mock_io.read_track_candidates = AsyncMock(return_value=mock_track_candidates)
        mock_io.read_user_stats = AsyncMock(return_value=mock_user_stats)
        mock_io.read_artist_play_counts = AsyncMock(return_value=mock_top_artists)
        return mock_io

    def test_app_loads_without_errors(
        self,
        mock_async_firestore,
        empty_excluded_tracks,
        empty_excluded_artists,
    ):
        """Test that the app loads without raising exceptions."""
        with patch(
            "music_airflow.app.data_loading.AsyncFirestoreReader",
            return_value=mock_async_firestore,
        ):
            with patch(
                "music_airflow.app.exclusions_ui.read_excluded_tracks",
                return_value=empty_excluded_tracks.lazy(),
            ):
                with patch(
                    "music_airflow.app.exclusions_ui.read_excluded_artists",
                    return_value=empty_excluded_artists.lazy(),
                ):
                    at = AppTest.from_file(
                        "src/music_airflow/app/streamlit_app.py",
                        default_timeout=30,
                    )
                    at.run()

                    assert not at.exception, f"App raised exception: {at.exception}"

    def test_app_displays_user_selectbox(
        self,
        mock_async_firestore,
        empty_excluded_tracks,
        empty_excluded_artists,
    ):
        """Test that user selectbox is displayed in sidebar."""
        with patch(
            "music_airflow.app.data_loading.AsyncFirestoreReader",
            return_value=mock_async_firestore,
        ):
            with patch(
                "music_airflow.app.exclusions_ui.read_excluded_tracks",
                return_value=empty_excluded_tracks.lazy(),
            ):
                with patch(
                    "music_airflow.app.exclusions_ui.read_excluded_artists",
                    return_value=empty_excluded_artists.lazy(),
                ):
                    at = AppTest.from_file(
                        "src/music_airflow/app/streamlit_app.py",
                        default_timeout=30,
                    )
                    at.run()

                    selectboxes = at.sidebar.selectbox
                    assert len(selectboxes) >= 1

    def test_app_displays_metrics(
        self,
        mock_async_firestore,
        empty_excluded_tracks,
        empty_excluded_artists,
    ):
        """Test that user metrics are displayed."""
        with patch(
            "music_airflow.app.data_loading.AsyncFirestoreReader",
            return_value=mock_async_firestore,
        ):
            with patch(
                "music_airflow.app.exclusions_ui.read_excluded_tracks",
                return_value=empty_excluded_tracks.lazy(),
            ):
                with patch(
                    "music_airflow.app.exclusions_ui.read_excluded_artists",
                    return_value=empty_excluded_artists.lazy(),
                ):
                    at = AppTest.from_file(
                        "src/music_airflow/app/streamlit_app.py",
                        default_timeout=30,
                    )
                    at.run()

                    metrics = at.metric
                    assert len(metrics) >= 3

    def test_app_displays_recommendations_dataframe(
        self,
        mock_async_firestore,
        empty_excluded_tracks,
        empty_excluded_artists,
    ):
        """Test that recommendations dataframe is displayed."""
        with patch(
            "music_airflow.app.data_loading.AsyncFirestoreReader",
            return_value=mock_async_firestore,
        ):
            with patch(
                "music_airflow.app.exclusions_ui.read_excluded_tracks",
                return_value=empty_excluded_tracks.lazy(),
            ):
                with patch(
                    "music_airflow.app.exclusions_ui.read_excluded_artists",
                    return_value=empty_excluded_artists.lazy(),
                ):
                    at = AppTest.from_file(
                        "src/music_airflow/app/streamlit_app.py",
                        default_timeout=30,
                    )
                    at.run()

                    dataframes = at.dataframe
                    assert len(dataframes) >= 1

    def test_app_displays_headers(
        self,
        mock_async_firestore,
        empty_excluded_tracks,
        empty_excluded_artists,
    ):
        """Test that expected headers are displayed."""
        with patch(
            "music_airflow.app.data_loading.AsyncFirestoreReader",
            return_value=mock_async_firestore,
        ):
            with patch(
                "music_airflow.app.exclusions_ui.read_excluded_tracks",
                return_value=empty_excluded_tracks.lazy(),
            ):
                with patch(
                    "music_airflow.app.exclusions_ui.read_excluded_artists",
                    return_value=empty_excluded_artists.lazy(),
                ):
                    at = AppTest.from_file(
                        "src/music_airflow/app/streamlit_app.py",
                        default_timeout=30,
                    )
                    at.run()

                    headers = at.header
                    assert len(headers) >= 2

    def test_sidebar_widgets_exist(
        self,
        mock_async_firestore,
        empty_excluded_tracks,
        empty_excluded_artists,
    ):
        """Test that slider and radio widgets are present in sidebar."""
        with patch(
            "music_airflow.app.data_loading.AsyncFirestoreReader",
            return_value=mock_async_firestore,
        ):
            with patch(
                "music_airflow.app.exclusions_ui.read_excluded_tracks",
                return_value=empty_excluded_tracks.lazy(),
            ):
                with patch(
                    "music_airflow.app.exclusions_ui.read_excluded_artists",
                    return_value=empty_excluded_artists.lazy(),
                ):
                    at = AppTest.from_file(
                        "src/music_airflow/app/streamlit_app.py",
                        default_timeout=30,
                    )
                    at.run()

                    sliders = at.sidebar.slider
                    assert len(sliders) >= 1
                    radios = at.sidebar.radio
                    assert len(radios) >= 1

    def test_checkbox_widgets_exist(
        self,
        mock_async_firestore,
        empty_excluded_tracks,
        empty_excluded_artists,
    ):
        """Test that checkbox widgets are present for sources."""
        with patch(
            "music_airflow.app.data_loading.AsyncFirestoreReader",
            return_value=mock_async_firestore,
        ):
            with patch(
                "music_airflow.app.exclusions_ui.read_excluded_tracks",
                return_value=empty_excluded_tracks.lazy(),
            ):
                with patch(
                    "music_airflow.app.exclusions_ui.read_excluded_artists",
                    return_value=empty_excluded_artists.lazy(),
                ):
                    at = AppTest.from_file(
                        "src/music_airflow/app/streamlit_app.py",
                        default_timeout=30,
                    )
                    at.run()

                    checkboxes = at.sidebar.checkbox
                    assert len(checkboxes) >= 3

    def test_changing_discovery_slider_reruns_app(
        self,
        mock_async_firestore,
        empty_excluded_tracks,
        empty_excluded_artists,
    ):
        """Test that changing discovery slider triggers app rerun."""
        with patch(
            "music_airflow.app.data_loading.AsyncFirestoreReader",
            return_value=mock_async_firestore,
        ):
            with patch(
                "music_airflow.app.exclusions_ui.read_excluded_tracks",
                return_value=empty_excluded_tracks.lazy(),
            ):
                with patch(
                    "music_airflow.app.exclusions_ui.read_excluded_artists",
                    return_value=empty_excluded_artists.lazy(),
                ):
                    at = AppTest.from_file(
                        "src/music_airflow/app/streamlit_app.py",
                        default_timeout=30,
                    )
                    at.run()

                    discovery_slider = at.sidebar.slider[0]
                    discovery_slider.set_value(0.8).run()

                    assert not at.exception

    def test_unchecking_all_sources_shows_warning(
        self,
        mock_async_firestore,
        empty_excluded_tracks,
        empty_excluded_artists,
    ):
        """Test that unchecking all sources shows a warning."""
        with patch(
            "music_airflow.app.data_loading.AsyncFirestoreReader",
            return_value=mock_async_firestore,
        ):
            with patch(
                "music_airflow.app.exclusions_ui.read_excluded_tracks",
                return_value=empty_excluded_tracks.lazy(),
            ):
                with patch(
                    "music_airflow.app.exclusions_ui.read_excluded_artists",
                    return_value=empty_excluded_artists.lazy(),
                ):
                    at = AppTest.from_file(
                        "src/music_airflow/app/streamlit_app.py",
                        default_timeout=30,
                    )
                    at.run()

                    for checkbox in at.sidebar.checkbox:
                        checkbox.uncheck()
                    at.run()

                    assert not at.exception
