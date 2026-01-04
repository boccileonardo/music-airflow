"""
Tests for dimension table transformations and extraction.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import polars as pl
import pytest
from airflow.exceptions import AirflowSkipException

from music_airflow.extract.dimensions import (
    extract_tracks_to_bronze,
    extract_artists_to_bronze,
)
from music_airflow.transform.dimensions import (
    _transform_tracks_raw_to_structured,
    _transform_artists_raw_to_structured,
    transform_tracks_to_silver,
    transform_artists_to_silver,
    compute_dim_users,
)


@pytest.fixture
def sample_raw_tracks():
    """Sample raw track data from Last.fm API."""
    return pl.LazyFrame(
        {
            "name": ["Song A", "Song B"],
            "mbid": ["track-mbid-1", ""],
            "url": ["https://last.fm/track/a", "https://last.fm/track/b"],
            "duration": [180000, 240000],
            "artist": [
                {"name": "Artist X", "mbid": "artist-mbid-1"},
                {"name": "Artist Y", "mbid": ""},
            ],
            "album": [
                {"title": "Album 1", "mbid": "album-mbid-1"},
                {"title": "Album 2", "mbid": ""},
            ],
            "listeners": [5000, 3000],
            "playcount": [10000, 7000],
            "toptags": [
                {
                    "tag": [
                        {"name": "rock"},
                        {"name": "indie"},
                        {"name": "alternative"},
                    ]
                },
                {
                    "tag": [
                        {"name": "pop"},
                        {"name": "electronic"},
                    ]
                },
            ],
        }
    )


@pytest.fixture
def sample_raw_artists():
    """Sample raw artist data from Last.fm API."""
    return pl.LazyFrame(
        {
            "name": ["Artist X", "Artist Y"],
            "mbid": ["artist-mbid-1", ""],
            "url": ["https://last.fm/artist/x", "https://last.fm/artist/y"],
            "stats": [
                {"listeners": 50000, "playcount": 100000},
                {"listeners": 30000, "playcount": 70000},
            ],
            "tags": [
                {
                    "tag": [
                        {"name": "rock"},
                        {"name": "indie"},
                        {"name": "alternative"},
                    ]
                },
                {
                    "tag": [
                        {"name": "pop"},
                        {"name": "electronic"},
                    ]
                },
            ],
            "bio": [
                {"summary": "This is a bio summary for Artist X. " * 50},
                {"summary": "This is a bio summary for Artist Y. " * 50},
            ],
        }
    )


def test_transform_tracks_raw_to_structured(sample_raw_tracks):
    """Test track transformation logic."""
    result = _transform_tracks_raw_to_structured(sample_raw_tracks).collect()

    # Check schema (track_id and artist_id added during deduplication, not in raw transform)
    expected_cols = [
        "track_name",
        "track_mbid",
        "artist_name",
        "artist_mbid",
        "album_name",
        "duration_ms",
        "listeners",
        "playcount",
        "tags",
        "track_url",
    ]
    assert result.columns == expected_cols

    # Check first row
    assert result["track_name"][0] == "Song A"
    assert result["track_mbid"][0] == "track-mbid-1"
    assert result["artist_name"][0] == "Artist X"
    assert result["artist_mbid"][0] == "artist-mbid-1"
    assert result["album_name"][0] == "Album 1"
    assert result["duration_ms"][0] == 180000
    assert result["listeners"][0] == 5000
    assert result["playcount"][0] == 10000
    assert result["tags"][0] == "rock, indie, alternative"

    # Check second row (missing mbids)
    assert result["track_mbid"][1] == ""
    assert result["artist_mbid"][1] == ""
    assert result["tags"][1] == "pop, electronic"


def test_transform_artists_raw_to_structured(sample_raw_artists):
    """Test artist transformation logic."""
    result = _transform_artists_raw_to_structured(sample_raw_artists).collect()

    # Check schema (artist_id added during deduplication, not in raw transform)
    expected_cols = [
        "artist_name",
        "artist_mbid",
        "listeners",
        "playcount",
        "tags",
        "bio_summary",
        "artist_url",
    ]
    assert result.columns == expected_cols

    # Check first row
    assert result["artist_name"][0] == "Artist X"
    assert result["artist_mbid"][0] == "artist-mbid-1"
    assert result["listeners"][0] == 50000
    assert result["playcount"][0] == 100000
    assert result["tags"][0] == "rock, indie, alternative"
    assert "Artist X" in result["bio_summary"][0]
    assert len(result["bio_summary"][0]) >= 500
    assert result["artist_url"][0] == "https://last.fm/artist/x"

    # Check second row
    assert result["artist_mbid"][1] == ""
    assert result["tags"][1] == "pop, electronic"


def test_tracks_tags_truncation():
    """Test that only top 5 tags are kept."""
    raw_tracks = pl.LazyFrame(
        {
            "name": ["Song X"],
            "mbid": [""],
            "url": ["https://last.fm/track/x"],
            "duration": [180000],
            "artist": [{"name": "Artist", "mbid": ""}],
            "album": [{"title": "Album", "mbid": ""}],
            "listeners": [1000],
            "playcount": [2000],
            "toptags": [
                {
                    "tag": [
                        {"name": "tag1"},
                        {"name": "tag2"},
                        {"name": "tag3"},
                        {"name": "tag4"},
                        {"name": "tag5"},
                        {"name": "tag6"},
                        {"name": "tag7"},
                    ]
                }
            ],
        }
    )

    result = _transform_tracks_raw_to_structured(raw_tracks).collect()

    # Should only have first 5 tags
    tags = result["tags"][0].split(", ")
    assert len(tags) == 5
    assert tags == ["tag1", "tag2", "tag3", "tag4", "tag5"]


class TestExtractTracksToBronze:
    """Test extract_tracks_to_bronze function."""

    @pytest.mark.asyncio
    @patch("music_airflow.extract.dimensions.LastFMClient")
    @patch("music_airflow.extract.dimensions.PolarsDeltaIOManager")
    @patch("music_airflow.extract.dimensions.JSONIOManager")
    async def test_extract_new_tracks(
        self, mock_json_io, mock_delta_io, mock_client_class, test_data_dir
    ):
        """Test extracting metadata for new tracks."""
        # Setup: Create silver plays table with tracks
        plays_df = pl.LazyFrame(
            {
                "track_name": ["Track A", "Track B"],
                "artist_name": ["Artist X", "Artist Y"],
                "track_mbid": ["mbid1", ""],
            }
        )

        # Mock Delta IO for plays (silver)
        mock_plays_io = MagicMock()
        mock_plays_io.read_delta.return_value = plays_df

        # Mock Delta IO for tracks (no existing tracks table)
        from deltalake.exceptions import TableNotFoundError

        mock_tracks_io = MagicMock()
        mock_tracks_io.read_delta.side_effect = TableNotFoundError("Table not found")

        # Use a simple counter approach
        call_count = [0]

        def get_io_manager(medallion_layer: str):  # type: ignore[misc]
            call_count[0] += 1
            if call_count[0] == 1:  # First call - plays
                return mock_plays_io
            else:  # Second call - tracks (raises error)
                mock_io = MagicMock()
                mock_io.read_delta.side_effect = TableNotFoundError("Table not found")
                return mock_io

        mock_delta_io.side_effect = get_io_manager

        # Mock LastFM client with AsyncMock for async methods
        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get_track_info = AsyncMock(
            side_effect=[
                {
                    "name": "Track A",
                    "artist": {"name": "Artist X", "mbid": "artist1"},
                    "album": {"title": "Album 1", "mbid": "album1"},
                    "mbid": "mbid1",
                    "url": "url1",
                    "duration": 180000,
                    "listeners": 5000,
                    "playcount": 10000,
                    "toptags": {"tag": [{"name": "rock"}, {"name": "indie"}]},
                },
                {
                    "name": "Track B",
                    "artist": {"name": "Artist Y", "mbid": "artist2"},
                    "album": {"title": "Album 2", "mbid": "album2"},
                    "mbid": "",
                    "url": "url2",
                    "duration": 200000,
                    "listeners": 3000,
                    "playcount": 7000,
                    "toptags": {"tag": [{"name": "pop"}]},
                },
            ]
        )
        mock_client_class.return_value = mock_client

        # Mock JSON IO
        from music_airflow.utils.polars_io_manager import JSONIOManager

        json_mgr = JSONIOManager(medallion_layer="bronze")
        json_mgr.base_dir = test_data_dir / "bronze"
        mock_json_io.return_value = json_mgr

        # Execute
        result = await extract_tracks_to_bronze()

        # Verify
        assert result["rows"] == 2
        assert result["tracks_fetched"] == 2
        assert result["format"] == "json"
        assert "tracks/" in result["filename"]
        assert mock_client.get_track_info.call_count == 2

    @patch("music_airflow.extract.dimensions.PolarsDeltaIOManager")
    def test_extract_no_new_tracks_raises_skip(self, mock_delta_io):
        """Test that no new tracks raises AirflowSkipException."""
        # Setup: plays and existing tracks are the same
        plays_df = pl.LazyFrame(
            {
                "track_name": ["Track A"],
                "artist_name": ["Artist X"],
                "track_mbid": ["mbid1"],
            }
        )

        existing_tracks_df = pl.LazyFrame(
            {
                "track_name": ["Track A"],
                "artist_name": ["Artist X"],
            }
        )

        mock_plays_io = MagicMock()
        mock_plays_io.read_delta.return_value = plays_df

        mock_tracks_io = MagicMock()
        mock_tracks_io.read_delta.return_value = existing_tracks_df

        call_count = [0]

        def get_io_manager(medallion_layer):
            call_count[0] += 1
            return mock_plays_io if call_count[0] == 1 else mock_tracks_io

        mock_delta_io.side_effect = get_io_manager

        # Execute and verify
        with pytest.raises(AirflowSkipException, match="No new tracks to fetch"):
            extract_tracks_to_bronze()


class TestExtractArtistsToBronze:
    """Test extract_artists_to_bronze function."""

    @pytest.mark.asyncio
    @patch("music_airflow.extract.dimensions.LastFMClient")
    @patch("music_airflow.extract.dimensions.PolarsDeltaIOManager")
    @patch("music_airflow.extract.dimensions.JSONIOManager")
    async def test_extract_new_artists(
        self, mock_json_io, mock_delta_io, mock_client_class, test_data_dir
    ):
        """Test extracting metadata for new artists."""
        # Setup: Create silver plays table with artists
        plays_df = pl.LazyFrame(
            {
                "artist_name": ["Artist X", "Artist Y"],
                "artist_mbid": ["artist1", ""],
            }
        )

        # Mock Delta IO
        mock_plays_io = MagicMock()
        mock_plays_io.read_delta.return_value = plays_df

        from deltalake.exceptions import TableNotFoundError

        mock_artists_io = MagicMock()
        mock_artists_io.read_delta.side_effect = TableNotFoundError("Table not found")

        call_count = [0]

        def get_io_manager(medallion_layer):
            call_count[0] += 1
            if call_count[0] == 1:
                return mock_plays_io
            else:
                mock_io = MagicMock()
                mock_io.read_delta.side_effect = TableNotFoundError("Table not found")
                return mock_io

        mock_delta_io.side_effect = get_io_manager

        # Mock LastFM client with AsyncMock for async methods
        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get_artist_info = AsyncMock(
            side_effect=[
                {
                    "name": "Artist X",
                    "mbid": "artist1",
                    "url": "url1",
                    "stats": {"listeners": 50000, "playcount": 100000},
                    "tags": {"tag": [{"name": "rock"}, {"name": "indie"}]},
                    "bio": {"summary": "Bio for Artist X"},
                },
                {
                    "name": "Artist Y",
                    "mbid": "",
                    "url": "url2",
                    "stats": {"listeners": 30000, "playcount": 70000},
                    "tags": {"tag": [{"name": "pop"}]},
                    "bio": {"summary": "Bio for Artist Y"},
                },
            ]
        )
        mock_client_class.return_value = mock_client

        # Mock JSON IO
        from music_airflow.utils.polars_io_manager import JSONIOManager

        json_mgr = JSONIOManager(medallion_layer="bronze")
        json_mgr.base_dir = test_data_dir / "bronze"
        mock_json_io.return_value = json_mgr

        # Execute
        result = await extract_artists_to_bronze()

        # Verify
        assert result["rows"] == 2
        assert result["artists_fetched"] == 2
        assert result["format"] == "json"
        assert "artists/" in result["filename"]
        assert mock_client.get_artist_info.call_count == 2


class TestTransformTracksToSilver:
    """Test transform_tracks_to_silver integration function."""

    def test_transform_tracks_integration(self, test_data_dir):
        """Test full transformation pipeline for tracks."""
        # Setup: Create bronze JSON data
        import json

        bronze_dir = test_data_dir / "bronze" / "tracks"
        bronze_dir.mkdir(parents=True, exist_ok=True)
        silver_dir = test_data_dir / "silver"
        silver_dir.mkdir(parents=True, exist_ok=True)

        tracks_data = [
            {
                "name": "Track A",
                "artist": {"name": "Artist X", "mbid": "artist1"},
                "album": {"title": "Album 1", "mbid": "album1"},
                "mbid": "track1",
                "url": "url1",
                "duration": 180000,
                "listeners": 5000,
                "playcount": 10000,
                "toptags": {"tag": [{"name": "rock"}, {"name": "indie"}]},
            }
        ]

        tracks_file = bronze_dir / "tracks_test.json"
        with open(tracks_file, "w") as f:
            json.dump(tracks_data, f)

        # Patch IO managers
        with (
            patch("music_airflow.transform.dimensions.JSONIOManager") as mock_json_io,
            patch(
                "music_airflow.transform.dimensions.PolarsDeltaIOManager"
            ) as mock_delta_io,
        ):
            from music_airflow.utils.polars_io_manager import (
                JSONIOManager,
                PolarsDeltaIOManager,
            )

            json_mgr = JSONIOManager(medallion_layer="bronze")
            json_mgr.base_dir = test_data_dir / "bronze"

            delta_mgr = PolarsDeltaIOManager(medallion_layer="silver")
            delta_mgr.base_dir = test_data_dir / "silver"

            mock_json_io.return_value = json_mgr
            mock_delta_io.return_value = delta_mgr

            # Execute
            fetch_metadata = {"filename": "tracks/tracks_test.json"}
            result = transform_tracks_to_silver(fetch_metadata)

        # Verify
        assert result["table_name"] == "tracks"
        assert result["format"] == "delta"
        assert result["mode"] == "merge"
        assert result["rows"] == 1

        # Verify Delta table
        delta_table_path = silver_dir / "tracks"
        assert delta_table_path.exists()
        df = pl.read_delta(str(delta_table_path))
        assert len(df) == 1
        assert df["track_name"][0] == "Track A"


class TestTransformArtistsToSilver:
    """Test transform_artists_to_silver integration function."""

    def test_transform_artists_integration(self, test_data_dir):
        """Test full transformation pipeline for artists."""
        # Setup: Create bronze JSON data
        import json

        bronze_dir = test_data_dir / "bronze" / "artists"
        bronze_dir.mkdir(parents=True, exist_ok=True)
        silver_dir = test_data_dir / "silver"
        silver_dir.mkdir(parents=True, exist_ok=True)

        artists_data = [
            {
                "name": "Artist X",
                "mbid": "artist1",
                "url": "url1",
                "stats": {"listeners": 50000, "playcount": 100000},
                "tags": {"tag": [{"name": "rock"}, {"name": "indie"}]},
                "bio": {"summary": "Bio for Artist X"},
            }
        ]

        artists_file = bronze_dir / "artists_test.json"
        with open(artists_file, "w") as f:
            json.dump(artists_data, f)

        # Patch IO managers
        with (
            patch("music_airflow.transform.dimensions.JSONIOManager") as mock_json_io,
            patch(
                "music_airflow.transform.dimensions.PolarsDeltaIOManager"
            ) as mock_delta_io,
        ):
            from music_airflow.utils.polars_io_manager import (
                JSONIOManager,
                PolarsDeltaIOManager,
            )

            json_mgr = JSONIOManager(medallion_layer="bronze")
            json_mgr.base_dir = test_data_dir / "bronze"

            delta_mgr = PolarsDeltaIOManager(medallion_layer="silver")
            delta_mgr.base_dir = test_data_dir / "silver"

            mock_json_io.return_value = json_mgr
            mock_delta_io.return_value = delta_mgr

            # Execute
            fetch_metadata = {"filename": "artists/artists_test.json"}
            result = transform_artists_to_silver(fetch_metadata)

        # Verify
        assert result["table_name"] == "artists"
        assert result["format"] == "delta"
        assert result["mode"] == "merge"
        assert result["rows"] == 1

        # Verify Delta table
        delta_table_path = silver_dir / "artists"
        assert delta_table_path.exists()
        df = pl.read_delta(str(delta_table_path))
        assert len(df) == 1
        assert df["artist_name"][0] == "Artist X"

    @patch("music_airflow.transform.dimensions.LastFMClient")
    def test_transform_artists_enriches_mbid_and_filters_invalid(
        self, mock_client_class, test_data_dir
    ):
        """Enrich missing MBIDs and drop invalid artists (listeners < 1000)."""
        import json

        bronze_dir = test_data_dir / "bronze" / "artists"
        bronze_dir.mkdir(parents=True, exist_ok=True)
        silver_dir = test_data_dir / "silver"
        silver_dir.mkdir(parents=True, exist_ok=True)

        # Two artists: one missing MBID with high listeners/tags (to enrich),
        # one missing MBID with low listeners (should be excluded)
        artists_data = [
            {
                "name": "Likely Valid",
                "mbid": "",
                "url": "urlA",
                "stats": {"listeners": 20000, "playcount": 30000},
                "tags": {"tag": [{"name": "rock"}]},
                "bio": {"summary": "Some bio"},
            },
            {
                "name": "Likely Invalid",
                "mbid": "",
                "url": "urlB",
                "stats": {"listeners": 500, "playcount": 1000},
                "tags": {"tag": [{"name": "pop"}]},
                "bio": {"summary": "short"},
            },
        ]

        artists_file = bronze_dir / "artists_enrich_test.json"
        with open(artists_file, "w") as f:
            json.dump(artists_data, f)

        # Mock LastFMClient.search_artist to return an MBID for "Likely Valid"
        mock_client = MagicMock()

        def _search_artist(name, limit=1):  # type: ignore[unused-argument]
            if name == "Likely Valid":
                return [{"name": name, "mbid": "mbid-123"}]
            return []

        mock_client.search_artist.side_effect = _search_artist
        mock_client_class.return_value = mock_client

        # Patch IO managers
        with (
            patch("music_airflow.transform.dimensions.JSONIOManager") as mock_json_io,
            patch(
                "music_airflow.transform.dimensions.PolarsDeltaIOManager"
            ) as mock_delta_io,
        ):
            from music_airflow.utils.polars_io_manager import (
                JSONIOManager,
                PolarsDeltaIOManager,
            )

            json_mgr = JSONIOManager(medallion_layer="bronze")
            json_mgr.base_dir = test_data_dir / "bronze"

            delta_mgr = PolarsDeltaIOManager(medallion_layer="silver")
            delta_mgr.base_dir = test_data_dir / "silver"

            mock_json_io.return_value = json_mgr
            mock_delta_io.return_value = delta_mgr

            fetch_metadata = {"filename": "artists/artists_enrich_test.json"}
            result = transform_artists_to_silver(fetch_metadata)

        assert result["rows"] == 1
        df = pl.read_delta(str(silver_dir / "artists"))
        assert set(df["artist_name"]) == {"Likely Valid"}
        # Enriched MBID should be applied and artist_id should use MBID
        row = df.filter(pl.col("artist_name") == "Likely Valid").to_dict(
            as_series=False
        )
        assert row["artist_mbid"][0] == "mbid-123"
        assert row["artist_id"][0] == "mbid-123"

    @patch("music_airflow.transform.dimensions.LastFMClient")
    @patch("music_airflow.transform.dimensions._search_musicbrainz_artist_mbid")
    def test_transform_artists_enriches_via_musicbrainz_when_lastfm_has_no_mbid(
        self, mock_mbz, mock_client_class, test_data_dir
    ):
        """Fallback to MusicBrainz if Last.fm search returns empty/blank MBID."""
        import json

        bronze_dir = test_data_dir / "bronze" / "artists"
        bronze_dir.mkdir(parents=True, exist_ok=True)
        silver_dir = test_data_dir / "silver"
        silver_dir.mkdir(parents=True, exist_ok=True)

        artists_data = [
            {
                "name": "Fallback Artist",
                "mbid": "",
                "url": "urlC",
                "stats": {"listeners": 5000, "playcount": 8000},
                "tags": {"tag": [{"name": "alt"}]},
                "bio": {"summary": "bio"},
            }
        ]

        artists_file = bronze_dir / "artists_enrich_mbz.json"
        with open(artists_file, "w") as f:
            json.dump(artists_data, f)

        # LastFM returns match but empty mbid, forcing MBZ fallback
        mock_client = MagicMock()
        mock_client.search_artist.return_value = [
            {"name": "Fallback Artist", "mbid": ""}
        ]
        mock_client_class.return_value = mock_client
        mock_mbz.return_value = "mbid-fallback-999"

        with (
            patch("music_airflow.transform.dimensions.JSONIOManager") as mock_json_io,
            patch(
                "music_airflow.transform.dimensions.PolarsDeltaIOManager"
            ) as mock_delta_io,
        ):
            from music_airflow.utils.polars_io_manager import (
                JSONIOManager,
                PolarsDeltaIOManager,
            )

            json_mgr = JSONIOManager(medallion_layer="bronze")
            json_mgr.base_dir = test_data_dir / "bronze"

            delta_mgr = PolarsDeltaIOManager(medallion_layer="silver")
            delta_mgr.base_dir = test_data_dir / "silver"

            mock_json_io.return_value = json_mgr
            mock_delta_io.return_value = delta_mgr

            fetch_metadata = {"filename": "artists/artists_enrich_mbz.json"}
            result = transform_artists_to_silver(fetch_metadata)

        assert result["rows"] == 1
        df = pl.read_delta(str(silver_dir / "artists"))
        assert df["artist_mbid"][0] == "mbid-fallback-999"
        assert df["artist_id"][0] == "mbid-fallback-999"


class TestExtractWithoutPlaysData:
    """Test extraction functions when plays data doesn't exist yet."""

    @patch("music_airflow.extract.dimensions.PolarsDeltaIOManager")
    def test_extract_tracks_skips_without_plays(self, mock_delta_io):
        """Test that extract_tracks_to_bronze skips when no plays data exists."""
        # Mock IO manager to raise FileNotFoundError when reading plays
        mock_plays_io = MagicMock()
        mock_plays_io.read_delta.side_effect = FileNotFoundError(
            "No such file or directory: 'data/silver/plays'"
        )
        mock_delta_io.return_value = mock_plays_io

        # Should raise AirflowSkipException
        with pytest.raises(AirflowSkipException) as exc_info:
            extract_tracks_to_bronze()

        assert "No plays data available yet" in str(exc_info.value)

    @patch("music_airflow.extract.dimensions.PolarsDeltaIOManager")
    def test_extract_artists_skips_without_plays(self, mock_delta_io):
        """Test that extract_artists_to_bronze skips when no plays data exists."""
        # Mock IO manager to raise FileNotFoundError when reading plays
        mock_plays_io = MagicMock()
        mock_plays_io.read_delta.side_effect = FileNotFoundError(
            "No such file or directory: 'data/silver/plays'"
        )
        mock_delta_io.return_value = mock_plays_io

        # Should raise AirflowSkipException
        with pytest.raises(AirflowSkipException) as exc_info:
            extract_artists_to_bronze()

        assert "No plays data available yet" in str(exc_info.value)


class TestComputeDimUsersWithoutPlaysData:
    """Test compute_dim_users when plays data doesn't exist yet."""

    @patch("music_airflow.transform.dimensions.PolarsDeltaIOManager")
    def test_compute_dim_users_skips_without_plays(self, mock_delta_io):
        """Test that compute_dim_users skips when no plays data exists."""
        import datetime as dt

        # Mock IO manager to raise FileNotFoundError when reading plays
        mock_plays_io = MagicMock()
        mock_plays_io.read_delta.side_effect = FileNotFoundError(
            "No such file or directory: 'data/silver/plays'"
        )
        mock_delta_io.return_value = mock_plays_io

        execution_date = dt.datetime(2025, 1, 20, tzinfo=dt.timezone.utc)

        # Should raise AirflowSkipException
        with pytest.raises(AirflowSkipException) as exc_info:
            compute_dim_users(execution_date)

        assert "No plays data available yet" in str(exc_info.value)
