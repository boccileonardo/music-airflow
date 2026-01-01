"""
Tests for candidate track generation (silver + gold consolidation).

All tests are fully isolated: they change CWD to a tmp folder
and create minimal Delta tables under ./data/* so no real data is touched.
Last.fm API client is mocked to avoid network calls.
"""

from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest

from music_airflow.utils.polars_io_manager import PolarsDeltaIOManager

from music_airflow.transform.candidate_generation import (
    generate_similar_artist_candidates,
    generate_similar_tag_candidates,
    generate_deep_cut_candidates,
    merge_candidate_sources,
)


@pytest.fixture
def patched_delta_io(test_data_dir):
    """Provide patched IO managers with base_dir pointing to test_data_dir subfolders."""
    silver_mgr = PolarsDeltaIOManager(medallion_layer="silver")
    silver_mgr.base_dir = test_data_dir / "silver"

    gold_mgr = PolarsDeltaIOManager(medallion_layer="gold")
    gold_mgr.base_dir = test_data_dir / "gold"

    def factory(layer: str = "silver"):
        if layer == "gold":
            return gold_mgr
        return silver_mgr

    return factory


def _write_silver_base_tables(patched_delta_io):
    """Create minimal silver tables for plays, tracks, artists."""
    # Plays: one user with a played track (track_id format: "track|artist")
    plays_df = pl.DataFrame(
        {
            "username": ["user1"],
            "track_id": ["Known|Artist A"],
        }
    )
    patched_delta_io("silver").write_delta(
        plays_df, table_name="plays", mode="overwrite"
    )

    # Tracks: map track_id -> artist_id and include tags
    tracks_df = pl.DataFrame(
        {
            "track_id": [
                "Known|Artist A",
                "New Track|Artist B",
                "Tag Track|Artist C",
            ],
            "artist_id": ["a1", "b1", "c1"],
            "tags": ["rock,alt", "indie", "rock"],
        }
    )
    patched_delta_io("silver").write_delta(
        tracks_df, table_name="tracks", mode="overwrite"
    )

    # Artists: map artist_id -> artist_name and optional tags
    artists_df = pl.DataFrame(
        {
            "artist_id": ["a1", "b1", "c1"],
            "artist_name": ["Artist A", "Artist B", "Artist C"],
            "tags": ["indie,alt", "indie", "rock"],
        }
    )
    patched_delta_io("silver").write_delta(
        artists_df, table_name="artists", mode="overwrite"
    )


class TestSimilarArtistCandidates:
    def test_generates_and_filters(self, test_data_dir, patched_delta_io):
        _write_silver_base_tables(patched_delta_io)

        with (
            patch(
                "music_airflow.transform.candidate_generation.LastFMClient"
            ) as MockClient,
            patch(
                "music_airflow.transform.candidate_generation.PolarsDeltaIOManager"
            ) as MockDeltaIO,
        ):
            client = MockClient.return_value
            # Similar artists for Artist A: include a clone (match>0.9) which should be filtered
            client.get_similar_artists.return_value = [
                {"name": "Artist B", "match": 0.5},
                {"name": "Artist A", "match": 0.95},  # filtered out
            ]
            # Top tracks for Artist B: include one below min_listeners to be filtered
            client.get_artist_top_tracks.return_value = [
                {
                    "name": "New Track",
                    "mbid": "tmbid",
                    "artist": {"name": "Artist B", "mbid": "bmbid"},
                    "listeners": 5000,
                    "playcount": 10000,
                },
                {
                    "name": "Too Small",
                    "mbid": "",
                    "artist": {"name": "Artist B", "mbid": "bmbid"},
                    "listeners": 10,
                    "playcount": 20,
                },
            ]

            # Patch IO manager construction to use our preconfigured instances
            MockDeltaIO.side_effect = lambda medallion_layer="silver": patched_delta_io(
                medallion_layer
            )

            result = generate_similar_artist_candidates(
                username="user1",
                artist_sample_rate=1.0,
            )

        # Validate metadata and Delta output
        assert result["table_name"] == "candidate_similar_artist"
        out_path = Path(result["path"])  # data/silver/candidate_similar_artist
        assert out_path.exists()
        df = pl.read_delta(str(out_path))
        # Only New Track should remain (clone and small filtered); prefer MBID when present
        assert df["track_id"].to_list() == ["tmbid"]
        assert df["username"].to_list() == ["user1"]


class TestSimilarTagCandidates:
    def test_excludes_played_and_respects_min_listeners(
        self, test_data_dir, patched_delta_io
    ):
        _write_silver_base_tables(patched_delta_io)

        with (
            patch(
                "music_airflow.transform.candidate_generation.LastFMClient"
            ) as MockClient,
            patch(
                "music_airflow.transform.candidate_generation.PolarsDeltaIOManager"
            ) as MockDeltaIO,
        ):
            client = MockClient.return_value
            # Expand tags for user's tags (rock, alt, indie)
            client.get_similar_tags.side_effect = lambda tag: [
                {"name": "indie"},
                {"name": "alternative"},
            ]
            # For top tracks per tag, include one already played and one valid
            client.get_tag_top_tracks.side_effect = lambda tag, limit=10: [
                {
                    "name": "Known",
                    "mbid": "",
                    "artist": {"name": "Artist A", "mbid": "a_mbid"},
                    "listeners": 5000,
                    "playcount": 8000,
                },
                {
                    "name": "Tag Track",
                    "mbid": "tm2",
                    "artist": {"name": "Artist C", "mbid": "c_mbid"},
                    "listeners": 6000,
                    "playcount": 9000,
                },
                {
                    "name": "Too Small",
                    "mbid": "",
                    "artist": {"name": "Artist Z", "mbid": "z_mbid"},
                    "listeners": 100,  # filtered by min_listeners default 1000
                    "playcount": 200,
                },
            ]

            MockDeltaIO.side_effect = lambda medallion_layer="silver": patched_delta_io(
                medallion_layer
            )

            result = generate_similar_tag_candidates(
                username="user1",
                tag_sample_rate=1.0,
            )

        assert result["table_name"] == "candidate_similar_tag"
        out_path = Path(result["path"])  # data/silver/candidate_similar_tag
        assert out_path.exists()
        df = pl.read_delta(str(out_path))
        # Should exclude already played "Known|Artist A" and the too-small one; prefer MBID when present
        assert df["track_id"].to_list() == ["tm2"]
        assert df["username"].to_list() == ["user1"]


class TestDeepCutCandidates:
    def test_generation_and_filters(self, test_data_dir, patched_delta_io):
        _write_silver_base_tables(patched_delta_io)

        with (
            patch(
                "music_airflow.transform.candidate_generation.LastFMClient"
            ) as MockClient,
            patch(
                "music_airflow.transform.candidate_generation.PolarsDeltaIOManager"
            ) as MockDeltaIO,
        ):
            client = MockClient.return_value
            # Top albums for Artist A, include one valid album
            client.get_artist_top_albums.return_value = [
                {
                    "name": "Obscure Album",
                    "playcount": 10000,
                    "artist": {"mbid": "a_mbid"},
                },
                {
                    "name": "Too Popular",
                    "playcount": 1000000,
                    "artist": {"mbid": "a_mbid"},
                },
            ]
            # Album info returns track list including one already played
            client.get_album_info.side_effect = lambda album_name, artist_name: {
                "tracks": {
                    "track": [
                        {"name": "Hidden Gem", "mbid": "hg_mbid"},
                        {"name": "Known", "mbid": ""},
                    ]
                }
            }

            MockDeltaIO.side_effect = lambda medallion_layer="silver": patched_delta_io(
                medallion_layer
            )
            result = generate_deep_cut_candidates(username="user1")

        assert result["table_name"] == "candidate_deep_cut"
        out_path = Path(result["path"])  # data/silver/candidate_deep_cut
        assert out_path.exists()
        df = pl.read_delta(str(out_path))
        # Only Hidden Gem should remain; Too Popular filtered by max_listeners; prefer MBID when present
        assert df["track_id"].to_list() == ["hg_mbid"]
        assert df["album_name"].to_list() == ["Obscure Album"]


class TestMergeCandidateSources:
    def test_dedup_and_source_flags(self, test_data_dir, patched_delta_io):
        # Write synthetic silver candidate tables directly to test merge logic
        similar_artist_df = pl.DataFrame(
            {
                "username": ["user1"],
                "track_id": ["artist b::new track"],
                "track_name": ["New Track"],
                "track_mbid": ["tmbid"],
                "artist_name": ["Artist B"],
                "artist_mbid": ["bmbid"],
                "listeners": [5000],
                "playcount": [10000],
                "score": [10000.0],
            }
        )
        patched_delta_io("silver").write_delta(
            similar_artist_df, table_name="candidate_similar_artist", mode="overwrite"
        )

        similar_tag_df = pl.DataFrame(
            {
                "username": ["user1", "user1"],
                "track_id": ["artist b::new track", "artist c::tag track"],
                "track_name": ["New Track", "Tag Track"],
                "track_mbid": ["tmbid", "tm2"],
                "artist_name": ["Artist B", "Artist C"],
                "artist_mbid": ["bmbid", "cmbid"],
                "listeners": [5000, 6000],
                "playcount": [10000, 9000],
                "score": [10000.0, 9000.0],
            }
        )
        patched_delta_io("silver").write_delta(
            similar_tag_df, table_name="candidate_similar_tag", mode="overwrite"
        )

        # Ensure column order aligns with similar_* tables for concat
        deep_cut_df = pl.DataFrame(
            {
                "username": ["user1"],
                "track_id": ["artist c::tag track"],
                "track_name": ["Tag Track"],
                "track_mbid": ["tm2"],
                "artist_name": ["Artist C"],
                "artist_mbid": ["cmbid"],
                "listeners": [6000],
                "playcount": [6000],
                "score": [0.0001],
                "album_name": ["Album X"],
            }
        )
        patched_delta_io("silver").write_delta(
            deep_cut_df, table_name="candidate_deep_cut", mode="overwrite"
        )

        with patch(
            "music_airflow.transform.candidate_generation.PolarsDeltaIOManager"
        ) as MockDeltaIO:
            MockDeltaIO.side_effect = lambda medallion_layer="silver": patched_delta_io(
                medallion_layer
            )
            result = merge_candidate_sources(username="user1")
        assert result["table_name"] == "track_candidates"
        out_path = Path(result["path"])  # data/gold/track_candidates
        assert out_path.exists()
        merged = pl.read_delta(str(out_path)).sort("track_id")

        # Expect two deduped rows with correct flags
        assert merged.shape[0] == 2
        rows = {row["track_id"]: row for row in merged.iter_rows(named=True)}

        b_row = rows["artist b::new track"]
        assert b_row["similar_artist"] is True
        assert b_row["similar_tag"] is True
        assert b_row["deep_cut_same_artist"] is False

        c_row = rows["artist c::tag track"]
        assert c_row["similar_artist"] is False
        assert c_row["similar_tag"] is True
        assert c_row["deep_cut_same_artist"] is True
