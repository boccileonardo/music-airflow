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
    # Plays: one user with a played track (track_id format: "track|artist" normalized)
    plays_df = pl.DataFrame(
        {
            "username": ["user1"],
            "track_id": ["known|artist a"],  # normalized
        }
    )
    patched_delta_io("silver").write_delta(
        plays_df, table_name="plays", mode="overwrite"
    )

    # Tracks: map track_id -> artist_id and include tags
    tracks_df = pl.DataFrame(
        {
            "track_id": [
                "known|artist a",  # normalized
                "new track|artist b",
                "tag track|artist c",
            ],
            "artist_id": ["a1", "b1", "c1"],
        }
    )
    patched_delta_io("silver").write_delta(
        tracks_df, table_name="tracks", mode="overwrite"
    )

    # Artists: map artist_id -> artist_name and tags
    # Include artist_mbid for deep cut filtering
    artists_df = pl.DataFrame(
        {
            "artist_id": ["a1", "b1", "c1"],
            "artist_name": ["Artist A", "Artist B", "Artist C"],
            "artist_mbid": ["a_mbid", "b_mbid", "c_mbid"],
            "tags": ["rock,indie,alt", "indie", "rock"],
        }
    )
    patched_delta_io("silver").write_delta(
        artists_df, table_name="artists", mode="overwrite"
    )

    # Gold artist_play_count: user's play stats for each artist
    artist_play_counts_df = pl.DataFrame(
        {
            "username": ["user1", "user1", "user1"],
            "artist_id": ["a1", "b1", "c1"],
            "artist_name": ["Artist A", "Artist B", "Artist C"],
            "play_count": [10, 5, 3],
        }
    )
    patched_delta_io("gold").write_delta(
        artist_play_counts_df, table_name="artist_play_count", mode="overwrite"
    )


class TestSimilarArtistCandidates:
    @pytest.mark.asyncio
    async def test_generates_and_filters(self, test_data_dir, patched_delta_io):
        _write_silver_base_tables(patched_delta_io)

        with (
            patch(
                "music_airflow.transform.candidate_generation.LastFMClient"
            ) as MockClient,
            patch(
                "music_airflow.transform.candidate_generation.PolarsDeltaIOManager"
            ) as MockDeltaIO,
        ):
            from unittest.mock import AsyncMock

            client = MockClient.return_value
            client.__aenter__ = AsyncMock(return_value=client)
            client.__aexit__ = AsyncMock(return_value=None)
            # Similar artists for Artist A: include a clone (match>0.9) which should be filtered
            client.get_similar_artists = AsyncMock(
                return_value=[
                    {"name": "Artist B", "match": 0.5},
                    {"name": "Artist A", "match": 0.95},  # filtered out
                ]
            )
            # Top tracks for Artist B: include one below min_listeners to be filtered
            client.get_artist_top_tracks = AsyncMock(
                return_value=[
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
            )

            # Patch IO manager construction to use our preconfigured instances
            MockDeltaIO.side_effect = lambda medallion_layer="silver": patched_delta_io(
                medallion_layer
            )

            result = await generate_similar_artist_candidates(
                username="user1",
                artist_sample_rate=1.0,
            )

        # Validate metadata and Delta output
        assert result["table_name"] == "candidate_similar_artist"
        out_path = Path(result["path"])  # data/silver/candidate_similar_artist
        assert out_path.exists()
        df = pl.read_delta(str(out_path))
        # Only New Track should remain (clone and small filtered); uses canonical ID
        assert df["track_id"].to_list() == ["new track|artist b"]
        assert df["username"].to_list() == ["user1"]


class TestSimilarTagCandidates:
    @pytest.mark.asyncio
    async def test_tag_profile_matching_with_min_matches(
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
            from unittest.mock import AsyncMock

            client = MockClient.return_value
            client.__aenter__ = AsyncMock(return_value=client)
            client.__aexit__ = AsyncMock(return_value=None)

            # Return top tracks for each tag (rock, indie, alt)
            # Track A appears in multiple tags (meets min_tag_matches=3)
            # Track B appears in only 1 tag (filtered out)
            async def mock_get_tag_top_tracks(tag, limit=30):
                if tag == "rock":
                    return [
                        {
                            "name": "Rock Track A",
                            "mbid": "rta_mbid",
                            "artist": {"name": "Artist D", "mbid": "d_mbid"},
                            "@attr": {"rank": "1"},
                        },
                        {
                            "name": "Rock Track B",
                            "mbid": "rtb_mbid",
                            "artist": {"name": "Artist E", "mbid": "e_mbid"},
                            "@attr": {"rank": "2"},
                        },
                    ]
                elif tag == "indie":
                    return [
                        {
                            "name": "Rock Track A",  # Same track, different tag
                            "mbid": "rta_mbid",
                            "artist": {"name": "Artist D", "mbid": "d_mbid"},
                            "@attr": {"rank": "3"},
                        },
                    ]
                elif tag == "alt":
                    return [
                        {
                            "name": "Rock Track A",  # Same track, third tag
                            "mbid": "rta_mbid",
                            "artist": {"name": "Artist D", "mbid": "d_mbid"},
                            "@attr": {"rank": "5"},
                        },
                    ]
                return []

            client.get_tag_top_tracks = mock_get_tag_top_tracks

            MockDeltaIO.side_effect = lambda medallion_layer="silver": patched_delta_io(
                medallion_layer
            )

            result = await generate_similar_tag_candidates(
                username="user1",
                top_tags_count=3,
                min_tag_matches=3,  # explicitly test with 3 matches
            )

        assert result["table_name"] == "candidate_similar_tag"
        out_path = Path(result["path"])
        assert out_path.exists()
        df = pl.read_delta(str(out_path))

        # Only Rock Track A should be in results (appears in 3 tags)
        # Rock Track B only appears in 1 tag, filtered out
        assert len(df) == 1
        assert df["track_id"][0] == "rock track a|artist d"
        assert df["tag_match_count"][0] == 3
        # Tags are sorted alphabetically when collected
        assert set(df["source_tags"][0].split(",")) == {"rock", "indie", "alt"}
        # Score = tag_match_count
        assert df["score"][0] == 3.0


class TestDeepCutCandidates:
    @pytest.mark.asyncio
    async def test_generation_and_filters(self, test_data_dir, patched_delta_io):
        _write_silver_base_tables(patched_delta_io)

        with (
            patch(
                "music_airflow.transform.candidate_generation.LastFMClient"
            ) as MockClient,
            patch(
                "music_airflow.transform.candidate_generation.PolarsDeltaIOManager"
            ) as MockDeltaIO,
        ):
            from unittest.mock import AsyncMock

            client = MockClient.return_value
            client.__aenter__ = AsyncMock(return_value=client)
            client.__aexit__ = AsyncMock(return_value=None)

            # Mock get_artist_top_albums to return albums only for Artist A
            async def mock_top_albums(artist_name, limit=10):
                if artist_name == "Artist A":
                    return [
                        {
                            "name": "Album One",
                            "playcount": 10000,
                            "artist": {"mbid": "a_mbid"},
                        },
                        {
                            "name": "Album Two",
                            "playcount": 1000000,
                            "artist": {"mbid": "a_mbid"},
                        },
                    ]
                else:
                    # Other artists have no albums above threshold
                    return []

            client.get_artist_top_albums = mock_top_albums

            # Album info returns track list including one already played
            async def mock_album_info(album_name, artist_name):
                if album_name == "Album One":
                    return {
                        "tracks": {
                            "track": [
                                {"name": "Hidden Gem", "mbid": "hg_mbid"},
                                {"name": "Known", "mbid": ""},
                            ]
                        }
                    }
                else:  # Album Two
                    return {
                        "tracks": {
                            "track": [
                                {"name": "Popular Track", "mbid": "pt_mbid"},
                            ]
                        }
                    }

            client.get_album_info = mock_album_info

            MockDeltaIO.side_effect = lambda medallion_layer="silver": patched_delta_io(
                medallion_layer
            )
            result = await generate_deep_cut_candidates(username="user1")

        assert result["table_name"] == "candidate_deep_cut"
        out_path = Path(result["path"])  # data/silver/candidate_deep_cut
        assert out_path.exists()
        df = pl.read_delta(str(out_path)).sort("track_id")
        # Both Hidden Gem and Popular Track should be included (no max_listeners filter)
        # Known|Artist A is filtered because already played
        assert df["track_id"].to_list() == [
            "hidden gem|artist a",
            "popular track|artist a",
        ]


class TestMergeCandidateSources:
    def test_dedup_and_source_flags(self, test_data_dir, patched_delta_io):
        # Write plays table (empty for this user to not filter candidates)
        plays_df = pl.DataFrame(
            {
                "username": ["user1"],
                "track_id": ["some_other_track"],
            }
        )
        patched_delta_io("silver").write_delta(
            plays_df, table_name="plays", mode="overwrite"
        )

        # Write dimension tables needed for joins
        tracks_df = pl.DataFrame(
            {
                "track_id": ["artist b::new track", "artist c::tag track"],
                "track_name": ["New Track", "Tag Track"],
                "track_mbid": ["tmbid", "tm2"],
                "artist_name": ["Artist B", "Artist C"],
                "artist_id": ["b1", "c1"],
                "duration_ms": [180000, 240000],
                "tags": ["rock,indie", "rock,pop"],
                "listeners": [5000, 6000],
                "playcount": [10000, 6000],
                "youtube_url": [
                    "https://youtube.com/watch?v=1",
                    "https://youtube.com/watch?v=2",
                ],
                "spotify_url": [
                    "https://open.spotify.com/track/1",
                    "https://open.spotify.com/track/2",
                ],
            }
        )
        patched_delta_io("silver").write_delta(
            tracks_df, table_name="tracks", mode="overwrite"
        )

        artists_df = pl.DataFrame(
            {
                "artist_id": ["b1", "c1"],
                "artist_mbid": ["bmbid", "cmbid"],
            }
        )
        patched_delta_io("silver").write_delta(
            artists_df, table_name="artists", mode="overwrite"
        )

        # Write synthetic silver candidate tables with updated schema (includes artist_mbid)
        similar_artist_df = pl.DataFrame(
            {
                "username": ["user1"],
                "track_id": ["artist b::new track"],
                "track_name": ["New Track"],
                "artist_name": ["Artist B"],
                "track_mbid": ["tmbid"],
                "artist_mbid": ["bmbid"],
                "score": [10000],
                "source_artist_id": ["a1"],
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
                "artist_name": ["Artist B", "Artist C"],
                "track_mbid": ["tmbid", "tm2"],
                "artist_mbid": ["bmbid", "cmbid"],
                "tag_match_count": [3, 4],
                "avg_rank": [5.0, 3.0],
                "score": [3095.0, 4097.0],
                "source_tags": ["rock,indie,alt", "rock,indie,alt,pop"],
            }
        )
        patched_delta_io("silver").write_delta(
            similar_tag_df, table_name="candidate_similar_tag", mode="overwrite"
        )

        deep_cut_df = pl.DataFrame(
            {
                "username": ["user1"],
                "track_id": ["artist c::tag track"],
                "track_name": ["Tag Track"],
                "artist_name": ["Artist C"],
                "track_mbid": ["tm2"],
                "artist_mbid": ["cmbid"],
                "score": [6000.0],
                "source_artist_id": ["c1"],
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
        assert b_row["old_favorite"] is False
        # Track metadata SHOULD be present (needed for dimension extraction)
        assert b_row["track_name"] == "New Track"
        assert b_row["artist_name"] == "Artist B"
        # YouTube/Spotify URLs should be present (joined from tracks dimension)
        assert b_row["youtube_url"] == "https://youtube.com/watch?v=1"
        assert b_row["spotify_url"] == "https://open.spotify.com/track/1"

        c_row = rows["artist c::tag track"]
        assert c_row["similar_artist"] is False
        assert c_row["similar_tag"] is True
        assert c_row["deep_cut_same_artist"] is True
        assert c_row["old_favorite"] is False
        # Track metadata SHOULD be present
        assert c_row["track_name"] == "Tag Track"
        assert c_row["artist_name"] == "Artist C"
        # YouTube/Spotify URLs should be present
        assert c_row["youtube_url"] == "https://youtube.com/watch?v=2"
        assert c_row["spotify_url"] == "https://open.spotify.com/track/2"
