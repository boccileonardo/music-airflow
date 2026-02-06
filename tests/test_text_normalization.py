"""
Unit tests for text normalization utilities.
"""

import polars as pl

from music_airflow.utils.text_normalization import (
    normalize_text,
    is_music_video,
    generate_canonical_track_id,
    generate_canonical_artist_id,
    normalize_text_expr,
    generate_canonical_track_id_expr,
    generate_canonical_artist_id_expr,
    is_music_video_expr,
)


class TestNormalizeText:
    """Tests for normalize_text function."""

    def test_basic_normalization(self):
        """Test basic lowercase and whitespace normalization."""
        assert normalize_text("Highway Star") == "highway star"
        assert normalize_text("  Track Name  ") == "track name"
        assert normalize_text("UPPERCASE") == "uppercase"

    def test_remaster_removal(self):
        """Test removal of remaster indicators."""
        assert normalize_text("Song (Remastered)") == "song"
        assert normalize_text("Song (Remastered 2012)") == "song"
        assert normalize_text("Song (2012 Remaster)") == "song"
        assert normalize_text("Song [Remastered]") == "song"
        assert normalize_text("Song (Re-Mastered)") == "song"

    def test_live_removal(self):
        """Test removal of live indicators."""
        assert normalize_text("Song (Live)") == "song"
        assert normalize_text("Song (Live at Wembley)") == "song"
        assert normalize_text("Song (Live from NYC)") == "song"
        assert normalize_text("Song [Live 1995]") == "song"
        assert normalize_text("Song (Live Recording)") == "song"

    def test_version_removal(self):
        """Test removal of version indicators."""
        assert normalize_text("Song (Radio Edit)") == "song"
        assert normalize_text("Song (Album Version)") == "song"
        assert normalize_text("Song (Extended Mix)") == "song"
        assert normalize_text("Song (Acoustic Version)") == "song"
        assert normalize_text("Song (Single Version)") == "song"
        assert normalize_text("Song (Remix)") == "song"

    def test_explicit_removal(self):
        """Test removal of explicit/clean indicators."""
        assert normalize_text("Song (Explicit)") == "song"
        assert normalize_text("Song (Clean)") == "song"
        assert normalize_text("Song (Censored)") == "song"

    def test_audio_format_removal(self):
        """Test removal of stereo/mono indicators."""
        assert normalize_text("Song (Stereo)") == "song"
        assert normalize_text("Song (Mono)") == "song"
        assert normalize_text("Song (Stereo Mix)") == "song"
        assert normalize_text("Song (Mono Version)") == "song"
        assert normalize_text("Surfin' U.S.A. (Stereo)") == "surfin usa"

    def test_featured_artist_removal(self):
        """Test removal of featured artist credits."""
        assert normalize_text("Song (feat. Artist)") == "song"
        assert normalize_text("Song (ft. Artist)") == "song"
        assert normalize_text("Song (featuring Artist)") == "song"
        assert normalize_text("Song feat. Artist") == "song"
        assert normalize_text("Song (with Artist)") == "song"
        assert normalize_text("Song (vs. Artist)") == "song"

    def test_year_removal(self):
        """Test removal of years."""
        assert normalize_text("Song (2020)") == "song"
        assert normalize_text("Song [1999]") == "song"
        assert normalize_text("Song 2015") == "song"

    def test_punctuation_removal(self):
        """Test punctuation removal while keeping hyphens."""
        assert normalize_text("Song! Name?") == "song name"
        assert normalize_text("Song's Name") == "songs name"
        assert normalize_text("Hip-Hop") == "hip-hop"
        assert normalize_text("Song, Name") == "song name"

    def test_complex_normalization(self):
        """Test complex cases with multiple patterns."""
        assert (
            normalize_text("Highway Star (Remastered 2012) (Live at Wembley)")
            == "highway star"
        )
        # Radio Edit should be removed - it's a version indicator
        assert (
            normalize_text("Song Name (feat. Artist) - Radio Edit (2020)")
            == "song name"
        )
        assert normalize_text("Track!!! (Explicit) [Remastered]") == "track"

    def test_dash_separated_suffixes(self):
        """Test removal of dash-separated suffixes like '- Demo', '- Early Take'."""
        # Demo patterns
        assert normalize_text("Song - Demo") == "song"
        assert normalize_text("Song - Early Demo") == "song"
        assert normalize_text("Song - Band Instrumental Demo") == "song"

        # Take patterns
        assert normalize_text("Song - Early Take") == "song"
        assert normalize_text("Dreams - Take 2") == "dreams"
        assert normalize_text("Songbird - Instrumental Take 10") == "songbird"

        # Live/venue patterns
        assert (
            normalize_text("Pictures Of You - Live At Wembley / 1989")
            == "pictures of you"
        )
        assert (
            normalize_text("Summer Of '69 - Live At Hammersmith Odeon/1985")
            == "summer of 69"
        )

        # Video/audio patterns
        assert normalize_text("Song - Official Video") == "song"
        assert normalize_text("Song - Official Music Video") == "song"
        assert normalize_text("Song - Lyric Video") == "song"
        assert normalize_text("Song - Visualizer") == "song"

        # Recording session patterns
        assert normalize_text("Gold Dust Woman - Early Take") == "gold dust woman"
        assert normalize_text("Song - Sessions Roughs Outtakes") == "song"
        assert normalize_text("Song - Studio Guide Vocal Rough") == "song"
        assert normalize_text("Song - Band Instrumental Rehearsal") == "song"

        # Alternate versions
        assert normalize_text("Four Sticks - Alternate Mix") == "four sticks"
        assert normalize_text("Song - Acoustic Version") == "song"
        assert (
            normalize_text("Doesn't Anything Last - Acoustic Duet")
            == "doesnt anything last"
        )

        # Audio format with dash
        assert (
            normalize_text("From Me To You - Mono / Remastered 2015")
            == "from me to you"
        )

    def test_trailing_suffixes_without_dash(self):
        """Test removal of trailing suffixes without dash separator."""
        # Demo at end
        assert normalize_text("Song Name demo") == "song name"
        assert normalize_text("Some Might Say demo") == "some might say"

        # Video/audio at end
        assert normalize_text("Song official video") == "song"
        assert normalize_text("Song official music video") == "song"
        assert normalize_text("Song lyric video") == "song"
        assert normalize_text("Song Name official audio") == "song name"
        assert normalize_text("Stairway To Heaven visualizer") == "stairway to heaven"

        # Live location at end
        assert normalize_text("Lullaby at Wembley") == "lullaby"
        assert normalize_text("Pictures of You at Wembley") == "pictures of you"

        # Audio format at end
        assert normalize_text("God Only Knows mono") == "god only knows"
        assert normalize_text("Surfin USA stereo") == "surfin usa"

        # Excerpt at end
        assert normalize_text("The Swamp Song excerpt 1") == "the swamp song"
        assert normalize_text("The Swamp Song excerpt 2") == "the swamp song"

        # Dub at end
        assert (
            normalize_text("Big Love House on the Hill dub")
            == "big love house on the hill"
        )

    def test_empty_and_none(self):
        """Test edge cases with empty strings."""
        assert normalize_text("") == ""
        assert normalize_text("   ") == ""

    def test_special_characters(self):
        """Test handling of special characters."""
        assert normalize_text("Café") == "café"
        assert normalize_text("Naïve") == "naïve"


class TestIsMusicVideo:
    """Tests for is_music_video function."""

    def test_music_video_detection(self):
        """Test detection of music video indicators."""
        assert is_music_video("Song (Official Video)") is True
        assert is_music_video("Song (Music Video)") is True
        assert is_music_video("Song (Video Clip)") is True
        assert is_music_video("Song [Official Audio]") is True
        assert is_music_video("Song (Visualizer)") is True
        assert is_music_video("Song (Lyric Video)") is True

    def test_non_video_tracks(self):
        """Test that non-video tracks return False."""
        assert is_music_video("Song Name") is False
        assert is_music_video("Song (Live)") is False
        assert is_music_video("Song (Remastered)") is False
        assert is_music_video("Song (Radio Edit)") is False

    def test_case_insensitive(self):
        """Test case-insensitive detection."""
        assert is_music_video("Song (OFFICIAL VIDEO)") is True
        assert is_music_video("Song (music video)") is True

    def test_empty_and_none(self):
        """Test edge cases."""
        assert is_music_video("") is False


class TestGenerateCanonicalTrackId:
    """Tests for generate_canonical_track_id function."""

    def test_basic_id_generation(self):
        """Test basic track ID generation."""
        assert (
            generate_canonical_track_id("Highway Star", "Deep Purple")
            == "highway star|deep purple"
        )
        assert (
            generate_canonical_track_id("Bohemian Rhapsody", "Queen")
            == "bohemian rhapsody|queen"
        )

    def test_normalized_id_generation(self):
        """Test ID generation with normalization."""
        # Same track with different versions should produce same ID
        assert generate_canonical_track_id(
            "Highway Star (Remastered)", "Deep Purple"
        ) == generate_canonical_track_id("Highway Star", "Deep Purple")

        assert generate_canonical_track_id(
            "Song (Live)", "Artist"
        ) == generate_canonical_track_id("Song", "Artist")

        assert generate_canonical_track_id(
            "Track (feat. Other)", "Artist"
        ) == generate_canonical_track_id("Track", "Artist")

    def test_consistent_ids_across_versions(self):
        """Test that different recordings produce the same canonical ID."""
        artist_name = "Led Zeppelin"

        versions = [
            "Stairway to Heaven",
            "Stairway to Heaven (Remastered)",
            "Stairway to Heaven (Live)",
            "Stairway to Heaven (2012 Remaster)",
            "Stairway to Heaven [Remastered]",
        ]

        canonical_ids = [
            generate_canonical_track_id(version, artist_name) for version in versions
        ]

        # All should produce the same canonical ID
        assert len(set(canonical_ids)) == 1
        assert canonical_ids[0] == "stairway to heaven|led zeppelin"

    def test_stereo_mono_versions_same_id(self):
        """Test that stereo/mono versions produce the same canonical ID."""
        # The Beach Boys Surfin' U.S.A. example
        artist = "The Beach Boys"
        versions = [
            "Surfin' U.S.A.",
            "Surfin' U.S.A. (Stereo)",
            "Surfin' U.S.A. (Mono)",
            "Surfin' U.S.A. (Stereo Mix)",
        ]

        canonical_ids = [
            generate_canonical_track_id(version, artist) for version in versions
        ]

        # All should produce the same canonical ID
        assert len(set(canonical_ids)) == 1
        assert canonical_ids[0] == "surfin usa|the beach boys"

    def test_edge_cases(self):
        """Test edge cases."""
        # Empty strings should fallback to lowercase original
        assert generate_canonical_track_id("", "") == "|"
        assert generate_canonical_track_id("Track", "") == "track|"
        assert generate_canonical_track_id("", "Artist") == "|artist"


class TestGenerateCanonicalArtistId:
    """Tests for generate_canonical_artist_id function."""

    def test_basic_artist_id(self):
        """Test basic artist ID generation."""
        assert generate_canonical_artist_id("Deep Purple") == "deep purple"
        assert generate_canonical_artist_id("The Beatles") == "the beatles"

    def test_artist_normalization(self):
        """Test artist name normalization."""
        assert generate_canonical_artist_id("Queen") == "queen"
        assert generate_canonical_artist_id("QUEEN") == "queen"
        assert generate_canonical_artist_id("  Queen  ") == "queen"

    def test_edge_cases(self):
        """Test edge cases."""
        assert generate_canonical_artist_id("") == ""


class TestIntegration:
    """Integration tests for common use cases."""

    def test_real_world_track_variations(self):
        """Test with real-world track name variations."""
        # These should all produce the same canonical ID
        variations = [
            ("Highway Star", "Deep Purple"),
            ("Highway Star (Remastered 2012)", "Deep Purple"),
            ("Highway Star (Live at Wembley)", "Deep Purple"),
            ("Highway Star - Remastered", "Deep Purple"),
            ("Highway Star (Official Audio)", "Deep Purple"),
        ]

        canonical_ids = [
            generate_canonical_track_id(track, artist) for track, artist in variations
        ]

        base_id = "highway star|deep purple"
        for cid in canonical_ids:
            assert cid == base_id

    def test_fleetwood_mac_variations(self):
        """Test Fleetwood Mac track variations from real data."""
        artist = "Fleetwood Mac"

        # Gold Dust Woman variations - all should be same
        gdw_versions = [
            "Gold Dust Woman",
            "Gold Dust Woman - Early Take",
            "Gold Dust Woman - 2004 Remastered Edition",
            "Gold Dust Woman - 2004 Remaster",
        ]
        gdw_ids = [generate_canonical_track_id(v, artist) for v in gdw_versions]
        assert len(set(gdw_ids)) == 1
        assert gdw_ids[0] == "gold dust woman|fleetwood mac"

        # Never Going Back Again variations
        ngba_versions = [
            "Never Going Back Again",
            "Never Going Back Again - Acoustic Duet",
            "Never Going Back Again - (Instrumental",
            "Never Going Back Again - 2004 Remastered Edition",
            "Never Going Back Again (early demo)",
        ]
        ngba_ids = [generate_canonical_track_id(v, artist) for v in ngba_versions]
        assert len(set(ngba_ids)) == 1
        assert ngba_ids[0] == "never going back again|fleetwood mac"

    def test_oasis_demo_variations(self):
        """Test Oasis demo track variations."""
        artist = "Oasis"

        # Bonehead's Bank Holiday
        boneheads_versions = [
            "Bonehead's Bank Holiday",
            "Bonehead's Bank Holiday (demo)",
            "Bonehead's Bank Holiday (Remastered)",
        ]
        boneheads_ids = [
            generate_canonical_track_id(v, artist) for v in boneheads_versions
        ]
        assert len(set(boneheads_ids)) == 1
        assert boneheads_ids[0] == "boneheads bank holiday|oasis"

    def test_van_morrison_official_audio(self):
        """Test Van Morrison official audio variation."""
        artist = "Van Morrison"

        versions = [
            "Brown Eyed Girl",
            "Brown Eyed Girl (Official Audio)",
        ]
        ids = [generate_canonical_track_id(v, artist) for v in versions]
        assert len(set(ids)) == 1
        assert ids[0] == "brown eyed girl|van morrison"

    def test_led_zeppelin_variations(self):
        """Test Led Zeppelin track variations."""
        artist = "Led Zeppelin"

        # Stairway variations
        stairway_versions = [
            "Stairway To Heaven",
            "Stairway To Heaven (Remaster)",
            "Stairway To Heaven - Sunset Sound Mix",
            "Stairway To Heaven - Remaster (Visualizer)",
        ]
        stairway_ids = [
            generate_canonical_track_id(v, artist) for v in stairway_versions
        ]
        assert len(set(stairway_ids)) == 1

    def test_cure_live_and_demo_variations(self):
        """Test The Cure live and demo variations."""
        artist = "The Cure"

        # Pictures of You
        poy_versions = [
            "Pictures of You",
            "Pictures Of You - RS Home Instrumental Demo",
            "Pictures Of You - Live At Wembley / 1989",
        ]
        poy_ids = [generate_canonical_track_id(v, artist) for v in poy_versions]
        assert len(set(poy_ids)) == 1
        assert poy_ids[0] == "pictures of you|the cure"

    def test_video_suffix_removal(self):
        """Test removal of video-related suffixes."""
        # Official music video
        assert generate_canonical_track_id(
            "Wanted Dead Or Alive (Official Music Video)", "Bon Jovi"
        ) == generate_canonical_track_id("Wanted Dead or Alive", "Bon Jovi")

        # Lyric video
        assert generate_canonical_track_id(
            "Sweet Home Alabama (Lyric Video)", "Lynyrd Skynyrd"
        ) == generate_canonical_track_id("Sweet Home Alabama", "Lynyrd Skynyrd")

        # HD video
        assert generate_canonical_track_id(
            "We Didn't Start the Fire (Official HD Video)", "Billy Joel"
        ) == generate_canonical_track_id("We Didn't Start the Fire", "Billy Joel")

    def test_music_video_vs_audio(self):
        """Test music video detection on real examples."""
        assert is_music_video("Bohemian Rhapsody (Official Video)") is True
        assert is_music_video("Bohemian Rhapsody") is False
        assert is_music_video("Bohemian Rhapsody (Official Audio)") is True

    def test_unicode_handling(self):
        """Test handling of unicode characters."""
        # Should preserve unicode in normalized form
        track_id = generate_canonical_track_id("Café del Mar", "Artíst")
        assert "café" in track_id.lower()
        assert "artíst" in track_id.lower()


class TestPolarsExpressions:
    """Tests verifying Polars expressions produce same results as Python functions."""

    def test_normalize_text_expr_matches_python(self):
        """Verify normalize_text_expr produces same output as normalize_text."""
        test_cases = [
            "Highway Star",
            "Song (Remastered 2012)",
            "Bohemian Rhapsody (Live)",
            "Track - Radio Edit",
            "Song (feat. Artist)",
            "Song Name demo",
            "Track (Official Video)",
            "  Spaced  Text  ",
            "UPPERCASE TRACK",
        ]

        df = pl.DataFrame({"text": test_cases})
        result = (
            df.with_columns(normalized=normalize_text_expr("text"))
            .select("normalized")["normalized"]
            .to_list()
        )

        expected = [normalize_text(t) for t in test_cases]
        assert result == expected

    def test_generate_canonical_track_id_expr_matches_python(self):
        """Verify generate_canonical_track_id_expr produces same output as Python function."""
        test_cases = [
            ("Highway Star", "Deep Purple"),
            ("Song (Remastered)", "Artist"),
            ("Track (Live at Wembley)", "Band Name"),
            ("Bohemian Rhapsody", "Queen"),
        ]

        df = pl.DataFrame(
            {
                "track": [t[0] for t in test_cases],
                "artist": [t[1] for t in test_cases],
            }
        )
        result = (
            df.with_columns(
                track_id=generate_canonical_track_id_expr("track", "artist")
            )
            .select("track_id")["track_id"]
            .to_list()
        )

        expected = [generate_canonical_track_id(t, a) for t, a in test_cases]
        assert result == expected

    def test_generate_canonical_artist_id_expr_matches_python(self):
        """Verify generate_canonical_artist_id_expr produces same output as Python function."""
        test_cases = ["Deep Purple", "The Beatles", "QUEEN", "  Spaced Band  "]

        df = pl.DataFrame({"artist": test_cases})
        result = (
            df.with_columns(artist_id=generate_canonical_artist_id_expr("artist"))
            .select("artist_id")["artist_id"]
            .to_list()
        )

        expected = [generate_canonical_artist_id(a) for a in test_cases]
        assert result == expected

    def test_is_music_video_expr_matches_python(self):
        """Verify is_music_video_expr produces same output as Python function."""
        test_cases = [
            "Song Name",
            "Song (Official Video)",
            "Track (Music Video)",
            "Album Track",
            "Song - Lyric Video",
            "Track (Visualizer)",
            "Regular Song",
        ]

        df = pl.DataFrame({"track": test_cases})
        result = (
            df.with_columns(is_video=is_music_video_expr("track"))
            .select("is_video")["is_video"]
            .to_list()
        )

        expected = [is_music_video(t) for t in test_cases]
        assert result == expected
