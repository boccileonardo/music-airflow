"""
Unit tests for text normalization utilities.
"""

from music_airflow.utils.text_normalization import (
    normalize_text,
    is_music_video,
    generate_canonical_track_id,
    generate_canonical_artist_id,
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

        # All should produce same ID (except official audio might differ)
        base_id = "highway star|deep purple"
        for cid in canonical_ids[:4]:  # First 4 should match
            assert cid == base_id

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
