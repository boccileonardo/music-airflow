"""Tests for Last.fm scraper utility."""

import pytest
from music_airflow.utils.lastfm_scraper import LastFMScraper


class TestLastFMScraper:
    """Tests for LastFMScraper class."""

    def test_extract_streaming_links_with_all_platforms(self):
        """Test extracting links when all platforms are present."""
        html = """
        <html>
        <body>
            <a class="play-this-track-playlink play-this-track-playlink--youtube"
               data-youtube-url="https://www.youtube.com/watch?v=kDFFHLrzTDM">
                YouTube
            </a>
            <a class="play-this-track-playlink play-this-track-playlink--spotify"
               data-spotify-url="https://open.spotify.com/track/abc123">
                Spotify
            </a>
        </body>
        </html>
        """

        scraper = LastFMScraper()
        links = scraper._extract_streaming_links(html)

        assert links["youtube_url"] == "https://www.youtube.com/watch?v=kDFFHLrzTDM"
        assert links["spotify_url"] == "https://open.spotify.com/track/abc123"

    def test_extract_streaming_links_partial(self):
        """Test extracting links when only some platforms are present."""
        html = """
        <html>
        <body>
            <a class="play-this-track-playlink play-this-track-playlink--youtube"
               href="https://www.youtube.com/watch?v=abc">
                YouTube
            </a>
        </body>
        </html>
        """

        scraper = LastFMScraper()
        links = scraper._extract_streaming_links(html)

        assert links["youtube_url"] == "https://www.youtube.com/watch?v=abc"
        assert links["spotify_url"] is None

    def test_extract_streaming_links_none(self):
        """Test extracting links when no platforms are present."""
        html = """
        <html>
        <body>
            <p>No streaming links available</p>
        </body>
        </html>
        """

        scraper = LastFMScraper()
        links = scraper._extract_streaming_links(html)

        assert links["youtube_url"] is None
        assert links["spotify_url"] is None

    def test_extract_streaming_links_uses_href_fallback(self):
        """Test that href attribute is used as fallback when data-*-url is missing."""
        html = """
        <html>
        <body>
            <a class="play-this-track-playlink play-this-track-playlink--youtube"
               href="https://www.youtube.com/watch?v=fallback">
                YouTube
            </a>
        </body>
        </html>
        """

        scraper = LastFMScraper()
        links = scraper._extract_streaming_links(html)

        assert links["youtube_url"] == "https://www.youtube.com/watch?v=fallback"

    @pytest.mark.asyncio
    async def test_get_streaming_links_handles_errors(self):
        """Test that errors during scraping are handled gracefully."""
        scraper = LastFMScraper()

        # Use an invalid URL to trigger error handling
        links = await scraper.get_streaming_links("not-a-valid-url")

        # Should return None for all platforms on error
        assert links["youtube_url"] is None
        assert links["spotify_url"] is None

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test that scraper can be used as async context manager."""
        async with LastFMScraper() as scraper:
            assert scraper._session is not None

        # Session should be closed after exit
        # Note: _session becomes None after close
        assert scraper._session is None
