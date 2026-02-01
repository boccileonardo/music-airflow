"""
Scraper for extracting streaming platform links from Last.fm track pages.

Last.fm track pages contain links to YouTube, Spotify, and Apple Music that are not
exposed in the API. This module scrapes those links from the HTML.
"""

import asyncio
import logging

import aiohttp
from bs4 import BeautifulSoup
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

logger = logging.getLogger(__name__)


class LastFMScraper:
    """
    Async scraper for extracting streaming platform links from Last.fm track pages.

    Implements rate limiting and retries to be respectful of Last.fm's servers.
    """

    RATE_LIMIT_DELAY = 0.1  # 10 requests per second

    def __init__(self):
        """Initialize the scraper."""
        self._session: aiohttp.ClientSession | None = None
        self._last_request_time = 0.0
        self._rate_limit_lock = asyncio.Lock()

    async def __aenter__(self):
        """Create aiohttp session on context manager entry."""
        self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close aiohttp session on context manager exit."""
        if self._session:
            await self._session.close()
            self._session = None

    async def _ensure_session(self):
        """Ensure session exists, create if needed."""
        if self._session is None:
            self._session = aiohttp.ClientSession()

    async def close(self):
        """Close the client session."""
        if self._session:
            await self._session.close()
            self._session = None

    async def _rate_limit(self):
        """Enforce rate limiting between requests."""
        async with self._rate_limit_lock:
            elapsed = asyncio.get_event_loop().time() - self._last_request_time
            if elapsed < self.RATE_LIMIT_DELAY:
                await asyncio.sleep(self.RATE_LIMIT_DELAY - elapsed)
            self._last_request_time = asyncio.get_event_loop().time()

    @retry(
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
        wait=wait_random_exponential(multiplier=1, max=60),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def _fetch_page(self, url: str) -> str:
        """
        Fetch HTML content from URL with retries.

        Args:
            url: URL to fetch

        Returns:
            HTML content as string

        Raises:
            aiohttp.ClientError: On HTTP errors
            asyncio.TimeoutError: On timeout
        """
        await self._ensure_session()
        await self._rate_limit()

        async with self._session.get(  # type: ignore
            url,
            timeout=aiohttp.ClientTimeout(total=30),
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
            },
        ) as response:
            response.raise_for_status()
            return await response.text()

    def _extract_streaming_links(self, html: str) -> dict[str, str | None]:
        """
        Extract streaming platform links from Last.fm track page HTML.

        Looks for these link patterns:
        - YouTube: <a class="play-this-track-playlink--youtube" data-youtube-url="...">
        - Spotify: <a class="play-this-track-playlink--spotify" data-spotify-url="...">

        Args:
            html: HTML content of Last.fm track page

        Returns:
            Dict with keys: youtube_url, spotify_url
            Values are URLs or None if not found
        """
        soup = BeautifulSoup(html, "html.parser")

        # Extract YouTube link
        youtube_url = None
        youtube_link = soup.find("a", class_="play-this-track-playlink--youtube")
        if youtube_link:
            youtube_url = youtube_link.get("data-youtube-url") or youtube_link.get(
                "href"
            )

        # Extract Spotify link
        spotify_url = None
        spotify_link = soup.find("a", class_="play-this-track-playlink--spotify")
        if spotify_link:
            spotify_url = spotify_link.get("data-spotify-url") or spotify_link.get(
                "href"
            )

        return {
            "youtube_url": youtube_url,
            "spotify_url": spotify_url,
        }

    async def get_streaming_links(self, track_url: str) -> dict[str, str | None]:
        """
        Get streaming platform links for a Last.fm track.

        Args:
            track_url: Last.fm track URL (e.g., https://www.last.fm/music/Artist/_/Track)

        Returns:
            Dict with keys: youtube_url, spotify_url
            Values are URLs or None if not found or on error

        Example:
            >>> async with LastFMScraper() as scraper:
            ...     links = await scraper.get_streaming_links(
            ...         "https://www.last.fm/music/The+Spinners/_/It's+A+Shame"
            ...     )
            ...     print(links["youtube_url"])
            https://www.youtube.com/watch?v=kDFFHLrzTDM
        """
        try:
            html = await self._fetch_page(track_url)
            return self._extract_streaming_links(html)
        except Exception as e:
            logger.warning(f"Failed to scrape {track_url}: {e}")
            return {
                "youtube_url": None,
                "spotify_url": None,
            }

    async def get_streaming_links_batch(
        self, track_urls: list[str]
    ) -> list[dict[str, str | None]]:
        """
        Get streaming links for multiple tracks concurrently.

        Args:
            track_urls: List of Last.fm track URLs

        Returns:
            List of dicts with streaming links (same order as input)

        Example:
            >>> urls = [
            ...     "https://www.last.fm/music/Artist1/_/Track1",
            ...     "https://www.last.fm/music/Artist2/_/Track2",
            ... ]
            >>> async with LastFMScraper() as scraper:
            ...     results = await scraper.get_streaming_links_batch(urls)
        """
        tasks = [self.get_streaming_links(url) for url in track_urls]
        return await asyncio.gather(*tasks, return_exceptions=False)
