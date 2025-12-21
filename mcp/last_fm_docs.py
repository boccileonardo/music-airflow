"""
Last.fm API Documentation MCP Server

This MCP server provides tools for AI agents to query Last.fm API documentation.
It helps agents understand available API methods, parameters, and response formats.
"""

from fastmcp import FastMCP
from typing import Optional

mcp = FastMCP("Last.fm API Documentation")

# Last.fm API method categories
API_METHODS = {
    "album": ["addTags", "getInfo", "getTags", "getTopTags", "removeTag", "search"],
    "artist": [
        "addTags",
        "getCorrection",
        "getInfo",
        "getSimilar",
        "getTags",
        "getTopAlbums",
        "getTopTags",
        "getTopTracks",
        "removeTag",
        "search",
    ],
    "auth": ["getMobileSession", "getSession", "getToken"],
    "chart": ["getTopArtists", "getTopTags", "getTopTracks"],
    "geo": ["getTopArtists", "getTopTracks"],
    "library": ["getArtists"],
    "tag": [
        "getInfo",
        "getSimilar",
        "getTopAlbums",
        "getTopArtists",
        "getTopTags",
        "getTopTracks",
        "getWeeklyChartList",
    ],
    "track": [
        "addTags",
        "getCorrection",
        "getInfo",
        "getSimilar",
        "getTags",
        "getTopTags",
        "love",
        "removeTag",
        "scrobble",
        "search",
        "unlove",
        "updateNowPlaying",
    ],
    "user": [
        "getFriends",
        "getInfo",
        "getLovedTracks",
        "getPersonalTags",
        "getRecentTracks",
        "getTopAlbums",
        "getTopArtists",
        "getTopTags",
        "getTopTracks",
        "getWeeklyAlbumChart",
        "getWeeklyArtistChart",
        "getWeeklyChartList",
        "getWeeklyTrackChart",
    ],
}

# API Guide pages
API_GUIDES = {
    "introduction": "https://www.last.fm/api/intro",
    "authentication": "https://www.last.fm/api/authentication",
    "web_auth": "https://www.last.fm/api/webauth",
    "mobile_auth": "https://www.last.fm/api/mobileauth",
    "desktop_auth": "https://www.last.fm/api/desktopauth",
    "auth_spec": "https://www.last.fm/api/authspec",
    "scrobbling": "https://www.last.fm/api/scrobbling",
    "radio": "https://www.last.fm/api/radio",
    "playlists": "https://www.last.fm/api/playlists",
    "rest": "https://www.last.fm/api/rest",
    "xmlrpc": "https://www.last.fm/api/xmlrpc",
    "terms": "https://www.last.fm/api/tos",
}


@mcp.tool()
async def get_lastfm_api_intro() -> str:
    """
    Fetch the Last.fm API introduction and overview documentation.

    Returns:
        Introduction to Last.fm API including authentication, rate limits, and general usage
    """
    url = "https://www.last.fm/api/intro"
    return f"""Last.fm API Introduction
            Documentation URL: {url}

            Getting Started with Last.fm API:
            1. Sign up for an API account at https://www.last.fm/api/account/create
            2. Get your API key and shared secret
            3. Choose authentication method based on your app type (web, mobile, or desktop)
            4. Make REST API calls with your API key

            Key Concepts:
            - Most read methods require only an API key
            - Write methods (scrobbling, tagging, etc.) require full authentication
            - Rate limit: Generally lenient for personal use
            - Response formats: JSON (recommended) or XML

            Use get_lastfm_api_guide() to access specific guide topics like authentication or scrobbling."""


@mcp.tool()
async def get_lastfm_api_guide(guide: str) -> str:
    """
    Get documentation for Last.fm API guides.

    Args:
        guide: The guide topic - one of:
               'introduction', 'authentication', 'web_auth', 'mobile_auth',
               'desktop_auth', 'auth_spec', 'scrobbling', 'radio',
               'playlists', 'rest', 'xmlrpc', 'terms'

    Returns:
        URL and description for the requested guide

    Examples:
        - get_lastfm_api_guide('authentication') - Learn about auth methods
        - get_lastfm_api_guide('scrobbling') - Scrobbling documentation
        - get_lastfm_api_guide('web_auth') - Web application authentication
    """
    guide_lower = guide.lower()

    if guide_lower not in API_GUIDES:
        available = ", ".join(sorted(API_GUIDES.keys()))
        return f"Guide '{guide}' not found. Available guides: {available}"

    url = API_GUIDES[guide_lower]

    guide_descriptions = {
        "introduction": "Overview of Last.fm API, getting started, and basic concepts",
        "authentication": "Authentication overview - choosing the right method for your app",
        "web_auth": "Web Application Authentication - OAuth-style flow for web apps",
        "mobile_auth": "Mobile Application Authentication - auth flow for mobile apps",
        "desktop_auth": "Desktop Application Authentication - auth flow for desktop apps",
        "auth_spec": "Complete authentication specification and technical details",
        "scrobbling": "Scrobbling 2.0 Documentation - how to submit listening data",
        "radio": "Radio API - streaming and radio functionality",
        "playlists": "Playlists API - working with Last.fm playlists",
        "rest": "REST API requests - making HTTP calls to Last.fm",
        "xmlrpc": "XML-RPC API (legacy) - older API access method",
        "terms": "Terms of Service - API usage terms and conditions",
    }

    description = guide_descriptions.get(guide_lower, "Last.fm API guide")

    return f"""Last.fm API Guide: {guide}
    {description}
    Documentation URL: {url}
    Visit the URL for complete documentation including examples, parameters, and implementation details."""


@mcp.tool()
async def get_lastfm_method_docs(method_category: str, method_name: str) -> str:
    """
    Get documentation for a specific Last.fm API method.

    Args:
        method_category: The API category (e.g., 'album', 'artist', 'track', 'user')
        method_name: The method name (e.g., 'getInfo', 'search', 'getTopTracks')

    Returns:
        Documentation for the specified API method including parameters and response format

    Examples:
        - get_lastfm_method_docs('artist', 'getInfo') - Get info about an artist
        - get_lastfm_method_docs('track', 'search') - Search for tracks
        - get_lastfm_method_docs('user', 'getRecentTracks') - Get user's recent tracks
    """
    # Construct the URL for the API method documentation
    full_method = f"{method_category}.{method_name}"
    url = f"https://www.last.fm/api/show/{full_method}"

    return f"""Last.fm API Method: {full_method}

            Documentation URL: {url}

            To view full documentation including:
            - Required and optional parameters
            - Authentication requirements
            - Response format and examples
            - Error codes

            Please visit: {url}

            Common parameters for most methods:
            - api_key (required): Your Last.fm API key
            - format (optional): Response format (json or xml, default: xml)

            Note: Most read methods require only an API key, while write methods require full authentication."""


@mcp.tool()
async def list_lastfm_methods(category: Optional[str] = None) -> str:
    """
    List available Last.fm API methods, optionally filtered by category.

    Args:
        category: Optional category to filter by (e.g., 'album', 'artist', 'track', 'user')
                 If None, lists all categories and their methods

    Returns:
        List of available API methods
    """
    if category:
        if category.lower() not in API_METHODS:
            available = ", ".join(sorted(API_METHODS.keys()))
            return f"Category '{category}' not found. Available categories: {available}"

        methods = API_METHODS[category.lower()]
        result = f"Last.fm API Methods for '{category}':\n\n"
        for method in methods:
            result += f"  - {category}.{method}\n"
        result += f"\nUse get_lastfm_method_docs('{category}', '<method_name>') for detailed documentation."
        return result

    # List all categories and methods
    result = "Last.fm API Methods by Category:\n\n"
    for cat, methods in sorted(API_METHODS.items()):
        result += f"{cat}:\n"
        for method in methods:
            result += f"  - {cat}.{method}\n"
        result += "\n"

    result += "Use get_lastfm_method_docs('<category>', '<method_name>') for detailed documentation."
    return result


@mcp.tool()
async def search_lastfm_methods(query: str) -> str:
    """
    Search for Last.fm API methods by keyword.

    Args:
        query: Search term (e.g., 'search', 'top', 'recent', 'info')

    Returns:
        List of matching API methods
    """
    query_lower = query.lower()
    matches = []

    for category, methods in API_METHODS.items():
        for method in methods:
            full_method = f"{category}.{method}"
            if query_lower in full_method.lower():
                matches.append(full_method)

    if not matches:
        return f"No methods found matching '{query}'. Use list_lastfm_methods() to see all available methods."

    result = f"Last.fm API methods matching '{query}':\n\n"
    for match in sorted(matches):
        result += f"  - {match}\n"

    result += "\nUse get_lastfm_method_docs() for detailed documentation on any method."
    return result


if __name__ == "__main__":
    # Run the MCP server
    mcp.run()
