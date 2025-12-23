"""
Dimension Tables DAG - Fetch and maintain track/artist metadata.

This DAG incrementally builds dimension tables with Last.fm metadata:
- tracks (bronze → silver): Track details, tags, listeners, playcount
- artists (bronze → silver): Artist details, tags, listeners, bio

Only fetches metadata for new tracks/artists across all users' play history.
Dimensions are global (not per-user).

Configuration:
- Runs weekly (dimensions change slowly)
- Incremental: only fetches new tracks/artists
- Rate limited to respect Last.fm API limits
"""

import datetime as dt
from typing import Any

from airflow.sdk import Asset, dag, task

# Assets produced by this DAG
tracks_asset = Asset("delta://data/silver/tracks")
artists_asset = Asset("delta://data/silver/artists")


@dag(
    schedule="@weekly",
    start_date=dt.datetime(2025, 11, 1, tzinfo=dt.timezone.utc),
    catchup=False,  # Don't backfill dimensions
    max_active_runs=1,
    tags=["dimensions", "bronze", "silver"],
    doc_md=__doc__,
)
def lastfm_dimensions():
    """
    Fetch and maintain track and artist dimension tables.

    Incrementally fetches Last.fm metadata for tracks and artists
    that appear in any user's play history but aren't yet in dimension tables.
    Dimensions are global across all users.
    """

    @task(multiple_outputs=False)
    def fetch_tracks() -> dict[str, Any]:
        """
        Extract track metadata from Last.fm API for new tracks.

        Identifies tracks across all users' play history not yet in bronze,
        fetches metadata from Last.fm API, and saves to bronze layer.

        Returns:
            Metadata dict with filename, path, rows, tracks_fetched
        """
        from music_airflow.extract import extract_tracks_to_bronze

        return extract_tracks_to_bronze()

    @task(multiple_outputs=False)
    def fetch_artists() -> dict[str, Any]:
        """
        Extract artist metadata from Last.fm API for new artists.

        Identifies artists across all users' play history not yet in bronze,
        fetches metadata from Last.fm API, and saves to bronze layer.

        Returns:
            Metadata dict with filename, path, rows, artists_fetched
        """
        from music_airflow.extract import extract_artists_to_bronze

        return extract_artists_to_bronze()

    @task(multiple_outputs=False, outlets=[tracks_asset])
    def transform_tracks(fetch_metadata: dict[str, Any]) -> dict[str, Any]:
        """
        Transform raw track metadata from bronze to silver layer.

        Cleans and structures track metadata, merges into silver Delta table.

        Args:
            fetch_metadata: Metadata from extraction

        Returns:
            Metadata dict with path, rows, table_name, merge_metrics
        """
        from music_airflow.transform import transform_tracks_to_silver

        return transform_tracks_to_silver(fetch_metadata)

    @task(multiple_outputs=False, outlets=[artists_asset])
    def transform_artists(fetch_metadata: dict[str, Any]) -> dict[str, Any]:
        """
        Transform raw artist metadata from bronze to silver layer.

        Cleans and structures artist metadata, merges into silver Delta table.

        Args:
            fetch_metadata: Metadata from extraction

        Returns:
            Metadata dict with path, rows, table_name, merge_metrics
        """
        from music_airflow.transform import transform_artists_to_silver

        return transform_artists_to_silver(fetch_metadata)

    # Process tracks and artists sequentially (no per-user expansion)
    track_metadata = fetch_tracks()
    transform_tracks(track_metadata)  # type: ignore[arg-type]

    artist_metadata = fetch_artists()
    transform_artists(artist_metadata)  # type: ignore[arg-type]


# Instantiate the DAG
lastfm_dimensions()
