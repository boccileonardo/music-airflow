"""
Transform dimension data (tracks, artists) from bronze to silver layer.

Cleans and structures raw Last.fm metadata for analytics.
"""

from datetime import datetime
from typing import Any

import polars as pl
from airflow.exceptions import AirflowSkipException
from deltalake.exceptions import TableNotFoundError

from music_airflow.utils.polars_io_manager import JSONIOManager, PolarsDeltaIOManager
from music_airflow.lastfm_client import LastFMClient
import requests

# Minimum half-life for new users (30 days)
MIN_HALF_LIFE_DAYS = 30.0

__all__ = [
    "transform_tracks_to_silver",
    "transform_artists_to_silver",
    "compute_dim_users",
    "_transform_tracks_raw_to_structured",
    "_transform_artists_raw_to_structured",
    "_deduplicate_tracks",
    "_deduplicate_artists",
]


def transform_tracks_to_silver(fetch_metadata: dict[str, Any]) -> dict[str, Any]:
    """
    Transform raw track metadata from bronze to structured Delta table in silver layer.

    Reads raw JSON track data from bronze, extracts and flattens relevant fields,
    and merges into silver Delta table. Uses upsert based on MBID when available,
    otherwise falls back to track_name + artist_name + album_name.

    Args:
        fetch_metadata: Metadata from extraction containing filename, tracks_fetched

    Returns:
        Metadata dict with:
        - path: Path to Delta table
        - table_name: "tracks"
        - rows: Number of tracks processed
        - schema: Column names and types
        - format: "delta"
        - medallion_layer: "silver"
        - mode: "merge"
        - merge_metrics: Dict with merge statistics

    Raises:
        AirflowSkipException: If no tracks to process
    """
    # Read raw JSON using Polars
    io_manager = JSONIOManager(medallion_layer="bronze")
    filename = fetch_metadata["filename"]
    tracks_lf = io_manager.read_json(filename)

    # Apply transformations
    df = _transform_tracks_raw_to_structured(tracks_lf)
    df = _deduplicate_tracks(df)

    # Write to silver layer Delta table with merge/upsert
    table_name = "tracks"
    predicate = """
        s.track_id = t.track_id
    """

    silver_io = PolarsDeltaIOManager(medallion_layer="silver")
    write_metadata = silver_io.write_delta(
        df,
        table_name=table_name,
        mode="merge",
        predicate=predicate,
    )

    return write_metadata


def transform_artists_to_silver(fetch_metadata: dict[str, Any]) -> dict[str, Any]:
    """
    Transform raw artist metadata from bronze to structured Delta table in silver layer.

    Reads raw JSON artist data from bronze, extracts and flattens relevant fields,
    and merges into silver Delta table. Uses upsert based on MBID when available.

    Args:
        fetch_metadata: Metadata from extraction containing filename, artists_fetched

    Returns:
        Metadata dict with:
        - path: Path to Delta table
        - table_name: "artists"
        - rows: Number of artists processed
        - schema: Column names and types
        - format: "delta"
        - medallion_layer: "silver"
        - mode: "merge"
        - merge_metrics: Dict with merge statistics

    Raises:
        AirflowSkipException: If no artists to process
    """
    # Read raw JSON using Polars
    io_manager = JSONIOManager(medallion_layer="bronze")
    filename = fetch_metadata["filename"]
    artists_lf = io_manager.read_json(filename)

    # Apply transformations
    df = _transform_artists_raw_to_structured(artists_lf)
    df = _deduplicate_artists(df)

    # Exclude invalid artists (no mbid and listeners < 1000)
    df = df.filter(
        ~(
            (pl.col("artist_mbid").is_null() | (pl.col("artist_mbid") == ""))
            & (pl.col("listeners") < 1000)
        )
    )

    # Enrich missing MBIDs for remaining artists
    df = _enrich_missing_artist_mbids(df)

    # Write to silver layer Delta table with merge/upsert
    table_name = "artists"
    predicate = """
       s.artist_id = t.artist_id
    """

    silver_io = PolarsDeltaIOManager(medallion_layer="silver")
    write_metadata = silver_io.write_delta(
        df,
        table_name=table_name,
        mode="merge",
        predicate=predicate,
    )

    return write_metadata


def _search_musicbrainz_artist_mbid(artist_name: str) -> str | None:
    """Search MusicBrainz for an artist MBID (first hit)."""
    try:
        resp = requests.get(
            "https://musicbrainz.org/ws/2/artist",
            params={"query": artist_name, "fmt": "json", "limit": 1},
            headers={
                "User-Agent": "github.com/boccileonardo/music-airflow",
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        artists = data.get("artists", [])
        if artists:
            return artists[0].get("id")
        return None
    except requests.RequestException:
        # todo: log
        print("Unable to get MBID from MusicBrainz for artist:", artist_name)
        return None


def _enrich_missing_artist_mbids(artists_lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Enrich missing artist MBIDs by consulting Last.fm artist.search and
    falling back to MusicBrainz search. Then recompute artist_id.
    """
    artist_names = (
        artists_lf.select("artist_name")
        .unique()
        .collect(engine="streaming")
        .to_series()
        .to_list()
    )

    if not artist_names:
        return artists_lf

    client = LastFMClient(api_key=None)
    mapping_df = pl.DataFrame(
        schema={"artist_name": pl.Utf8, "enriched_artist_mbid": pl.Utf8}
    )
    for name in artist_names:
        print("Enriching MBID for artist:", name)
        mbid: str | None = None
        last_fm_search_result = client.search_artist(name, limit=1)
        if last_fm_search_result:
            mbid = last_fm_search_result[0].get("mbid")
        if not mbid:
            print("Falling back to MusicBrainz search for artist:", name)
            mbid = _search_musicbrainz_artist_mbid(name)
        if mbid:
            mapping_df = mapping_df.vstack(
                pl.DataFrame([{"artist_name": name, "enriched_artist_mbid": mbid}])
            )

    if mapping_df.is_empty():
        return artists_lf

    enriched = (
        artists_lf.join(mapping_df.lazy(), on="artist_name", how="left")
        .with_columns(
            # only overwrite artist_mbid if it's missing and enriched_artist_mbid is not null or empty
            pl.when(
                (pl.col("artist_mbid").is_null() | (pl.col("artist_mbid") == ""))
                & (pl.col("enriched_artist_mbid").is_not_null())
                & (pl.col("enriched_artist_mbid") != "")
            )
            .then(pl.col("enriched_artist_mbid"))
            .otherwise(pl.col("artist_mbid"))
            .alias("artist_mbid")
        )
        .with_columns(
            # recompute artist_id to prefer MBID when available
            pl.when(
                (pl.col("artist_mbid").is_not_null()) & (pl.col("artist_mbid") != "")
            )
            .then(pl.col("artist_mbid"))
            .otherwise(pl.col("artist_name"))
            .alias("artist_id")
        )
    )

    return enriched


def _transform_tracks_raw_to_structured(raw_tracks: pl.LazyFrame) -> pl.LazyFrame:
    """
    Transform raw Last.fm track metadata to structured format.

    Extracts relevant fields from nested JSON structure:
    - Basic info: name, artist, album, duration
    - MBIDs for linking
    - Popularity metrics: listeners, playcount
    - Tags (flattened to comma-separated string)
    - URLs

    Args:
        raw_tracks: LazyFrame with raw JSON structure from Last.fm API

    Returns:
        Transformed LazyFrame with flattened columns
    """
    df = raw_tracks.with_columns(
        [
            # Basic info
            pl.col("name").alias("track_name"),
            pl.col("mbid").alias("track_mbid"),
            pl.col("url").alias("track_url"),
            pl.col("duration").cast(pl.Int64).alias("duration_ms"),
            # Artist info (nested struct)
            pl.col("artist").struct.field("name").alias("artist_name"),
            pl.col("artist").struct.field("mbid").alias("artist_mbid"),
            # Album info (nested struct)
            pl.when(pl.col("album").is_not_null())
            .then(pl.col("album").struct.field("title"))
            .otherwise(None)
            .alias("album_name"),
            # Popularity metrics
            pl.col("listeners").cast(pl.Int64).alias("listeners"),
            pl.col("playcount").cast(pl.Int64).alias("playcount"),
            # Tags - extract top 5 tag names as comma-separated string
            pl.when(pl.col("toptags").is_not_null())
            .then(
                pl.col("toptags")
                .struct.field("tag")
                .list.head(5)
                .list.eval(pl.element().struct.field("name"))
                .list.join(", ")
            )
            .otherwise(None)
            .alias("tags"),
        ]
    ).select(
        [
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
    )

    return df


def _transform_artists_raw_to_structured(raw_artists: pl.LazyFrame) -> pl.LazyFrame:
    """
    Transform raw Last.fm artist metadata to structured format.

    Extracts relevant fields from nested JSON structure:
    - Basic info: name, mbid
    - Popularity metrics: listeners, playcount
    - Tags (flattened to comma-separated string)
    - Bio summary (first 500 chars)
    - URLs

    Args:
        raw_artists: LazyFrame with raw JSON structure from Last.fm API

    Returns:
        Transformed LazyFrame with flattened columns
    """
    df = raw_artists.with_columns(
        [
            # Basic info
            pl.col("name").alias("artist_name"),
            pl.col("mbid").alias("artist_mbid"),
            pl.col("url").alias("artist_url"),
            # Popularity metrics (nested in stats struct)
            pl.when(pl.col("stats").is_not_null())
            .then(pl.col("stats").struct.field("listeners").cast(pl.Int64))
            .otherwise(None)
            .alias("listeners"),
            pl.when(pl.col("stats").is_not_null())
            .then(pl.col("stats").struct.field("playcount").cast(pl.Int64))
            .otherwise(None)
            .alias("playcount"),
            # Tags - extract top 5 tag names as comma-separated string
            pl.when(pl.col("tags").is_not_null())
            .then(
                pl.col("tags")
                .struct.field("tag")
                .list.head(5)
                .list.eval(pl.element().struct.field("name"))
                .list.join(", ")
            )
            .otherwise(None)
            .alias("tags"),
            # Bio summary
            pl.when(pl.col("bio").is_not_null())
            .then(pl.col("bio").struct.field("summary"))
            .otherwise(None)
            .alias("bio_summary"),
        ]
    ).select(
        [
            "artist_name",
            "artist_mbid",
            "listeners",
            "playcount",
            "tags",
            "bio_summary",
            "artist_url",
        ]
    )

    return df


def _deduplicate_tracks(tracks_lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Deduplicate tracks with same attributes, maintaining non-null metadata.

    When multiple rows exist, consolidate by:
    - Keeping non-null values preferentially
    - Prioritizing rows with MBIDs
    - Taking max for numeric fields (listeners, playcount, duration)

    Args:
        tracks_lf: LazyFrame with potentially duplicate tracks

    Returns:
        Deduplicated LazyFrame with one row per track
    """
    # same track name by same artist with same album name should be considered same track
    deduplicated = (
        tracks_lf.group_by(["track_name", "artist_name", "album_name"]).agg(
            [
                # non-null and non "", prefer rows with MBID
                pl.col("track_mbid")
                .filter(
                    (pl.col("track_mbid") != "") & pl.col("track_mbid").is_not_null()
                )
                .first()
                .alias("track_mbid"),
                pl.col("artist_mbid")
                .filter(
                    (pl.col("artist_mbid") != "") & pl.col("artist_mbid").is_not_null()
                )
                .first()
                .alias("artist_mbid"),
                pl.col("duration_ms")
                .filter(pl.col("duration_ms").is_not_null())
                .first()
                .alias("duration_ms"),
                pl.col("track_url")
                .filter((pl.col("track_url") != "") & pl.col("track_url").is_not_null())
                .first()
                .alias("track_url"),
                pl.col("tags")
                .filter((pl.col("tags") != "") & pl.col("tags").is_not_null())
                .first()
                .alias("tags"),
                # Take max for numeric popularity metrics
                pl.col("listeners").max().alias("listeners"),
                pl.col("playcount").max().alias("playcount"),
                # artist and track id
            ]
        )
    ).with_columns(
        # Create track_id: prefer MBID when available, else synthetic from names
        pl.when((pl.col("track_mbid").is_not_null()) & (pl.col("track_mbid") != ""))
        .then(pl.col("track_mbid"))
        .otherwise(
            pl.concat_str([pl.col("track_name"), pl.col("artist_name")], separator="|")
        )
        .alias("track_id"),
        # Create artist_id: use MBID if available, else artist name
        # TODO: instead of artist name as fallback, use lastfm search for mbid based on fuzzy name search
        pl.when((pl.col("artist_mbid").is_not_null()) & (pl.col("artist_mbid") != ""))
        .then(pl.col("artist_mbid"))
        .otherwise(pl.col("artist_name"))
        .alias("artist_id"),
    )

    return deduplicated


def _deduplicate_artists(artists_lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Deduplicate artists.

    When multiple rows exist, consolidate by:
    - Keeping non-null values preferentially
    - Prioritizing rows with MBIDs
    - Taking max for numeric fields (listeners, playcount)

    Args:
        artists_lf: LazyFrame with potentially duplicate artists

    Returns:
        Deduplicated LazyFrame with one row per artist
    """
    deduplicated = (
        artists_lf.group_by("artist_name").agg(
            [
                # non-null and non "", prefer rows with MBID
                pl.col("artist_mbid")
                .filter(
                    (pl.col("artist_mbid") != "") & pl.col("artist_mbid").is_not_null()
                )
                .first()
                .alias("artist_mbid"),
                pl.col("artist_url")
                .filter(
                    (pl.col("artist_url") != "") & pl.col("artist_url").is_not_null()
                )
                .first()
                .alias("artist_url"),
                pl.col("tags")
                .filter((pl.col("tags") != "") & pl.col("tags").is_not_null())
                .first()
                .alias("tags"),
                pl.col("bio_summary")
                .filter(
                    (pl.col("bio_summary") != "") & pl.col("bio_summary").is_not_null()
                )
                .first()
                .alias("bio_summary"),
                # Take max for numeric popularity metrics
                pl.col("listeners").max().alias("listeners"),
                pl.col("playcount").max().alias("playcount"),
            ]
        )
    ).with_columns(
        # Create artist_id: use MBID if available, else artist name
        pl.when((pl.col("artist_mbid").is_not_null()) & (pl.col("artist_mbid") != ""))
        .then(pl.col("artist_mbid"))
        .otherwise(pl.col("artist_name"))
        .alias("artist_id"),
    )

    return deduplicated


def compute_dim_users(execution_date: datetime) -> dict[str, Any]:
    """
    Compute user dimension table with listening profile metadata.

    Creates a dimension table with per-user metrics:
    - Listening span (first to last play dates)
    - User-specific half-life for recency calculations
    - Total plays count

    Half-life is calculated as: max(listening_span_days / 3, 30 days)
    This ensures:
    - New users (30 day history): 30 day half-life
    - Users with 6 months history: 60 day half-life
    - Users with 1 year history: 120 day half-life

    Args:
        execution_date: Reference date for calculations (typically DAG run date)

    Returns:
        Metadata dict with:
        - path: Path to Delta table
        - table_name: "dim_users"
        - rows: Number of unique users
        - schema: Column schema
        - format: "delta"
        - medallion_layer: "silver"
        - execution_date: Reference date used

    Raises:
        AirflowSkipException: If no plays data available yet
    """
    # Read silver plays table
    try:
        io_manager = PolarsDeltaIOManager(medallion_layer="silver")
        plays_lf: pl.LazyFrame = io_manager.read_delta("plays")  # type: ignore[assignment]
    except (FileNotFoundError, TableNotFoundError):
        # No plays data yet - nothing to process
        raise AirflowSkipException("No plays data available yet - run plays DAG first")

    # Compute user-level aggregations
    dim_users_lf = (
        plays_lf.group_by("username")
        .agg(
            [
                pl.col("scrobbled_at_utc").min().alias("first_play_date"),
                pl.col("scrobbled_at_utc").max().alias("last_play_date"),
                pl.len().alias("total_plays"),
            ]
        )
        .with_columns(
            [
                # Listening span in days
                (pl.col("last_play_date") - pl.col("first_play_date"))
                .dt.total_days()
                .alias("listening_span_days"),
            ]
        )
        .with_columns(
            [
                # Half-life = max(listening_span / 3, 30 days)
                pl.max_horizontal(
                    pl.col("listening_span_days") / 3.0, pl.lit(MIN_HALF_LIFE_DAYS)
                ).alias("user_half_life_days"),
            ]
        )
        .sort("username")
    )

    # Write to silver layer
    silver_io = PolarsDeltaIOManager(medallion_layer="silver")
    write_metadata = silver_io.write_delta(
        dim_users_lf,
        table_name="dim_users",
        mode="overwrite",  # Full refresh
    )

    return {
        **write_metadata,
        "execution_date": execution_date.isoformat(),
    }
