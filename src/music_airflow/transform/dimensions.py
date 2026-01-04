"""
Transform dimension data (tracks, artists) from bronze to silver layer.

Cleans and structures raw Last.fm metadata for analytics.
"""

import asyncio
from datetime import datetime
import logging
from typing import Any

import polars as pl
from airflow.exceptions import AirflowSkipException
from deltalake.exceptions import TableNotFoundError

from music_airflow.utils.polars_io_manager import JSONIOManager, PolarsDeltaIOManager
from music_airflow.lastfm_client import LastFMClient
import requests

logger = logging.getLogger(__name__)

# Minimum half-life for new users (30 days)
MIN_HALF_LIFE_DAYS = 30.0

__all__ = [
    "transform_tracks_to_silver",
    "transform_artists_to_silver",
    "compute_dim_users",
    "enrich_track_metadata",
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

    Also checks for and incorporates enriched candidate tracks from candidate_enriched_tracks
    table if it exists.

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
    df = _union_enriched_recommended_tracks(df)

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


def _search_musicbrainz_track_mbid(track_name: str, artist_name: str) -> str | None:
    """Search MusicBrainz for a track MBID (first hit)."""
    try:
        query = f'recording:"{track_name}" AND artist:"{artist_name}"'
        resp = requests.get(
            "https://musicbrainz.org/ws/2/recording",
            params={"query": query, "fmt": "json", "limit": 1},
            headers={
                "User-Agent": "github.com/boccileonardo/music-airflow",
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        recordings = data.get("recordings", [])
        if recordings:
            return recordings[0].get("id")
        return None
    except requests.RequestException:
        print(
            f"Unable to get MBID from MusicBrainz for track: {track_name} by {artist_name}"
        )
        return None


def _union_enriched_recommended_tracks(df: pl.LazyFrame) -> pl.LazyFrame:
    # Check for enriched candidate tracks and merge them in
    silver_io = PolarsDeltaIOManager(medallion_layer="silver")
    try:
        candidate_tracks = silver_io.read_delta("candidate_enriched_tracks")
        # Deduplicate by track_id, keeping the most recent enrichment
        candidate_tracks = (
            candidate_tracks.sort("recommended_at", descending=True)
            .group_by("track_id")
            .agg(
                [
                    pl.first("track_name"),
                    pl.first("track_mbid"),
                    pl.first("artist_name"),
                    pl.first("artist_mbid"),
                    pl.first("album_name"),
                    pl.first("duration_ms"),
                    pl.first("listeners"),
                    pl.first("playcount"),
                    pl.first("tags"),
                    pl.first("track_url"),
                ]
            )
        )
        track_count: int = candidate_tracks.select(pl.len()).collect().item()
        if track_count > 0:
            logger.info(
                f"Merging {track_count} enriched candidate tracks into dimension"
            )
            # Concatenate and deduplicate
            combined_lf = pl.concat([df, candidate_tracks])
            df = _deduplicate_tracks(combined_lf)
    except (FileNotFoundError, TableNotFoundError):
        # No enriched candidates yet, continue with just the fetched tracks
        logger.info("No enriched candidate tracks found to merge into dimension")
        pass
    return df


def _enrich_missing_artist_mbids(artists_lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Enrich missing artist MBIDs by consulting Last.fm artist.search and
    falling back to MusicBrainz search. Then recompute artist_id.
    """
    import asyncio

    artist_names = (
        artists_lf.select("artist_name")
        .unique()
        .collect(engine="streaming")
        .to_series()
        .to_list()
    )

    if not artist_names:
        return artists_lf

    async def _enrich_artist_names():
        async with LastFMClient(api_key=None) as client:
            # Fetch all Last.fm searches concurrently
            search_tasks = [
                client.search_artist(name, limit=1) for name in artist_names
            ]

            print(
                f"Searching for MBIDs for {len(artist_names)} artists concurrently..."
            )
            all_search_results = await asyncio.gather(
                *search_tasks, return_exceptions=True
            )

            # Process results and build mapping
            mapping_data = []
            for name, search_result in zip(artist_names, all_search_results):
                mbid: str | None = None

                if isinstance(search_result, Exception):
                    print(f"Last.fm search failed for artist '{name}': {search_result}")
                elif search_result and len(search_result) > 0:
                    mbid = search_result[0].get("mbid")

                # Fallback to MusicBrainz if Last.fm didn't return MBID
                if not mbid:
                    print("Falling back to MusicBrainz search for artist:", name)
                    mbid = _search_musicbrainz_artist_mbid(name)

                if mbid:
                    mapping_data.append(
                        {"artist_name": name, "enriched_artist_mbid": mbid}
                    )

            return pl.DataFrame(
                mapping_data,
                schema={"artist_name": pl.Utf8, "enriched_artist_mbid": pl.Utf8},
            )

    mapping_df = asyncio.run(_enrich_artist_names())

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


def enrich_track_metadata(
    tracks_lf: pl.LazyFrame,
) -> pl.LazyFrame:
    """
    Enrich track metadata by fetching from Last.fm and MusicBrainz.

    For tracks without full metadata, fetches:
    - Track MBID, duration, listeners, playcount, tags, URL
    - Artist MBID
    - Queries Last.fm track.getInfo first
    - Falls back to MusicBrainz for missing track MBIDs

    Args:
        tracks_lf: LazyFrame with track_id, track_name, artist_name (min required)

    Returns:
        Enriched LazyFrame with full track metadata
    """
    # Collect unique track/artist combinations to enrich
    tracks_to_enrich = (
        tracks_lf.select(["track_id", "track_name", "artist_name"])
        .unique()
        .collect(engine="streaming")
    )

    if tracks_to_enrich.is_empty():
        return tracks_lf

    async def _enrich_tracks():
        async with LastFMClient(api_key=None) as client:
            # Fetch all track info concurrently
            fetch_tasks = [
                client.get_track_info(
                    track=row["track_name"], artist=row["artist_name"]
                )
                for row in tracks_to_enrich.to_dicts()
            ]

            print(f"Enriching {len(fetch_tasks)} tracks from Last.fm...")
            all_track_results = await asyncio.gather(
                *fetch_tasks, return_exceptions=True
            )

            # Process results and build enriched data
            enriched_data = []
            for row, track_info in zip(tracks_to_enrich.to_dicts(), all_track_results):
                track_name = row["track_name"]
                artist_name = row["artist_name"]
                track_id = row["track_id"]

                # Handle errors or empty results from Last.fm
                if isinstance(track_info, Exception):
                    print(
                        f"Last.fm fetch failed for track '{track_name}' by '{artist_name}': {track_info}"
                    )
                    track_info = {}

                # Extract data from Last.fm response
                track_mbid = track_info.get("mbid") or None
                duration_ms = track_info.get("duration")
                listeners = track_info.get("listeners")
                playcount = track_info.get("playcount")
                track_url = track_info.get("url")

                # Extract artist info
                artist_info = track_info.get("artist", {})
                artist_mbid = artist_info.get("mbid") or None

                # Extract album info
                album_info = track_info.get("album", {})
                album_name = album_info.get("title") if album_info else None

                # Extract tags
                toptags = track_info.get("toptags", {})
                tag_list = toptags.get("tag", []) if toptags else []
                if isinstance(tag_list, list):
                    tags = ", ".join([t.get("name", "") for t in tag_list[:5]])
                else:
                    tags = None

                # Fallback to MusicBrainz for track MBID if missing
                if not track_mbid:
                    print(
                        f"Falling back to MusicBrainz for track: {track_name} by {artist_name}"
                    )
                    track_mbid = _search_musicbrainz_track_mbid(track_name, artist_name)

                enriched_data.append(
                    {
                        "track_id": track_id,
                        "track_name": track_name,
                        "track_mbid": track_mbid,
                        "artist_name": artist_name,
                        "artist_mbid": artist_mbid,
                        "album_name": album_name,
                        "duration_ms": int(duration_ms) if duration_ms else None,
                        "listeners": int(listeners) if listeners else None,
                        "playcount": int(playcount) if playcount else None,
                        "tags": tags,
                        "track_url": track_url,
                    }
                )

            return pl.DataFrame(
                enriched_data,
                schema={
                    "track_id": pl.Utf8,
                    "track_name": pl.Utf8,
                    "track_mbid": pl.Utf8,
                    "artist_name": pl.Utf8,
                    "artist_mbid": pl.Utf8,
                    "album_name": pl.Utf8,
                    "duration_ms": pl.Int64,
                    "listeners": pl.Int64,
                    "playcount": pl.Int64,
                    "tags": pl.Utf8,
                    "track_url": pl.Utf8,
                },
            )

    enriched_df = asyncio.run(_enrich_tracks())

    if enriched_df.is_empty():
        return tracks_lf

    # Merge enriched data back into the original lazyframe
    result = tracks_lf.join(
        enriched_df.lazy(),
        on=["track_id", "track_name", "artist_name"],
        how="left",
        suffix="_enriched",
    )

    # Coalesce to prefer enriched values where available
    schema_names = result.collect_schema().names()

    coalesce_columns = []
    for col in [
        "track_mbid",
        "artist_mbid",
        "album_name",
        "duration_ms",
        "listeners",
        "playcount",
        "tags",
        "track_url",
    ]:
        enriched_col = f"{col}_enriched"
        if enriched_col in schema_names:
            coalesce_columns.append(
                pl.coalesce([pl.col(enriched_col), pl.col(col)]).alias(col)
            )

    if coalesce_columns:
        result = result.with_columns(coalesce_columns)

    # Drop enriched suffix columns
    cols_to_drop = [c for c in schema_names if c.endswith("_enriched")]
    if cols_to_drop:
        result = result.drop(cols_to_drop)

    return result


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
