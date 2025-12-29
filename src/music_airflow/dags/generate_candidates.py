"""
Candidate Generation DAG - Generate track candidates for recommender system.

Generates three types of candidates (similar artists, similar tags, deep cuts),
saves each to silver tables, then consolidates into a single gold table with
type indicators and deduplication.

Configuration:
- Runs when silver layer assets are updated (plays, artists, tracks)
- Generates candidates for all active users
- Saves intermediate results to silver, final unified table to gold
"""

# todo: review outputs

from typing import Any
from music_airflow.utils.constants import DAG_START_DATE
from airflow.sdk import Asset, dag, task

# Assets consumed by this DAG
plays_asset = Asset("delta://data/silver/plays")
artists_asset = Asset("delta://data/silver/artists")
tracks_asset = Asset("delta://data/silver/tracks")

# Asset produced by this DAG
candidates_asset = Asset("delta://data/gold/track_candidates")


@dag(
    schedule=[plays_asset, artists_asset, tracks_asset],
    start_date=DAG_START_DATE,
    catchup=False,
    max_active_runs=1,
    tags=["gold", "candidates", "recommendations"],
    doc_md=__doc__,
)
def candidate_generation():
    """
    Generate recommendation candidate tracks from silver layer data.

    Produces three silver candidate tables (similar_artist, similar_tag, deep_cut),
    then consolidates into single unified gold table with type indicators:
    - similar_artist: Tracks from artists similar to user's played artists
    - similar_tag: Tracks with tags matching user's library
    - deep_cut_same_artist: Obscure tracks from known artists

    Each candidate can belong to multiple types (one-hot encoded).
    """

    @task
    def get_active_users() -> list[str]:
        """
        Get list of users with plays in the system.

        Returns:
            List of usernames
        """
        import polars as pl

        plays_lf = pl.scan_delta("data/silver/plays")
        users = plays_lf.select("username").unique().collect(engine="streaming")
        return users["username"].to_list()

    @task(
        multiple_outputs=False,
        inlets=[plays_asset, artists_asset],
    )
    def generate_similar_artist(username: str) -> dict[str, Any]:
        """
        Generate similar artist candidates and save to silver.

        Args:
            username: Target user

        Returns:
            Metadata dict with path, rows, table_name
        """
        from music_airflow.transform.candidate_generation import (
            generate_similar_artist_candidates,
        )

        return generate_similar_artist_candidates(
            username=username,
        )

    @task(
        multiple_outputs=False,
        inlets=[plays_asset, tracks_asset],
    )
    def generate_similar_tag(username: str) -> dict[str, Any]:
        """
        Generate similar tag candidates and save to silver.

        Args:
            username: Target user

        Returns:
            Metadata dict with path, rows, table_name
        """
        from music_airflow.transform.candidate_generation import (
            generate_similar_tag_candidates,
        )

        return generate_similar_tag_candidates(
            username=username,
        )

    @task(
        multiple_outputs=False,
        inlets=[plays_asset, artists_asset, tracks_asset],
    )
    def generate_deep_cut(username: str) -> dict[str, Any]:
        """
        Generate deep cut candidates and save to silver.

        Args:
            username: Target user

        Returns:
            Metadata dict with path, rows, table_name
        """
        from music_airflow.transform.candidate_generation import (
            generate_deep_cut_candidates,
        )

        return generate_deep_cut_candidates(
            username=username,
        )

    @task(
        multiple_outputs=False,
        outlets=[candidates_asset],
    )
    def consolidate_to_gold(username: str) -> dict[str, Any]:
        """
        Consolidate silver candidate tables into unified gold table.

        Reads from silver candidate tables and merges them with type indicators.

        Args:
            username: Target user

        Returns:
            Metadata dict with path, rows, table_name
        """
        from music_airflow.transform.candidate_generation import merge_candidate_sources

        return merge_candidate_sources(username=username)

    @task(
        multiple_outputs=False,
    )
    def consolidate_results(results: list) -> dict[str, Any]:
        """
        Consolidate candidate generation results across all users.

        Args:
            results: List of metadata dicts from candidate generation tasks

        Returns:
            Summary metadata
        """
        total_candidates = sum(r.get("rows", 0) for r in results if isinstance(r, dict))
        num_users = len([r for r in results if isinstance(r, dict)])
        return {
            "total_candidates": total_candidates,
            "num_users": num_users,
            "table_name": "track_candidates",
        }

    # Define workflow
    users = get_active_users()

    # Generate each candidate type in parallel for each user
    similar_artist_results = generate_similar_artist.expand(username=users)
    similar_tag_results = generate_similar_tag.expand(username=users)
    deep_cut_results = generate_deep_cut.expand(username=users)

    # Consolidate silver tables to gold for each user
    # Wait for all three silver tables to be ready before consolidating
    gold_results = consolidate_to_gold.expand(username=users)
    gold_results.set_upstream(
        [similar_artist_results, similar_tag_results, deep_cut_results]
    )

    # Final summary
    consolidate_results([gold_results])  # type: ignore[arg-type]


candidate_generation()
