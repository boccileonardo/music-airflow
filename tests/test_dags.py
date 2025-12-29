"""
Tests for DAG structure and validity.

Ensures DAGs load without errors and meet custom requirements.
Based on Astronomer testing best practices for CI/CD.
"""

import datetime as dt
from pathlib import Path

import pytest
from airflow.models import DagBag
from airflow.utils import db


@pytest.fixture(scope="session", autouse=True)
def initialize_airflow_db():
    """Initialize Airflow database for DAG tests."""
    db.initdb()
    yield
    db.resetdb()


@pytest.fixture(scope="module")
def dag_bag():
    """Load all DAGs from the dags directory once for all tests."""
    dags_folder = Path(__file__).parent.parent / "src" / "music_airflow" / "dags"
    return DagBag(dag_folder=str(dags_folder), include_examples=False)


class TestDagValidation:
    """DAG validation tests - ensure all DAGs meet requirements."""

    def test_no_import_errors(self, dag_bag):
        """Test that all DAGs load without import errors."""
        assert len(dag_bag.import_errors) == 0, (
            f"DAG import failures: {dag_bag.import_errors}"
        )

    def test_expected_dags_present(self, dag_bag):
        """Test that expected DAGs are present."""
        expected_dags = {
            "lastfm_plays",
            "gold_play_aggregations",
            "lastfm_dimensions",
            "candidate_generation",
        }
        dag_ids = set(dag_bag.dag_ids)
        assert expected_dags.issubset(dag_ids), (
            f"Missing DAGs: {expected_dags - dag_ids}"
        )

    def test_all_dags_have_tags(self, dag_bag):
        """Test that all DAGs have at least one tag."""
        for dag_id, dag in dag_bag.dags.items():
            assert dag.tags, f"{dag_id} has no tags"

    def test_all_dags_have_start_date(self, dag_bag):
        """Test that all DAGs have a start_date."""
        for dag_id, dag in dag_bag.dags.items():
            assert dag.start_date is not None, f"{dag_id} has no start_date"

    def test_all_dags_have_tasks(self, dag_bag):
        """Test that all DAGs have at least one task."""
        for dag_id, dag in dag_bag.dags.items():
            assert len(dag.tasks) > 0, f"{dag_id} has no tasks"


class TestLastFmPlaysDag:
    """Test lastfm_plays DAG structure."""

    def test_dag_properties(self, dag_bag):
        """Test basic DAG configuration."""
        dag = dag_bag.get_dag("lastfm_plays")

        assert dag is not None
        assert dag.schedule == "@daily"
        assert dag.start_date == dt.datetime(2025, 12, 25, tzinfo=dt.timezone.utc)
        assert dag.catchup is True
        assert dag.max_active_runs == 1
        assert "lastfm" in dag.tags

    def test_task_count(self, dag_bag):
        """Test that DAG has expected tasks."""
        dag = dag_bag.get_dag("lastfm_plays")

        # fetch_plays and transform_and_save are dynamically mapped
        # so we count unique task_ids
        task_ids = [task.task_id for task in dag.tasks]

        # Should have fetch_plays and transform_and_save
        assert "fetch_plays" in task_ids
        assert "transform_and_save" in task_ids

    def test_task_dependencies(self, dag_bag):
        """Test that task dependencies are correct."""
        dag = dag_bag.get_dag("lastfm_plays")

        fetch_task = dag.get_task("fetch_plays")
        transform_task = dag.get_task("transform_and_save")

        # transform_and_save should depend on fetch_plays
        assert fetch_task in transform_task.upstream_list

    def test_tasks_have_outlets(self, dag_bag):
        """Test that transform_and_save produces the plays asset."""
        dag = dag_bag.get_dag("lastfm_plays")

        transform_task = dag.get_task("transform_and_save")
        assert len(transform_task.outlets) == 1
        assert "plays" in str(transform_task.outlets[0])


class TestGoldPlayAggregationsDag:
    """Test gold_play_aggregations DAG structure."""

    def test_dag_properties(self, dag_bag):
        """Test basic DAG configuration."""
        dag = dag_bag.get_dag("gold_play_aggregations")

        assert dag is not None
        assert dag.schedule is not None
        assert dag.catchup is False
        assert "gold" in dag.tags

    def test_task_count(self, dag_bag):
        """Test that DAG has expected tasks."""
        dag = dag_bag.get_dag("gold_play_aggregations")

        task_ids = [task.task_id for task in dag.tasks]

        # Should have artist and track aggregation tasks
        assert "compute_artist_aggregations" in task_ids
        assert "compute_track_aggregations" in task_ids

    def test_asset_dependency(self, dag_bag):
        """Test that DAG depends on plays asset."""
        dag = dag_bag.get_dag("gold_play_aggregations")

        # Check that DAG is triggered by plays asset
        assert dag.schedule is not None
        # For Airflow 3.0 asset scheduling
        schedule_str = str(dag.schedule)
        assert "plays" in schedule_str.lower() or "asset" in schedule_str.lower()

    def test_tasks_run_in_parallel(self, dag_bag):
        """Test that aggregation tasks have no dependencies (can run in parallel)."""
        dag = dag_bag.get_dag("gold_play_aggregations")

        artist_task = dag.get_task("compute_artist_aggregations")
        track_task = dag.get_task("compute_track_aggregations")

        # Tasks should not depend on each other
        assert artist_task not in track_task.upstream_list
        assert track_task not in artist_task.upstream_list


class TestLastFmDimensionsDag:
    """Test lastfm_dimensions DAG structure."""

    def test_dag_properties(self, dag_bag):
        """Test basic DAG configuration."""
        dag = dag_bag.get_dag("lastfm_dimensions")

        assert dag is not None
        assert dag.schedule == "@weekly"
        assert dag.catchup is False
        assert "dimensions" in dag.tags

    def test_task_count(self, dag_bag):
        """Test that DAG has expected tasks."""
        dag = dag_bag.get_dag("lastfm_dimensions")

        task_ids = [task.task_id for task in dag.tasks]

        # Should have extract and transform tasks for both tracks and artists
        assert "fetch_tracks" in task_ids
        assert "transform_tracks" in task_ids
        assert "fetch_artists" in task_ids
        assert "transform_artists" in task_ids

    def test_task_dependencies(self, dag_bag):
        """Test that task dependencies are correct."""
        dag = dag_bag.get_dag("lastfm_dimensions")

        fetch_tracks = dag.get_task("fetch_tracks")
        transform_tracks = dag.get_task("transform_tracks")
        fetch_artists = dag.get_task("fetch_artists")
        transform_artists = dag.get_task("transform_artists")

        # transform should depend on extract
        assert fetch_tracks in transform_tracks.upstream_list
        assert fetch_artists in transform_artists.upstream_list

        # tracks and artists pipelines can run in parallel
        assert fetch_artists not in fetch_tracks.upstream_list
        assert fetch_tracks not in fetch_artists.upstream_list


class TestCandidateGenerationDAG:
    """Test candidate_generation DAG structure."""

    def test_dag_properties(self, dag_bag):
        """Test basic DAG configuration."""
        dag = dag_bag.get_dag("candidate_generation")

        assert dag is not None
        assert dag.catchup is False
        assert "gold" in dag.tags
        assert "candidates" in dag.tags

    def test_task_count(self, dag_bag):
        """Test that DAG has expected tasks."""
        dag = dag_bag.get_dag("candidate_generation")

        task_ids = [task.task_id for task in dag.tasks]

        # Should have core tasks
        assert "get_active_users" in task_ids
        assert "generate_similar_artist" in task_ids
        assert "generate_similar_tag" in task_ids
        assert "generate_deep_cut" in task_ids
        assert "consolidate_to_gold" in task_ids
        assert "consolidate_results" in task_ids

    def test_task_dependencies(self, dag_bag):
        """Test that task dependencies are correct."""
        dag = dag_bag.get_dag("candidate_generation")

        consolidate = dag.get_task("consolidate_to_gold")

        # Consolidate should have generation tasks as upstream
        assert len(consolidate.upstream_list) > 0
