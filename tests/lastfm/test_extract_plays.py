"""
Tests for plays extraction logic.

Tests extract_plays_to_bronze function with mocked API and file I/O.
"""

import datetime as dt
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from airflow.exceptions import AirflowSkipException

from music_airflow.extract.plays import extract_plays_to_bronze


class TestExtractPlaysIntegration:
    """Integration tests for extract_plays_to_bronze."""

    @patch("music_airflow.extract.plays.LastFMClient")
    @patch("music_airflow.extract.plays.JSONIOManager")
    def test_successful_extraction(
        self,
        mock_io_manager_class,
        mock_client_class,
        sample_tracks_response,
        test_data_dir,
    ):
        """Test successful extraction of plays to bronze layer."""
        # Mock the LastFMClient
        mock_client = MagicMock()
        mock_client.get_recent_tracks.return_value = sample_tracks_response[
            "recenttracks"
        ]["track"]
        mock_client_class.return_value = mock_client

        # Mock JSONIOManager to use test directory
        from music_airflow.utils.polars_io_manager import JSONIOManager

        mock_io_manager = JSONIOManager(medallion_layer="bronze")
        mock_io_manager.base_dir = test_data_dir / "bronze"
        mock_io_manager_class.return_value = mock_io_manager

        from_dt = dt.datetime(2021, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc)
        to_dt = dt.datetime(2021, 1, 2, 0, 0, 0, tzinfo=dt.timezone.utc)

        result = extract_plays_to_bronze(
            username="testuser", from_dt=from_dt, to_dt=to_dt
        )

        # Verify result metadata
        assert result["username"] == "testuser"
        assert result["rows"] == 3
        assert result["format"] == "json"
        assert result["medallion_layer"] == "bronze"
        assert "path" in result
        assert "filename" in result
        assert result["filename"] == "plays/testuser/20210101.json"

        # Verify file was written
        output_path = Path(result["path"])
        assert output_path.exists()

        # Verify content
        with open(output_path) as f:
            data = json.load(f)
            assert len(data) == 3
            assert data[0]["name"] == "Creep"

    @patch("music_airflow.extract.plays.LastFMClient")
    def test_empty_response_raises_skip(self, mock_client_class):
        """Test that empty response raises AirflowSkipException."""
        mock_client = MagicMock()
        mock_client.get_recent_tracks.return_value = []
        mock_client_class.return_value = mock_client

        from_dt = dt.datetime(2021, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc)
        to_dt = dt.datetime(2021, 1, 2, 0, 0, 0, tzinfo=dt.timezone.utc)

        # Should raise before any file I/O happens
        with pytest.raises(
            AirflowSkipException, match="No plays found for testuser on 20210101"
        ):
            extract_plays_to_bronze(username="testuser", from_dt=from_dt, to_dt=to_dt)

    @patch("music_airflow.extract.plays.LastFMClient")
    @patch("music_airflow.extract.plays.JSONIOManager")
    def test_timestamp_conversion(
        self,
        mock_io_manager_class,
        mock_client_class,
        sample_tracks_response,
        test_data_dir,
    ):
        """Test that datetime is correctly converted to Unix timestamp."""
        mock_client = MagicMock()
        mock_client.get_recent_tracks.return_value = sample_tracks_response[
            "recenttracks"
        ]["track"]
        mock_client_class.return_value = mock_client

        # Mock JSONIOManager to use test directory
        from music_airflow.utils.polars_io_manager import JSONIOManager

        mock_io_manager = JSONIOManager(medallion_layer="bronze")
        mock_io_manager.base_dir = test_data_dir / "bronze"
        mock_io_manager_class.return_value = mock_io_manager

        from_dt = dt.datetime(2021, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc)
        to_dt = dt.datetime(2021, 1, 2, 0, 0, 0, tzinfo=dt.timezone.utc)

        # Call the function (will succeed with mocked IO)
        extract_plays_to_bronze(username="testuser", from_dt=from_dt, to_dt=to_dt)

        # Verify timestamps were passed correctly to the API client
        mock_client.get_recent_tracks.assert_called_once_with(
            from_timestamp=1609459200,  # 2021-01-01 00:00:00 UTC
            to_timestamp=1609545600,  # 2021-01-02 00:00:00 UTC
            extended=True,
        )

    @patch("music_airflow.extract.plays.LastFMClient")
    @patch("music_airflow.extract.plays.JSONIOManager")
    def test_filename_format(
        self,
        mock_io_manager_class,
        mock_client_class,
        sample_tracks_response,
        test_data_dir,
    ):
        """Test that filename follows expected format."""
        mock_client = MagicMock()
        mock_client.get_recent_tracks.return_value = sample_tracks_response[
            "recenttracks"
        ]["track"]
        mock_client_class.return_value = mock_client

        # Mock JSONIOManager to use test directory
        from music_airflow.utils.polars_io_manager import JSONIOManager

        mock_io_manager = JSONIOManager(medallion_layer="bronze")
        mock_io_manager.base_dir = test_data_dir / "bronze"
        mock_io_manager_class.return_value = mock_io_manager

        from_dt = dt.datetime(2023, 12, 25, 0, 0, 0, tzinfo=dt.timezone.utc)
        to_dt = dt.datetime(2023, 12, 26, 0, 0, 0, tzinfo=dt.timezone.utc)

        result = extract_plays_to_bronze(
            username="john_doe", from_dt=from_dt, to_dt=to_dt
        )
        assert result["filename"] == "plays/john_doe/20231225.json"

    @patch("music_airflow.extract.plays.LastFMClient")
    @patch("music_airflow.extract.plays.JSONIOManager")
    def test_result_includes_datetime_strings(
        self,
        mock_io_manager_class,
        mock_client_class,
        sample_tracks_response,
        test_data_dir,
    ):
        """Test that result includes ISO format datetime strings."""
        mock_client = MagicMock()
        mock_client.get_recent_tracks.return_value = sample_tracks_response[
            "recenttracks"
        ]["track"]
        mock_client_class.return_value = mock_client

        # Mock JSONIOManager to use test directory
        from music_airflow.utils.polars_io_manager import JSONIOManager

        mock_io_manager = JSONIOManager(medallion_layer="bronze")
        mock_io_manager.base_dir = test_data_dir / "bronze"
        mock_io_manager_class.return_value = mock_io_manager

        from_dt = dt.datetime(2021, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc)
        to_dt = dt.datetime(2021, 1, 2, 0, 0, 0, tzinfo=dt.timezone.utc)

        result = extract_plays_to_bronze(
            username="testuser", from_dt=from_dt, to_dt=to_dt
        )

        assert result["from_datetime"] == "2021-01-01T00:00:00+00:00"
        assert result["to_datetime"] == "2021-01-02T00:00:00+00:00"
