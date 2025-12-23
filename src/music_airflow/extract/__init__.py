"""
Extract module

Write raw data to bronze layer.
"""

from music_airflow.extract.plays import extract_plays_to_bronze
from music_airflow.extract.dimensions import (
    extract_tracks_to_bronze,
    extract_artists_to_bronze,
)

__all__ = [
    "extract_plays_to_bronze",
    "extract_tracks_to_bronze",
    "extract_artists_to_bronze",
]
