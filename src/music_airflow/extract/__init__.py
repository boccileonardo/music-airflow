"""
Extract module

Write raw data to bronze layer.
"""

from music_airflow.extract.plays import extract_plays_to_bronze

__all__ = ["extract_plays_to_bronze"]
