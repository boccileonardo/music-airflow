from dotenv import load_dotenv
import os
import datetime as dt

LAST_FM_USERNAMES = ["lelopolel", "Martazie"]
DEFAULT_USERNAME = "lelopolel"

# YouTube playlist limits (API quota is per-app, not per-user)
YOUTUBE_MAX_TRACKS_DEFAULT = 30  # Non-default users
YOUTUBE_MAX_TRACKS_OWNER = 100  # Default username only

load_dotenv()
DAG_START_DATE = dt.datetime.strptime(
    os.getenv("DAG_START_DATE", "2025-11-01"), "%Y-%m-%d"
).replace(tzinfo=dt.timezone.utc)
