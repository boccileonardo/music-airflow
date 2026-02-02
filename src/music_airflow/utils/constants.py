from dotenv import load_dotenv
import os
import datetime as dt

LAST_FM_USERNAMES = ["lelopolel", "Martazie"]

load_dotenv()
DAG_START_DATE = dt.datetime.strptime(
    os.getenv("DAG_START_DATE", "2025-11-01"), "%Y-%m-%d"
).replace(tzinfo=dt.timezone.utc)
