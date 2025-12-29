# Last.FM Music Recommender with Airflow and Streamlit

Use Airflow to collect liked and recently played songs in the music platforms I use (yt music and deezer) from last.fm and have a recommender system that supports various modes (favorites, discover new, etc.) in a streamlit app.

Quickstart:

```sh
chmod +x ./scripts/install-airflow.sh
./scripts/install-airflow.sh

chmod +x ./scripts/start-airflow.sh
./scripts/start-airflow.sh
```

Add api_key and shared_secret to .env file.  
[LastFM API Accounts Page](https://www.last.fm/api/accounts).  
Optionally, add DAG_START_DATE to .env file in YYYY-MM-DD format to set the start date for DAGs. Default is 2025-11-01.
