# Last.FM Music Recommender with Airflow and Streamlit

Use Airflow to collect liked and recently played songs in the music platforms I use (yt music and deezer) from last.fm and have a recommender system that supports various modes (favorites, discover new, etc.) in a streamlit app.

## Quickstart

### 1. Set up Airflow

```sh
chmod +x ./scripts/install-airflow.sh
./scripts/install-airflow.sh

chmod +x ./scripts/start-airflow.sh
./scripts/start-airflow.sh
```

Add api_key and shared_secret to .env file.
[LastFM API Accounts Page](https://www.last.fm/api/accounts).
Optionally, add DAG_START_DATE to .env file in YYYY-MM-DD format to set the start date for DAGs. Default is 2025-11-01.

### 2. Run the Streamlit App

After your Airflow DAGs have run and generated gold tables with recommendations:

```sh
uv run streamlit run src/music_airflow/app/streamlit_app.py
```

The app will be available at http://localhost:8501

### 3. Oauth setup on GCP for Youtube API and Streamlit Cloud Deployment

Create a streamlit community cloud app.
In GCP, create a project, enable youtube v3 api, create oauth credentials for web app with streamlit cloud url.
Run the app locally first as above and authenticate to get the token.
Copy the token from the `.youtube_token.json` file.
On streamlit cloud, add the following to secrets toml:

```toml
[youtube]
client_id = "ID"
client_secret = "SECRET"

[youtube_token]
# Copy from .youtube_token.json after one-time authentication locally
token = "TOKEN"
refresh_token = ""
token_uri = "https://oauth2.googleapis.com/token"
client_id = "ID"
client_secret = "SECRET"
scopes = ["https://www.googleapis.com/auth/youtube.force-ssl"]
