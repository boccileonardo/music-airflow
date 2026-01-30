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

### 3. YouTube Playlist Integration (OAuth Setup)

The app supports creating YouTube playlists from recommendations using the YouTube Data API v3.

**Note:** The API has daily quota limits (10,000 units/day). Each search costs 100 units, playlist creation costs 50 units.

#### GCP Setup

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a project and enable the **YouTube Data API v3**
3. Go to **Credentials** → **Create Credentials** → **OAuth client ID**
4. Choose **Desktop app** (for local) or **Web application** (for Streamlit Cloud)
5. For web apps, add authorized redirect URIs:
   - `http://localhost:8501` (local development)
   - Your Streamlit Cloud URL

#### Local Setup (.env)

Add OAuth client credentials to your `.env` file:

```bash
YOUTUBE_CLIENT_ID=your-client-id.apps.googleusercontent.com
YOUTUBE_CLIENT_SECRET=your-client-secret
```

Then run the app and click **"🔐 Start YouTube API OAuth"** to complete the OAuth flow. The app will display the access and refresh tokens to add to your `.env`:

```bash
YOUTUBE_ACCESS_TOKEN=your-access-token
YOUTUBE_REFRESH_TOKEN=your-refresh-token
```

#### Streamlit Cloud Deployment

Add credentials as flat keys in Streamlit secrets (Settings → Secrets):

```toml
YOUTUBE_CLIENT_ID = "your-client-id.apps.googleusercontent.com"
YOUTUBE_CLIENT_SECRET = "your-client-secret"
YOUTUBE_ACCESS_TOKEN = "your-access-token"
YOUTUBE_REFRESH_TOKEN = "your-refresh-token"
```

**Note:** If tokens expire (you see `invalid_grant` errors), re-run the OAuth flow locally and update the tokens.
