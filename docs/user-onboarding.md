# User Onboarding Guide

This guide covers the steps required to add a new user to the AirStream.FM music recommendation system.

## Access Modes

The app supports three access modes:

| Mode | Who | Capabilities |
|------|-----|--------------|
| **Dev Mode** | Local development (no auth configured) | Full access, user selector dropdown |
| **Demo Mode** | Unauthenticated or unmapped users | View-only access to default user's recommendations |
| **Full Access** | Authenticated users with mapped Last.fm username | Full access to own data, exclusions, playlist export |

Users in demo mode can browse the app and see sample recommendations, but cannot:
- Manage exclusions
- Export playlists to Spotify/YouTube

## Step 1: Add User to Code Configuration

### 1.1 Add Email-to-Username Mapping

Edit `src/music_airflow/app/auth.py` and add the user's email to the `EMAIL_TO_USERNAME` mapping:

```python
EMAIL_TO_USERNAME: dict[str, str] = {
    "existing.user@gmail.com": "existing_lastfm_user",
    "new.user@gmail.com": "new_lastfm_username",  # Add new user here
}
```

This mapping:
- Controls who gets full access to the app
- Links the user's Google login email to their Last.fm username
- Ensures users can only access their own data and OAuth tokens

### 1.2 Add to LAST_FM_USERNAMES

Edit `src/music_airflow/utils/constants.py` and add the Last.fm username:

```python
LAST_FM_USERNAMES = ["existing_user", "new_lastfm_username"]  # Add new username
```

This list is used by:
- The data pipeline to know which users to fetch data for
- Development mode (when auth is disabled) for the user dropdown

## Step 2: Configure OAuth Access

These steps are only needed if the user wants to export playlists to Spotify or YouTube.

### 2.1 Spotify Developer Dashboard

Spotify requires explicit user authorization during the "Development Mode" phase:

1. Go to [Spotify Developer Dashboard](https://developer.spotify.com/dashboard/321528f081c14851accde7f93f331277)
2. Add the user's Spotify email address to the list of allowed users

## Step 3: Run the Data Pipeline

After adding the user to the configuration, run the data pipeline to fetch their Last.fm listening history:

1. Trigger the `lastfm_plays` DAG for the new user, or wait for the daily scheduled run
2. The pipeline will automatically cascade through:
   - `lastfm_plays` → `candidate_generation` → `lastfm_dimensions` → `gold_play_aggregations`

## Step 4: Deploy Changes

1. Commit and push the code changes
2. Streamlit Cloud will automatically redeploy with the new user configuration
