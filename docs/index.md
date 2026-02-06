# Music Recommendation System

Personalized music recommendation engine using Apache Airflow, Last.fm data, and Delta Lake. Balances music discovery with forgotten favorites using exponential decay scoring to avoid feedback loops.

## Documentation

- [System Architecture](complete-architecture.md) - Complete pipeline overview with DAG orchestration, data layers, and key techniques
- [User Onboarding](user-onboarding.md) - How to add new users to the system

## Quick Reference

### Data Flow

```
Last.fm API → Bronze (JSON) → Silver (Delta) → Gold (Aggregations) → Streamlit
```

### DAGs

| DAG | Schedule | Purpose |
|-----|----------|---------|
| lastfm_plays | @daily | Extract listening history, trigger candidate generation |
| candidate_generation | Asset-triggered (plays) | Generate recommendations, trigger dimensions |
| lastfm_dimensions | Asset-triggered (candidates) | Extract track/artist metadata with streaming links |
| gold_play_aggregations | Asset-triggered (dimensions OR) | Compute recency scores |

**Flow**: plays → candidates → dimensions → aggregations

### Key Tables

**Silver**:

- `plays` - Normalized play events (partitioned by user)
- `tracks` - Track dimension with metadata
- `artists` - Artist dimension with similar artists
- `dim_users` - User profiles with half-life values
- `candidate_*` - Four types of recommendation candidates

**Gold**:

- `artist_play_count` - Artist stats with recency
- `track_play_count` - Track stats with recency
- `track_candidates` - Unified recommendations

### Tech Stack

- Apache Airflow 3.0 (TaskFlow API, asset scheduling)
- Polars (LazyFrame API)
- Delta Lake (ACID, partitioning)
- Last.fm API
- Streamlit
- pytest, ruff, ty, uv
