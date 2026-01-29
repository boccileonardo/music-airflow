# Music Recommendation System

Personalized music recommendation engine using Apache Airflow, Last.fm data, and Delta Lake. Balances music discovery with forgotten favorites using exponential decay scoring to avoid feedback loops.

## Architecture

[System Architecture](06-complete-architecture.md) - Complete pipeline overview with DAG orchestration, data layers, and key techniques.

## Quick Reference

### Data Flow

```
Last.fm API → Bronze (JSON) → Silver (Delta) → Gold (Aggregations) → Streamlit
```

### DAGs

| DAG | Schedule | Purpose |
|-----|----------|---------|
| lastfm_plays | @daily | Extract listening history |
| lastfm_dimensions | @weekly | Extract track/artist metadata |
| gold_play_aggregations | Asset-triggered | Compute recency scores |
| candidate_generation | Asset-triggered | Generate recommendations |

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
