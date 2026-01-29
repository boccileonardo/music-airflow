# Fuzzy Search Refactoring Implementation Plan

## Overview

This refactoring eliminates MusicBrainz ID dependencies and implements robust fuzzy text matching for track identification. It also removes the circular dependency by ensuring only one DAG writes to each table.

## Core Changes

### 1. Fuzzy Text Matching Strategy

**Problem**: MusicBrainz IDs are unreliable and different recordings of the same song get different IDs.

**Solution**: Text-based normalization and canonical IDs
- Normalize track/artist names (lowercase, remove suffixes like "(Remastered)", "(Live)")
- Generate canonical ID: `normalize(track_name)|normalize(artist_name)`
- When deduplicating, prefer: non-music-video versions with highest playcount

### 2. Eliminate Circular Dependency & Concurrent Writes

**Problem**: `candidate_enriched_tracks` creates circular flow and potential concurrent writes.

**Solution**: Remove intermediate table, accept delayed enrichment
- Only `lastfm_dimensions` writes to `tracks` dimension (weekly)
- `candidate_generation` writes track_id only (no metadata enrichment)
- `lastfm_dimensions` reads candidates and enriches them on next weekly run
- Result: 1-7 day delay for new track metadata (acceptable tradeoff)

## Implementation Phases

### Phase 1: Text Normalization Utilities (3 hours)

**Create**: `src/music_airflow/utils/text_normalization.py`

Functions:
- `normalize_text(text: str) -> str`: Remove suffixes, lowercase, strip punctuation
- `is_music_video(text: str) -> bool`: Detect video versions
- `generate_canonical_track_id(track_name: str, artist_name: str) -> str`

**Update**: `_deduplicate_tracks()` in `dimensions.py`
- Use normalized grouping
- Prefer non-video, highest playcount
- Remove album-based grouping

**Update**: Track ID generation everywhere:
- `plays.py`: `transform_plays_raw_to_structured()`
- `dimensions.py`: `_transform_tracks_raw_to_structured()`
- `candidate_generation.py`: All track_id generation

### Phase 2: Remove Enrichment from candidate_generation (2 hours)

**Update**: `merge_candidate_sources()` in `candidate_generation.py`
- Remove calls to `enrich_track_metadata()`
- Remove writes to `candidate_enriched_tracks`
- Write track_id only to gold (metadata optional)

**Delete**: `_union_enriched_recommended_tracks()` from `dimensions.py`

### Phase 3: Update lastfm_dimensions to Read Candidates (2 hours)

**Update**: `fetch_tracks()` in `extract/tracks.py`
- Read from `gold/track_candidates`
- Parse track_ids to extract track/artist names
- Include in fetch queue alongside play-based tracks

### Phase 4: Update Streamlit App (2 hours)

**Update**: `load_track_candidates()` in `streamlit_app.py`
- Join with tracks dimension (don't parse track_id)
- Handle missing metadata gracefully
- Remove reference to `candidate_enriched_tracks`

### Phase 5: Remove MusicBrainz Dependencies (1 hour)

**Delete** from `dimensions.py`:
- `_search_musicbrainz_track_mbid()`
- `_search_musicbrainz_artist_mbid()`
- `_enrich_missing_artist_mbids()`

**Update**: Remove MBID enrichment calls

### Phase 6: Testing & Validation (2 hours)

- Unit tests for normalization
- Full pipeline end-to-end test
- Verify no concurrent writes
- Check recommendation quality
- App functionality test

**Total Estimated Time**: 12 hours

## Data Flow After Refactoring

```
Bronze → Silver → Gold

Plays (daily):
  lastfm_plays → bronze/plays → silver/plays

Dimensions (weekly):
  lastfm_dimensions → bronze/tracks,artists → silver/tracks,artists,dim_users
                   ↑ also reads gold/track_candidates for new track discovery

Aggregations (asset-triggered):
  gold_play_aggregations → gold/artist_play_count, track_play_count

Candidates (asset-triggered):
  candidate_generation → silver/candidate_* → gold/track_candidates
  (writes track_id only, no metadata)

App reads: gold/track_candidates + silver/tracks
```

## DAG → Table Write Matrix (No Conflicts!)

| Table | Writer DAG | Readers |
|-------|------------|---------|
| `plays` | `lastfm_plays` | All |
| `tracks` | `lastfm_dimensions` | All |
| `artists` | `lastfm_dimensions` | All |
| `dim_users` | `lastfm_dimensions` | All |
| `candidate_*` | `candidate_generation` | `candidate_generation` |
| `track_candidates` | `candidate_generation` | App, `lastfm_dimensions` |
| `*_play_count` | `gold_play_aggregations` | `candidate_generation` |

✅ Each table has exactly ONE writer - no concurrent conflicts!

## Key Technical Details

### Track ID Format

**Before**: `track_mbid` or `track_name|artist_name` (original text)

**After**: Always `normalize(track_name)|normalize(artist_name)`

Example:
- "Highway Star (Remastered 2012)" + "Deep Purple" → `"highway star|deep purple"`
- "Highway Star (Live)" + "Deep Purple" → `"highway star|deep purple"`
- Same canonical ID = same track!

### Deduplication Logic

```python
# Group by normalized names
# Sort by: is_music_video (False first), playcount (desc)
# Take first = best version
# Keep metadata from best version
# Use max(playcount) across all versions
```

### App Handling of Missing Metadata

```python
# If track not yet in dimensions:
track_name, artist_name = track_id.split("|")  # Parse from ID
display_name = f"{track_name.title()} - {artist_name.title()}"
metadata_status = "Pending enrichment..."
```

## Migration Steps

1. **Backup data**: Snapshot all Delta tables
2. **Implement changes**: Follow phases 1-6
3. **Test thoroughly**: Run all tests
4. **Regenerate track_ids**: One-time migration script to recompute all track_ids with new logic
5. **Deploy**: Roll out to production

## Success Criteria

- ✅ Zero MusicBrainz API calls
- ✅ No concurrent write conflicts
- ✅ No circular dependencies
- ✅ Track deduplication handles recording variations
- ✅ All tests pass
- ✅ Pipeline runs end-to-end
- ✅ App displays recommendations correctly
