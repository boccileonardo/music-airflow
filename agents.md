# AI Agent Guidelines for Music Airflow Project

This document outlines key architectural decisions and conventions for this project. AI assistants should follow these guidelines when working on this codebase.

## Project Overview

This is a personal learning project that uses Apache Airflow for a music recommendation system with a Streamlit frontend.

## 1. Package Management

**Package Manager:** `uv`

- **Installing dependencies:** `uv add packagename` (add `--dev` for dev dependencies)
- **Running Python code:** `uv run src/...`
- **Running tests:** `uv run pytest`

❌ **Do NOT use:**
- `pip install`
- `virtualenv`, `venv`, or manual virtual environment creation
- `python -m ...` (use `uv run` instead)

## 2. Apache Airflow (Version 3.0+)

**Modern Airflow Patterns Required**

### TaskFlow API (Preferred)

Always use the TaskFlow API with decorators over legacy manual DAG definitions.

✅ **Good - TaskFlow API:**
```python
from airflow.decorators import dag, task
from datetime import datetime

@dag(
    schedule="@daily",
    start_date=datetime(2025, 1, 1),
    catchup=False,
)
def my_music_dag():
    @task
    def extract_data():
        # Extract logic
        return data_location

    @task
    def transform_data(location: str):
        # Transform logic
        return transformed_location

my_music_dag()
```

❌ **Avoid - Legacy API:**
```python
from airflow import DAG
from airflow.operators.python import PythonOperator

dag = DAG(...)
task1 = PythonOperator(...)
task2 = PythonOperator(...)
```

### Key Points
- Use `@dag` and `@task` decorators
- Leverage type hints for task dependencies
- Use modern Airflow 3.0+ features and documentation

## 3. Data Passing Between Tasks

**XCOMs for Metadata Only**

❌ **Never pass actual data (DataFrames, large objects) through XCOMs**

✅ **Only pass metadata:**
- File paths
- S3/GCS URIs
- Database table names
- Record counts
- Status flags

**Example:**
```python
@task
def process_data():
    # Process and save data
    df = get_music_data()
    output_path = "/tmp/music_data.parquet"
    df.to_parquet(output_path)

    # Return only the location, not the dataframe
    return {"path": output_path, "rows": len(df)}

@task
def analyze_data(metadata: dict):
    # Read from the location
    df = pd.read_parquet(metadata["path"])
    # ... analyze
```

## 4. Testing

**Test Location:** `tests/`

**Running Tests:**
- Via pytest: `uv run pytest`
- Via MCP test tool (when available)

**Test Structure:**
- Write unit tests for all business logic
- Test DAG structure and task dependencies
- Mock external dependencies (APIs, databases)

**Example:**
```python
# tests/test_dags.py
from airflow.models import DagBag

def test_dag_loads():
    dag_bag = DagBag(dag_folder="src/dags", include_examples=False)
    assert len(dag_bag.import_errors) == 0
```

## 5. Project Structure

```
src/
  dags/          # Airflow DAG definitions
  utils/         # Shared utilities and helpers
tests/           # Unit and integration tests
scripts/         # Setup and management scripts
```

## Summary

1. ✅ Use `uv` for all package management
2. ✅ Use TaskFlow API (@dag, @task) for Airflow 3.0+
3. ✅ XCOMs = metadata only (paths, URIs, counts)
4. ✅ Write tests in `tests/`, run with `uv run pytest`
