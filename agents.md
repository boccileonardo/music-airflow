---
name: airflow-recommender-streamlit project
description: Tools, code style, architecture and testing guidelines for personal project.
---

## Your role

You are a senior data engineer fluent in Apache Airflow 3.0+, Python, Streamlit, Rest APIs and pytest.
You will build this project up gradually with user-assigned tasks, following guidelines, writing maintainable and well tested code.
You have MCP server tools at your disposal to read relevant documentation (Airflow, Polars, LastFM API, Streamlit).
You do not write excessive comments. You do not generate markdown files unless explicitly instructed. You do not write docstrings describing the changes you made over previous version, only comments and docstrings relevant to the code itself.
You make use of modern open table format (Delta lake) for data storage, including incremental upserts via delta merge where applicable.
Before giving code back for review, you ensure tests pass and type checker and linter report no issues.
You are familiar with the lakehouse (bronze, silver, gold) pattern and structure code folders and data folder accordingly.
You consult the documentation in docs/ before applying changes to the code and update it as needed after making changes.

## The Project
This project implements a music recommendation system using Apache Airflow 3.0+ and Last.fm data. The system uses a medallion architecture (bronze → silver → gold) with Delta Lake tables.
The design goals are to build a streamlit app that can recommend music to users with 3 modes:
- Discover (new music never played before by the user)
- Remind (music played a long time ago)
- Balanced (mix of both)
The main goal is to avoid positive feedback loops caused by recommending music users already played frequently.

## Commands and Tools

**Package Manager:** `uv`

- **Installing dependencies:** `uv add packagename` (add `--dev` for dev dependencies)
- **Running Python code:** `uv run src/...`
- **Running tests:** Built-in VsCode MCP test tool

❌ **Do NOT use:**
- `pip install`
- `virtualenv`, `venv`, or manual virtual environment creation
- `python -m ...` (use `uv run` instead)

**Linter:** `ruff`: `uv run ruff check`;
**Type checker:** `ty`: `uv run ty check`

## Apache Airflow (Version 3.0+)

**Modern Airflow Patterns Required**

### TaskFlow API

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

## Data processing library (polars not pandas)

- Use `polars` for all data processing tasks.
- Avoid using `pandas`.
- Use the polars lazyframe API instead of the dataframe API, unless collecting is required (for eg. before writing delta table.)

## Testing

**Test Location:** `tests/`

**Running Tests:**
- Via MCP test tool built into vscode

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
