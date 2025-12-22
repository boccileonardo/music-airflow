#!/bin/bash

# Get the project root directory (parent of scripts/)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd .. && pwd)"

export AIRFLOW_HOME="$PROJECT_ROOT/airflow"
export AIRFLOW__CORE__LOAD_EXAMPLES=False
export AIRFLOW__CORE__DAGS_FOLDER="$PROJECT_ROOT/src/music_airflow/dags"

cd "$PROJECT_ROOT"
touch .env

# Run airflow and watch for credentials in the output
uv run airflow standalone 2>&1 | while IFS= read -r line; do
    echo "$line"

    # Extract credentials when they appear
    if [[ $line =~ Password\ for\ user\ \'([^\']+)\':[[:space:]]*(.+) ]]; then
        user="${BASH_REMATCH[1]}"
        password="${BASH_REMATCH[2]}"

        # Update .env file (remove old entries, add new ones)
        grep -v "AIRFLOW_UI_USER\|AIRFLOW_UI_PASSWORD" .env > .env.tmp 2>/dev/null || true
        mv .env.tmp .env
        echo "AIRFLOW_UI_USER=$user" >> .env
        echo "AIRFLOW_UI_PASSWORD=$password" >> .env

        echo ""
        echo "✓ Credentials saved to .env"
        echo ""
    fi
done
