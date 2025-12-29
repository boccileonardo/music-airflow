#!/bin/bash

# Get the project root directory (parent of scripts/)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd .. && pwd)"

# Args
CLEAN=false
SHOW_HELP=false
for arg in "$@"; do
    case "$arg" in
        --clean)
            CLEAN=true
            ;;
        -h|--help)
            SHOW_HELP=true
            ;;
    esac
done

if [[ "$SHOW_HELP" == "true" ]]; then
    echo "Usage: $(basename "$0") [--clean]"
    echo "  --clean    Remove data/ and airflow/ before startup"
    exit 0
fi

export AIRFLOW_HOME="$PROJECT_ROOT/airflow"
export AIRFLOW__CORE__LOAD_EXAMPLES=False
export AIRFLOW__CORE__DAGS_FOLDER="$PROJECT_ROOT/src/music_airflow/dags"
export AIRFLOW__CORE__PARALLELISM=1 # prevent local oom

# Optional cleanup
if [[ "$CLEAN" == "true" ]]; then
    echo "Cleaning directories before startup..."
    # Safeguard: only remove within the known project root
    if [[ -d "$PROJECT_ROOT/data" ]]; then
        rm -rf "$PROJECT_ROOT/data"
        echo "✓ Removed $PROJECT_ROOT/data"
    else
        echo "- Skipped: $PROJECT_ROOT/data not found"
    fi
    if [[ -d "$AIRFLOW_HOME" ]]; then
        rm -rf "$AIRFLOW_HOME"
        echo "✓ Removed $AIRFLOW_HOME"
    else
        echo "- Skipped: $AIRFLOW_HOME not found"
    fi
fi

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
