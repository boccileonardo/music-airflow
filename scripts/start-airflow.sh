#!/bin/bash

export AIRFLOW_HOME=/home/leo/repos/music-airflow/airflow
export AIRFLOW__CORE__LOAD_EXAMPLES=False

# Create .env if it doesn't exist
touch .env

# Run airflow standalone in the background and capture its output
LOG_FILE=$(mktemp)
# Use tee to redirect output to the log file AND the terminal
uv run airflow standalone 2>&1 | tee "$LOG_FILE" &
AIRFLOW_PID=$!

echo "Waiting for Airflow to start and generate credentials..."

# Function to extract credentials
extract_creds() {
    local user_line
    local pass_line
    user_line=$(grep -oP "user '\K[^']+" "$LOG_FILE" | head -n 1)
    pass_line=$(grep -oP "Password for user '.*': \K.*" "$LOG_FILE" | head -n 1)

    if [ -n "$user_line" ] && [ -n "$pass_line" ]; then
        # Remove old Airflow creds and append new ones
        grep -v "AIRFLOW_UI_USER" .env | grep -v "AIRFLOW_UI_PASSWORD" > .env.tmp
        mv .env.tmp .env
        echo "AIRFLOW_UI_USER=$user_line" >> .env
        echo "AIRFLOW_UI_PASSWORD=$pass_line" >> .env
        return 0 # Success
    else
        return 1 # Not found yet
    fi
}

# Poll for credentials
creds_found=false
for i in {1..30}; do
    if extract_creds; then
        echo "Credentials captured in .env file:"
        cat .env
        creds_found=true
        break
    fi
    sleep 1
done

if [ "$creds_found" = "false" ]; then
    echo "Could not find credentials after 30 seconds."
    echo "Airflow logs:"
    cat "$LOG_FILE"
    kill $AIRFLOW_PID
    rm "$LOG_FILE"
    exit 1
fi

# Clean up the log file, we don't need it anymore
rm "$LOG_FILE"

# Bring Airflow to the foreground and wait for it
echo "Airflow is running. Press Ctrl+C to stop."
wait $AIRFLOW_PID