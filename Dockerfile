FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    ALGO_DB_URL=sqlite:///data/algo.db \
    SUPERVISOR_TOKEN=changeme \
    DASHBOARD_PASSWORD=password

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends build-essential git && \
    rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    curl -sSL https://install.python-poetry.org | python3 - --version 1.8.0 && \
    rm -rf /var/lib/apt/lists/*
ENV PATH="/root/.local/bin:$PATH"

# Copy project files and install dependencies
WORKDIR /app
COPY pyproject.toml poetry.lock ./
RUN poetry install --no-dev --only main

# Copy the rest of the application code
COPY src/ ./src/
COPY configs/ ./configs/
COPY .streamlit/ ./.streamlit/
# Ensure the data directory exists and has correct permissions if needed
RUN mkdir -p /app/data && chmod -R 777 /app/data

# Entrypoint
ENTRYPOINT ["python", "-m", "algo_mvp.supervisor", "--config", "configs/supervisor_sample.yaml"]
