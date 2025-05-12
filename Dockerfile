FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    ALGO_DB_URL=sqlite:///data/algo.db \
    SUPERVISOR_TOKEN=changeme \
    DASHBOARD_PASSWORD=password \
    PIP_NO_CACHE_DIR=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends build-essential git curl && \
    rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 - --version 1.8.0 && \
    rm -rf /var/lib/apt/lists/*
ENV PATH="/root/.local/bin:$PATH"

# Set up working directory
WORKDIR /app

# Copy all project files first
COPY README.md pyproject.toml poetry.lock ./
COPY src/ ./src/
COPY configs/ ./configs/
COPY .streamlit/ ./.streamlit/

# Install dependencies and the package
RUN poetry config virtualenvs.create false && \
    poetry install --no-dev --only main

# Ensure the data directory exists and has correct permissions
RUN mkdir -p /app/data && chmod -R 777 /app/data

# Correct Entrypoint to run the FastAPI app using uvicorn
ENTRYPOINT ["uvicorn", "algo_mvp.supervisor.server:app", "--host", "0.0.0.0", "--port", "8000"]
