"""Smoke tests for the Docker Compose setup."""

import os
import subprocess
import time
import pytest
import requests

# Define the maximum wait time in seconds
MAX_WAIT_SECONDS = 60
HEALTH_CHECK_INTERVAL = 5  # seconds


@pytest.mark.skipif(
    os.getenv("CI_DOCKER") != "true",
    reason="Docker tests are skipped unless CI_DOCKER is true (requires Docker daemon)",
)
def test_docker_compose_smoke():
    """Test that docker-compose up starts services and they become healthy."""
    try:
        # Build and start services in detached mode
        print("Building and starting Docker Compose services...")
        compose_up_command = ["docker", "compose", "up", "-d", "--build"]
        subprocess.run(compose_up_command, check=True, capture_output=True, text=True)
        print("Docker Compose services started.")

        start_time = time.time()
        backend_healthy = False
        dashboard_accessible = False

        # Wait for backend health check
        print("Waiting for backend health check (http://localhost:8000/health)...")
        while time.time() - start_time < MAX_WAIT_SECONDS:
            try:
                response = requests.get("http://localhost:8000/health", timeout=5)
                if response.status_code == 200:
                    print("Backend is healthy.")
                    backend_healthy = True
                    break
            except requests.exceptions.ConnectionError:
                pass  # Service not yet available
            except requests.exceptions.Timeout:
                print("Backend health check timed out, retrying...")
            time.sleep(HEALTH_CHECK_INTERVAL)

        assert backend_healthy, (
            f"Backend did not become healthy within {MAX_WAIT_SECONDS} seconds."
        )

        # Wait for dashboard to be accessible
        print("Waiting for dashboard (http://localhost:8501)...")
        while time.time() - start_time < MAX_WAIT_SECONDS:
            try:
                response = requests.get("http://localhost:8501", timeout=5)
                # Check for any successful HTML response
                if response.status_code == 200 and "<html" in response.text.lower():
                    print("Dashboard is accessible.")
                    dashboard_accessible = True
                    break
            except requests.exceptions.ConnectionError:
                pass  # Service not yet available
            except requests.exceptions.Timeout:
                print("Dashboard accessibility check timed out, retrying...")
            time.sleep(HEALTH_CHECK_INTERVAL)

        assert dashboard_accessible, (
            f"Dashboard did not become accessible within {MAX_WAIT_SECONDS} seconds."
        )

    finally:
        # Stop and remove containers, networks, and volumes
        print("Stopping and cleaning up Docker Compose services...")
        compose_down_command = ["docker", "compose", "down", "-v"]
        subprocess.run(
            compose_down_command, check=False, capture_output=True, text=True
        )
        print("Docker Compose services stopped and cleaned up.")
