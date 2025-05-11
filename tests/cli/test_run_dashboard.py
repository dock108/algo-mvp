"""Tests for the dashboard CLI command."""

import os
from unittest.mock import patch, MagicMock
import pytest
import subprocess

from cli import run_dashboard


@pytest.fixture
def mock_subprocess():
    """Mock the subprocess module for testing."""
    with patch("cli.run_dashboard.subprocess.run") as mock_run:
        # Configure the mock to return a simple object with returncode 0
        process_mock = MagicMock()
        process_mock.returncode = 0
        mock_run.return_value = process_mock
        yield mock_run


def test_run_dashboard_success(mock_subprocess):
    """Test that run_dashboard executes the correct Streamlit command successfully."""
    # Setup
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    project_root = os.path.dirname(script_dir)
    expected_app_path = os.path.join(
        project_root, "src", "algo_mvp", "dashboard", "app.py"
    )

    # Mock os.path.exists to always return True
    with patch("os.path.exists", return_value=True):
        # Execute
        result = run_dashboard.main()

        # Assert
        assert result == 0
        mock_subprocess.assert_called_once()

        # Get the args used to call subprocess.run
        call_args = mock_subprocess.call_args[0][0]

        # Verify the command structure
        assert call_args[0] == "streamlit"
        assert call_args[1] == "run"
        assert expected_app_path in call_args[2]
        assert call_args[3] == "--server.address"
        assert call_args[4] == "0.0.0.0"


def test_run_dashboard_file_not_found():
    """Test that the CLI fails when the app file is not found."""
    # Mock os.path.exists to return False and capture sys.stderr output
    with (
        patch("os.path.exists", return_value=False),
        patch("sys.stderr.write"),
    ):  # Suppress error output during test
        # Execute the main function
        result = run_dashboard.main()

        # Verify proper error code returned
        assert result == 1


def test_run_dashboard_subprocess_error():
    """Test handling of subprocess errors."""
    # Mock subprocess.run to raise CalledProcessError
    error = subprocess.CalledProcessError(1, "streamlit")
    with (
        patch("os.path.exists", return_value=True),
        patch("cli.run_dashboard.subprocess.run", side_effect=error),
    ):
        # Execute
        result = run_dashboard.main()

        # Assert error handling
        assert result == 1


def test_run_dashboard_streamlit_not_found():
    """Test handling of missing Streamlit executable."""
    # Mock subprocess.run to raise FileNotFoundError
    with (
        patch("os.path.exists", return_value=True),
        patch(
            "cli.run_dashboard.subprocess.run",
            side_effect=FileNotFoundError("No such file: streamlit"),
        ),
    ):
        # Execute
        result = run_dashboard.main()

        # Assert error handling
        assert result == 1
