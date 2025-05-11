"""Tests for the dashboard module entry point."""

from unittest.mock import patch, MagicMock
import pytest
import subprocess

from algo_mvp.dashboard import __main__


@pytest.fixture
def mock_subprocess():
    """Mock the subprocess module for testing."""
    with patch("algo_mvp.dashboard.__main__.subprocess.run") as mock_run:
        # Configure the mock to return a simple object with returncode 0
        process_mock = MagicMock()
        process_mock.returncode = 0
        mock_run.return_value = process_mock
        yield mock_run


def test_main_success(mock_subprocess):
    """Test that the main function executes the correct Streamlit command successfully."""
    # Mock os.path.exists to always return True
    with patch("os.path.exists", return_value=True):
        # Execute
        result = __main__.main()

        # Assert
        assert result == 0
        mock_subprocess.assert_called_once()

        # Get the args used to call subprocess.run
        call_args = mock_subprocess.call_args[0][0]

        # Verify the command structure
        assert call_args[0] == "streamlit"
        assert call_args[1] == "run"
        assert "app.py" in call_args[2]
        assert call_args[3] == "--server.address"
        assert call_args[4] == "0.0.0.0"


def test_main_file_not_found():
    """Test that the main function fails when the app file is not found."""
    # Mock os.path.exists to return False and capture sys.stderr output
    with (
        patch("os.path.exists", return_value=False),
        patch("sys.stderr.write"),
    ):  # Suppress error output during test
        # Execute the main function
        result = __main__.main()

        # Verify proper error code returned
        assert result == 1


def test_main_subprocess_error():
    """Test handling of subprocess errors."""
    # Mock subprocess.run to raise CalledProcessError
    error = subprocess.CalledProcessError(1, "streamlit")
    with (
        patch("os.path.exists", return_value=True),
        patch("algo_mvp.dashboard.__main__.subprocess.run", side_effect=error),
    ):
        # Execute
        result = __main__.main()

        # Assert error handling
        assert result == 1


def test_main_streamlit_not_found():
    """Test handling of missing Streamlit executable."""
    # Mock subprocess.run to raise FileNotFoundError
    with (
        patch("os.path.exists", return_value=True),
        patch(
            "algo_mvp.dashboard.__main__.subprocess.run",
            side_effect=FileNotFoundError("No such file: streamlit"),
        ),
    ):
        # Execute
        result = __main__.main()

        # Assert error handling
        assert result == 1
