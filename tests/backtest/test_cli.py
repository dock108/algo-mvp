"""Tests for the backtest CLI."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from cli.run_backtest import main


@pytest.fixture
def sample_config_file(tmp_path):
    """Create a sample configuration file."""
    config_content = """
provider: alpaca
symbol: AAPL
timeframe: 1Min
strategy: vwap_atr
params:
  band_mult: [1.5, 2.0]
  atr_len: [10, 14]
cash: 10000
commission: 0.0
metrics: [sharpe, max_drawdown]
"""
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(config_content)
    return config_file


@patch("cli.run_backtest.BacktestEngine")
def test_cli_success(mock_engine, sample_config_file, monkeypatch):
    """Test that the CLI runs successfully."""
    # Mock the BacktestEngine
    mock_engine_instance = MagicMock()
    mock_engine_instance.run.return_value = True
    mock_engine.return_value = mock_engine_instance

    # Mock sys.argv
    monkeypatch.setattr(
        sys, "argv", ["run_backtest.py", "--config", str(sample_config_file)]
    )

    # Mock sys.exit to avoid exiting the test
    with patch("sys.exit") as mock_exit:
        main()
        mock_exit.assert_called_once_with(0)

    # Check that the engine was initialized with the correct parameters
    mock_engine.assert_called_once()
    assert mock_engine.call_args[1]["config_file"] == str(sample_config_file)
    assert mock_engine.call_args[1]["verbose"] is False
    assert "output_dir" not in mock_engine.call_args[1]

    # Check that the run method was called
    mock_engine_instance.run.assert_called_once()


@patch("cli.run_backtest.BacktestEngine")
def test_cli_failure(mock_engine, sample_config_file, monkeypatch):
    """Test that the CLI handles engine failure correctly."""
    # Mock the BacktestEngine
    mock_engine_instance = MagicMock()
    mock_engine_instance.run.return_value = False
    mock_engine.return_value = mock_engine_instance

    # Mock sys.argv
    monkeypatch.setattr(
        sys, "argv", ["run_backtest.py", "--config", str(sample_config_file)]
    )

    # Mock sys.exit to avoid exiting the test
    with patch("sys.exit") as mock_exit:
        main()
        mock_exit.assert_called_once_with(1)


@patch("cli.run_backtest.BacktestEngine")
def test_cli_with_output_dir(mock_engine, sample_config_file, tmp_path, monkeypatch):
    """Test that the CLI handles custom output directory correctly."""
    # Mock the BacktestEngine
    mock_engine_instance = MagicMock()
    mock_engine_instance.run.return_value = True
    mock_engine.return_value = mock_engine_instance

    # Create output directory
    output_dir = tmp_path / "custom_output"

    # Mock sys.argv
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_backtest.py",
            "--config",
            str(sample_config_file),
            "--output-dir",
            str(output_dir),
        ],
    )

    # Mock sys.exit to avoid exiting the test
    with patch("sys.exit") as mock_exit:
        main()
        mock_exit.assert_called_once_with(0)

    # Check that the engine was initialized with the correct parameters
    mock_engine.assert_called_once()
    assert mock_engine.call_args[1]["config_file"] == str(sample_config_file)
    assert mock_engine.call_args[1]["output_dir"] == str(output_dir)


def test_cli_config_not_found(monkeypatch):
    """Test that the CLI handles missing config file correctly."""
    # Mock sys.argv with a non-existent config file
    monkeypatch.setattr(
        sys, "argv", ["run_backtest.py", "--config", "nonexistent.yaml"]
    )

    # Mock sys.exit to avoid exiting the test
    with patch("sys.exit") as mock_exit:
        main()
        mock_exit.assert_called_once_with(1)


@patch("cli.run_backtest.BacktestEngine")
def test_cli_verbose(mock_engine, sample_config_file, monkeypatch):
    """Test that the CLI handles verbose flag correctly."""
    # Mock the BacktestEngine
    mock_engine_instance = MagicMock()
    mock_engine_instance.run.return_value = True
    mock_engine.return_value = mock_engine_instance

    # Mock sys.argv
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_backtest.py", "--config", str(sample_config_file), "--verbose"],
    )

    # Mock sys.exit to avoid exiting the test
    with patch("sys.exit") as mock_exit:
        main()
        mock_exit.assert_called_once_with(0)

    # Check that the engine was initialized with verbose=True
    mock_engine.assert_called_once()
    assert mock_engine.call_args[1]["verbose"] is True
