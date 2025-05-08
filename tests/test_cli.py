import argparse
from unittest.mock import MagicMock, patch

import pytest
import yaml  # Import yaml for the dump

from algo_mvp.fetch import main
from algo_mvp.models import (  # Added TradovateConfig for completeness
    AlpacaConfig,
    TradovateConfig,
)


@pytest.fixture
def valid_alpaca_config_path(tmp_path):
    config_file = tmp_path / "valid_alpaca_config.yaml"
    config_data = {
        "provider": "alpaca",
        "symbol": "AAPL",
        "timeframe": "1Day",
        "start": "2023-01-01",  # Ensure this is a string
        "end": "2023-01-10",  # Ensure this is a string
        "adjust": True,
    }
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)
    return str(config_file)


@pytest.fixture
def valid_tradovate_config_path(tmp_path):
    config_file = tmp_path / "valid_tradovate_config.yaml"
    config_data = {
        "provider": "tradovate",
        "symbol": "MESM25",
        "timeframe": "1Min",
        "start": "2024-03-01T10:00:00Z",  # Ensure this is a string
        "end": "2024-03-01T10:05:00Z",  # Ensure this is a string
    }
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)
    return str(config_file)


@pytest.fixture
def invalid_config_path(tmp_path):
    config_file = tmp_path / "invalid_config.yaml"
    config_data = {
        "provider": "unknown",  # Invalid provider
        "symbol": "AAPL",
        "timeframe": "1Day",
        "start": "2023-01-01",
        "end": "2023-01-10",
    }
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)
    return str(config_file)


@patch("argparse.ArgumentParser.parse_args")
@patch(
    "algo_mvp.data.fetcher.DataFetcher"
)  # Patch where DataFetcher is imported in cli/fetch_data.py
def test_main_with_valid_alpaca_config(
    mock_data_fetcher_class, mock_parse_args, valid_alpaca_config_path
):
    """Test the main function with a valid Alpaca configuration file."""
    mock_args = argparse.Namespace(
        config=valid_alpaca_config_path, force=False, verbose=False
    )
    mock_parse_args.return_value = mock_args

    mock_fetcher_instance = MagicMock()
    mock_fetcher_instance.fetch_data.return_value = True
    mock_data_fetcher_class.return_value = mock_fetcher_instance

    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 0  # Expect exit code 0 on success

    mock_data_fetcher_class.assert_called_once()
    call_args_list = mock_data_fetcher_class.call_args_list
    assert len(call_args_list) == 1
    actual_pos_args, actual_kwargs = call_args_list[0]
    assert not actual_pos_args  # Ensure no positional args were used

    assert "config" in actual_kwargs
    assert isinstance(actual_kwargs["config"], AlpacaConfig)
    assert actual_kwargs["config"].symbol == "AAPL"
    assert "verbose" in actual_kwargs
    assert actual_kwargs["verbose"] is False
    mock_fetcher_instance.fetch_data.assert_called_once_with(force=False)


@patch("argparse.ArgumentParser.parse_args")
@patch(
    "algo_mvp.data.fetcher.DataFetcher"
)  # Patch where DataFetcher is imported in cli/fetch_data.py
def test_main_with_valid_tradovate_config(
    mock_data_fetcher_class, mock_parse_args, valid_tradovate_config_path
):
    """Test the main function with a valid Tradovate configuration file."""
    mock_args = argparse.Namespace(
        config=valid_tradovate_config_path, force=True, verbose=True
    )
    mock_parse_args.return_value = mock_args

    mock_fetcher_instance = MagicMock()
    mock_fetcher_instance.fetch_data.return_value = True
    mock_data_fetcher_class.return_value = mock_fetcher_instance

    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 0  # Expect exit code 0 on success

    mock_data_fetcher_class.assert_called_once()
    actual_pos_args, actual_kwargs = mock_data_fetcher_class.call_args
    assert not actual_pos_args  # Ensure no positional args were used

    assert "config" in actual_kwargs
    assert isinstance(actual_kwargs["config"], TradovateConfig)
    assert actual_kwargs["config"].symbol == "MESM25"
    assert "verbose" in actual_kwargs
    assert actual_kwargs["verbose"] is True
    mock_fetcher_instance.fetch_data.assert_called_once_with(force=True)


@patch("argparse.ArgumentParser.parse_args")
# No need to patch DataFetcher here as it won't be reached if config is invalid
def test_main_with_invalid_provider_config(mock_parse_args, invalid_config_path):
    """Test main with a config having an unsupported provider."""
    mock_args = argparse.Namespace(
        config=invalid_config_path, force=False, verbose=False
    )
    mock_parse_args.return_value = mock_args

    # Expect SystemExit because main() calls sys.exit(1) for bad provider
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 1


@patch("argparse.ArgumentParser.parse_args")
@patch(
    "algo_mvp.data.fetcher.DataFetcher"
)  # Patch where DataFetcher is imported in cli/fetch_data.py
def test_main_fetcher_fails(
    mock_data_fetcher_class, mock_parse_args, valid_alpaca_config_path
):
    """Test the main function when the fetcher's fetch_data fails."""
    mock_args = argparse.Namespace(
        config=valid_alpaca_config_path, force=False, verbose=False
    )
    mock_parse_args.return_value = mock_args

    mock_fetcher_instance = MagicMock()
    mock_fetcher_instance.fetch_data.return_value = False  # Simulate fetch failure
    mock_data_fetcher_class.return_value = mock_fetcher_instance

    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 1
    mock_fetcher_instance.fetch_data.assert_called_once_with(force=False)


@patch("argparse.ArgumentParser.parse_args")
@patch("yaml.safe_load")  # Patch where yaml.safe_load is imported in cli/fetch_data.py
def test_main_config_validation_error(
    mock_safe_load, mock_parse_args, valid_alpaca_config_path
):
    """Test main handling Pydantic ValidationError during config parsing."""
    mock_args = argparse.Namespace(
        config=valid_alpaca_config_path, force=False, verbose=False
    )
    mock_parse_args.return_value = mock_args

    # Make safe_load return data that will cause a Pydantic error (e.g., missing required field)
    mock_safe_load.return_value = {
        "provider": "alpaca",
        "symbol": "AAPL",
    }  # Missing timeframe, start, end

    with pytest.raises(
        SystemExit
    ) as exc_info:  # main() should catch ValidationError and sys.exit(1)
        main()
    assert exc_info.value.code == 1
    mock_safe_load.assert_called_once()
