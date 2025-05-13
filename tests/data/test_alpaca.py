# import pytest # pytest is already imported if pytestmark was used, but ensure it is
# pytestmark = pytest.mark.skip(
#     reason="Temporarily skipping all tests in test_alpaca.py to focus on live trading module and avoid alpaca_trade_api dependency issue."
# ) # REMOVING THIS SKIP

# alpaca_trade_api = pytest.importorskip("alpaca_trade_api") # Option 1: Skip if not importable

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest  # Ensure pytest is imported
from alpaca.common.exceptions import APIError  # This import is from alpaca-py
import datetime  # Import datetime

# Update imports for alpaca-py
from alpaca.data.historical import StockHistoricalDataClient

from algo_mvp.data.alpaca import AlpacaFetcher, is_retryable_alpaca_error
from algo_mvp.models import AlpacaConfig

# Only import these if not skipping the whole file, or handle potential ImportError
# from alpaca_trade_api.rest import REST, APIError, TimeFrame, TimeFrameUnit
# from algo_mvp.data.alpaca import AlpacaFetcher
# from algo_mvp.models import AlpacaConfig


# from algo_mvp.models import AlpacaConfig


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Mocks environment variables for Alpaca keys."""
    monkeypatch.setenv("ALPACA_KEY_ID", "test_key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test_secret")


@pytest.fixture
def mock_stock_historical_client():
    client = MagicMock(spec=StockHistoricalDataClient)

    # Mock the get_stock_bars method
    bars_response = MagicMock()
    bars_response.data = []
    bars_response.df = pd.DataFrame()
    client.get_stock_bars.return_value = bars_response

    # Mock other methods if used in tests
    return client


@pytest.fixture
def alpaca_fetcher(mock_stock_historical_client):
    # Patch the StockHistoricalDataClient instantiation within AlpacaFetcher
    with patch(
        "algo_mvp.data.alpaca.StockHistoricalDataClient",
        return_value=mock_stock_historical_client,
    ) as _mock_client_init:  # Variable needed for context manager, not directly used
        fetcher = AlpacaFetcher(api_key="test_key", secret_key="test_secret")
        fetcher.client = mock_stock_historical_client  # Ensure the mock is used
    return fetcher


@pytest.fixture
def sample_symbol_config():
    return AlpacaConfig(
        symbol="AAPL",
        start="2023-01-01",
        end="2023-01-05",
        timeframe="1Day",
        provider="alpaca",
        adjust=True,
        force=False,
        verbose=False,
    )


@pytest.mark.asyncio
async def test_alpaca_fetch_data_success(
    alpaca_fetcher, mock_stock_historical_client, sample_symbol_config, tmp_path
):
    # Create mock bar data
    mock_bar_data = {
        "AAPL": [
            {
                # Keep timestamp naive initially for parsing
                "timestamp": pd.Timestamp("2023-01-03 09:30:00"),
                "open": 130.0,
                "high": 131.0,
                "low": 129.0,
                "close": 130.5,
                "volume": 100000,
                "trade_count": 1000,
                "vwap": 130.2,
            }
        ]
    }

    # Create a mock response
    mock_response = MagicMock()
    mock_response.data = mock_bar_data

    # Create a DataFrame similar to what the real alpaca-py client would return
    df = pd.DataFrame(mock_bar_data["AAPL"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    # --- Make index timezone-aware (e.g., New York) to test tz_convert ---
    df = df.set_index("timestamp").tz_localize("America/New_York")
    mock_response.df = df

    # Set the mock response for get_stock_bars
    mock_stock_historical_client.get_stock_bars.return_value = mock_response

    # Call fetch directly instead of fetch_data
    result_df = alpaca_fetcher.fetch(
        symbol=sample_symbol_config.symbol,
        timeframe_str=sample_symbol_config.timeframe,
        start_date_str="2023-01-03",  # Adjust start/end to match data
        end_date_str="2023-01-03",
    )

    mock_stock_historical_client.get_stock_bars.assert_called_once()
    assert not result_df.empty
    assert result_df.iloc[0]["open"] == 130.0
    assert (
        result_df.index.tz == datetime.timezone.utc
    )  # Verify it was converted to UTC (covers line 131)
    # We don't check if the file was saved since we're calling fetch directly


@pytest.mark.asyncio
async def test_alpaca_fetch_data_api_error(
    alpaca_fetcher, mock_stock_historical_client, sample_symbol_config, tmp_path, capfd
):
    # --- Mock an API error using a real subclass with a mocked response property ---
    class MockAPIErrorWithResponse(APIError):
        def __init__(self, msg="API Error", response_text="Default Response Text"):
            super().__init__(msg)
            # Store the text we want the response property to return
            self._text = response_text

        @property
        def response(self):
            # Return a mock object simulating the response attribute
            resp_mock = MagicMock()
            resp_mock.text = self._text
            return resp_mock

    # Instantiate our custom error and set it as the side_effect
    error_instance = MockAPIErrorWithResponse(
        msg="API Error Occurred", response_text="Detailed error message from API"
    )
    mock_stock_historical_client.get_stock_bars.side_effect = error_instance

    # Call fetch directly
    result_df = alpaca_fetcher.fetch(
        symbol=sample_symbol_config.symbol,
        timeframe_str=sample_symbol_config.timeframe,
        start_date_str=sample_symbol_config.start,
        end_date_str=sample_symbol_config.end,
    )

    assert result_df is None  # fetch returns None on error
    # --- Check captured stderr using capfd ---
    captured = capfd.readouterr()
    # Check for the actual error message string and the response.text
    # Note: The __str__ of the base APIError might just be the message.
    assert "API Error Occurred" in captured.err
    assert "Detailed error message from API" in captured.err  # Verify line 144 was hit


@pytest.mark.asyncio
async def test_alpaca_fetch_data_connection_error(
    alpaca_fetcher, mock_stock_historical_client, sample_symbol_config, capfd
):
    """Tests fetch when a ConnectionError occurs."""
    # Mock a ConnectionError
    mock_stock_historical_client.get_stock_bars.side_effect = ConnectionError(
        "Network down"
    )

    # Call fetch directly
    result_df = alpaca_fetcher.fetch(
        symbol=sample_symbol_config.symbol,
        timeframe_str=sample_symbol_config.timeframe,
        start_date_str=sample_symbol_config.start,
        end_date_str=sample_symbol_config.end,
    )

    assert result_df is None
    # --- Check captured stderr using capfd ---
    captured = capfd.readouterr()
    assert "Alpaca Connection Error" in captured.err
    assert "Network down" in captured.err


@pytest.mark.asyncio
async def test_alpaca_fetch_data_no_data(
    alpaca_fetcher, mock_stock_historical_client, sample_symbol_config, tmp_path, caplog
):
    # Mock empty data response
    mock_response = MagicMock()
    mock_response.data = {}  # Empty data
    mock_response.df = pd.DataFrame()  # Empty DataFrame
    mock_stock_historical_client.get_stock_bars.return_value = mock_response

    # Call fetch directly
    result_df = alpaca_fetcher.fetch(
        symbol=sample_symbol_config.symbol,
        timeframe_str=sample_symbol_config.timeframe,
        start_date_str=sample_symbol_config.start,
        end_date_str=sample_symbol_config.end,
    )

    assert result_df.empty  # Should return an empty DataFrame


def test_alpaca_fetch_invalid_timeframe(alpaca_fetcher, sample_symbol_config, capfd):
    """Tests fetch with an invalid timeframe string."""
    result_df = alpaca_fetcher.fetch(
        symbol=sample_symbol_config.symbol,
        timeframe_str="1Year",  # Invalid timeframe
        start_date_str=sample_symbol_config.start,
        end_date_str=sample_symbol_config.end,
    )

    assert result_df is None
    captured = capfd.readouterr()
    assert "Error parsing timeframe '1Year'" in captured.err


@patch("algo_mvp.data.alpaca.StockHistoricalDataClient")
def test_alpaca_fetch_verbose(mock_client, mock_env_vars, capsys):
    """Tests verbose output during fetch operation."""
    # Create a verbose fetcher instance directly
    verbose_fetcher = AlpacaFetcher(verbose=True)

    # Set up mock client response (similar to test_alpaca_fetch_data_success)
    mock_response = MagicMock()
    mock_df = pd.DataFrame(
        {
            "open": [100],
            "high": [101],
            "low": [99],
            "close": [100.5],
            "volume": [1000],
        },
        index=pd.to_datetime(["2023-01-01 09:30:00+00:00"], utc=True),
    )
    mock_response.df = mock_df
    mock_response.data = {"AAPL": [{}]}  # Need some data for it not to exit early
    mock_client.return_value.get_stock_bars.return_value = mock_response
    verbose_fetcher.client = mock_client.return_value  # Assign mock client

    # Call fetch
    verbose_fetcher.fetch("AAPL", "1Min", "2023-01-01", "2023-01-01")

    # Check captured output
    captured = capsys.readouterr()
    assert "Alpaca: Fetching AAPL (1Min)" in captured.out  # Line 99-101
    assert "Alpaca API request params:" in captured.out  # Line 111
    assert "Alpaca: Fetched 1 bars for AAPL." in captured.out  # Line 153


# --- Tests for helper functions ---


def test_is_retryable_alpaca_error():
    """Tests the is_retryable_alpaca_error helper function."""
    # Retryable status codes
    retryable_error_429 = MagicMock(spec=APIError)
    retryable_error_429.status_code = 429
    retryable_error_429.__str__ = lambda self: "Rate limit"  # For completeness
    assert is_retryable_alpaca_error(retryable_error_429) is True

    retryable_error_500 = MagicMock(spec=APIError)
    retryable_error_500.status_code = 500
    retryable_error_500.__str__ = lambda self: "Server error"
    assert is_retryable_alpaca_error(retryable_error_500) is True

    retryable_error_503 = MagicMock(spec=APIError)
    retryable_error_503.status_code = 503
    retryable_error_503.__str__ = lambda self: "Server error"
    assert is_retryable_alpaca_error(retryable_error_503) is True

    # Non-retryable status code
    non_retryable_error_400 = MagicMock(spec=APIError)
    non_retryable_error_400.status_code = 400
    non_retryable_error_400.__str__ = lambda self: "Bad request"
    assert is_retryable_alpaca_error(non_retryable_error_400) is False

    non_retryable_error_401 = MagicMock(spec=APIError)
    non_retryable_error_401.status_code = 401
    non_retryable_error_401.__str__ = lambda self: "Unauthorized"
    assert is_retryable_alpaca_error(non_retryable_error_401) is False

    # Retryable based on message content (when status_code might not be present)
    retryable_msg_rate_limit = MagicMock(spec=APIError)
    del retryable_msg_rate_limit.status_code  # Ensure status_code is not present
    retryable_msg_rate_limit.__str__ = (
        lambda self: "Something went wrong: rate limit exceeded"
    )
    assert is_retryable_alpaca_error(retryable_msg_rate_limit) is True

    retryable_msg_server_err = MagicMock(spec=APIError)
    del retryable_msg_server_err.status_code
    retryable_msg_server_err.__str__ = lambda self: "Internal server error occurred"
    assert is_retryable_alpaca_error(retryable_msg_server_err) is True

    # Non-retryable message content
    non_retryable_msg = MagicMock(spec=APIError)
    del non_retryable_msg.status_code
    non_retryable_msg.__str__ = lambda self: "Invalid symbol"
    assert is_retryable_alpaca_error(non_retryable_msg) is False

    # Non-APIError exceptions
    assert is_retryable_alpaca_error(ValueError("Some other error")) is False
    assert (
        is_retryable_alpaca_error(ConnectionError("Connection failed")) is False
    )  # is_retryable only checks APIError


# --- Tests for AlpacaFetcher class ---


def test_alpaca_fetcher_init_verbose(capsys, mock_env_vars):
    """Tests AlpacaFetcher initialization with verbose=True."""
    fetcher = AlpacaFetcher(verbose=True)
    assert fetcher.api_key == "test_key"
    assert fetcher.secret_key == "test_secret"
    assert fetcher.verbose is True
    captured = capsys.readouterr()
    assert "AlpacaFetcher initialized" in captured.out
