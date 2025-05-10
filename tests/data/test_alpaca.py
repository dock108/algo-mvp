# import pytest # pytest is already imported if pytestmark was used, but ensure it is
# pytestmark = pytest.mark.skip(
#     reason="Temporarily skipping all tests in test_alpaca.py to focus on live trading module and avoid alpaca_trade_api dependency issue."
# ) # REMOVING THIS SKIP

# alpaca_trade_api = pytest.importorskip("alpaca_trade_api") # Option 1: Skip if not importable

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest  # Ensure pytest is imported
from alpaca.common.exceptions import APIError  # This import is from alpaca-py

# Update imports for alpaca-py
from alpaca.data.historical import StockHistoricalDataClient

from algo_mvp.data.alpaca import AlpacaFetcher
from algo_mvp.models import AlpacaConfig

# Only import these if not skipping the whole file, or handle potential ImportError
# from alpaca_trade_api.rest import REST, APIError, TimeFrame, TimeFrameUnit
# from algo_mvp.data.alpaca import AlpacaFetcher
# from algo_mvp.models import AlpacaConfig


# from algo_mvp.models import AlpacaConfig


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
                "timestamp": pd.Timestamp("2023-01-03 00:00:00", tz="UTC"),
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
    df.set_index("timestamp", inplace=True)
    mock_response.df = df

    # Set the mock response for get_stock_bars
    mock_stock_historical_client.get_stock_bars.return_value = mock_response

    # Call fetch directly instead of fetch_data
    result_df = alpaca_fetcher.fetch(
        symbol=sample_symbol_config.symbol,
        timeframe_str=sample_symbol_config.timeframe,
        start_date_str=sample_symbol_config.start,
        end_date_str=sample_symbol_config.end,
    )

    mock_stock_historical_client.get_stock_bars.assert_called_once()
    assert not result_df.empty
    assert result_df.iloc[0]["open"] == 130.0
    # We don't check if the file was saved since we're calling fetch directly


@pytest.mark.asyncio
async def test_alpaca_fetch_data_api_error(
    alpaca_fetcher, mock_stock_historical_client, sample_symbol_config, tmp_path, caplog
):
    # Mock an API error from alpaca-py - remove message parameter
    mock_stock_historical_client.get_stock_bars.side_effect = APIError("Test API Error")

    # Call fetch directly
    result_df = alpaca_fetcher.fetch(
        symbol=sample_symbol_config.symbol,
        timeframe_str=sample_symbol_config.timeframe,
        start_date_str=sample_symbol_config.start,
        end_date_str=sample_symbol_config.end,
    )

    assert result_df is None  # fetch returns None on error


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


# Add more tests for edge cases, different timeframes, adjustments, force flag, etc.
