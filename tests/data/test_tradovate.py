# import pytest # pytest is already imported if pytestmark was used

# pytestmark = pytest.mark.skip(
#     reason="Temporarily skipping all tests in test_tradovate.py to focus on live trading module and avoid alpaca_trade_api dependency issue."
# ) # REMOVING THIS SKIP

# import asyncio # Removed unused import
from datetime import timezone
from unittest.mock import patch

import pandas as pd
import pytest  # Ensure pytest is imported
import requests
import responses

from algo_mvp.data.tradovate import TRADOVATE_DEMO_API_URL, TradovateFetcher
from algo_mvp.models import TradovateConfig


@pytest.fixture
def tradovate_config_fixture():
    return TradovateConfig(
        provider="tradovate",
        symbol="MESM25",
        timeframe="1Min",
        start="2024-03-01T10:00:00Z",
        end="2024-03-01T10:05:00Z",
    )


@pytest.fixture
def sample_tradovate_tick_response():
    # Sample response for tick data from Tradovate /history endpoint (hypothetical)
    return {
        "ticks": [
            {"t": "2024-03-01T10:00:00.123Z", "p": 5123.25, "s": 2},
            {"t": "2024-03-01T10:00:30.456Z", "p": 5123.50, "s": 1},
            {"t": "2024-03-01T10:01:00.789Z", "p": 5124.00, "s": 3},
        ]
    }


@pytest.fixture
def tradovate_fetcher_instance(tradovate_config_fixture):
    with patch("os.getenv") as mock_getenv:
        mock_getenv.side_effect = lambda k: {
            "TRADOVATE_CLIENT_ID": "dummy_client_id",
            "TRADOVATE_ACCESS_TOKEN": "dummy_access_token",
        }.get(k)

        fetcher = TradovateFetcher(
            client_id="test_client_id", access_token="test_access_token", verbose=True
        )
        return fetcher


@responses.activate
def test_tradovate_fetcher_init(tradovate_fetcher_instance, tradovate_config_fixture):
    """Test initialization of TradovateFetcher."""
    fetcher = tradovate_fetcher_instance
    assert fetcher.client_id == "test_client_id"
    assert fetcher.access_token == "test_access_token"
    assert fetcher.verbose is True


@responses.activate
def test_tradovate_fetch_successful(
    tradovate_fetcher_instance, tradovate_config_fixture, sample_tradovate_tick_response
):
    """Test the main fetch method that returns a DataFrame from resampled ticks."""
    fetcher = tradovate_fetcher_instance

    responses.add(
        responses.GET,
        f"{TRADOVATE_DEMO_API_URL}/history",
        json=sample_tradovate_tick_response,
        status=200,
    )

    result_df = fetcher.fetch(
        symbol=tradovate_config_fixture.symbol,
        timeframe_str=tradovate_config_fixture.timeframe,
        start_date_str=tradovate_config_fixture.start,
        end_date_str=tradovate_config_fixture.end,
    )

    assert isinstance(result_df, pd.DataFrame)
    assert not result_df.empty
    assert list(result_df.columns) == ["open", "high", "low", "close", "volume"]
    assert result_df.index.name == "timestamp"
    assert result_df.index.tz == timezone.utc

    assert len(responses.calls) == 1
    request_url = responses.calls[0].request.url
    assert f"symbol={tradovate_config_fixture.symbol}" in request_url
    assert "type=Tick" in request_url


@responses.activate
def test_tradovate_fetch_api_http_error(
    tradovate_fetcher_instance, tradovate_config_fixture
):
    """Test fetch method with API HTTP error."""
    fetcher = tradovate_fetcher_instance

    # Instead of using responses to mock the HTTP call (which still triggers retries),
    # directly patch the _make_request method to raise an HTTPError
    with patch.object(fetcher, "_make_request") as mock_make_request:
        mock_make_request.side_effect = requests.exceptions.HTTPError(
            "500 Server Error"
        )

        result = fetcher.fetch(
            symbol=tradovate_config_fixture.symbol,
            timeframe_str=tradovate_config_fixture.timeframe,
            start_date_str=tradovate_config_fixture.start,
            end_date_str=tradovate_config_fixture.end,
        )

        # Verify _make_request was called with expected parameters
        mock_make_request.assert_called_once()

        assert result is None


@responses.activate
def test_tradovate_fetch_empty_ticks(
    tradovate_fetcher_instance, tradovate_config_fixture
):
    """Test fetch method when API returns no ticks."""
    fetcher = tradovate_fetcher_instance
    responses.add(
        responses.GET,
        f"{TRADOVATE_DEMO_API_URL}/history",
        json={"ticks": []},
        status=200,
    )
    result = fetcher.fetch(
        symbol=tradovate_config_fixture.symbol,
        timeframe_str=tradovate_config_fixture.timeframe,
        start_date_str=tradovate_config_fixture.start,
        end_date_str=tradovate_config_fixture.end,
    )
    assert result is not None
    assert result.empty


def test_tradovate_resample_ticks_to_ohlcv(tradovate_fetcher_instance):
    """Test the internal _resample_ticks_to_ohlcv method."""
    fetcher = tradovate_fetcher_instance
    tick_data = [
        {"timestamp": "2024-03-01T10:00:05Z", "price": 100, "size": 10},
        {"timestamp": "2024-03-01T10:00:15Z", "price": 101, "size": 5},
        {"timestamp": "2024-03-01T10:00:55Z", "price": 100.5, "size": 8},
        {"timestamp": "2024-03-01T10:01:05Z", "price": 102, "size": 12},
        {"timestamp": "2024-03-01T10:01:35Z", "price": 101.5, "size": 6},
    ]
    ticks_df = pd.DataFrame(tick_data)
    ticks_df["timestamp"] = pd.to_datetime(ticks_df["timestamp"])
    ticks_df = ticks_df.set_index("timestamp")

    ohlcv_df = fetcher._resample_ticks_to_ohlcv(ticks_df, "1Min")

    assert len(ohlcv_df) == 2
    assert ohlcv_df.iloc[0]["open"] == 100
    assert ohlcv_df.iloc[0]["high"] == 101
    assert ohlcv_df.iloc[0]["low"] == 100
    assert ohlcv_df.iloc[0]["close"] == 100.5
    assert ohlcv_df.iloc[0]["volume"] == 23

    assert ohlcv_df.iloc[1]["open"] == 102
    assert ohlcv_df.iloc[1]["high"] == 102
    assert ohlcv_df.iloc[1]["low"] == 101.5
    assert ohlcv_df.iloc[1]["close"] == 101.5
    assert ohlcv_df.iloc[1]["volume"] == 18
    assert ohlcv_df.index.name == "timestamp"
    assert ohlcv_df.index.tz == timezone.utc


def test_tradovate_fetcher_init_missing_token(tradovate_config_fixture):
    """Test TradovateFetcher raises ValueError if access token is missing."""
    with patch.dict("os.environ", {"TRADOVATE_CLIENT_ID": "dummy_id"}, clear=True):
        with pytest.raises(ValueError, match="Tradovate access token must be provided"):
            TradovateFetcher(client_id="dummy_id")


def test_tradovate_fetcher_init_missing_client_id(tradovate_config_fixture, capsys):
    """Test TradovateFetcher prints warning if client ID is missing but token is present."""
    with patch.dict(
        "os.environ", {"TRADOVATE_ACCESS_TOKEN": "dummy_token"}, clear=True
    ):
        TradovateFetcher(access_token="dummy_token")
        captured = capsys.readouterr()
        assert "Warning: TRADOVATE_CLIENT_ID is not set" in captured.out


@responses.activate
def test_tradovate_fetch_connection_error(
    tradovate_fetcher_instance, tradovate_config_fixture
):
    """Test fetch handling ConnectionError during API request."""
    fetcher = tradovate_fetcher_instance

    # Use patch instead of responses to avoid retry behavior
    with patch.object(fetcher, "_make_request") as mock_make_request:
        mock_make_request.side_effect = requests.exceptions.ConnectionError(
            "Test connection error"
        )

        result = fetcher.fetch(
            symbol=tradovate_config_fixture.symbol,
            timeframe_str=tradovate_config_fixture.timeframe,
            start_date_str=tradovate_config_fixture.start,
            end_date_str=tradovate_config_fixture.end,
        )

        # Verify _make_request was called with expected parameters
        mock_make_request.assert_called_once()

        assert result is None


def test_tradovate_fetch_invalid_date_format(
    tradovate_fetcher_instance, tradovate_config_fixture
):
    """Test fetch handling errors during start/end date parsing."""
    fetcher = tradovate_fetcher_instance
    result = fetcher.fetch(
        symbol=tradovate_config_fixture.symbol,
        timeframe_str=tradovate_config_fixture.timeframe,
        start_date_str="invalid-date-format",
        end_date_str=tradovate_config_fixture.end,
    )
    assert result is None


def test_tradovate_resample_missing_columns(tradovate_fetcher_instance):
    """Test _resample_ticks_to_ohlcv raises ValueError if columns are missing."""
    fetcher = tradovate_fetcher_instance
    # Create DataFrame missing the 'price' column
    tick_data = [{"timestamp": "2024-03-01T10:00:05Z", "size": 10}]
    ticks_df = pd.DataFrame(tick_data)
    ticks_df["timestamp"] = pd.to_datetime(ticks_df["timestamp"])
    ticks_df = ticks_df.set_index("timestamp")

    with pytest.raises(ValueError, match="must contain 'price' and 'size' columns"):
        fetcher._resample_ticks_to_ohlcv(ticks_df, "1Min")
