import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from alpaca_trade_api.rest import REST

from algo_mvp.data.alpaca import AlpacaFetcher
from algo_mvp.models import AlpacaConfig


@pytest.fixture
def alpaca_config_fixture():
    return AlpacaConfig(
        provider="alpaca",
        symbol="AAPL",
        timeframe="1Day",
        start="2023-01-01",
        end="2023-01-10",
        adjust=True,
    )


@pytest.fixture
def mock_alpaca_client_rest():
    with patch("algo_mvp.data.alpaca.REST") as mock_rest_class:
        mock_client_instance = MagicMock(spec=REST)
        mock_rest_class.return_value = mock_client_instance
        yield mock_client_instance


@pytest.fixture
def alpaca_fetcher_instance(alpaca_config_fixture, mock_alpaca_client_rest):
    with patch(
        "os.getenv",
        side_effect=lambda k: (
            "fake_key" if k in ["ALPACA_KEY_ID", "ALPACA_SECRET_KEY"] else None
        ),
    ):
        fetcher = AlpacaFetcher(
            api_key="dummy_key",
            secret_key="dummy_secret",
            adjust=alpaca_config_fixture.adjust,
            verbose=True,
        )
        fetcher.client = mock_alpaca_client_rest
        return fetcher


@pytest.fixture
def sample_alpaca_bars():
    """Mock the return from alpaca API get_bars().
    The alpaca-trade-api library returns a DataFrame with standard column names.
    """
    data = {
        "timestamp": [  # Index needs a name for consistency, though Alpaca might not set it
            datetime.datetime(2023, 1, 1, 0, 0, tzinfo=datetime.timezone.utc),
            datetime.datetime(2023, 1, 2, 0, 0, tzinfo=datetime.timezone.utc),
        ],
        "open": [100.0, 101.0],
        "high": [102.0, 103.0],
        "low": [99.0, 100.0],
        "close": [101.0, 102.0],
        "volume": [10000, 12000],
    }
    df = pd.DataFrame(data)
    df.set_index("timestamp", inplace=True)  # Set the index
    mock_bars_container = (
        MagicMock()
    )  # Alpaca returns a container object with a .df attribute
    mock_bars_container.df = df
    return mock_bars_container


def test_alpaca_fetcher_init(
    alpaca_fetcher_instance, alpaca_config_fixture, mock_alpaca_client_rest
):
    """Test initialization of AlpacaFetcher with proper configuration."""
    fetcher = alpaca_fetcher_instance
    assert fetcher.adjust == alpaca_config_fixture.adjust
    assert fetcher.verbose is True
    assert fetcher.client is mock_alpaca_client_rest


def test_alpaca_fetcher_fetch(
    alpaca_fetcher_instance, alpaca_config_fixture, sample_alpaca_bars
):
    """Test the fetch method of AlpacaFetcher."""
    fetcher = alpaca_fetcher_instance
    mock_client_get_bars = fetcher.client.get_bars
    mock_client_get_bars.return_value = sample_alpaca_bars

    result = fetcher.fetch(
        symbol=alpaca_config_fixture.symbol,
        timeframe_str=alpaca_config_fixture.timeframe,
        start_date_str=alpaca_config_fixture.start,
        end_date_str=alpaca_config_fixture.end,
    )

    mock_client_get_bars.assert_called_once()
    _, called_kwargs = mock_client_get_bars.call_args
    assert called_kwargs["symbol_or_symbols"] == alpaca_config_fixture.symbol
    assert called_kwargs["adjustment"] == "split"

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2
    assert list(result.columns) == ["open", "high", "low", "close", "volume"]
    assert result.index.name == "timestamp"


def test_alpaca_fetcher_fetch_no_adjust(
    alpaca_fetcher_instance, alpaca_config_fixture, sample_alpaca_bars
):
    """Test the fetch method with adjust=False."""
    fetcher = alpaca_fetcher_instance
    fetcher.adjust = False
    mock_client_get_bars = fetcher.client.get_bars
    mock_client_get_bars.return_value = sample_alpaca_bars

    fetcher.fetch(
        symbol=alpaca_config_fixture.symbol,
        timeframe_str=alpaca_config_fixture.timeframe,
        start_date_str=alpaca_config_fixture.start,
        end_date_str=alpaca_config_fixture.end,
    )

    _, called_kwargs = mock_client_get_bars.call_args
    assert called_kwargs["adjustment"] == "raw"


def test_alpaca_fetcher_fetch_api_error(alpaca_fetcher_instance, alpaca_config_fixture):
    """Test error handling during fetching."""
    fetcher = alpaca_fetcher_instance
    mock_client_get_bars = fetcher.client.get_bars
    mock_client_get_bars.side_effect = Exception("API Error")

    result = fetcher.fetch(
        symbol=alpaca_config_fixture.symbol,
        timeframe_str=alpaca_config_fixture.timeframe,
        start_date_str=alpaca_config_fixture.start,
        end_date_str=alpaca_config_fixture.end,
    )
    assert result is None


def test_alpaca_fetcher_fetch_empty_result(
    alpaca_fetcher_instance, alpaca_config_fixture
):
    """Test handling of empty results from the API."""
    fetcher = alpaca_fetcher_instance
    empty_bars_container = MagicMock()
    empty_bars_container.df = pd.DataFrame()
    mock_client_get_bars = fetcher.client.get_bars
    mock_client_get_bars.return_value = empty_bars_container

    result = fetcher.fetch(
        symbol=alpaca_config_fixture.symbol,
        timeframe_str=alpaca_config_fixture.timeframe,
        start_date_str=alpaca_config_fixture.start,
        end_date_str=alpaca_config_fixture.end,
    )
    assert result is not None
    assert result.empty


def test_alpaca_fetcher_transform_df_not_needed(
    alpaca_fetcher_instance, sample_alpaca_bars
):
    pass
