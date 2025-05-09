"""Tests for strategy implementations."""

import pandas as pd
import pytest

from algo_mvp.backtest.strategies.three_day_momo import ThreeDayMomoStrategy
from algo_mvp.backtest.strategies.vwap_atr import VwapAtrStrategy


@pytest.fixture
def sample_price_data():
    """Create a small price DataFrame for testing."""
    # Create a simple 20-row DataFrame with OHLCV data
    dates = pd.date_range(start="2023-01-01", periods=20, freq="1min")

    data = {
        "open": [
            100,
            101,
            102,
            101,
            102,
            103,
            102,
            101,
            102,
            103,
            104,
            105,
            106,
            105,
            106,
            107,
            106,
            105,
            106,
            107,
        ],
        "high": [
            102,
            103,
            104,
            103,
            104,
            105,
            104,
            103,
            104,
            105,
            106,
            107,
            108,
            107,
            108,
            109,
            108,
            107,
            108,
            109,
        ],
        "low": [
            99,
            100,
            101,
            100,
            101,
            102,
            101,
            100,
            101,
            102,
            103,
            104,
            105,
            104,
            105,
            106,
            105,
            104,
            105,
            106,
        ],
        "close": [
            101,
            102,
            103,
            102,
            103,
            104,
            103,
            102,
            103,
            104,
            105,
            106,
            107,
            106,
            107,
            108,
            107,
            106,
            107,
            108,
        ],
        "volume": [
            1000,
            1100,
            1200,
            1100,
            1200,
            1300,
            1200,
            1100,
            1200,
            1300,
            1400,
            1500,
            1600,
            1500,
            1600,
            1700,
            1600,
            1500,
            1600,
            1700,
        ],
    }

    return pd.DataFrame(data, index=dates)


def test_vwap_atr_strategy(sample_price_data):
    """Test the VWAP-ATR strategy."""
    # Initialize strategy
    params = {"band_mult": 2.0, "atr_len": 5}
    strategy = VwapAtrStrategy(params)

    # Run strategy
    signals = strategy.run(sample_price_data)

    # Check that signals are returned
    assert "entries" in signals
    assert "exits" in signals

    # Check that signals are Series with the same index as the data
    assert isinstance(signals["entries"], pd.Series)
    assert isinstance(signals["exits"], pd.Series)
    assert signals["entries"].index.equals(sample_price_data.index)
    assert signals["exits"].index.equals(sample_price_data.index)

    # Check that signals are boolean
    assert signals["entries"].dtype == bool
    assert signals["exits"].dtype == bool


def test_three_day_momo_strategy(sample_price_data):
    """Test the Three Day Momentum strategy."""
    # Initialize strategy
    params = {"band_mult": 1.5, "atr_len": 5}
    strategy = ThreeDayMomoStrategy(params)

    # Run strategy
    signals = strategy.run(sample_price_data)

    # Check that signals are returned
    assert "entries" in signals
    assert "exits" in signals

    # Check that signals are Series with the same index as the data
    assert isinstance(signals["entries"], pd.Series)
    assert isinstance(signals["exits"], pd.Series)
    assert signals["entries"].index.equals(sample_price_data.index)
    assert signals["exits"].index.equals(sample_price_data.index)

    # Check that signals are boolean
    assert signals["entries"].dtype == bool
    assert signals["exits"].dtype == bool


def test_vwap_atr_missing_columns():
    """Test that the VWAP-ATR strategy raises an error for missing columns."""
    # Create data with missing columns
    data = pd.DataFrame(
        {
            "open": [100, 101],
            "high": [102, 103],
            "low": [99, 100],
            # Missing close and volume
        }
    )

    # Initialize strategy
    params = {"band_mult": 2.0, "atr_len": 5}
    strategy = VwapAtrStrategy(params)

    # Check that running with missing columns raises an error
    with pytest.raises(ValueError):
        strategy.run(data)


def test_three_day_momo_missing_columns():
    """Test that the Three Day Momentum strategy raises an error for missing columns."""
    # Create data with missing columns
    data = pd.DataFrame(
        {
            "open": [100, 101],
            "high": [102, 103],
            # Missing low, close, and volume
        }
    )

    # Initialize strategy
    params = {"band_mult": 1.5, "atr_len": 5}
    strategy = ThreeDayMomoStrategy(params)

    # Check that running with missing columns raises an error
    with pytest.raises(ValueError):
        strategy.run(data)
