"""Tests for the AnalyticsAPI class."""

from datetime import datetime, timedelta
import pandas as pd

from algo_mvp.analytics.api import AnalyticsAPI


def test_api_init(test_engine):
    """Test that AnalyticsAPI can be initialized with an engine."""
    api = AnalyticsAPI(engine=test_engine)
    assert api._engine == test_engine


def test_pnl_curve_empty():
    """Test that pnl_curve returns empty DataFrame when no data."""
    api = AnalyticsAPI(engine=None)

    # Mock _get_session to return a session with empty results
    class MockSession:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def execute(self, query):
            class EmptyResult:
                def fetchall(self):
                    return []

            return EmptyResult()

    api._get_session = lambda: MockSession()

    # Test with default params
    result = api.pnl_curve()
    assert isinstance(result, pd.DataFrame)
    assert result.empty
    assert list(result.columns) == ["timestamp", "equity"]


def test_pnl_curve(test_engine, seed_test_data):
    """Test that pnl_curve returns correct data."""
    api = AnalyticsAPI(engine=test_engine)

    # Test with default params
    result = api.pnl_curve()

    # Check structure
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert list(result.columns) == ["timestamp", "equity"]

    # Check forward-fill behavior
    # Our test data has 7 equity points at specific times
    # The result should have data points at 1-min intervals
    # The number of minutes between first and last equity point
    start_time = datetime(2025, 5, 1, 9, 0, 0)
    end_time = datetime(2025, 5, 3, 16, 0, 0)
    minutes_in_range = int((end_time - start_time).total_seconds() / 60) + 1

    assert len(result) == minutes_in_range

    # Check that dates are 1-min apart
    timestamps = pd.to_datetime(result["timestamp"])
    time_diffs = timestamps.diff().dropna()
    assert (time_diffs == timedelta(minutes=1)).all()

    # Check specific values from test data
    first_point = result[result["timestamp"] == pd.Timestamp("2025-05-01 09:00:00")]
    assert not first_point.empty
    assert first_point["equity"].values[0] == 10000.0

    # Check forward-fill worked
    mid_point = result[result["timestamp"] == pd.Timestamp("2025-05-01 15:00:00")]
    assert not mid_point.empty
    assert mid_point["equity"].values[0] == 10050.0  # Should be the value from 11:00

    # Test with date filters
    start_date = datetime(2025, 5, 2, 0, 0, 0)
    filtered_result = api.pnl_curve(start=start_date)
    assert len(filtered_result) < len(result)
    assert filtered_result["timestamp"].min() >= pd.Timestamp(start_date)


def test_trade_log_empty():
    """Test that trade_log returns empty DataFrame when no data."""
    api = AnalyticsAPI(engine=None)

    # Mock _get_session to return a session with empty results
    class MockSession:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def execute(self, query):
            class EmptyResult:
                def fetchall(self):
                    return []

            return EmptyResult()

    api._get_session = lambda: MockSession()

    # Test with default params
    result = api.trade_log()
    assert isinstance(result, pd.DataFrame)
    assert result.empty
    assert "pnl_per_trade" in result.columns
    assert "duration_s" in result.columns


def test_trade_log(test_engine, seed_test_data):
    """Test that trade_log returns correct data and calculates PnL."""
    api = AnalyticsAPI(engine=test_engine)

    # Test with default params
    result = api.trade_log()

    # Check structure
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert "pnl_per_trade" in result.columns
    assert "duration_s" in result.columns
    assert "symbol" in result.columns

    # Check all expected rows
    assert len(result) == 5  # We have 5 fills in our test data

    # Check PnL calculation for AAPL round-trip
    aapl_rows = result[result["symbol"] == "AAPL"]
    assert len(aapl_rows) == 2

    # Find the row with the completed trade (sell side)
    completed_trade = aapl_rows[~aapl_rows["pnl_per_trade"].isna()]
    assert len(completed_trade) == 1

    # Calculate expected PnL:
    # Buy 10 shares at $150 with $1 commission = -$1501
    # Sell 10 shares at $160 with $1 commission = $1599
    # Total P&L = $98
    expected_pnl = 98.0
    assert abs(completed_trade["pnl_per_trade"].values[0] - expected_pnl) < 0.01

    # Check duration calculation
    assert all(result["duration_s"] > 0)  # All durations should be positive

    # Test symbol filter
    msft_result = api.trade_log(symbol="MSFT")
    assert len(msft_result) == 2
    assert all(msft_result["symbol"] == "MSFT")


def test_drawdown_series(test_engine, seed_test_data):
    """Test that drawdown_series returns correct data."""
    api = AnalyticsAPI(engine=test_engine)

    # Get drawdown series
    result = api.drawdown_series()

    # Check structure
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert list(result.columns) == ["timestamp", "drawdown_pct"]

    # Since our test data has only increasing equity,
    # drawdowns should all be 0
    assert all(result["drawdown_pct"] == 0)


def test_open_positions(test_engine, seed_test_data):
    """Test that open_positions returns correct data."""
    api = AnalyticsAPI(engine=test_engine)

    # Get open positions
    result = api.open_positions()

    # Check structure
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert set(result.columns) >= {
        "symbol",
        "side",
        "open_qty",
        "avg_price",
        "unrealized_pnl",
        "entry_time",
    }

    # Check values
    # We should have 2 open positions:
    # 1. MSFT: Bought 5, sold 2, net +3
    # 2. GOOG: Bought 2, sold 0, net +2
    assert len(result) == 2

    # Check MSFT position
    msft_pos = result[result["symbol"] == "MSFT"]
    assert not msft_pos.empty
    assert msft_pos["side"].values[0] == "buy"
    assert msft_pos["open_qty"].values[0] == 3
    assert msft_pos["avg_price"].values[0] == 250.0

    # Check GOOG position
    goog_pos = result[result["symbol"] == "GOOG"]
    assert not goog_pos.empty
    assert goog_pos["side"].values[0] == "buy"
    assert goog_pos["open_qty"].values[0] == 2
    assert goog_pos["avg_price"].values[0] == 1500.0


def test_summary_stats(test_engine, seed_test_data):
    """Test that summary_stats returns correct values."""
    api = AnalyticsAPI(engine=test_engine)

    # Get summary stats
    result = api.summary_stats()

    # Check structure
    assert isinstance(result, dict)
    assert set(result.keys()) == {"cagr", "sharpe", "max_dd_pct", "hit_rate", "avg_r"}

    # Check values
    # Since our test data has only increasing equity and one completed trade,
    # we expect:
    # - CAGR to be positive
    # - Sharpe to be positive
    # - Max drawdown to be 0
    # - Hit rate to be 1.0 (one winning trade)
    # - Avg R to be positive
    assert result["cagr"] > 0
    assert result["sharpe"] > 0
    assert result["max_dd_pct"] == 0
    assert result["hit_rate"] == 1.0
    assert result["avg_r"] > 0

    # Test with period filter
    period_result = api.summary_stats(period="today")
    assert isinstance(period_result, dict)
    assert set(period_result.keys()) == {
        "cagr",
        "sharpe",
        "max_dd_pct",
        "hit_rate",
        "avg_r",
    }
