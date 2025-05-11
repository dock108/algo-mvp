"""Tests for analytics utility functions."""

import pandas as pd
from datetime import datetime, timedelta

from algo_mvp.analytics.utils import (
    calc_sharpe,
    calc_cagr,
    calc_max_drawdown,
    calc_drawdown_series,
)


def test_calc_sharpe():
    """Test Sharpe ratio calculation."""
    # Test with simple return series
    returns = pd.Series([0.01, 0.02, -0.01, 0.03, 0.005])
    sharpe = calc_sharpe(returns, freq="D")
    assert isinstance(sharpe, float)
    assert sharpe > 0  # Overall positive returns should give positive Sharpe

    # Test with negative returns
    neg_returns = pd.Series([-0.01, -0.02, -0.01, -0.03, -0.005])
    neg_sharpe = calc_sharpe(neg_returns, freq="D")
    assert neg_sharpe < 0  # Overall negative returns should give negative Sharpe

    # Test with equity series (not returns)
    equity = pd.Series([10000, 10100, 10050, 10200, 10250])
    equity_sharpe = calc_sharpe(equity, freq="D")
    assert isinstance(equity_sharpe, float)
    assert equity_sharpe > 0

    # Test with constant returns (zero standard deviation)
    const_returns = pd.Series([0.01, 0.01, 0.01, 0.01])
    const_sharpe = calc_sharpe(const_returns, freq="D")
    assert const_sharpe == 0  # Should handle division by zero

    # Test with different frequencies
    returns_hourly = pd.Series([0.001, 0.002, -0.001, 0.003, 0.0005])
    hourly_sharpe = calc_sharpe(returns_hourly, freq="H")
    # Hourly Sharpe should be lower than daily with same returns
    # because we annualize by multiplying by sqrt(trading_hours_per_year)
    daily_sharpe = calc_sharpe(returns_hourly, freq="D")
    assert hourly_sharpe > daily_sharpe


def test_calc_cagr():
    """Test CAGR calculation."""
    # Create a series with a datetime index spanning 2 years
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=i * 30) for i in range(25)]  # ~2 years of data

    # Test with 10% annual growth
    # If we grow 10% annually for 2 years, we end up with 1.10^2 = 1.21x
    equity = pd.Series(
        [10000 * (1.1) ** (i / 12) for i in range(25)],  # 10% annual growth
        index=dates,
    )
    cagr = calc_cagr(equity)
    assert abs(cagr - 0.10) < 0.01  # Should be close to 10%

    # Test with decreasing equity
    declining_equity = pd.Series(
        [10000 * (0.9) ** (i / 12) for i in range(25)],  # 10% annual decline
        index=dates,
    )
    neg_cagr = calc_cagr(declining_equity)
    assert neg_cagr < 0
    assert abs(neg_cagr + 0.10) < 0.01  # Should be close to -10%

    # Test with empty series
    empty_series = pd.Series(dtype=float)
    empty_cagr = calc_cagr(empty_series)
    assert empty_cagr == 0.0

    # Test with one day
    one_day = pd.Series(
        [10000, 10100], index=[datetime(2023, 1, 1), datetime(2023, 1, 1, 12)]
    )
    one_day_cagr = calc_cagr(one_day)
    assert one_day_cagr == 0.0  # Should handle zero years gracefully


def test_calc_max_drawdown():
    """Test maximum drawdown calculation."""
    # Test with a simple equity curve with one drawdown
    equity = pd.Series([10000, 10500, 10200, 10000, 10300, 10600])
    max_dd = calc_max_drawdown(equity)
    assert isinstance(max_dd, float)
    # Max drawdown should be (10000 - 10500) / 10500 = -0.0476...
    assert abs(max_dd - 0.0476) < 0.001

    # Test with multiple drawdowns
    equity2 = pd.Series([10000, 10500, 10000, 10800, 10400, 11000])
    max_dd2 = calc_max_drawdown(equity2)
    # Max drawdown should be (10000 - 10500) / 10500 = -0.0476... or (10400 - 10800) / 10800 = -0.037...
    # The first one is larger
    assert abs(max_dd2 - 0.0476) < 0.001

    # Test with empty series
    empty_series = pd.Series(dtype=float)
    empty_max_dd = calc_max_drawdown(empty_series)
    assert empty_max_dd == 0.0

    # Test with always increasing equity
    increasing = pd.Series([10000, 10100, 10200, 10300, 10400])
    inc_max_dd = calc_max_drawdown(increasing)
    assert inc_max_dd == 0.0


def test_calc_drawdown_series():
    """Test drawdown series calculation."""
    # Test with a simple equity curve
    equity = pd.Series([10000, 10500, 10200, 10000, 10300, 10600])
    dd_series = calc_drawdown_series(equity)
    assert isinstance(dd_series, pd.Series)
    assert len(dd_series) == len(equity)

    # Check specific values
    # At index 0, drawdown should be 0
    assert dd_series.iloc[0] == 0

    # At index 1, drawdown should be 0 (new high)
    assert dd_series.iloc[1] == 0

    # At index 2, drawdown should be (10200 - 10500) / 10500 = -0.0286...
    assert abs(dd_series.iloc[2] + 0.0286) < 0.001

    # At index 3, drawdown should be (10000 - 10500) / 10500 = -0.0476...
    assert abs(dd_series.iloc[3] + 0.0476) < 0.001

    # Test with empty series
    empty_series = pd.Series(dtype=float)
    empty_dd = calc_drawdown_series(empty_series)
    assert isinstance(empty_dd, pd.Series)
    assert empty_dd.empty
