"""Utility functions for financial calculations used by the Analytics API."""

import numpy as np
import pandas as pd


def calc_sharpe(series: pd.Series, freq: str = "D") -> float:
    """Calculate Sharpe ratio for a returns series.

    Args:
        series: Series of returns or equity values
        freq: Frequency of the data ('D' for daily, etc.)

    Returns:
        Sharpe ratio (annualized)
    """
    # If equity curve (not returns), convert to returns
    if not ((-1 <= series) & (series <= 1)).all():
        returns = series.pct_change().dropna()
    else:
        returns = series

    # Calculate annualization factor based on frequency
    freq_factors = {
        "D": 252,  # Trading days in a year
        "W": 52,  # Weeks in a year
        "M": 12,  # Months in a year
        "Q": 4,  # Quarters in a year
        "Y": 1,  # Years
        "H": 252 * 6.5,  # Trading hours in a year (approx)
        "min": 252 * 6.5 * 60,  # Trading minutes in a year (approx)
        "T": 252 * 6.5 * 60,  # Trading minutes in a year (alias for min)
    }

    annualization_factor = freq_factors.get(freq, 252)  # Default to daily

    # Calculate Sharpe
    mean_return = returns.mean()
    std_return = returns.std()

    # Avoid division by zero
    if std_return == 0:
        return 0

    sharpe = mean_return / std_return * np.sqrt(annualization_factor)
    return sharpe


def calc_cagr(equity_series: pd.Series) -> float:
    """Calculate Compound Annual Growth Rate from an equity curve.

    Args:
        equity_series: Series of equity values with datetime index

    Returns:
        CAGR as a decimal (e.g., 0.15 for 15% annual return)
    """
    if equity_series.empty:
        return 0.0

    # Get first and last values
    start_value = equity_series.iloc[0]
    end_value = equity_series.iloc[-1]

    # Avoid division by zero
    if start_value == 0:
        return 0.0

    # Get time difference in years
    start_date = equity_series.index[0]
    end_date = equity_series.index[-1]
    years = (end_date - start_date).days / 365.25

    # Avoid division by zero
    if years == 0:
        return 0.0

    # Calculate CAGR
    cagr = (end_value / start_value) ** (1 / years) - 1
    return cagr


def calc_max_drawdown(equity_series: pd.Series) -> float:
    """Calculate maximum drawdown percentage from an equity curve.

    Args:
        equity_series: Series of equity values

    Returns:
        Maximum drawdown as a decimal (e.g., 0.25 for 25% drawdown)
    """
    if equity_series.empty:
        return 0.0

    # Calculate the running maximum
    running_max = equity_series.cummax()

    # Calculate drawdown percentage at each point
    drawdown = (equity_series - running_max) / running_max

    # Get the maximum drawdown
    max_dd = drawdown.min()

    return abs(max_dd)  # Return as a positive value


def calc_drawdown_series(equity_series: pd.Series) -> pd.Series:
    """Calculate drawdown percentage series from an equity curve.

    Args:
        equity_series: Series of equity values with datetime index

    Returns:
        Series of drawdown percentages with same index as equity_series
    """
    if equity_series.empty:
        return pd.Series(dtype=float)

    # Calculate the running maximum
    running_max = equity_series.cummax()

    # Calculate drawdown percentage at each point
    drawdown = (equity_series - running_max) / running_max

    return drawdown
