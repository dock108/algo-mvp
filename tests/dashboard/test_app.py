"""Tests for the dashboard app."""

from unittest.mock import patch, MagicMock
import pandas as pd
import pytest

from algo_mvp.dashboard import app


@pytest.fixture
def mock_analytics_api():
    """Mock the AnalyticsAPI class for testing."""
    with patch("algo_mvp.dashboard.app.AnalyticsAPI") as mock_api_class:
        # Create a mock instance
        mock_api = MagicMock()
        mock_api_class.return_value = mock_api

        # Setup mock return values
        # Mock equity curve data
        equity_data = pd.DataFrame(
            {
                "timestamp": pd.date_range(start="2025-01-01", periods=100, freq="1h"),
                "equity": [100 + i * 0.5 for i in range(100)],
            }
        )
        mock_api.pnl_curve.return_value = equity_data

        # Mock summary stats
        mock_api.summary_stats.return_value = {
            "cagr": 12.5,
            "sharpe": 1.8,
            "max_dd_pct": 5.2,
            "hit_rate": 0.65,
            "avg_r": 1.2,
        }

        # Mock open positions
        positions_data = pd.DataFrame(
            {
                "symbol": ["AAPL", "MSFT"],
                "side": ["buy", "sell"],
                "open_qty": [100, 50],
                "avg_price": [150.25, 280.75],
                "unrealized_pnl": [250.75, 120.50],
                "entry_time": pd.date_range(start="2025-04-01", periods=2),
            }
        )
        mock_api.open_positions.return_value = positions_data

        # Mock trade log
        trades_data = pd.DataFrame(
            {
                "symbol": ["AAPL", "MSFT", "GOOG"] * 7,
                "side": ["buy", "sell", "buy"] * 7,
                "fill_qty": [100, 50, 75] * 7,
                "fill_price": [150.25, 280.75, 2100.50] * 7,
                "commission": [1.25, 0.95, 1.75] * 7,
                "filled_at": pd.date_range(start="2025-04-01", periods=21),
                "pnl_per_trade": [125.50, -75.25, 200.00] * 7,
            }
        )
        mock_api.trade_log.return_value = trades_data

        yield mock_api


def test_format_dataframe():
    """Test the dataframe formatting function."""
    # Test with an empty dataframe
    empty_df = pd.DataFrame()
    result = app.format_dataframe(empty_df)
    assert result.empty

    # Test with a dataframe containing timestamps and currency values
    test_df = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2025-05-01 12:34:56")],
            "equity": [1234.56],
            "drawdown_pct": [5.25],
        }
    )

    result = app.format_dataframe(test_df)
    assert result["timestamp"][0] == "2025-05-01 12:34:56"
    assert result["equity"][0] == "$1234.56"
    assert result["drawdown_pct"][0] == "5.25%"


def test_streamlit_dashboard_renders(mock_analytics_api):
    """Test that the Streamlit dashboard renders without errors."""
    # Mock Streamlit components to prevent actual rendering
    with (
        patch("streamlit.title"),
        patch("streamlit.subheader"),
        patch("streamlit.columns", return_value=[MagicMock(), MagicMock()]),
        patch("streamlit.line_chart"),
        patch("streamlit.metric"),
        patch("streamlit.dataframe"),
        patch("streamlit.caption"),
        patch("streamlit.info"),
        patch("streamlit.set_page_config"),
        patch("streamlit.container", return_value=MagicMock()),
        patch("streamlit.session_state", {}),
        patch("streamlit.markdown"),
        patch("streamlit.button", return_value=False),
        patch("streamlit.sidebar", return_value=MagicMock()),
        patch("streamlit.rerun"),
    ):
        # Call the main function with auto_refresh=False to avoid time.sleep and rerun
        app.main(auto_refresh=False)

        # Verify the API was called correctly - this is what we really care about testing
        mock_analytics_api.pnl_curve.assert_called_once()
        mock_analytics_api.summary_stats.assert_called_once()
        mock_analytics_api.open_positions.assert_called_once()
        mock_analytics_api.trade_log.assert_called_once()


def test_trade_table_rows(mock_analytics_api):
    """Test that the Last 20 Trades table contains at most 20 rows."""
    # Create a mock DataFrame for trade log
    mock_trades_df = MagicMock()
    mock_trades_df.empty = False
    mock_trades_df.columns = [
        "symbol",
        "side",
        "fill_qty",
        "fill_price",
        "commission",
        "filled_at",
        "pnl_per_trade",
    ]

    # Set up the mock API to return our mocked DataFrame
    mock_analytics_api.trade_log.return_value = mock_trades_df

    # Mock all the Streamlit functions to prevent actual rendering
    with (
        patch("streamlit.title"),
        patch("streamlit.subheader"),
        patch("streamlit.columns", return_value=[MagicMock(), MagicMock()]),
        patch("streamlit.line_chart"),
        patch("streamlit.metric"),
        patch("streamlit.dataframe"),
        patch("streamlit.caption"),
        patch("streamlit.info"),
        patch("streamlit.set_page_config"),
    ):
        # Call the main function with auto_refresh=False to avoid time.sleep and rerun
        app.main(auto_refresh=False)

        # Verify the tail(20) was called on the trades dataframe
        mock_trades_df.tail.assert_called_once_with(20)
