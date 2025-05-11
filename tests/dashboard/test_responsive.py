"""Tests for dashboard responsive design elements."""

from algo_mvp.dashboard import app
from unittest.mock import MagicMock


def test_metric_card_structure():
    """Test that the metric_card function generates correct HTML structure."""
    # Basic card without delta or icon
    card_html = app.metric_card(label="Test Metric", value="123.45")

    # Check the basic structure
    assert "Test Metric" in card_html
    assert "123.45" in card_html
    assert "border-radius: 0.5rem" in card_html

    # Card with positive delta and icon
    card_html = app.metric_card(
        label="Net P&L", value="$500.00", delta="+2.5%", icon="💰"
    )

    # Check for icon and delta
    assert "💰" in card_html
    assert "Net P&L" in card_html
    assert "$500.00" in card_html
    assert "+2.5%" in card_html
    assert "green" in card_html.lower()

    # Card with negative delta
    card_html = app.metric_card(
        label="Drawdown", value="-15.2%", delta="-3.1%", icon="📉"
    )

    assert "📉" in card_html
    assert "Drawdown" in card_html
    assert "-15.2%" in card_html
    assert "-3.1%" in card_html
    assert "red" in card_html.lower()


def test_responsive_column_layout():
    """
    Test that the dashboard uses responsive columns with appropriate settings.

    This test verifies the column configuration rather than launching a browser.
    """
    # These tests don't need to launch a browser - they just check
    # that we're using the right configuration in our code

    # Patch streamlit.columns to capture the parameters
    captured_params = []

    def mock_columns(*args, **kwargs):
        captured_params.append((args, kwargs))
        # Return mock columns that support context manager
        col1 = MagicMock()
        col2 = MagicMock()
        col1.__enter__ = MagicMock(return_value=col1)
        col1.__exit__ = MagicMock(return_value=None)
        col2.__enter__ = MagicMock(return_value=col2)
        col2.__exit__ = MagicMock(return_value=None)
        return [col1, col2]

    # Run the main function with mocked dependencies
    import unittest.mock as mock

    # Create a container that supports context manager
    mock_container = mock.MagicMock()
    mock_container.__enter__ = mock.MagicMock(return_value=mock_container)
    mock_container.__exit__ = mock.MagicMock(return_value=None)

    with (
        mock.patch("streamlit.columns", side_effect=mock_columns),
        mock.patch("streamlit.container", return_value=mock_container),
        mock.patch("streamlit.session_state", {"theme": "dark", "auto_refresh": False}),
        mock.patch("streamlit.title"),
        mock.patch("streamlit.sidebar.title"),
        mock.patch("streamlit.sidebar.button", return_value=False),
        mock.patch("streamlit.subheader"),
        mock.patch("streamlit.line_chart"),
        mock.patch("streamlit.markdown"),
        mock.patch("streamlit.dataframe"),
        mock.patch("streamlit.info"),
        mock.patch("streamlit.caption"),
        mock.patch("streamlit.set_page_config"),
        mock.patch("streamlit.rerun"),
        mock.patch("algo_mvp.dashboard.app.AnalyticsAPI") as mock_api,
    ):
        # Setup mock API returns
        mock_api_instance = mock.MagicMock()
        mock_api.return_value = mock_api_instance
        mock_api_instance.pnl_curve.return_value = mock.MagicMock(empty=True)
        mock_api_instance.summary_stats.return_value = {}
        mock_api_instance.open_positions.return_value = mock.MagicMock(empty=True)
        mock_api_instance.trade_log.return_value = mock.MagicMock(empty=True)
        mock_api_instance.trade_log.return_value.tail.return_value = mock.MagicMock(
            empty=True
        )

        # Run the main function
        app.main(auto_refresh=False)

    # Verify that we have column configuration with the expected responsive settings
    # Find the call for the main columns (3:1 ratio with large gap)
    main_columns_call = None
    for args, kwargs in captured_params:
        if args == ([3, 1],) and kwargs.get("gap") == "large":
            main_columns_call = (args, kwargs)
            break

    # Assert that we found the main columns with responsive settings
    assert main_columns_call is not None, (
        "Could not find main columns with 3:1 ratio and large gap"
    )
