"""Streamlit dashboard for Algo-MVP."""

import time
import pandas as pd
import streamlit as st
from datetime import datetime
import os

from algo_mvp.analytics.api import AnalyticsAPI


def set_display_options():
    """Set pandas display options for formatting currencies and percentages."""
    pd.options.display.float_format = "${:.2f}".format


def format_dataframe(df, is_positions=False):
    """Format dataframe columns for display."""
    if df.empty:
        return df

    # Make a copy to avoid SettingWithCopyWarning
    df_display = df.copy()

    # Format timestamps
    for col in df_display.columns:
        if "time" in col.lower() or "date" in col.lower() or col == "timestamp":
            if col in df_display.columns and not df_display[col].empty:
                df_display[col] = pd.to_datetime(df_display[col]).dt.strftime(
                    "%Y-%m-%d %H:%M:%S"
                )

    # Format currency columns
    currency_cols = [
        "equity",
        "pnl_per_trade",
        "commission",
        "fill_price",
        "avg_price",
        "unrealized_pnl",
    ]
    for col in currency_cols:
        if col in df_display.columns:
            df_display[col] = df_display[col].apply(
                lambda x: f"${x:.2f}" if pd.notnull(x) else ""
            )

    # Format percentage columns
    pct_cols = ["drawdown_pct"]
    for col in pct_cols:
        if col in df_display.columns:
            df_display[col] = df_display[col].apply(
                lambda x: f"{x:.2f}%" if pd.notnull(x) else ""
            )

    return df_display


def main(auto_refresh=True):
    """Main function to create the Streamlit dashboard.

    Args:
        auto_refresh: Whether to automatically refresh the dashboard.
                     Set to False during testing.
    """
    # Configure the Streamlit page
    st.set_page_config(page_title="Algo-MVP", layout="wide")

    # Set pandas display options
    set_display_options()

    # Page header
    st.title("Algo-MVP – Live Overview")

    # Initialize AnalyticsAPI
    api = AnalyticsAPI()

    # Get data from API
    equity_df = api.pnl_curve()
    stats = api.summary_stats()
    open_pos = api.open_positions()
    trades_df = api.trade_log().tail(20)

    # Create two columns with 70/30 split
    col1, col2 = st.columns([7, 3])

    # Column 1: Equity curve chart
    with col1:
        if not equity_df.empty:
            st.subheader("Equity Curve")
            # Convert timestamp to datetime if it isn't already
            equity_df["timestamp"] = pd.to_datetime(equity_df["timestamp"])
            # Plot the equity curve
            st.line_chart(equity_df.set_index("timestamp")["equity"])
        else:
            st.info("No equity data available.")

    # Column 2: Key metrics
    with col2:
        st.subheader("Performance Metrics")

        # Format stats for display
        net_pnl = (
            equity_df["equity"].iloc[-1] - equity_df["equity"].iloc[0]
            if not equity_df.empty
            else 0
        )

        # Display metrics
        st.metric("Net P&L", f"${net_pnl:.2f}")
        st.metric("CAGR", f"{stats.get('cagr', 0):.2f}%")
        st.metric("Max Drawdown", f"{stats.get('max_dd_pct', 0):.2f}%")
        st.metric("Hit Rate", f"{stats.get('hit_rate', 0) * 100:.2f}%")

    # Open positions table
    st.subheader("Open Positions")
    if not open_pos.empty:
        st.dataframe(
            format_dataframe(open_pos, is_positions=True), use_container_width=True
        )
    else:
        st.info("No open positions.")

    # Recent trades table
    st.subheader("Last 20 Trades")
    if not trades_df.empty:
        # Select only relevant columns for display
        display_cols = [
            "symbol",
            "side",
            "fill_qty",
            "fill_price",
            "commission",
            "filled_at",
            "pnl_per_trade",
        ]
        display_cols = [col for col in display_cols if col in trades_df.columns]
        trades_display = trades_df[display_cols].copy()

        st.dataframe(format_dataframe(trades_display), use_container_width=True)
    else:
        st.info("No trades available.")

    # Add a timestamp at the bottom to show when data was last refreshed
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Auto-refresh every 5 seconds if not in testing mode
    if auto_refresh:
        time.sleep(5)
        st.experimental_rerun()


if __name__ == "__main__":
    # Check if running in test environment
    is_test = os.environ.get("PYTEST_CURRENT_TEST") is not None
    main(auto_refresh=not is_test)
