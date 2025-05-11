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


def style_dataframe(df):
    """Apply pandas Styler to dataframe for better display."""
    if df.empty:
        return df

    # Create a styled dataframe
    styled_df = df.style.format(
        {
            "pnl_per_trade": "${:,.2f}",
            "fill_price": "${:,.2f}",
            "qty": "{:.0f}",
            "fill_qty": "{:.0f}",
            "commission": "${:,.2f}",
            "unrealized_pnl": "${:,.2f}",
            "equity": "${:,.2f}",
            "avg_price": "${:,.2f}",
        },
        na_rep="—",
    ).set_table_styles([{"selector": "th", "props": [("text-align", "center")]}])

    return styled_df


def metric_card(label, value, delta=None, icon=None):
    """Create a styled metric card with label, value, delta, and optional icon.

    Returns a styled HTML container via st.markdown.
    """
    # Determine if delta is positive, negative or neutral for styling
    if delta is not None:
        if isinstance(delta, str):
            # Try to extract numeric value from string (assuming format like "+2.5%")
            try:
                delta_value = float(delta.strip("%+-").replace(",", ""))
                delta_sign = delta[0] if delta[0] in ["+", "-"] else ""
            except ValueError:
                delta_value = 0
                delta_sign = ""
        else:
            delta_value = float(delta)
            delta_sign = "+" if delta_value > 0 else "-" if delta_value < 0 else ""

        delta_color = (
            "green" if delta_sign == "+" else "red" if delta_sign == "-" else "gray"
        )
    else:
        delta_value = None
        delta_sign = ""
        delta_color = "gray"

    # Build the HTML for the metric card
    icon_html = (
        f'<span style="font-size: 1.3rem; margin-right: 0.5rem;">{icon}</span>'
        if icon
        else ""
    )

    html = f"""
    <div style="
        background-color: var(--background-color);
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
    ">
        <div style="color: rgba(255, 255, 255, 0.6); font-size: 0.8rem; margin-bottom: 0.5rem;">
            {icon_html}{label}
        </div>
        <div style="font-size: 1.5rem; font-weight: bold; margin-bottom: 0.2rem;">
            {value}
        </div>
    """

    if delta is not None:
        delta_formatted = (
            delta if isinstance(delta, str) else f"{delta_sign}{abs(delta_value):.2f}%"
        )
        html += f"""
        <div style="font-size: 0.8rem; color: var(--{delta_color}-80);">
            {delta_formatted}
        </div>
        """

    html += "</div>"

    return html


def toggle_theme():
    """Toggle between light and dark theme."""
    if st.session_state.get("theme", "dark") == "dark":
        st.session_state["theme"] = "light"
    else:
        st.session_state["theme"] = "dark"

    # Update the config.toml file
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
        ".streamlit",
        "config.toml",
    )

    with open(config_path, "r") as f:
        config = f.read()

    if st.session_state["theme"] == "light":
        config = config.replace('base="dark"', 'base="light"')
    else:
        config = config.replace('base="light"', 'base="dark"')

    with open(config_path, "w") as f:
        f.write(config)

    # Force a rerun to apply theme change
    st.rerun()


def main(auto_refresh=True):
    """Main function to create the Streamlit dashboard.

    Args:
        auto_refresh: Whether to automatically refresh the dashboard.
                     Set to False during testing.
    """
    # Initialize session state for auto-refresh and theme if not already set
    if "auto_refresh" not in st.session_state:
        st.session_state["auto_refresh"] = auto_refresh

    if "theme" not in st.session_state:
        st.session_state["theme"] = "dark"  # Default theme

    # Configure the Streamlit page
    st.set_page_config(page_title="Algo-MVP", layout="wide")

    # Set pandas display options
    set_display_options()

    # Sidebar with theme toggle
    with st.sidebar:
        st.title("Settings")

        # Theme toggle
        theme_label = "🌗 Dark / Light"
        if st.button(theme_label):
            toggle_theme()

        # Manual refresh button
        refresh_label = (
            "🔄 Live Refresh (5 s)"
            if st.session_state["auto_refresh"]
            else "▶️ Resume Refresh"
        )
        if st.button(refresh_label):
            st.session_state["auto_refresh"] = not st.session_state["auto_refresh"]
            st.rerun()

    # Page header
    st.title("Algo-MVP – Live Overview")

    # Initialize AnalyticsAPI
    api = AnalyticsAPI()

    # Get data from API
    equity_df = api.pnl_curve()
    stats = api.summary_stats()
    open_pos = api.open_positions()
    trades_df = api.trade_log().tail(20)

    # Create responsive columns that stack on mobile (3:1 ratio with large gap)
    with st.container():
        col1, col2 = st.columns([3, 1], gap="large")

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

        # Column 2: Key metrics as styled cards
        with col2:
            st.subheader("Performance Metrics")

            # Format stats for display
            net_pnl = (
                equity_df["equity"].iloc[-1] - equity_df["equity"].iloc[0]
                if not equity_df.empty
                else 0
            )

            # Calculate daily performance (placeholder - replace with actual calculation if available)
            daily_change_pct = stats.get("daily_pct", 0)

            # Display metrics as cards
            st.markdown(
                metric_card("Net P&L", f"${net_pnl:.2f}", daily_change_pct, "💰"),
                unsafe_allow_html=True,
            )
            st.markdown(
                metric_card("CAGR", f"{stats.get('cagr', 0):.2f}%", None, "📈"),
                unsafe_allow_html=True,
            )
            st.markdown(
                metric_card(
                    "Max Drawdown", f"{stats.get('max_dd_pct', 0):.2f}%", None, "📉"
                ),
                unsafe_allow_html=True,
            )
            st.markdown(
                metric_card(
                    "Hit Rate", f"{stats.get('hit_rate', 0) * 100:.2f}%", None, "🎯"
                ),
                unsafe_allow_html=True,
            )

    # Open positions table
    st.subheader("Open Positions")
    if not open_pos.empty:
        st.dataframe(
            style_dataframe(open_pos),
            use_container_width=True,
            hide_index=True,
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

        st.dataframe(
            style_dataframe(trades_display),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No trades available.")

    # Add a timestamp at the bottom to show when data was last refreshed
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Auto-refresh every 5 seconds if enabled and not in testing mode
    if st.session_state["auto_refresh"] and auto_refresh:
        time.sleep(5)
        st.rerun()


if __name__ == "__main__":
    # Check if running in test environment
    is_test = os.environ.get("PYTEST_CURRENT_TEST") is not None
    main(auto_refresh=not is_test)
