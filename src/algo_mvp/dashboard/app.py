"""Streamlit dashboard for Algo-MVP."""

import time
import pandas as pd
import streamlit as st
from datetime import datetime
import os
import requests

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
        box-shadow: 0 0.0625rem 0.125rem 0 rgba(0, 0, 0, 0.05);
        border: 0.0625rem solid rgba(255, 255, 255, 0.1);
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
    # Toggle the theme in session state
    if st.session_state.get("theme", "dark") == "dark":
        new_theme = "light"
    else:
        new_theme = "dark"

    st.session_state["theme"] = new_theme

    # Update the config.toml file in the background
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
        ".streamlit",
        "config.toml",
    )

    try:
        with open(config_path, "r") as f:
            config = f.read()

        if new_theme == "light":
            config = config.replace('base="dark"', 'base="light"')
        else:
            config = config.replace('base="light"', 'base="dark"')

        with open(config_path, "w") as f:
            f.write(config)

        # We'll try to use JavaScript for immediate theme toggle without reload
        # but will keep the ability to rerun if needed
        return new_theme
    except Exception as e:
        st.error(f"Failed to update theme: {e}")
        # Force a rerun if modifying the config file directly failed
        st.rerun()


def main(auto_refresh=True):
    """Main function to create the Streamlit dashboard.

    Args:
        auto_refresh: Whether to automatically refresh the dashboard.
                     Set to False during testing.
    """
    # Get password from environment variable or secrets
    password = os.environ.get("DASHBOARD_PASSWORD", None)
    if password is None:
        try:
            password = st.secrets.get("dashboard_password", "password")
        except Exception:
            password = "password"  # Default fallback

    # Check authentication state
    if "auth" not in st.session_state:
        st.title("Dashboard Login")
        pwd = st.text_input("Dashboard password", type="password")

        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("Login"):
                if pwd == password:
                    st.session_state["auth"] = True
                    st.rerun()
                else:
                    st.error("Incorrect password")

        # Stop rendering the rest of the dashboard if not authenticated
        st.stop()

    # Initialize session state for auto-refresh and theme if not already set
    if "auto_refresh" not in st.session_state:
        st.session_state["auto_refresh"] = auto_refresh

    if "theme" not in st.session_state:
        st.session_state["theme"] = "dark"  # Default theme

    # Get supervisor URL from environment or secrets
    supervisor_url = os.environ.get("SUPERVISOR_URL", None)
    if not supervisor_url and hasattr(st, "secrets") and "supervisor_url" in st.secrets:
        supervisor_url = st.secrets["supervisor_url"]

    # Default for local development if not specified
    if not supervisor_url:
        supervisor_url = "http://localhost:8000"

    # Configure the Streamlit page
    st.set_page_config(page_title="Algo-MVP", layout="wide")

    # JavaScript to manage theme persistence with localStorage
    js_code = """
    <script>
    // Function to set theme in localStorage
    function setThemePreference(theme) {
        localStorage.setItem('algo_mvp_theme', theme);
    }

    // Function to get theme from localStorage
    function getThemePreference() {
        return localStorage.getItem('algo_mvp_theme') || 'dark';
    }

    // Function to apply theme without full page reload
    function applyTheme(theme) {
        // This targets Streamlit's theme CSS variables
        document.body.dataset.theme = theme;

        // Toggle dark/light class on body for additional custom styling
        if (theme === 'light') {
            document.body.classList.remove('dark');
            document.body.classList.add('light');
        } else {
            document.body.classList.remove('light');
            document.body.classList.add('dark');
        }

        // Store the preference
        setThemePreference(theme);
    }

    // Initialize theme from localStorage (will be called when page loads)
    document.addEventListener('DOMContentLoaded', function() {
        const savedTheme = getThemePreference();

        // Apply the theme immediately
        applyTheme(savedTheme);

        // Create a custom event to communicate with Streamlit
        const event = new CustomEvent('streamlit:themeInit', {
            detail: { theme: savedTheme }
        });
        window.dispatchEvent(event);
    });

    // Listen for theme changes to save to localStorage
    window.addEventListener('streamlit:themeChange', function(e) {
        applyTheme(e.detail.theme);
    });
    </script>
    """

    # Inject the JavaScript
    st.markdown(js_code, unsafe_allow_html=True)

    # Read theme from localStorage via custom event listener
    # We'll use a query param to get the initial theme without requiring a reload
    query_params = st.experimental_get_query_params()
    if "theme" in query_params and query_params["theme"][0] in ["light", "dark"]:
        st.session_state["theme"] = query_params["theme"][0]

        # Update the config.toml file to match localStorage preference
        config_path = os.path.join(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            ),
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

    # Custom CSS for theme transitions
    theme_transitions = """
    <style>
    /* Smooth theme transitions */
    body {
        transition: background-color 0.3s ease, color 0.3s ease;
    }

    /* Additional theme-specific styles can be added here */
    </style>
    """
    st.markdown(theme_transitions, unsafe_allow_html=True)

    # Set pandas display options
    set_display_options()

    # Sidebar with theme toggle
    with st.sidebar:
        st.title("Settings")

        # Logout button
        if st.button("🔑 Logout"):
            st.session_state.pop("auth", None)
            st.rerun()

        # Theme toggle
        theme_label = "🌗 Dark / Light"
        if st.button(theme_label):
            # Toggle theme and get the new theme
            new_theme = toggle_theme()

            # Dispatch a custom event to update localStorage (via st.markdown JavaScript)
            js_update = f"""
            <script>
                // Notify our listener that theme changed
                window.dispatchEvent(new CustomEvent('streamlit:themeChange', {{
                    detail: {{ theme: '{new_theme}' }}
                }}));
            </script>
            """
            st.markdown(js_update, unsafe_allow_html=True)

        # Manual refresh button
        refresh_label = (
            "🔄 Live Refresh (5 s)"
            if st.session_state["auto_refresh"]
            else "▶️ Resume Refresh"
        )
        if st.button(refresh_label):
            st.session_state["auto_refresh"] = not st.session_state["auto_refresh"]
            st.rerun()

        # Admin Controls section in an expander
        with st.expander("🔐 Admin Controls"):
            st.caption(
                "These controls require authentication and directly affect live strategies."
            )

            # Token input field
            token = st.text_input("Token", type="password", key="ctrl_token")

            # Flatten All button with API call
            if st.button("🛑 Flatten All"):
                if token:
                    try:
                        r = requests.post(
                            f"{supervisor_url}/action/flatten_all",
                            params={"token": token},
                        )
                        if r.ok:
                            st.success("All positions flattened")
                        else:
                            st.error(f"Error: {r.text}")
                    except Exception as e:
                        st.error(f"Connection error: {str(e)}")
                else:
                    st.warning("Please enter a token")

            # Strategy selection and Pause button
            # For now hardcoding "mes_scalp" as the example, but could be made dynamic
            strategy_name = "mes_scalp"

            if st.button(f"⏸ Pause Strategy ({strategy_name})"):
                if token:
                    try:
                        r = requests.post(
                            f"{supervisor_url}/action/pause",
                            params={"runner": strategy_name, "token": token},
                        )
                        if r.ok:
                            response_data = r.json()
                            state = (
                                "Paused"
                                if response_data.get("paused", False)
                                else "Active"
                            )
                            st.info(f"{strategy_name} is now {state}")
                        else:
                            st.error(f"Error: {r.text}")
                    except Exception as e:
                        st.error(f"Connection error: {str(e)}")
                else:
                    st.warning("Please enter a token")

            # Add Reload Config button
            if st.button("🔄 Reload Config"):
                if token:
                    try:
                        r = requests.post(
                            f"{supervisor_url}/action/reload_config",
                            params={"token": token},
                        )
                        if r.ok:
                            response_data = r.json()
                            if response_data.get("reloaded", False):
                                st.success("Configuration reloaded successfully")
                                # Display runner status
                                for runner_name, status in response_data.get(
                                    "runners", {}
                                ).items():
                                    st.info(f"Runner '{runner_name}': {status}")
                            else:
                                st.warning(
                                    "Reload operation completed but may not have been successful"
                                )
                        else:
                            st.error(f"Error: {r.text}")
                    except Exception as e:
                        st.error(f"Connection error: {str(e)}")
                else:
                    st.warning("Please enter a token")

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
