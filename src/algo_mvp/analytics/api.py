"""Analytics API for querying and analyzing trading data from the database."""

from datetime import datetime, timedelta
from typing import Dict, Optional

import pandas as pd
import numpy as np
from sqlalchemy import Engine, select, join
from sqlalchemy.orm import Session

from algo_mvp.db import SessionLocal
from algo_mvp.db.models import Order, Fill, Equity
from algo_mvp.analytics.utils import (
    calc_cagr,
    calc_sharpe,
    calc_max_drawdown,
    calc_drawdown_series,
)


class AnalyticsAPI:
    """API for retrieving and analyzing trading data.

    This class provides methods to query orders, fills, and equity data
    from the database and transform them into useful DataFrames and statistics
    for analysis and reporting.
    """

    def __init__(self, engine: Optional[Engine] = None):
        """Initialize the AnalyticsAPI.

        Args:
            engine: Optional SQLAlchemy engine. If not provided,
                   uses SessionLocal from algo_mvp.db.
        """
        self._engine = engine

    def _get_session(self) -> Session:
        """Get a database session.

        Returns:
            SQLAlchemy Session object
        """
        if self._engine:
            return Session(self._engine)
        return SessionLocal()

    def pnl_curve(
        self, start: Optional[datetime] = None, end: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Get the P&L curve from equity table, forward-filling gaps to 1-min frequency.

        Args:
            start: Optional start date/time filter
            end: Optional end date/time filter

        Returns:
            DataFrame with timestamp index and equity column
        """
        with self._get_session() as session:
            # Build the query with optional date filters
            query = select(Equity.timestamp, Equity.equity)

            if start:
                query = query.where(Equity.timestamp >= start)
            if end:
                query = query.where(Equity.timestamp <= end)

            query = query.order_by(Equity.timestamp)

            # Execute the query
            result = session.execute(query)

            # Convert to DataFrame
            df = pd.DataFrame(result.fetchall(), columns=["timestamp", "equity"])

            if df.empty:
                return pd.DataFrame(columns=["timestamp", "equity"])

            # Set timestamp as index and ensure datetime type
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp")

            # Forward-fill to 1-min frequency
            if not df.empty:
                # Create a continuous 1-min range from start to end
                if start is None:
                    start = df.index.min()
                if end is None:
                    end = df.index.max()

                # Create a continuous time range at 1-min frequency
                full_range = pd.date_range(start=start, end=end, freq="1min")

                # Reindex and forward-fill missing values
                df = df.reindex(full_range).ffill()

                # Reset index to get timestamp as a column
                df = df.reset_index().rename(columns={"index": "timestamp"})

            return df

    def trade_log(self, symbol: Optional[str] = None) -> pd.DataFrame:
        """Get a DataFrame of fills joined with orders, including P&L per trade.

        Args:
            symbol: Optional symbol filter

        Returns:
            DataFrame with trade details including PnL
        """
        with self._get_session() as session:
            # Build the SQL query
            query = (
                select(
                    Order.id.label("order_id"),
                    Order.broker_order_id,
                    Order.symbol,
                    Order.side,
                    Order.qty,
                    Order.limit_price,
                    Order.created_at.label("order_time"),
                    Fill.fill_qty,
                    Fill.fill_price,
                    Fill.commission,
                    Fill.filled_at,
                )
                .select_from(join(Order, Fill, Order.id == Fill.order_id))
                .order_by(Fill.filled_at)
            )

            if symbol:
                query = query.where(Order.symbol == symbol)

            result = session.execute(query)

            # Convert to DataFrame
            df = pd.DataFrame(
                result.fetchall(),
                columns=[
                    "order_id",
                    "broker_order_id",
                    "symbol",
                    "side",
                    "order_qty",
                    "limit_price",
                    "order_time",
                    "fill_qty",
                    "fill_price",
                    "commission",
                    "filled_at",
                ],
            )

            if df.empty:
                return pd.DataFrame(
                    columns=[
                        "order_id",
                        "broker_order_id",
                        "symbol",
                        "side",
                        "order_qty",
                        "limit_price",
                        "order_time",
                        "fill_qty",
                        "fill_price",
                        "commission",
                        "filled_at",
                        "pnl_per_trade",
                        "duration_s",
                    ]
                )

            # Calculate duration in seconds between order and fill
            df["duration_s"] = (
                pd.to_datetime(df["filled_at"]) - pd.to_datetime(df["order_time"])
            ).dt.total_seconds()

            # Calculate P&L per trade (based on round-trips)
            # Group by symbol to handle multiple symbols
            pnl_dfs = []

            for sym in df["symbol"].unique():
                symbol_df = df[df["symbol"] == sym].copy()

                # Process each symbol separately
                # Calculate position and cash flows
                symbol_df["position"] = np.where(
                    symbol_df["side"] == "buy",
                    symbol_df["fill_qty"],
                    -symbol_df["fill_qty"],
                )

                symbol_df["cash_flow"] = -1 * (
                    symbol_df["position"] * symbol_df["fill_price"]
                    + symbol_df["commission"]
                )

                # Calculate cumulative position
                symbol_df["cum_position"] = symbol_df["position"].cumsum()

                # Find round trips (position crosses zero)
                symbol_df["position_change"] = symbol_df["cum_position"].diff()
                symbol_df["trip_complete"] = (symbol_df["cum_position"] == 0) | (
                    symbol_df["cum_position"].shift(1) * symbol_df["cum_position"] < 0
                )

                # Calculate P&L for each trip
                symbol_df["pnl_per_trade"] = np.nan

                # Calculate running P&L for each trip
                current_pnl = 0
                for i in range(len(symbol_df)):
                    current_pnl += symbol_df.iloc[i]["cash_flow"]

                    # If trip is complete, assign P&L to this row
                    if symbol_df.iloc[i]["trip_complete"]:
                        symbol_df.loc[symbol_df.index[i], "pnl_per_trade"] = current_pnl
                        current_pnl = 0

                pnl_dfs.append(symbol_df)

            # Combine symbol-specific DataFrames
            if pnl_dfs:
                df = pd.concat(pnl_dfs, axis=0)

            # Sort by fill time
            df = df.sort_values("filled_at")

            # Drop intermediate calculation columns
            df = df.drop(
                columns=[
                    "position",
                    "cash_flow",
                    "cum_position",
                    "position_change",
                    "trip_complete",
                ],
                errors="ignore",
            )

            return df

    def summary_stats(self, period: str = "all") -> Dict[str, float]:
        """Calculate summary statistics for trading performance.

        Args:
            period: Time period for statistics calculation.
                  'all' (default): All available data
                  'YTD': Year to date
                  'MTD': Month to date
                  'WTD': Week to date
                  'today': Only today

        Returns:
            Dictionary with key metrics
        """
        # Define date range based on period
        end = datetime.now()
        start = None

        if period.lower() == "ytd":
            start = datetime(end.year, 1, 1)
        elif period.lower() == "mtd":
            start = datetime(end.year, end.month, 1)
        elif period.lower() == "wtd":
            # Go back to last Monday (or fewer days if today is Monday)
            start = end - timedelta(days=end.weekday())
            start = datetime(start.year, start.month, start.day)
        elif period.lower() == "today":
            start = datetime(end.year, end.month, end.day)

        # Get the data needed for calculations
        equity_df = self.pnl_curve(start=start, end=end)
        trade_df = self.trade_log()

        # Set default values in case of empty data
        stats = {
            "cagr": 0.0,
            "sharpe": 0.0,
            "max_dd_pct": 0.0,
            "hit_rate": 0.0,
            "avg_r": 0.0,
        }

        # Calculate metrics only if we have data
        if not equity_df.empty:
            equity_series = pd.Series(
                equity_df["equity"].values, index=pd.to_datetime(equity_df["timestamp"])
            )

            # Calculate CAGR and Sharpe ratio
            stats["cagr"] = calc_cagr(equity_series)
            stats["sharpe"] = calc_sharpe(equity_series)
            stats["max_dd_pct"] = calc_max_drawdown(equity_series)

        # Calculate trade-based metrics
        if not trade_df.empty and "pnl_per_trade" in trade_df.columns:
            # Filter only completed trades
            completed_trades = trade_df.dropna(subset=["pnl_per_trade"])

            if not completed_trades.empty:
                # Calculate hit rate
                winning_trades = completed_trades[completed_trades["pnl_per_trade"] > 0]
                stats["hit_rate"] = len(winning_trades) / len(completed_trades)

                # Calculate average R (average winner / average loser)
                avg_winner = (
                    winning_trades["pnl_per_trade"].mean()
                    if not winning_trades.empty
                    else 0
                )
                losing_trades = completed_trades[completed_trades["pnl_per_trade"] < 0]
                avg_loser = (
                    abs(losing_trades["pnl_per_trade"].mean())
                    if not losing_trades.empty
                    else 1
                )

                # Avoid division by zero
                if avg_loser == 0:
                    stats["avg_r"] = 0
                else:
                    stats["avg_r"] = avg_winner / avg_loser

        return stats

    def drawdown_series(self) -> pd.DataFrame:
        """Get a DataFrame of drawdown percentages over time.

        Returns:
            DataFrame with timestamp and drawdown_pct columns
        """
        # Get the equity curve
        equity_df = self.pnl_curve()

        if equity_df.empty:
            return pd.DataFrame(columns=["timestamp", "drawdown_pct"])

        # Create a Series from the DataFrame
        equity_series = pd.Series(
            equity_df["equity"].values, index=pd.to_datetime(equity_df["timestamp"])
        )

        # Calculate drawdown series
        drawdown = calc_drawdown_series(equity_series)

        # Convert back to DataFrame
        result = pd.DataFrame(
            {"timestamp": drawdown.index, "drawdown_pct": drawdown.values}
        )

        return result

    def open_positions(self) -> pd.DataFrame:
        """Get current open positions based on orders and fills.

        Returns:
            DataFrame with position details
        """
        with self._get_session() as session:
            # Get all orders
            orders_query = select(Order)
            orders_result = session.execute(orders_query)
            orders_df = pd.DataFrame(
                [row[0].__dict__ for row in orders_result.fetchall()]
            )

            if orders_df.empty:
                return pd.DataFrame(
                    columns=[
                        "symbol",
                        "side",
                        "open_qty",
                        "avg_price",
                        "unrealized_pnl",
                        "entry_time",
                    ]
                )

            # Get all fills
            fills_query = select(Fill)
            fills_result = session.execute(fills_query)
            fills_df = pd.DataFrame(
                [row[0].__dict__ for row in fills_result.fetchall()]
            )

            if fills_df.empty:
                return pd.DataFrame(
                    columns=[
                        "symbol",
                        "side",
                        "open_qty",
                        "avg_price",
                        "unrealized_pnl",
                        "entry_time",
                    ]
                )

            # Remove SQLAlchemy state column
            if "_sa_instance_state" in orders_df.columns:
                orders_df = orders_df.drop("_sa_instance_state", axis=1)
            if "_sa_instance_state" in fills_df.columns:
                fills_df = fills_df.drop("_sa_instance_state", axis=1)

            # Merge orders and fills
            merged_df = pd.merge(
                orders_df,
                fills_df,
                left_on="id",
                right_on="order_id",
                suffixes=("_order", "_fill"),
            )

            # Calculate positions by symbol
            position_data = []

            for symbol in merged_df["symbol"].unique():
                symbol_data = merged_df[merged_df["symbol"] == symbol]

                # Calculate net position
                buy_qty = symbol_data[symbol_data["side"] == "buy"]["fill_qty"].sum()
                sell_qty = symbol_data[symbol_data["side"] == "sell"]["fill_qty"].sum()
                net_qty = buy_qty - sell_qty

                # Only include symbols with open positions
                if abs(net_qty) > 0:
                    # Determine side of open position
                    current_side = "buy" if net_qty > 0 else "sell"

                    # Calculate average price for the position
                    if current_side == "buy":
                        # Average price of buys
                        position_fills = symbol_data[symbol_data["side"] == "buy"]
                    else:
                        # Average price of sells
                        position_fills = symbol_data[symbol_data["side"] == "sell"]

                    # Calculate quantity-weighted average price
                    avg_price = (
                        (
                            (
                                position_fills["fill_qty"]
                                * position_fills["fill_price"]
                            ).sum()
                            / position_fills["fill_qty"].sum()
                        )
                        if not position_fills.empty
                        else 0
                    )

                    # Get entry time (earliest fill time for this side)
                    entry_time = (
                        position_fills["filled_at"].min()
                        if not position_fills.empty
                        else None
                    )

                    # Add to position data
                    position_data.append(
                        {
                            "symbol": symbol,
                            "side": current_side,
                            "open_qty": abs(net_qty),
                            "avg_price": avg_price,
                            "unrealized_pnl": 0.0,  # Placeholder for market data
                            "entry_time": entry_time,
                        }
                    )

            # Create DataFrame from position data
            result_df = pd.DataFrame(position_data)

            return result_df
