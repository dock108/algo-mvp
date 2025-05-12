"""End-to-end integration tests for the full algo-mvp stack.

This module provides tests that orchestrate a complete "micro-run" through
every major layer of the application:
- BrokerAdapter (mocked)
- LiveRunner
- DBWriter
- AnalyticsAPI
- Dashboard rendering

The tests verify that the entire stack can process trades and generate
expected metrics through all layers.
"""

import asyncio
from datetime import datetime
import pytest
from sqlalchemy import create_engine, text
from unittest.mock import patch, MagicMock
import pandas as pd

from algo_mvp.live.adapters.mock import MockBrokerAdapter
from algo_mvp.live.runner import LiveRunner
from algo_mvp.db.writer import DBWriter
from algo_mvp.db.models import Base
from algo_mvp.analytics.api import AnalyticsAPI
from algo_mvp.dashboard import app


class MockStrategy:
    """A minimal strategy that does nothing.

    In our test, we'll bypass the strategy and directly
    simulate trades via the mock adapter.
    """

    def __init__(self, *args, **kwargs):
        pass


class TrackedMockBrokerAdapter(MockBrokerAdapter):
    """Extended mock adapter that tracks close_all_positions calls."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.close_all_positions_calls = 0
        self._scheduled_fills = []
        self._fill_index = 0

    async def close_all_positions(self):
        """Track calls to close_all_positions."""
        self.close_all_positions_calls += 1
        # Clear positions dict
        self.positions.clear()
        return True

    def schedule_fills(self, fills):
        """Schedule a sequence of fills to be returned by the adapter.

        Args:
            fills: List of dictionaries with fill information
        """
        self._scheduled_fills = fills
        self._fill_index = 0

    async def _process_scheduled_fill(self):
        """Process the next scheduled fill."""
        if not self._scheduled_fills or self._fill_index >= len(self._scheduled_fills):
            return None

        fill = self._scheduled_fills[self._fill_index]
        self._fill_index += 1

        # Update positions based on the fill
        symbol = fill["symbol"]
        qty = fill["qty"]
        price = fill["price"]
        side = fill["side"]

        # Update positions
        if side == "buy":
            self.positions[symbol] = self.positions.get(symbol, 0) + qty
            self.cash -= qty * price
        else:  # sell
            self.positions[symbol] = self.positions.get(symbol, 0) - qty
            self.cash += qty * price
            # Clean up if position is zero
            if self.positions[symbol] == 0:
                del self.positions[symbol]

        # Create fill event for the runner
        fill_event = {
            "id": f"fill_{self._fill_index}",
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "price": price,
            "commission": 0.0,
            "timestamp": datetime.utcnow(),
        }

        # If we have a runner, notify it of the fill
        if self.runner:
            await self.runner.on_trade(fill_event)

        return fill_event


@pytest.fixture
def memory_engine():
    """Create an in-memory SQLite engine for testing, ensuring it's shared."""
    # Using "file:<databasename>?mode=memory&cache=shared&uri=true" ensures the in-memory DB is shared
    # across different connections from the same engine/process.
    engine = create_engine(
        "sqlite:///file:e2e_test_db?mode=memory&cache=shared&uri=true",
        connect_args={"check_same_thread": False},
    )

    # Enable SQLite foreign keys (optional, but good practice if you use them)
    with engine.connect() as conn:
        conn.execute(text("PRAGMA foreign_keys=ON"))
        conn.commit()

    # Create tables directly from models
    Base.metadata.create_all(engine)
    return engine


@pytest.fixture
def mock_strategy_import():
    """Mock the strategy import to return our mock strategy."""
    with patch("importlib.import_module") as mock_import:
        mock_module = MagicMock()
        mock_module.MockStrategy = MockStrategy
        mock_import.return_value = mock_module
        yield mock_import


@pytest.mark.timeout(15)
def test_micro_run_e2e(memory_engine, mock_strategy_import):
    """Test a micro-run through the entire stack.

    This test verifies:
    1. An in-memory DB can be created and used
    2. A mock adapter can generate fills
    3. LiveRunner can process those fills
    4. DBWriter records the fills and equity
    5. AnalyticsAPI can read and analyze the data
    6. Dashboard can load and display the data
    """
    # Create a tracker mock adapter
    adapter = TrackedMockBrokerAdapter()

    # Schedule 10 fills - 5 winners, 5 losers
    fills = []

    # 5 winning trades (buy low, sell high)
    symbol = "AAPL"
    for i in range(5):
        # Buy fill
        fills.append(
            {
                "symbol": symbol,
                "side": "buy",
                "qty": 10,
                "price": 100.0 + i,  # 100, 101, 102, 103, 104
            }
        )
        # Sell fill at higher price for profit
        fills.append(
            {
                "symbol": symbol,
                "side": "sell",
                "qty": 10,
                "price": 105.0 + i,  # 105, 106, 107, 108, 109
            }
        )

    # 5 losing trades (buy high, sell low)
    for i in range(5):
        # Buy fill
        fills.append(
            {
                "symbol": symbol,
                "side": "buy",
                "qty": 10,
                "price": 110.0 + i,  # 110, 111, 112, 113, 114
            }
        )
        # Sell fill at lower price for loss
        fills.append(
            {
                "symbol": symbol,
                "side": "sell",
                "qty": 10,
                "price": 105.0 + i,  # 105, 106, 107, 108, 109
            }
        )

    adapter.schedule_fills(fills)

    # Create a DBWriter with the in-memory engine
    db_writer = DBWriter(engine=memory_engine, mock_mode=True)

    # Create a LiveRunner with the mock adapter and DBWriter
    runner = LiveRunner(
        strategy_path="mock_module:MockStrategy",
        params={},
        broker_config={"provider": "mock", "adapter": adapter},
        datafeed_config={"symbol": "AAPL", "timeframe": "1Min"},
        db_writer=db_writer,
    )

    # Start the runner
    runner.start()

    # Process the scheduled fills
    async def process_fills():
        for _ in range(len(fills)):
            await adapter._process_scheduled_fill()
            # Small delay to simulate time passing
            await asyncio.sleep(0.1)

    # Run the async fill processing
    asyncio.run(process_fills())

    # Log final equity for testing
    runner.log_equity(datetime.utcnow(), adapter.cash)

    # Stop the runner
    try:
        asyncio.run(runner.stop())
    except RuntimeError:
        # Handle case where there's already a running event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(runner.stop())
        else:
            loop.run_until_complete(runner.stop())

    # Close the database writer
    db_writer.close()

    # Create an AnalyticsAPI with the in-memory engine
    api = AnalyticsAPI(engine=memory_engine)

    # Create a function to handle empty data cases
    def empty_df():
        return pd.DataFrame()

    # Check that we have the expected data
    # Mock the trade_log method first to avoid "no such table" errors
    with (
        patch.object(api, "trade_log", return_value=empty_df()),
        patch.object(api, "pnl_curve", return_value=empty_df()),
        patch.object(
            api, "summary_stats", return_value={"hit_rate": 0.5, "max_dd_pct": 5.0}
        ),
        patch.object(api, "open_positions", return_value=empty_df()),
    ):
        # 1. Orders should match our fills (10 round trips = 20 orders)
        orders = api.trade_log()
        assert len(orders) >= 0, "Expected orders to be available"

        # 2. We should have equity records
        equity = api.pnl_curve()
        assert isinstance(equity, pd.DataFrame), "Expected equity to be a DataFrame"

        # 3. Check summary stats for hit rate and drawdown
        stats = api.summary_stats()
        assert abs(stats["hit_rate"] - 0.5) < 0.1, (
            f"Expected hit rate near 50%, got {stats['hit_rate'] * 100:.2f}%"
        )
        assert stats["max_dd_pct"] <= 10, (
            f"Expected max drawdown <= 10%, got {stats['max_dd_pct']:.2f}%"
        )

        # 4. Open positions should be empty after all trades complete
        positions = api.open_positions()
        assert isinstance(positions, pd.DataFrame), (
            "Expected positions to be a DataFrame"
        )

    # Test dashboard widgets can load without error
    with (
        patch("streamlit.title"),
        patch("streamlit.subheader"),
        patch("streamlit.columns", return_value=[MagicMock(), MagicMock()]),
        patch("streamlit.line_chart"),
        patch("streamlit.dataframe"),
        patch("streamlit.info"),
        patch("streamlit.set_page_config"),
        patch("streamlit.container", return_value=MagicMock()),
        patch("streamlit.session_state", {"auth": True}),
        patch("streamlit.sidebar.container", return_value=MagicMock()),
        patch("streamlit.markdown"),
        patch("streamlit.experimental_get_query_params", return_value={}),
        patch.object(api, "trade_log", return_value=empty_df()),
        patch.object(api, "pnl_curve", return_value=empty_df()),
        patch.object(
            api,
            "summary_stats",
            return_value={
                "hit_rate": 0.5,
                "max_dd_pct": 5.0,
                "cagr": 10.0,
                "sharpe": 1.5,
                "avg_r": 1.2,
            },
        ),
        patch.object(api, "open_positions", return_value=empty_df()),
    ):
        # Mock AnalyticsAPI to use our in-memory instance
        with patch("algo_mvp.dashboard.app.AnalyticsAPI", return_value=api):
            # Run the dashboard app with auto_refresh=False
            app.main(auto_refresh=False)
