# Placeholder for MockBrokerAdapter

import asyncio
import uuid
from collections import defaultdict
from typing import Any, Callable, DefaultDict, Union
from unittest.mock import MagicMock

import backtrader as bt

# It's good practice to define Order and Trade data structures
# if not using Backtrader's directly for external communication.
# For now, we'll use Backtrader's Order and Trade objects for simplicity.


class MockBrokerAdapter:
    def __init__(self, event_loop: asyncio.AbstractEventLoop | None = None):
        self.loop = event_loop or asyncio.get_event_loop()
        self.cash: float = 100000.0
        self.positions: DefaultDict[str, float] = defaultdict(float)
        self.open_orders: dict[str, bt.Order] = {}

        # Call counters
        self.submit_order_calls: int = 0
        self.cancel_order_calls: int = 0
        self.get_cash_calls: int = 0
        self.get_positions_calls: int = 0

        # Callbacks
        self.on_trade: Callable[[bt.Order, bt.Trade], None] | None = None
        self.on_order_status_change: Callable[[bt.Order], None] | None = (
            None  # For LiveRunner
        )

    def _generate_order_id(self) -> str:
        return str(uuid.uuid4())

    async def submit_order(
        self,
        owner: Any,  # Typically the strategy instance
        data: bt.DataBase,  # Data feed associated with the order
        side: str,  # 'buy' or 'sell'
        exectype: str,  # 'market', 'limit', 'stop', etc. (Backtrader order types)
        size: float,
        price: float | None = None,  # For limit/stop orders
        valid: str | None = None,  # e.g. bt.Order.DAY, bt.Order.GTC
        params: dict | None = None,  # Additional params
    ) -> Union[bt.Order, MagicMock]:
        self.submit_order_calls += 1

        # Simulate order properties for the mock object
        mock_order = MagicMock(
            spec=bt.Order
        )  # spec helps with asserting attributes later
        mock_order.ref = self._generate_order_id()  # Give it a unique ref
        mock_order.owner = owner
        mock_order.data = data
        mock_order.size = size
        mock_order.price = price
        mock_order.created = MagicMock()
        # Access datetime safely for preloaded PandasData (idx might be -1 initially)
        # data.lines.datetime[0] gives the float timestamp of the first bar if preloaded.
        # bt.num2date converts this float to a Python datetime object.
        current_bar_datetime = None
        if (
            hasattr(data, "lines")
            and hasattr(data.lines, "datetime")
            and len(data.lines.datetime) > 0
        ):
            try:
                current_bar_datetime = bt.num2date(data.lines.datetime[0])
            except IndexError:  # Should not happen if len > 0, but as a safeguard
                pass
        mock_order.created.dt = current_bar_datetime
        mock_order.created.size = size
        mock_order.created.price = (
            price
            if price is not None
            else (
                data.lines.close[0]
                if hasattr(data, "lines")
                and hasattr(data.lines, "close")
                and len(data.lines.close) > 0
                else 0
            )
        )

        mock_order.isbuy = MagicMock(return_value=side.lower() == "buy")
        mock_order.issell = MagicMock(return_value=side.lower() == "sell")

        # Simulate status progression
        mock_order.status = bt.Order.Submitted
        if self.on_order_status_change:  # Potentially an async callback
            # If on_order_status_change is an async def, it needs to be awaited
            # For now, assuming it's a regular callable or an AsyncMock that handles it
            if asyncio.iscoroutinefunction(
                self.on_order_status_change
            ) or asyncio.iscoroutine(self.on_order_status_change):
                asyncio.ensure_future(
                    self.on_order_status_change(mock_order), loop=self.loop
                )  # Check loop usage
            else:
                self.on_order_status_change(mock_order)

        await asyncio.sleep(0.01)  # Simulate network latency

        current_price = (
            data.lines.close[0]
            if hasattr(data, "lines")
            and hasattr(data.lines, "close")
            and len(data.lines.close) > 0
            else (price or 0)
        )

        if exectype.lower() == "market":
            mock_order.status = bt.Order.Completed
            mock_order.executed = MagicMock()
            mock_order.executed.price = current_price
            mock_order.executed.size = size
            mock_order.executed.dt = mock_order.created.dt  # Or slightly later
            # Update cash and positions
            cost = mock_order.executed.price * size
            if mock_order.isbuy():
                self.cash -= cost
                self.positions[
                    data._name if hasattr(data, "_name") else "UNKNOWN_ASSET"
                ] += size
            else:
                self.cash += cost
                self.positions[
                    data._name if hasattr(data, "_name") else "UNKNOWN_ASSET"
                ] -= size

            if self.on_order_status_change:
                self.on_order_status_change(mock_order)

            # Simulate trade creation for on_trade callback
            mock_trade = MagicMock(spec=bt.Trade)
            mock_trade.ref = self._generate_order_id() + "_trade"
            mock_trade.order = mock_order
            mock_trade.data = data
            mock_trade.size = mock_order.executed.size
            mock_trade.price = mock_order.executed.price
            mock_trade.value = mock_order.executed.size * mock_order.executed.price
            mock_trade.commission = 0.0  # Mock commission
            mock_trade.status = bt.Trade.Closed  # Assume closed for simplicity
            mock_trade.status_names = {bt.Trade.Closed: "Closed"}  # for logging

            if self.on_trade:  # Potentially an async callback
                if asyncio.iscoroutinefunction(self.on_trade) or asyncio.iscoroutine(
                    self.on_trade
                ):
                    asyncio.ensure_future(
                        self.on_trade(mock_order, mock_trade), loop=self.loop
                    )
                else:
                    self.on_trade(mock_order, mock_trade)

        elif exectype.lower() == "limit":
            mock_order.status = bt.Order.Accepted
            self.open_orders[mock_order.ref] = mock_order
            if self.on_order_status_change:  # Potentially an async callback
                if asyncio.iscoroutinefunction(
                    self.on_order_status_change
                ) or asyncio.iscoroutine(self.on_order_status_change):
                    asyncio.ensure_future(
                        self.on_order_status_change(mock_order), loop=self.loop
                    )
                else:
                    self.on_order_status_change(mock_order)
        else:
            print(f"Unsupported exectype for mock order simulation: {exectype}")
            mock_order.status = bt.Order.Rejected  # Or some other appropriate status
            if self.on_order_status_change:  # Potentially an async callback
                if asyncio.iscoroutinefunction(
                    self.on_order_status_change
                ) or asyncio.iscoroutine(self.on_order_status_change):
                    asyncio.ensure_future(
                        self.on_order_status_change(mock_order), loop=self.loop
                    )
                else:
                    self.on_order_status_change(mock_order)

        return mock_order

    async def cancel_order(self, order_ref: str) -> Union[bt.Order, MagicMock, None]:
        self.cancel_order_calls += 1
        order_to_cancel = self.open_orders.get(order_ref)
        if order_to_cancel:  # order_to_cancel is a MagicMock
            # Check mock_order.status if it was set, otherwise assume cancellable states for mock
            # if order_to_cancel.status in [bt.Order.Accepted, bt.Order.Submitted, bt.Order.Partial]:
            order_to_cancel.status = bt.Order.Canceled
            if self.on_order_status_change:  # Potentially an async callback
                if asyncio.iscoroutinefunction(
                    self.on_order_status_change
                ) or asyncio.iscoroutine(self.on_order_status_change):
                    asyncio.ensure_future(
                        self.on_order_status_change(order_to_cancel), loop=self.loop
                    )
                else:
                    self.on_order_status_change(order_to_cancel)
            del self.open_orders[order_ref]
            return order_to_cancel
        return None

    async def get_cash(self) -> float:
        self.get_cash_calls += 1
        return self.cash

    async def get_positions(self) -> dict[str, float]:
        self.get_positions_calls += 1
        return dict(self.positions)  # Return a copy

    def set_on_trade_callback(self, callback: Callable[[bt.Order, bt.Trade], None]):
        self.on_trade = callback

    def set_on_order_status_change_callback(self, callback: Callable[[bt.Order], None]):
        self.on_order_status_change = callback

    def get_backtrader_broker(self) -> bt.BrokerBase:
        """
        Returns a Backtrader compatible broker interface that uses this adapter.
        This is a simplified version. A full version would need to handle
        order notifications, cash/position updates correctly within Backtrader's flow.
        For now, LiveRunner will interact directly with the adapter's methods.
        This method is more for conceptual completeness or if we were to
        make Backtrader's Cerebro use this adapter more deeply.
        """
        # This is a placeholder. In a more integrated setup, we might create
        # a custom bt.BrokerBase subclass that calls these async adapter methods.
        # For the current LiveRunner design, LiveRunner directly calls adapter methods.
        # However, the strategy will create orders that Cerebro's default broker part
        # might try to handle. We need to ensure Cerebro's broker doesn't conflict.

        # A simple way is to use a BackBroker and override its methods,
        # but that's complex. The current approach is that strategy calls
        # `self.buy()`/`self.sell()`, which the `LiveRunner` will intercept
        # or the `LiveRunner`'s custom broker will handle.

        # For now, let's assume LiveRunner's cerebro might be given a very basic broker
        # or its broker interaction is managed more directly by LiveRunner.
        # The simplest approach is for LiveRunner to provide a `bt.Store`
        # and handle order execution via this adapter when `buy/sell` is called in strategy.

        # For this skeleton, we will assume LiveRunner doesn't replace cerebro.broker
        # with one that directly calls this async adapter yet. Instead, LiveRunner
        # will facilitate the calls based on strategy actions.
        # So, returning a simple BackBroker and relying on LiveRunner to bridge.
        mock_bt_broker = bt.brokers.BackBroker()
        mock_bt_broker.setcash(self.cash)
        # How to link its buy/sell to our async submit_order?
        # This is where the complexity lies if not building a full custom bt.BrokerBase.

        # For the skeleton, the LiveRunner will not use this method to get a full broker.
        # It will create a Cerebro instance and then directly use the adapter.
        print(
            "WARNING: get_backtrader_broker() in MockBrokerAdapter is conceptual and not fully implemented for direct Cerebro use."
        )
        return mock_bt_broker
