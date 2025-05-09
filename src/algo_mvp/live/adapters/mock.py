import asyncio
from collections import defaultdict


class MockBrokerAdapter:
    def __init__(self):
        self.cash = 100000.0
        self.positions = defaultdict(float)
        # self.call_counts = defaultdict(int) # Deprecated

        # Individual call counters
        self.submit_order_calls = 0
        self.cancel_order_calls = 0
        self.get_cash_calls = 0
        self.get_positions_calls = 0
        # Add more if other methods are tracked

        self._order_id_counter = 0  # To generate unique mock order IDs

    async def submit_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str,
        limit_price: float | None = None,
    ):
        # self.call_counts['submit_order'] += 1
        self.submit_order_calls += 1
        # print(
        #     f"MockBrokerAdapter: Submitting order - {side} {qty} {symbol} @ {limit_price if limit_price else 'market'}"
        # )
        await asyncio.sleep(0.01)  # Simulate network latency
        self._order_id_counter += 1
        order_id = f"mock_order_id_{self._order_id_counter}"

        # Simulate a simple fill logic
        price_to_use = (
            limit_price if order_type == "limit" and limit_price is not None else 100.0
        )  # Default for market orders

        if side == "buy":
            cost = qty * price_to_use
            if self.cash >= cost:
                self.cash -= cost
                self.positions[symbol] += qty
                # Crude price simulation; in a real mock, you might want a mock price feed
                # price = limit_price if limit_price else 100.0  # Assume a fill price
                # self.cash -= qty * price
                # print(f"MockBrokerAdapter: Bought {qty} of {symbol} @ {price_to_use}. Cash: {self.cash:.2f}")
                return {
                    "id": order_id,
                    "status": "filled",
                    "symbol": symbol,
                    "qty": qty,
                    "price": price_to_use,
                }
            else:
                # print(f"MockBrokerAdapter: Insufficient cash to buy {qty} of {symbol}. Cash: {self.cash:.2f}")
                return {
                    "id": order_id,
                    "status": "rejected",
                    "reason": "insufficient funds",
                }
        elif side == "sell":
            if self.positions[symbol] >= qty:
                self.cash += qty * price_to_use
                self.positions[symbol] -= qty
                if self.positions[symbol] == 0:
                    del self.positions[symbol]  # Clean up if position is zero
                # print(f"MockBrokerAdapter: Sold {qty} of {symbol} @ {price_to_use}. Cash: {self.cash:.2f}")
                return {
                    "id": order_id,
                    "status": "filled",
                    "symbol": symbol,
                    "qty": qty,
                    "price": price_to_use,
                }
            else:
                # print(f"MockBrokerAdapter: Insufficient position to sell {qty} of {symbol}. Position: {self.positions[symbol]}")
                return {
                    "id": order_id,
                    "status": "rejected",
                    "reason": "insufficient position",
                }
        return {"id": order_id, "status": "error", "reason": "invalid order side"}

    async def cancel_order(self, order_id: str):
        # self.call_counts['cancel_order'] += 1
        self.cancel_order_calls += 1
        # print(f"MockBrokerAdapter: Cancelling order {order_id}")
        await asyncio.sleep(0.01)  # Simulate network latency
        return {
            "id": order_id,
            "status": "cancelled",
        }  # Assume cancellation is always successful

    async def get_cash(self) -> float:
        # self.call_counts['get_cash'] += 1
        self.get_cash_calls += 1
        await asyncio.sleep(0.01)  # Simulate network latency
        return self.cash

    async def get_positions(self) -> dict:
        # self.call_counts['get_positions'] += 1
        self.get_positions_calls += 1
        await asyncio.sleep(0.01)  # Simulate network latency
        return dict(self.positions)  # Return a copy to prevent direct modification
