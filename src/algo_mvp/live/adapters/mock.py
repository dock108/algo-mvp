import asyncio
from collections import defaultdict


class MockBrokerAdapter:
    def __init__(self):
        self.cash = 100000.0
        self.positions = defaultdict(float)
        self.call_counts = defaultdict(int)

    async def submit_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str,
        limit_price: float | None = None,
    ):
        self.call_counts["submit_order"] += 1
        print(
            f"MockBrokerAdapter: Submitting order - {side} {qty} {symbol} @ {limit_price if limit_price else 'market'}"
        )
        # Simulate order fill after a short delay
        await asyncio.sleep(0.1)
        # Simulate a simple fill logic
        if side == "buy":
            self.positions[symbol] += qty
            # Crude price simulation; in a real mock, you might want a mock price feed
            price = limit_price if limit_price else 100.0  # Assume a fill price
            self.cash -= qty * price
        elif side == "sell":
            self.positions[symbol] -= qty
            price = limit_price if limit_price else 100.0
            self.cash += qty * price

        print(f"MockBrokerAdapter: Order filled - {side} {qty} {symbol}")
        # In a real adapter, you'd return an order object or ID
        return {"id": "mock_order_id", "status": "filled"}

    async def cancel_order(self, order_id: str):
        self.call_counts["cancel_order"] += 1
        print(f"MockBrokerAdapter: Cancelling order {order_id}")
        await asyncio.sleep(0.05)
        print(f"MockBrokerAdapter: Order {order_id} cancelled")
        return {"id": order_id, "status": "cancelled"}

    async def get_cash(self):
        self.call_counts["get_cash"] += 1
        return self.cash

    async def get_positions(self):
        self.call_counts["get_positions"] += 1
        return dict(self.positions)  # Return a copy
