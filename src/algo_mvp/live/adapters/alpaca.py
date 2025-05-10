import os
import threading
from datetime import datetime, timezone
from typing import List, Optional

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.models import Order as AlpacaOrderModel
from alpaca.trading.models import Position as AlpacaPositionModel
from alpaca.trading.requests import (
    LimitOrderRequest,
    MarketOrderRequest,
    StopOrderRequest,
)
from alpaca.trading.stream import TradingStream
from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential

from algo_mvp.live.broker_adapter_base import BrokerAdapterBase
from algo_mvp.live.models import Order, Position


class AlpacaBrokerAdapter(BrokerAdapterBase):
    """Broker adapter using the modern `alpaca-py` SDK (Trading API v2). Works in paper mode only for now."""

    def __init__(self, live_runner):
        self.live_runner = live_runner
        self.paper_url = "https://paper-api.alpaca.markets"

        self.key_id: str | None = os.getenv("ALPACA_KEY_ID")
        self.secret_key: str | None = os.getenv("ALPACA_SECRET_KEY")
        self.base_url: str = os.getenv("ALPACA_BASE_URL", self.paper_url)
        self.paper: bool = self.base_url == self.paper_url  # crude check

        # Set in connect()
        self.client: Optional[TradingClient] = None
        self.stream: Optional[TradingStream] = None
        self.stream_thread: Optional[threading.Thread] = None
        self._is_running = False

    # ---------------------------------------------------------------------
    # Connection helpers
    # ---------------------------------------------------------------------
    def connect(self):
        if not self.key_id or not self.secret_key:
            raise ValueError(
                "ALPACA_KEY_ID and ALPACA_SECRET_KEY must be set in environment variables."
            )

        # REST client
        self.client = TradingClient(self.key_id, self.secret_key, paper=self.paper)

        # Web-socket stream for order updates (async)
        self.stream = TradingStream(self.key_id, self.secret_key, paper=self.paper)

        # Register callbacks
        self.stream.subscribe_trade_updates(self._handle_trade_update)

        # Run the stream in a background thread to avoid blocking
        self._is_running = True
        self.stream_thread = threading.Thread(target=self._run_stream)
        self.stream_thread.daemon = (
            True  # Allow the thread to exit when the main thread exits
        )
        self.stream_thread.start()

    def _run_stream(self):
        """Run the stream in a loop that can be stopped."""
        try:
            while self._is_running and self.stream:
                self.stream.run(
                    timeout=60
                )  # Run with timeout to allow for clean shutdown
        except Exception as e:
            if self.live_runner and self.live_runner.on_error and self._is_running:
                self.live_runner.on_error(f"Stream error: {e}")

    def close(self):
        """Close all connections and clean up resources."""
        self._is_running = False

        # Stop the stream
        if self.stream:
            try:
                self.stream.stop()
                self.stream = None
            except Exception as e:
                print(f"Error stopping stream: {e}")

        # Wait for thread to join
        if self.stream_thread and self.stream_thread.is_alive():
            try:
                self.stream_thread.join(timeout=5)  # Give it 5 seconds to finish
                if self.stream_thread.is_alive():
                    print("Warning: Stream thread did not exit cleanly")
            except Exception as e:
                print(f"Error joining stream thread: {e}")

        # Clear client reference
        self.client = None
        self.stream_thread = None

    async def _handle_trade_update(self, data):
        """Translate Alpaca TradeUpdate object → internal Order, call LiveRunner callbacks."""
        try:
            event = data.event  # e.g. "fill", "partial_fill", "canceled"
            alpaca_order = data.order

            # Map event → status
            status_map = {
                "fill": "filled",
                "partial_fill": "partially_filled",
                "canceled": "canceled",
                "rejected": "rejected",
            }
            status = status_map.get(event, alpaca_order.status)

            mapped_order = self._map_order(alpaca_order, override_status=status)

            if event in ("fill", "partial_fill") and self.live_runner:
                self.live_runner.on_trade(mapped_order)
            elif "error" in event.lower() and self.live_runner:
                self.live_runner.on_error(
                    f"Alpaca trade stream error: {event}", {"raw": data.dict()}
                )
        except Exception as exc:  # pylint: disable=broad-except
            if self.live_runner:
                self.live_runner.on_error(
                    "Exception in trade update handler", {"exception": str(exc)}
                )

    # ------------------------------------------------------------------
    # Order helpers
    # ------------------------------------------------------------------
    async def submit_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
    ) -> Optional[Order]:
        if self.client is None:
            raise RuntimeError("Adapter not connected – call connect() first.")

        # Build request object based on order_type
        side_enum = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
        tif = TimeInForce.DAY

        if order_type == "market":
            req = MarketOrderRequest(
                symbol=symbol, qty=qty, side=side_enum, time_in_force=tif
            )
        elif order_type == "limit":
            if limit_price is None:
                raise ValueError("limit_price must be provided for limit orders")
            req = LimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side_enum,
                time_in_force=tif,
                limit_price=limit_price,
            )
        elif order_type == "stop":
            if stop_price is None:
                raise ValueError("stop_price must be provided for stop orders")
            req = StopOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side_enum,
                time_in_force=tif,
                stop_price=stop_price,
            )
        else:
            raise ValueError(f"Unsupported order_type '{order_type}'.")

        # Using async retry pattern
        retries = 0
        max_retries = 3

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=10),
        ):
            with attempt:
                try:
                    # Using the async version of the API
                    alpaca_order: AlpacaOrderModel = (
                        await self.client.submit_order_async(order_data=req)
                    )
                    return self._map_order(alpaca_order)
                except Exception as exc:  # pylint: disable=broad-except
                    retries += 1
                    print(
                        f"Alpaca submit_order_async error (attempt {retries}/{max_retries}): {exc}"
                    )
                    raise  # Let AsyncRetrying handle the retry

    async def cancel_order(self, order_id: str) -> bool:
        if self.client is None:
            raise RuntimeError("Adapter not connected.")

        # Using async retry pattern
        retries = 0
        max_retries = 3

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=10),
        ):
            with attempt:
                try:
                    await self.client.cancel_order_by_id_async(order_id)
                    return True
                except Exception as exc:  # pylint: disable=broad-except
                    retries += 1
                    print(
                        f"Cancel order error (attempt {retries}/{max_retries}): {exc}"
                    )
                    raise  # Let AsyncRetrying handle the retry

    async def get_cash(self) -> dict[str, float]:
        """Retrieves cash balance."""
        if not self.client:
            self._raise_not_connected_error()

        # Using async retry pattern
        retries = 0
        max_retries = 3

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=10),
        ):
            with attempt:
                try:
                    account = await self.client.get_account_async()
                    return {"USD": float(account.cash)}
                except Exception as exc:  # pylint: disable=broad-except
                    retries += 1
                    print(f"Get cash error (attempt {retries}/{max_retries}): {exc}")
                    raise  # Let AsyncRetrying handle the retry

    async def get_positions(self) -> List[Position]:
        if self.client is None:
            raise RuntimeError("Adapter not connected.")

        # Using async retry pattern
        retries = 0
        max_retries = 3

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=10),
        ):
            with attempt:
                try:
                    alpaca_positions: List[AlpacaPositionModel] = (
                        await self.client.get_all_positions_async()
                    )
                    return [
                        Position(
                            symbol=p.symbol,
                            qty=float(p.qty),
                            avg_entry_price=float(p.avg_entry_price),
                        )
                        for p in alpaca_positions
                    ]
                except Exception as exc:  # pylint: disable=broad-except
                    retries += 1
                    print(
                        f"Get positions error (attempt {retries}/{max_retries}): {exc}"
                    )
                    raise  # Let AsyncRetrying handle the retry

    def _map_order(
        self, alpaca_order: AlpacaOrderModel, override_status: Optional[str] = None
    ) -> Order:
        """Convert alpaca-py Order model → internal Order dataclass."""
        created_at = (
            alpaca_order.submitted_at
            or alpaca_order.created_at
            or datetime.now(timezone.utc)
        )
        updated_at = alpaca_order.updated_at or created_at
        return Order(
            id=str(alpaca_order.id),
            symbol=alpaca_order.symbol,
            qty=float(alpaca_order.qty),
            side=(
                alpaca_order.side.value
                if hasattr(alpaca_order.side, "value")
                else alpaca_order.side
            ),
            order_type=(
                alpaca_order.order_type.value
                if hasattr(alpaca_order.order_type, "value")
                else alpaca_order.order_type
            ),
            status=override_status or alpaca_order.status,
            filled_qty=float(alpaca_order.filled_qty or 0),
            avg_fill_price=(
                float(alpaca_order.filled_avg_price or 0)
                if alpaca_order.filled_avg_price
                else None
            ),
            limit_price=(
                float(alpaca_order.limit_price) if alpaca_order.limit_price else None
            ),
            stop_price=(
                float(alpaca_order.stop_price) if alpaca_order.stop_price else None
            ),
            created_at=created_at,
            updated_at=updated_at,
        )
