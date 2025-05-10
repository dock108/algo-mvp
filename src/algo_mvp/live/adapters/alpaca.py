import os
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
from tenacity import retry, stop_after_attempt, wait_exponential

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
        import threading

        self.stream_thread = threading.Thread(target=self.stream.run)
        self.stream_thread.daemon = (
            True  # Allow the thread to exit when the main thread exits
        )
        self.stream_thread.start()

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
    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    def submit_order(
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

        try:
            alpaca_order: AlpacaOrderModel = self.client.submit_order(order_data=req)
            return self._map_order(alpaca_order)
        except Exception as exc:  # pylint: disable=broad-except
            # tenacity will retry; on final failure returns None
            print(f"Alpaca submit_order error: {exc}")
            raise  # Let tenacity retry this

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

    # ------------------------------------------------------------------
    # Cancel / cash / positions
    # ------------------------------------------------------------------
    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    def cancel_order(self, order_id: str) -> bool:
        if self.client is None:
            raise RuntimeError("Adapter not connected.")
        try:
            self.client.cancel_order_by_id(order_id)
            return True
        except Exception as exc:  # pylint: disable=broad-except
            print(f"Cancel order error: {exc}")
            raise

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    def get_cash(self) -> dict[str, float]:
        """Retrieves cash balance."""
        if not self.client:
            self._raise_not_connected_error()
        account = self.client.get_account()
        return {"USD": float(account.cash)}

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    def get_positions(self) -> List[Position]:
        if self.client is None:
            raise RuntimeError("Adapter not connected.")
        alpaca_positions: List[AlpacaPositionModel] = self.client.get_all_positions()
        return [
            Position(
                symbol=p.symbol,
                qty=float(p.qty),
                avg_entry_price=float(p.avg_entry_price),
            )
            for p in alpaca_positions
        ]
