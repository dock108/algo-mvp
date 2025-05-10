from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from algo_mvp.live.models import Order, Position


class BrokerAdapterBase(ABC):
    """Abstract base class for broker adapters.

    All broker adapters should implement these methods.
    """

    def __init__(self, runner):
        self.runner = runner

    @abstractmethod
    def connect(self):
        """Connect to the broker API."""
        pass

    @abstractmethod
    async def submit_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
    ) -> Optional[Order]:
        """Submit an order to the broker."""
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order by ID."""
        pass

    @abstractmethod
    async def get_cash(self) -> Dict[str, float]:
        """Get the current cash balance by currency."""
        pass

    @abstractmethod
    async def get_positions(self) -> List[Position]:
        """Get the current positions."""
        pass

    @abstractmethod
    def close(self):
        """Close all connections and clean up resources."""
        pass

    def _raise_not_connected_error(self):
        """Helper method to raise a consistent error when not connected."""
        raise RuntimeError("Adapter not connected – call connect() first.")
