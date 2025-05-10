from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from algo_mvp.live.models import Order, Position


class BrokerAdapterBase(ABC):
    """Abstract base class for broker adapters.

    All broker adapters should implement these methods.
    """

    @abstractmethod
    def connect(self):
        """Connect to the broker API."""
        pass

    @abstractmethod
    def submit_order(
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
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order by ID."""
        pass

    @abstractmethod
    def get_cash(self) -> Dict[str, float]:
        """Get the current cash balance by currency."""
        pass

    @abstractmethod
    def get_positions(self) -> List[Position]:
        """Get the current positions."""
        pass

    def _raise_not_connected_error(self):
        """Helper method to raise a consistent error when not connected."""
        raise RuntimeError("Adapter not connected – call connect() first.")
