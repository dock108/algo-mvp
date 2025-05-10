from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Order:
    id: str  # Broker-assigned ID
    symbol: str
    qty: float
    side: str  # 'buy' or 'sell'
    order_type: str  # 'market', 'limit', 'stop', etc.
    status: str = (
        "new"  # 'new', 'submitted', 'filled', 'partially_filled', 'canceled', 'expired', 'rejected'
    )
    filled_qty: float = 0.0
    avg_fill_price: Optional[float] = None
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    # We might need a client_order_id if we want to tag orders locally before submission
    # client_order_id: Optional[str] = None


@dataclass
class Position:
    symbol: str
    qty: float  # Can be positive (long) or negative (short)
    avg_entry_price: float
    # current_price: Optional[float] = None # This might be better handled by a separate market data feed
    # unrealized_pnl: Optional[float] = None # Can be calculated from avg_entry_price, qty, and current_price
    # realized_pnl: float = 0.0 # Should be updated upon closing/reducing position
    # last_trade_price: Optional[float] = None # Price of the last fill affecting this position
    # Add other relevant fields from Alpaca's Position entity if needed
    # e.g., cost_basis, market_value, etc.
