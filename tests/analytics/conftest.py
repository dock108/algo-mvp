"""Test fixtures for analytics module tests."""

import datetime
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from algo_mvp.db.models import Base, Order, Fill, Equity


@pytest.fixture
def test_engine():
    """Create an in-memory SQLite engine for testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return engine


@pytest.fixture
def test_session(test_engine):
    """Create a database session for testing."""
    SessionLocal = sessionmaker(bind=test_engine)
    session = SessionLocal()
    yield session
    session.close()


@pytest.fixture
def seed_test_data(test_session):
    """Seed test data for analytics tests."""
    # Create some orders
    orders = [
        Order(
            broker_order_id="order1",
            symbol="AAPL",
            side="buy",
            order_type="limit",
            qty=10,
            limit_price=150.0,
            status="filled",
            created_at=datetime.datetime(2025, 5, 1, 10, 0, 0),
        ),
        Order(
            broker_order_id="order2",
            symbol="AAPL",
            side="sell",
            order_type="limit",
            qty=10,
            limit_price=160.0,
            status="filled",
            created_at=datetime.datetime(2025, 5, 1, 14, 0, 0),
        ),
        Order(
            broker_order_id="order3",
            symbol="MSFT",
            side="buy",
            order_type="limit",
            qty=5,
            limit_price=250.0,
            status="filled",
            created_at=datetime.datetime(2025, 5, 2, 10, 0, 0),
        ),
        Order(
            broker_order_id="order4",
            symbol="MSFT",
            side="sell",
            order_type="limit",
            qty=2,
            limit_price=260.0,
            status="filled",
            created_at=datetime.datetime(2025, 5, 2, 15, 0, 0),
        ),
        Order(
            broker_order_id="order5",
            symbol="GOOG",
            side="buy",
            order_type="limit",
            qty=2,
            limit_price=1500.0,
            status="filled",
            created_at=datetime.datetime(2025, 5, 3, 10, 0, 0),
        ),
    ]
    test_session.add_all(orders)
    test_session.flush()  # Flush to get IDs

    # Create some fills
    fills = [
        Fill(
            order_id=1,
            fill_qty=10,
            fill_price=150.0,
            commission=1.0,
            filled_at=datetime.datetime(2025, 5, 1, 10, 1, 0),
        ),
        Fill(
            order_id=2,
            fill_qty=10,
            fill_price=160.0,
            commission=1.0,
            filled_at=datetime.datetime(2025, 5, 1, 14, 1, 0),
        ),
        Fill(
            order_id=3,
            fill_qty=5,
            fill_price=250.0,
            commission=1.0,
            filled_at=datetime.datetime(2025, 5, 2, 10, 1, 0),
        ),
        Fill(
            order_id=4,
            fill_qty=2,
            fill_price=260.0,
            commission=0.5,
            filled_at=datetime.datetime(2025, 5, 2, 15, 1, 0),
        ),
        Fill(
            order_id=5,
            fill_qty=2,
            fill_price=1500.0,
            commission=2.0,
            filled_at=datetime.datetime(2025, 5, 3, 10, 1, 0),
        ),
    ]
    test_session.add_all(fills)

    # Create equity snapshots
    equity_snapshots = [
        Equity(
            timestamp=datetime.datetime(2025, 5, 1, 9, 0, 0),
            equity=10000.0,
        ),
        Equity(
            timestamp=datetime.datetime(2025, 5, 1, 11, 0, 0),
            equity=10050.0,
        ),
        Equity(
            timestamp=datetime.datetime(2025, 5, 1, 16, 0, 0),
            equity=10100.0,
        ),
        Equity(
            timestamp=datetime.datetime(2025, 5, 2, 9, 0, 0),
            equity=10150.0,
        ),
        Equity(
            timestamp=datetime.datetime(2025, 5, 2, 16, 0, 0),
            equity=10200.0,
        ),
        Equity(
            timestamp=datetime.datetime(2025, 5, 3, 9, 0, 0),
            equity=10250.0,
        ),
        Equity(
            timestamp=datetime.datetime(2025, 5, 3, 16, 0, 0),
            equity=10300.0,
        ),
    ]
    test_session.add_all(equity_snapshots)

    test_session.commit()
    return test_session
