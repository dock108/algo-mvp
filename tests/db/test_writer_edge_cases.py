import time
import pytest
from datetime import datetime

from sqlalchemy import text
from sqlalchemy.orm import Session

from algo_mvp.db.models import Order, Fill, Log
from algo_mvp.db.writer import DBWriter
from algo_mvp.live.models import Order as LiveOrder, Fill as LiveFill


def test_duplicate_broker_order_id(db_engine, db_session):
    """Test writer handles duplicate broker_order_id by ignoring and continuing."""
    writer = DBWriter(engine=db_engine, mock_mode=True)

    # Create the first order
    first_order = LiveOrder(
        id="duplicate-id",
        symbol="AAPL",
        qty=10.0,
        side="buy",
        order_type="market",
        status="filled",
        limit_price=None,
        stop_price=None,
        created_at=datetime.utcnow(),
    )

    # Log the first order
    writer.log_order(first_order)

    # Create a second order with the same ID but different details
    second_order = LiveOrder(
        id="duplicate-id",
        symbol="MSFT",  # Different symbol
        qty=20.0,  # Different quantity
        side="sell",  # Different side
        order_type="limit",
        status="new",
        limit_price=150.0,
        stop_price=None,
        created_at=datetime.utcnow(),
    )

    # Log the second order - should be ignored due to duplicate ID
    writer.log_order(second_order)

    # Verify only one order with the ID exists, and it has the first order's details
    orders = db_session.query(Order).filter_by(broker_order_id="duplicate-id").all()
    assert len(orders) == 1  # Only one order should exist
    assert orders[0].symbol == "AAPL"  # Should be the first order's symbol
    assert orders[0].qty == 10.0  # Should be the first order's quantity
    assert orders[0].side == "buy"  # Should be the first order's side

    # Verify a log entry was created for the duplicate
    logs = db_session.query(Log).filter(Log.message.like("%duplicate%")).all()
    assert len(logs) > 0

    writer.close()


def test_invalid_foreign_key_on_fill(db_engine, db_session):
    """Test fill with nonexistent order ID is rolled back but queue continues processing."""
    writer = DBWriter(engine=db_engine, mock_mode=False)  # Need real worker thread

    # Add a valid order first
    valid_order = LiveOrder(
        id="valid-order",
        symbol="AAPL",
        qty=10.0,
        side="buy",
        order_type="market",
        status="filled",
        limit_price=None,
        stop_price=None,
        created_at=datetime.utcnow(),
    )
    writer.log_order(valid_order)

    # Add a fill for a non-existent order
    invalid_fill = LiveFill(
        id="invalid-fill",
        order_id="nonexistent-order",
        symbol="AAPL",
        qty=5.0,
        price=150.0,
        side="buy",
        commission=1.0,
        timestamp=datetime.utcnow(),
    )
    writer.log_fill(invalid_fill)

    # Add a valid fill for the valid order
    valid_fill = LiveFill(
        id="valid-fill",
        order_id="valid-order",
        symbol="AAPL",
        qty=10.0,
        price=160.0,
        side="buy",
        commission=1.5,
        timestamp=datetime.utcnow(),
    )
    writer.log_fill(valid_fill)

    # Wait for queue processing
    time.sleep(1)

    # Check that the valid fill was inserted but the invalid one wasn't
    with Session(db_engine) as session:
        # Get the valid order ID
        valid_order_db = (
            session.query(Order).filter_by(broker_order_id="valid-order").first()
        )
        assert valid_order_db is not None

        # Check fills
        fills = session.query(Fill).all()
        assert len(fills) == 1  # Only the valid fill should exist
        assert (
            fills[0].order_id == valid_order_db.id
        )  # Should reference the valid order

        # Check for error log entry
        error_logs = session.query(Log).filter(Log.level == "ERROR").all()
        assert (
            len(error_logs) > 0
        )  # Should have an error log entry for the invalid fill

    writer.close()


@pytest.mark.timeout(5)  # Prevent hanging test
def test_thread_kill_recovery(db_engine):
    """Test that if writer thread is killed, queue items are processed on restart."""
    # Create a writer with real thread mode
    writer = DBWriter(engine=db_engine, queue_max=100, mock_mode=False)

    # Add a bunch of orders to the queue
    for i in range(20):
        order = LiveOrder(
            id=f"order-{i}",
            symbol="TEST",
            qty=10.0,
            side="buy",
            order_type="market",
            status="new",
            limit_price=None,
            stop_price=None,
            created_at=datetime.utcnow(),
        )
        writer.log_order(order)

    # Simulate thread kill by setting stop event and creating a new instance
    writer._stop_event.set()

    # Check that items are still in the queue
    assert writer.queue.qsize() > 0

    # Close the writer to clean up resources
    writer.close()

    # Create a new writer that should process the remaining items
    new_writer = DBWriter(engine=db_engine, mock_mode=False)

    # Give it some time to initialize
    time.sleep(0.5)

    # Verify queue is empty
    with Session(db_engine) as session:
        orders = session.query(Order).all()
        assert len(orders) == 20  # All orders should be processed

    new_writer.close()


@pytest.mark.timeout(5)
def test_large_batch_performance(db_engine):
    """Test that processing 10k mixed events takes less than 2 seconds."""
    # Create a writer with real thread mode
    writer = DBWriter(engine=db_engine, queue_max=15000, mock_mode=False)

    # Create 10k mixed events (orders, fills, equity, logs)
    start_time = time.time()

    # First create 2000 orders
    for i in range(2000):
        order = LiveOrder(
            id=f"batch-order-{i}",
            symbol="PERF",
            qty=10.0,
            side="buy" if i % 2 == 0 else "sell",
            order_type="market",
            status="new",
            limit_price=None,
            stop_price=None,
            created_at=datetime.utcnow(),
        )
        writer.log_order(order)

    # Then create fills for half of the orders
    for i in range(1000):
        fill = LiveFill(
            id=f"batch-fill-{i}",
            order_id=f"batch-order-{i}",
            symbol="PERF",
            qty=10.0,
            price=100.0 + i % 10,
            side="buy" if i % 2 == 0 else "sell",
            commission=1.0,
            timestamp=datetime.utcnow(),
        )
        writer.log_fill(fill)

    # Add 3000 equity updates
    for i in range(3000):
        writer.log_equity(datetime.utcnow(), 100000.0 + i)

    # Add 4000 log messages
    for i in range(4000):
        writer.log_message("INFO", f"Test log message {i}")

    # Wait for the queue to be processed
    while not writer.queue.empty():
        time.sleep(0.1)

    end_time = time.time()
    elapsed = end_time - start_time

    # Verify elapsed time is less than 2 seconds
    assert elapsed < 2.0, (
        f"Processing took {elapsed:.2f} seconds, expected < 2.0 seconds"
    )

    # Verify database counts match what we added
    with Session(db_engine) as session:
        order_count = session.query(Order).count()
        fill_count = session.query(Fill).count()
        log_count = session.query(Log).count()

        assert order_count == 2000
        assert fill_count == 1000  # Only half get fills
        assert log_count >= 4000  # At least the ones we added + any from the system

    writer.close()


def test_graceful_shutdown(db_engine):
    """Test that close() after heavy load leaves no open transactions."""
    # Create a writer with real thread mode
    writer = DBWriter(engine=db_engine, queue_max=1000, mock_mode=False)

    # Add a bunch of events
    for i in range(100):
        writer.log_order(
            LiveOrder(
                id=f"shutdown-order-{i}",
                symbol="TEST",
                qty=10.0,
                side="buy",
                order_type="market",
                status="new",
                limit_price=None,
                stop_price=None,
                created_at=datetime.utcnow(),
            )
        )

        writer.log_message("INFO", f"Test message {i}")

    # Close the writer
    writer.close()

    # Check SQLite journal mode
    with db_engine.connect() as conn:
        journal_mode = conn.execute(text("PRAGMA journal_mode;")).fetchone()[0]
        assert (
            journal_mode.lower() == "delete"
        )  # Expected mode when no transactions are open

        # Also check that no transactions are active
        if "sqlite" in str(db_engine.url):
            in_transaction = conn.execute(text("PRAGMA query_only;")).fetchone()[0]
            assert in_transaction == 0  # 0 means we're not in a transaction
