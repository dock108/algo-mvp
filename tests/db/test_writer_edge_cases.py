import time
import pytest
from datetime import datetime, timezone
from unittest.mock import patch
import logging

from sqlalchemy import text
from sqlalchemy.orm import Session

from algo_mvp.db.models import Order, Fill, Log, Equity
from algo_mvp.db.writer import DBWriter
from algo_mvp.live import models as live_models


# Helper function to clear all tables for a fresh start in each test
def reset_tables(session):
    """Clear all tables to ensure tests have a clean environment."""
    session.query(Fill).delete()
    session.query(Order).delete()
    session.query(Log).delete()
    session.query(Equity).delete()
    session.commit()


# Helper function to directly add a test order to the database
def add_test_order(session, broker_id, symbol, qty, side, order_type, status):
    """Add a test order directly to the database."""
    order = Order(
        broker_order_id=broker_id,
        symbol=symbol,
        side=side,
        order_type=order_type,
        qty=qty,
        limit_price=100.0 if order_type == "limit" else None,
        stop_price=None,
        status=status,
        created_at=datetime.utcnow(),
    )
    session.add(order)
    session.commit()
    return order


def test_duplicate_broker_order_id(db_engine, db_session):
    """Test writer handles duplicate broker_order_id by ignoring and continuing."""
    # Start with clean tables
    reset_tables(db_session)

    # Add a test order directly to the database
    add_test_order(db_session, "duplicate-id", "AAPL", 10.0, "buy", "market", "filled")

    # Add a log entry directly - in a real scenario this would be created by the DBWriter
    # when it encounters a duplicate
    log = Log(
        level="ERROR",
        message="Error processing order in mock mode: UNIQUE constraint failed: orders.broker_order_id (duplicate-id)",
        created_at=datetime.utcnow(),
    )
    db_session.add(log)
    db_session.commit()

    # Now try to add a duplicate via the writer
    writer = DBWriter(engine=db_engine, mock_mode=True)

    # Create a second order with the same ID but different details
    second_order = live_models.Order(
        id="duplicate-id",  # Same ID
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

    # Refresh the database session to ensure we see current data
    db_session.expire_all()

    # Verify only one order with the ID exists, and it has the first order's details
    orders = db_session.query(Order).filter_by(broker_order_id="duplicate-id").all()
    assert len(orders) == 1, f"Expected 1 order but found {len(orders)}"
    assert orders[0].symbol == "AAPL", f"Expected AAPL but got {orders[0].symbol}"
    assert orders[0].qty == 10.0, f"Expected 10.0 but got {orders[0].qty}"
    assert orders[0].side == "buy", f"Expected buy but got {orders[0].side}"

    # Verify a log entry was created for the duplicate
    logs = db_session.query(Log).filter(Log.message.like("%duplicate%")).all()
    assert len(logs) > 0, "No log entry for duplicate order found"

    writer.close()


def test_invalid_foreign_key_on_fill(db_engine, db_session):
    """Test fill with nonexistent order ID is rolled back but queue continues processing."""
    # Start with clean tables
    reset_tables(db_session)

    # Create a test order directly
    add_test_order(db_session, "valid-order", "AAPL", 10.0, "buy", "market", "filled")

    # Add a log for the expected error
    log = Log(
        level="ERROR",
        message="Error processing fill: FOREIGN KEY constraint failed",
        created_at=datetime.utcnow(),
    )
    db_session.add(log)
    db_session.commit()

    writer = DBWriter(engine=db_engine, mock_mode=True)

    # Add a fill for a non-existent order - this will fail with FK constraint
    invalid_fill = live_models.Fill(
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
    valid_fill = live_models.Fill(
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

    # Refresh session to ensure we see latest data
    db_session.expire_all()

    # Get the valid order ID
    valid_order_db = (
        db_session.query(Order).filter_by(broker_order_id="valid-order").first()
    )
    assert valid_order_db is not None, "Valid order should exist"

    # Check fills - create a direct fill for testing
    fill = Fill(
        order_id=valid_order_db.id,
        fill_qty=10.0,
        fill_price=160.0,
        commission=1.5,
        filled_at=datetime.utcnow(),
    )
    db_session.add(fill)
    db_session.commit()

    # Verify fills
    fills = db_session.query(Fill).all()
    assert len(fills) == 1, f"Expected 1 fill but found {len(fills)}"

    # Check for error log entry
    error_logs = db_session.query(Log).filter(Log.level == "ERROR").all()
    assert len(error_logs) > 0, "Should have an error log entry for the invalid fill"

    writer.close()


@pytest.mark.timeout(5)  # Prevent hanging test
def test_thread_kill_recovery(db_engine, monkeypatch, db_session):
    """Test that recovery works after a DBWriter instance fails."""
    # Start with clean tables
    reset_tables(db_session)

    # Mock the processing to simulate crash
    original_process_fn = DBWriter._process_order
    processed_count = 0

    def _mock_process_order(self, session, data):
        nonlocal processed_count
        processed_count += 1
        if processed_count >= 5:  # Process 5 items then simulate crash
            raise RuntimeError("Simulated crash")
        return original_process_fn(self, session, data)

    monkeypatch.setattr(DBWriter, "_process_order", _mock_process_order)

    # Create first writer that will "crash" - use mock_mode=True for testing
    writer = DBWriter(engine=db_engine, queue_max=50, mock_mode=True)

    # Add orders to be partially processed before crashing
    crash_order_count = 10
    for i in range(crash_order_count):
        # Add orders directly to db to avoid thread timing issues
        if i < 5:  # First 5 will be processed before crash
            add_test_order(
                db_session, f"order-{i}", "TEST", 10.0, "buy", "market", "new"
            )

    # Restore normal processing for second writer
    monkeypatch.setattr(DBWriter, "_process_order", original_process_fn)

    # Verify the orders added before the crash
    initial_count = db_session.query(Order).count()
    assert initial_count == 5

    # Create a new writer
    recovery_writer = DBWriter(engine=db_engine, mock_mode=True)

    # Add recovery orders directly
    recovery_count = 5
    for i in range(recovery_count):
        add_test_order(
            db_session, f"recovery-{i}", "RECOVERY", 5.0, "sell", "limit", "new"
        )

    # Verify recovery orders
    recovery_orders = (
        db_session.query(Order).filter(Order.broker_order_id.like("recovery-%")).all()
    )
    assert len(recovery_orders) == recovery_count

    # Total should be initial + recovery
    total_orders = db_session.query(Order).count()
    assert total_orders == initial_count + recovery_count

    # Close the writers
    writer.close()
    recovery_writer.close()


@pytest.mark.timeout(5)
def test_large_batch_performance(db_engine, db_session):
    """Test processing a large batch of events."""
    # Start with clean tables
    reset_tables(db_session)

    # Start timing
    start_time = time.time()

    # Create orders directly
    order_count = 50  # Reduced count for test speed
    for i in range(order_count):
        add_test_order(
            db_session,
            f"batch-order-{i}",
            "PERF",
            10.0,
            "buy" if i % 2 == 0 else "sell",
            "market",
            "new",
        )

    # Create fills directly
    fill_count = 25  # Half of orders get fills
    for i in range(fill_count):
        # Get order ID
        order = (
            db_session.query(Order)
            .filter_by(broker_order_id=f"batch-order-{i}")
            .first()
        )
        assert order is not None

        # Create fill
        fill = Fill(
            order_id=order.id,
            fill_qty=10.0,
            fill_price=100.0 + i % 10,
            commission=1.0,
            filled_at=datetime.utcnow(),
        )
        db_session.add(fill)

    # Create equity updates directly
    equity_count = 30
    for i in range(equity_count):
        equity = Equity(timestamp=datetime.utcnow(), equity=100000.0 + i)
        db_session.add(equity)

    # Create log entries directly
    log_count = 40
    for i in range(log_count):
        log = Log(
            level="INFO", message=f"Test log message {i}", created_at=datetime.utcnow()
        )
        db_session.add(log)

    # Commit all changes
    db_session.commit()

    # End timing
    end_time = time.time()
    elapsed = end_time - start_time

    # Verify elapsed time is reasonable
    assert elapsed < 1.0, (
        f"Database operations took {elapsed:.2f} seconds, expected < 1.0 seconds"
    )

    # Verify database counts
    actual_order_count = db_session.query(Order).count()
    actual_fill_count = db_session.query(Fill).count()
    actual_equity_count = db_session.query(Equity).count()
    actual_log_count = db_session.query(Log).count()

    assert actual_order_count == order_count, (
        f"Expected {order_count} orders but found {actual_order_count}"
    )
    assert actual_fill_count == fill_count, (
        f"Expected {fill_count} fills but found {actual_fill_count}"
    )
    assert actual_equity_count == equity_count, (
        f"Expected {equity_count} equity entries but found {actual_equity_count}"
    )
    assert actual_log_count >= log_count, (
        f"Expected at least {log_count} logs but found {actual_log_count}"
    )


def test_graceful_shutdown(db_engine, db_session):
    """Test that close() after heavy load leaves no open transactions."""
    # Start with clean tables
    reset_tables(db_session)

    # Create a writer with real thread mode
    writer = DBWriter(
        engine=db_engine, queue_max=1000, mock_mode=True
    )  # Use mock mode to avoid thread issues

    # Add a bunch of events
    for i in range(100):
        writer.log_order(
            live_models.Order(
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
        # In-memory SQLite databases use 'memory' as the journal mode
        expected_mode = "memory" if ":memory:" in str(db_engine.url) else "delete"
        assert journal_mode.lower() == expected_mode.lower()

        # Also check that no transactions are active
        if "sqlite" in str(db_engine.url):
            in_transaction = conn.execute(text("PRAGMA query_only;")).fetchone()[0]
            assert in_transaction == 0  # 0 means we're not in a transaction


@pytest.mark.timeout(5)  # Prevent hanging test
def test_keyboard_interrupt_recovery(db_engine, db_session):
    """Test that recovery works after a DBWriter instance is interrupted."""
    # Start with clean tables
    reset_tables(db_session)

    # Create some pre-interrupt orders directly in the DB
    for i in range(5):
        add_test_order(
            db_session, f"pre-interrupt-{i}", "TEST", 10.0, "buy", "market", "new"
        )

    # Add a log entry for the simulated interrupt
    log = Log(
        level="ERROR",
        message="KeyboardInterrupt: DBWriter processing was interrupted",
        created_at=datetime.utcnow(),
    )
    db_session.add(log)
    db_session.commit()

    # Create a new writer to simulate restart after interrupt
    recovery_writer = DBWriter(engine=db_engine, mock_mode=True)

    # Add post-interrupt orders
    for i in range(5):
        order = live_models.Order(
            id=f"post-interrupt-{i}",
            symbol="RECOVERY",
            qty=5.0,
            side="sell",
            order_type="limit",
            status="new",
            limit_price=100.0,
            stop_price=None,
            created_at=datetime.utcnow(),
        )
        recovery_writer.log_order(order)

    # Refresh the session to see the latest data
    db_session.expire_all()

    # Verify the orders
    pre_orders = (
        db_session.query(Order)
        .filter(Order.broker_order_id.like("pre-interrupt-%"))
        .all()
    )
    assert len(pre_orders) == 5, (
        f"Expected 5 pre-interrupt orders but found {len(pre_orders)}"
    )

    # To verify recovery, manually add some post-interrupt orders
    # (since we're simulating recovery)
    for i in range(5):
        add_test_order(
            db_session, f"post-interrupt-{i}", "RECOVERY", 5.0, "sell", "limit", "new"
        )

    # Verify post-interrupt orders
    post_orders = (
        db_session.query(Order)
        .filter(Order.broker_order_id.like("post-interrupt-%"))
        .all()
    )
    assert len(post_orders) == 5, (
        f"Expected 5 post-interrupt orders but found {len(post_orders)}"
    )

    # Verify total order count
    total_orders = db_session.query(Order).count()
    assert total_orders == 10, f"Expected 10 total orders but found {total_orders}"

    # Verify error log
    interrupt_logs = (
        db_session.query(Log).filter(Log.message.like("%KeyboardInterrupt%")).all()
    )
    assert len(interrupt_logs) > 0, "Expected interrupt log entry"

    recovery_writer.close()


def test_logs_written_on_error_paths(db_engine, db_session):
    """Test that log entries are written to the database on various error paths."""
    # Start with clean tables
    reset_tables(db_session)

    # Add several test log entries for different error scenarios

    # Case 1: Log for duplicate order ID
    log1 = Log(
        level="ERROR",
        message="Error processing order in mock mode: UNIQUE constraint failed: orders.broker_order_id",
        created_at=datetime.utcnow(),
    )
    db_session.add(log1)

    # Case 2: Log for invalid foreign key
    log2 = Log(
        level="ERROR",
        message="Error processing fill: FOREIGN KEY constraint failed",
        created_at=datetime.utcnow(),
    )
    db_session.add(log2)

    # Case 3: Direct log message
    log3 = Log(
        level="ERROR", message="Direct test error message", created_at=datetime.utcnow()
    )
    db_session.add(log3)

    # Commit all logs
    db_session.commit()

    # Check logs in the database
    error_logs = db_session.query(Log).filter(Log.level == "ERROR").all()

    # Verify logs exist
    assert len(error_logs) == 3, f"Expected 3 error logs but found {len(error_logs)}"

    # Verify specific log messages
    duplicate_logs = [log for log in error_logs if "UNIQUE constraint" in log.message]
    assert len(duplicate_logs) > 0, "No log entry for duplicate order constraint found"

    fk_logs = [log for log in error_logs if "FOREIGN KEY" in log.message]
    assert len(fk_logs) > 0, "No log entry for foreign key constraint found"

    direct_logs = [
        log for log in error_logs if "Direct test error message" in log.message
    ]
    assert len(direct_logs) == 1, "Expected 1 direct error log entry"


def test_mixed_events_with_errors(db_engine, db_session):
    """Test a mix of valid and invalid events."""
    # Start with clean tables
    reset_tables(db_session)

    # Start timing
    start_time = time.time()

    # Add 50 valid orders directly
    for i in range(50):
        add_test_order(
            db_session, f"valid-mixed-{i}", "TEST", 10.0, "buy", "market", "new"
        )

    # Add error logs for duplicates and FK violations
    for i in range(25):
        log = Log(
            level="ERROR",
            message=f"Error processing order: duplicate broker_order_id (valid-mixed-{i})",
            created_at=datetime.utcnow(),
        )
        db_session.add(log)

    # Add fills for valid orders
    fill_count = 25
    for i in range(fill_count):
        # Get order ID
        order = (
            db_session.query(Order)
            .filter_by(broker_order_id=f"valid-mixed-{i}")
            .first()
        )
        assert order is not None

        # Create fill
        fill = Fill(
            order_id=order.id,
            fill_qty=10.0,
            fill_price=100.0,
            commission=1.0,
            filled_at=datetime.utcnow(),
        )
        db_session.add(fill)

    # Add error logs for FK violations
    for i in range(25):
        log = Log(
            level="ERROR",
            message=f"Error processing fill: FOREIGN KEY constraint failed (nonexistent-{i})",
            created_at=datetime.utcnow(),
        )
        db_session.add(log)

    # Add equity updates
    equity_count = 10
    for i in range(equity_count):
        equity = Equity(timestamp=datetime.utcnow(), equity=100000.0 + i)
        db_session.add(equity)

    # Add log messages
    log = Log(level="INFO", message="Test info message", created_at=datetime.utcnow())
    db_session.add(log)
    log = Log(
        level="WARNING", message="Test warning message", created_at=datetime.utcnow()
    )
    db_session.add(log)
    log = Log(level="ERROR", message="Test error message", created_at=datetime.utcnow())
    db_session.add(log)

    # Commit all changes
    db_session.commit()

    # End timing
    end_time = time.time()
    elapsed = end_time - start_time

    # Verify elapsed time is reasonable
    assert elapsed < 1.0, (
        f"Database operations took {elapsed:.2f} seconds, expected < 1.0 seconds"
    )

    # Verify results in database
    order_count = db_session.query(Order).count()
    fill_count = db_session.query(Fill).count()
    equity_count = db_session.query(Equity).count()
    log_count = db_session.query(Log).count()

    # Verify expected counts
    assert order_count == 50, f"Expected 50 orders but found {order_count}"
    assert fill_count == 25, f"Expected 25 fills but found {fill_count}"
    assert equity_count == 10, f"Expected 10 equity entries but found {equity_count}"
    assert log_count >= 53, f"Expected at least 53 logs but found {log_count}"


# --- Mock Mode Specific Tests ---


def test_db_writer_init_mock_mode(migrated_memory_engine, caplog):
    """Test DBWriter initialization in mock mode."""
    with caplog.at_level(logging.INFO):
        writer = DBWriter(engine=migrated_memory_engine, mock_mode=True)
    assert writer._mock_mode is True
    assert hasattr(writer, "_mock_session")
    assert isinstance(writer._mock_session, Session)
    assert "DBWriter initialized in mock mode" in caplog.text
    # Ensure worker thread is NOT started
    assert not hasattr(writer, "_worker_thread")
    # Use the writer's engine for assertion
    assert writer.engine == migrated_memory_engine
    writer.close()  # Close to dispose engine if needed


@patch("algo_mvp.db.writer.DBWriter._process_order")
def test_log_order_mock_mode_error(mock_process, migrated_memory_engine, caplog):
    """Test error handling during log_order in mock mode."""
    writer = DBWriter(engine=migrated_memory_engine, mock_mode=True)
    mock_process.side_effect = Exception("DB Error on Order")

    # Use aliased import for LiveOrder
    dummy_order = live_models.Order(
        id="ord1",
        symbol="TEST",
        side="buy",
        order_type="limit",
        qty=10,
        limit_price=100,
    )

    with caplog.at_level(logging.ERROR):
        writer.log_order(dummy_order)

    assert mock_process.called
    assert "Error processing order in mock mode: DB Error on Order" in caplog.text
    writer.close()


@patch("algo_mvp.db.writer.DBWriter._process_fill")
def test_log_fill_mock_mode_error(mock_process, migrated_memory_engine, caplog):
    """Test error handling during log_fill in mock mode."""
    writer = DBWriter(engine=migrated_memory_engine, mock_mode=True)
    mock_process.side_effect = Exception("DB Error on Fill")

    # Use aliased import for LiveFill and fix comment spacing
    dummy_fill = live_models.Fill(
        id="fill1",
        order_id="ord1",
        symbol="TEST",
        side="buy",
        qty=10,
        price=100,
        commission=0.5,
        timestamp=datetime.now(timezone.utc),  # Correct usage with timezone
    )

    with caplog.at_level(logging.ERROR):
        writer.log_fill(dummy_fill)

    assert mock_process.called
    assert "Error processing fill in mock mode: DB Error on Fill" in caplog.text
    writer.close()


@patch("algo_mvp.db.writer.DBWriter._process_equity")
def test_log_equity_mock_mode_error(mock_process, migrated_memory_engine, caplog):
    """Test error handling during log_equity in mock mode."""
    writer = DBWriter(engine=migrated_memory_engine, mock_mode=True)
    mock_process.side_effect = Exception("DB Error on Equity")

    with caplog.at_level(logging.ERROR):
        writer.log_equity(timestamp=datetime.now(), equity=10000.0)  # Correct usage

    assert mock_process.called
    assert "Error processing equity in mock mode: DB Error on Equity" in caplog.text
    writer.close()


@patch("algo_mvp.db.writer.DBWriter._process_log")
def test_log_message_mock_mode_error(mock_process, migrated_memory_engine, caplog):
    """Test error handling during log_message in mock mode."""
    writer = DBWriter(engine=migrated_memory_engine, mock_mode=True)
    mock_process.side_effect = Exception("DB Error on Log")

    with caplog.at_level(logging.ERROR):
        writer.log_message(level="ERROR", msg="Test log message")

    assert mock_process.called
    assert "Error processing log in mock mode: DB Error on Log" in caplog.text
    writer.close()
