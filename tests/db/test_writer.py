import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from freezegun import freeze_time

from sqlalchemy.orm import Session

from algo_mvp.db.models import Order, Fill, Equity, Log
from algo_mvp.db.writer import DBWriter
from algo_mvp.live.models import Order as LiveOrder, Fill as LiveFill


@pytest.fixture
def db_writer(migrated_memory_engine):
    """Create a DBWriter with an in-memory database for testing (mock_mode=True)."""
    writer = DBWriter(engine=migrated_memory_engine, queue_max=100, mock_mode=True)
    yield writer
    writer.close()


def test_init_with_default_engine():
    """Test DBWriter initialization with the default engine."""
    with patch("algo_mvp.db.writer.get_engine") as mock_get_engine:
        mock_engine = Mock()
        mock_get_engine.return_value = mock_engine

        writer = DBWriter(mock_mode=True)

        mock_get_engine.assert_called_once()
        assert writer.engine == mock_engine

        # Clean up
        writer.close()


def test_close():
    """Test that close properly stops the worker thread and disposes the engine."""
    mock_engine = Mock()

    writer = DBWriter(engine=mock_engine, mock_mode=True)
    writer.close()

    # Assert the closed flag was set
    assert writer._closed

    # Assert the engine was disposed
    mock_engine.dispose.assert_called_once()


def test_log_order(db_writer: DBWriter):
    """Test logging an order to the database."""
    # Create a mock order
    order = LiveOrder(
        id="test-order-1",
        symbol="AAPL",
        qty=10.0,
        side="buy",
        order_type="market",
        status="filled",
        limit_price=None,
        stop_price=None,
        created_at=datetime.utcnow(),
    )

    # Log the order
    db_writer.log_order(order)

    # Check that the order was inserted
    with Session(db_writer.engine) as session:
        db_order = (
            session.query(Order).filter_by(broker_order_id="test-order-1").first()
        )
        assert db_order is not None
        assert db_order.symbol == "AAPL"
        assert db_order.qty == 10.0
        assert db_order.side == "buy"
        assert db_order.order_type == "market"
        assert db_order.status == "filled"


def test_log_fill(db_writer: DBWriter):
    """Test logging a fill to the database."""
    # First create an order to reference
    order = LiveOrder(
        id="test-order-2",
        symbol="MSFT",
        qty=5.0,
        side="sell",
        order_type="limit",
        status="filled",
        limit_price=150.0,
        stop_price=None,
        created_at=datetime.utcnow(),
    )

    # Log the order first
    db_writer.log_order(order)

    # Create a mock fill
    fill = LiveFill(
        id="test-fill-1",
        order_id="test-order-2",
        symbol="MSFT",
        qty=5.0,
        price=150.0,
        side="sell",
        commission=1.5,
        timestamp=datetime.utcnow(),
    )

    # Log the fill
    db_writer.log_fill(fill)

    # Check that the fill was inserted
    with Session(db_writer.engine) as session:
        # First find the order ID
        db_order = (
            session.query(Order).filter_by(broker_order_id="test-order-2").first()
        )
        assert db_order is not None

        # Now find the fill
        db_fill = session.query(Fill).filter_by(order_id=db_order.id).first()
        assert db_fill is not None
        assert db_fill.fill_qty == 5.0
        assert db_fill.fill_price == 150.0
        assert db_fill.commission == 1.5


def test_log_equity(db_writer: DBWriter):
    """Test logging an equity snapshot to the database."""
    # Create a timestamp and equity value
    timestamp = datetime.utcnow()
    equity_value = 100000.0

    # Log the equity
    db_writer.log_equity(timestamp, equity_value)

    # Check that the equity was inserted
    with Session(db_writer.engine) as session:
        db_equity = session.query(Equity).first()
        assert db_equity is not None
        assert db_equity.equity == 100000.0
        assert db_equity.timestamp == timestamp


def test_log_message(db_writer: DBWriter):
    """Test logging a message to the database."""
    # Log a message
    db_writer.log_message("ERROR", "Test error message")

    # Check that the message was inserted
    with Session(db_writer.engine) as session:
        db_log = session.query(Log).first()
        assert db_log is not None
        assert db_log.level == "ERROR"
        assert db_log.message == "Test error message"


def test_log_methods_when_closed(db_writer: DBWriter):
    """Test that log methods don't add to the queue when the writer is closed."""
    # Close the writer
    db_writer.close()

    # Try to log various items
    db_writer.log_order(Mock(id="test"))
    db_writer.log_fill(Mock(order_id="test"))
    db_writer.log_equity(datetime.utcnow(), 1000.0)
    db_writer.log_message("INFO", "Test")

    # The queue should be empty since we're closed
    assert db_writer.queue.empty()


def test_process_fill_with_nonexistent_order(db_writer: DBWriter):
    """Test processing a fill for an order that doesn't exist."""
    # Create a fill for a non-existent order
    fill = LiveFill(
        id="nonexistent-fill",
        order_id="nonexistent-order",
        symbol="NONEXISTENT",
        qty=10.0,
        price=150.0,
        side="buy",
        commission=1.0,
        timestamp=datetime.utcnow(),
    )

    # Log the fill
    db_writer.log_fill(fill)

    # Should not have created a fill since the order doesn't exist
    with Session(db_writer.engine) as session:
        fill_count = session.query(Fill).count()
        assert fill_count == 0


def test_worker_handles_exceptions():
    """Test that the worker handles exceptions during processing."""
    with patch("algo_mvp.db.writer.logger") as mock_logger:
        # Create a writer that will raise an exception when log_order is called
        writer = DBWriter(mock_mode=True)

        # Mock a specific method to raise an exception
        with patch.object(
            writer, "_process_order", side_effect=Exception("Test exception")
        ):
            # Create a mock order
            order = Mock(
                id="exception-order",
                symbol="TEST",
                side="buy",
                order_type="market",
                qty=1.0,
                status="new",
                limit_price=None,
                stop_price=None,
                created_at=datetime.utcnow(),
            )

            # Try to log the order, which should trigger the exception
            writer.log_order(order)

            # Verify the error was logged
            mock_logger.error.assert_called()

        # Clean up
        writer.close()


# --- Threaded (non-mock mode) tests ---


@pytest.fixture
def db_writer_threaded(migrated_memory_engine):
    """Create a DBWriter with a worker thread for testing."""
    writer = DBWriter(engine=migrated_memory_engine, queue_max=100, mock_mode=False)
    yield writer
    # close() will wait for queue to drain and thread to stop, called by fixture teardown
    # Ensure writer is closed even if test fails to call it
    if not writer._closed:
        writer.close()


@freeze_time("2025-05-13 01:59:12.191949+00:00")
def test_log_order_threaded(db_writer_threaded: DBWriter):
    """Test logging an order via the worker thread."""
    order = LiveOrder(
        id="thread-order-1",
        symbol="TICK1",
        qty=1.0,
        side="buy",
        order_type="market",
        status="new",
        created_at=datetime.utcnow(),
    )
    db_writer_threaded.log_order(order)
    # Calling close() ensures all items in queue are processed before verification.
    db_writer_threaded.close()

    with Session(db_writer_threaded.engine) as session:
        db_order = (
            session.query(Order).filter_by(broker_order_id="thread-order-1").first()
        )
        assert db_order is not None
        assert db_order.symbol == "TICK1"


@freeze_time("2025-05-13 01:59:12.990313+00:00")
def test_log_fill_threaded(db_writer_threaded: DBWriter):
    """Test logging a fill via the worker thread."""
    order_created_at = datetime.utcnow()
    fill_timestamp = order_created_at + timedelta(seconds=1)

    order = LiveOrder(
        id="thread-order-fill",
        symbol="TICK2",
        qty=2.0,
        side="sell",
        order_type="limit",
        status="partially_filled",
        limit_price=10.0,
        created_at=order_created_at,
    )
    db_writer_threaded.log_order(order)

    fill = LiveFill(
        id="thread-fill-1",
        order_id="thread-order-fill",
        symbol="TICK2",
        qty=2.0,
        price=10.0,
        side="sell",
        commission=0.1,
        timestamp=fill_timestamp,
    )
    db_writer_threaded.log_fill(fill)
    db_writer_threaded.close()

    with Session(db_writer_threaded.engine) as session:
        db_order = (
            session.query(Order).filter_by(broker_order_id="thread-order-fill").first()
        )
        assert db_order is not None
        # Check that order status is updated by the fill processing logic if applicable
        # For now, just check fill exists
        db_fill = session.query(Fill).filter_by(order_id=db_order.id).first()
        assert db_fill is not None
        assert db_fill.fill_qty == 2.0
        assert db_fill.filled_at == fill_timestamp


@freeze_time("2025-05-13 01:59:13.761547+00:00")
def test_log_equity_threaded(db_writer_threaded: DBWriter):
    """Test logging equity via the worker thread."""
    ts = datetime.utcnow()
    db_writer_threaded.log_equity(ts, 50000.0)
    db_writer_threaded.close()

    with Session(db_writer_threaded.engine) as session:
        db_equity = session.query(Equity).filter_by(equity=50000.0).first()
        assert db_equity is not None
        assert db_equity.timestamp == ts


@freeze_time("2025-05-13 01:59:14.516644+00:00")
def test_log_message_threaded(db_writer_threaded: DBWriter):
    """Test logging a message via the worker thread."""
    db_writer_threaded.log_message("THREAD_INFO", "Worker processed this")
    db_writer_threaded.close()

    with Session(db_writer_threaded.engine) as session:
        db_log = session.query(Log).filter_by(level="THREAD_INFO").first()
        assert db_log is not None
        assert db_log.message == "Worker processed this"
