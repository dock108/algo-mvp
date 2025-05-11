import logging
import queue
import threading
import time
from datetime import datetime

from sqlalchemy.orm import Session

from algo_mvp.db import get_engine
from algo_mvp.db.models import Equity, Fill, Log, Order

logger = logging.getLogger(__name__)


class DBWriter:
    """Thread-safe database writer that pipes trading events to SQLite.

    This class manages a background worker thread that consumes events from a queue
    and writes them to the database, ensuring that database operations do not block
    the main trading threads.
    """

    def __init__(self, engine=None, queue_max=1000, mock_mode=False):
        """Initialize the DBWriter with an optional engine and queue size.

        Args:
            engine: SQLAlchemy engine (if None, uses default from get_engine())
            queue_max: Maximum size of the event queue
            mock_mode: If True, disables the worker thread for testing purposes
        """
        self.engine = engine or get_engine()
        self.queue = queue.Queue(maxsize=queue_max)
        self._stop_event = threading.Event()
        self._closed = False
        self._mock_mode = mock_mode

        # Only start the worker thread if not in mock mode
        if not mock_mode:
            self._worker_thread = threading.Thread(target=self._worker, daemon=True)
            self._worker_thread.start()
            logger.info("DBWriter initialized with engine %s", self.engine.url)
        else:
            logger.info("DBWriter initialized in mock mode (no worker thread)")

    def log_order(self, order):
        """Enqueue an order for database insertion.

        Args:
            order: Order dataclass from algo_mvp.live.models
        """
        if self._closed:
            logger.warning("Cannot log order - DBWriter is closed")
            return

        order_data = {
            "broker_order_id": order.id,
            "symbol": order.symbol,
            "side": order.side,
            "order_type": order.order_type,
            "qty": order.qty,
            "limit_price": order.limit_price,
            "stop_price": order.stop_price,
            "status": order.status,
            "created_at": order.created_at,
        }

        # If in mock mode, process immediately with a new session
        if self._mock_mode:
            with Session(self.engine) as session:
                try:
                    self._process_order(session, order_data)
                    session.commit()
                except Exception as e:
                    session.rollback()
                    logger.error(
                        f"Error processing order in mock mode: {e}", exc_info=True
                    )
            return

        self.queue.put(("order", order_data))

    def log_fill(self, fill):
        """Enqueue a fill for database insertion.

        Args:
            fill: Fill dataclass from algo_mvp.live.models
        """
        if self._closed:
            logger.warning("Cannot log fill - DBWriter is closed")
            return

        fill_data = {
            "broker_order_id": fill.order_id,  # This is the broker order ID
            "fill_qty": fill.qty,
            "fill_price": fill.price,
            "commission": fill.commission,
            "filled_at": fill.timestamp,
        }

        # If in mock mode, process immediately with a new session
        if self._mock_mode:
            with Session(self.engine) as session:
                try:
                    self._process_fill(session, fill_data)
                    session.commit()
                except Exception as e:
                    session.rollback()
                    logger.error(
                        f"Error processing fill in mock mode: {e}", exc_info=True
                    )
            return

        self.queue.put(("fill", fill_data))

    def log_equity(self, timestamp, equity):
        """Enqueue an equity snapshot for database insertion.

        Args:
            timestamp: Datetime representing when the equity was recorded
            equity: Float value of the account equity
        """
        if self._closed:
            logger.warning("Cannot log equity - DBWriter is closed")
            return

        equity_data = {"timestamp": timestamp, "equity": equity}

        # If in mock mode, process immediately with a new session
        if self._mock_mode:
            with Session(self.engine) as session:
                try:
                    self._process_equity(session, equity_data)
                    session.commit()
                except Exception as e:
                    session.rollback()
                    logger.error(
                        f"Error processing equity in mock mode: {e}", exc_info=True
                    )
            return

        self.queue.put(("equity", equity_data))

    def log_message(self, level, msg):
        """Enqueue a log message for database insertion.

        Args:
            level: String log level (INFO, WARNING, ERROR, etc.)
            msg: String message content
        """
        if self._closed:
            logger.warning("Cannot log message - DBWriter is closed")
            return

        log_data = {"level": level, "message": msg, "created_at": datetime.utcnow()}

        # If in mock mode, process immediately with a new session
        if self._mock_mode:
            with Session(self.engine) as session:
                try:
                    self._process_log(session, log_data)
                    session.commit()
                except Exception as e:
                    session.rollback()
                    logger.error(
                        f"Error processing log in mock mode: {e}", exc_info=True
                    )
            return

        self.queue.put(("log", log_data))

    def close(self):
        """Flush remaining queue items, stop worker, and dispose engine.

        This method should be called when shutting down the application to ensure
        all queued events are written to the database.
        """
        if self._closed:
            logger.warning("DBWriter already closed")
            return

        logger.info("Closing DBWriter - waiting for queue to drain")
        self._closed = True

        # Skip worker thread cleanup in mock mode
        if self._mock_mode:
            logger.info("Mock mode - no worker thread to close")
            # Clean up the engine
            if self.engine:
                self.engine.dispose()
                logger.info("Database engine disposed")
            return

        # Wait for the queue to be empty (processed)
        while not self.queue.empty():
            time.sleep(0.1)

        # Signal the worker to stop and wait for it
        self._stop_event.set()
        self._worker_thread.join(timeout=5)

        if self._worker_thread.is_alive():
            logger.warning("DBWriter worker thread did not stop gracefully")
        else:
            logger.info("DBWriter worker thread stopped")

        # Clean up the engine
        if self.engine:
            self.engine.dispose()
            logger.info("Database engine disposed")

    def _worker(self):
        """Worker thread function that processes the event queue.

        This method runs in a separate thread and processes events from the queue,
        writing them to the database as they arrive.
        """
        session = None
        try:
            session = Session(self.engine)

            while not self._stop_event.is_set():
                try:
                    # Get an item from the queue with a timeout
                    try:
                        event_type, data = self.queue.get(timeout=0.5)
                    except queue.Empty:
                        continue

                    # Process the event based on its type
                    try:
                        if event_type == "order":
                            self._process_order(session, data)
                        elif event_type == "fill":
                            self._process_fill(session, data)
                        elif event_type == "equity":
                            self._process_equity(session, data)
                        elif event_type == "log":
                            self._process_log(session, data)
                        else:
                            logger.warning(f"Unknown event type: {event_type}")

                        # Commit the session
                        session.commit()
                    except Exception as e:
                        # Rollback on error
                        session.rollback()
                        logger.error(
                            f"Error processing {event_type} event: {e}", exc_info=True
                        )
                    finally:
                        # Mark task as done regardless of success/failure
                        self.queue.task_done()

                except Exception as e:
                    logger.error(
                        f"Unexpected error in DBWriter worker: {e}", exc_info=True
                    )
                    # If we get an unexpected error, sleep briefly to avoid tight loop
                    time.sleep(0.1)
        finally:
            # Ensure the session is closed
            if session:
                session.close()
                logger.info("Database session closed")

    def _process_order(self, session, data):
        """Process an order event by inserting it into the database."""
        # Create the order object
        order = Order(
            broker_order_id=data["broker_order_id"],
            symbol=data["symbol"],
            side=data["side"],
            order_type=data["order_type"],
            qty=data["qty"],
            limit_price=data["limit_price"],
            stop_price=data["stop_price"],
            status=data["status"],
            created_at=data["created_at"],
        )

        # Add to session
        session.add(order)
        logger.debug(f"Order processed: {data['broker_order_id']}")

    def _process_fill(self, session, data):
        """Process a fill event by inserting it into the database.

        This links the fill to the corresponding order via its broker_order_id.
        """
        # First, find the order by broker_order_id
        order = (
            session.query(Order)
            .filter_by(broker_order_id=data["broker_order_id"])
            .first()
        )

        if not order:
            logger.warning(
                f"Could not find order with broker_order_id {data['broker_order_id']} for fill"
            )
            return

        # Create the fill object with the order_id
        fill = Fill(
            order_id=order.id,
            fill_qty=data["fill_qty"],
            fill_price=data["fill_price"],
            commission=data["commission"],
            filled_at=data["filled_at"],
        )

        # Add to session
        session.add(fill)
        logger.debug(f"Fill processed for order: {data['broker_order_id']}")

    def _process_equity(self, session, data):
        """Process an equity event by inserting it into the database."""
        # Create the equity object
        equity = Equity(timestamp=data["timestamp"], equity=data["equity"])

        # Add to session
        session.add(equity)
        logger.debug(f"Equity snapshot processed: {data['equity']}")

    def _process_log(self, session, data):
        """Process a log event by inserting it into the database."""
        # Create the log object
        log = Log(
            level=data["level"], message=data["message"], created_at=data["created_at"]
        )

        # Add to session
        session.add(log)
        logger.debug(f"Log processed: {data['message'][:30]}...")
