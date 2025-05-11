import pytest
import logging
from datetime import datetime
from unittest.mock import MagicMock, patch

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from algo_mvp.db.models import Base, Equity, Log
from algo_mvp.live.runner import LiveRunner
from algo_mvp.db.writer import DBWriter


@pytest.fixture
def memory_engine():
    """Create an in-memory SQLite engine for testing."""
    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}
    )

    # Enable SQLite foreign keys
    with engine.connect() as conn:
        conn.execute(text("PRAGMA foreign_keys=ON"))
        conn.commit()

    Base.metadata.create_all(engine)
    return engine


@pytest.fixture
def db_writer(memory_engine):
    """Create a DBWriter with an in-memory database for testing."""
    writer = DBWriter(engine=memory_engine, queue_max=100, mock_mode=True)
    yield writer
    writer.close()


def test_log_direct_from_runner(db_writer, memory_engine):
    """Test that LiveRunner logs messages correctly."""
    # Create a runner that just has a db_writer
    runner = MagicMock()
    runner.db_writer = db_writer

    # Create the original LiveRunner._log method
    original_log = LiveRunner._log

    # Test logging a message directly
    message = "Test error message"
    original_log(runner, message, level=logging.ERROR)

    # Check that the message was written to the database
    with Session(memory_engine) as session:
        db_log = session.query(Log).first()
        assert db_log is not None
        assert db_log.level == "ERROR"
        assert message in db_log.message


def test_log_equity_from_runner(db_writer, memory_engine):
    """Test that LiveRunner logs equity correctly."""
    # Create a runner that just has a db_writer
    runner = MagicMock()
    runner.db_writer = db_writer

    # Create the original LiveRunner.log_equity method
    original_log_equity = LiveRunner.log_equity

    # Test logging equity directly
    timestamp = datetime.utcnow()
    equity = 100000.0
    original_log_equity(runner, timestamp, equity)

    # Check that the equity was written to the database
    with Session(memory_engine) as session:
        db_equity = session.query(Equity).first()
        assert db_equity is not None
        assert db_equity.equity == equity
        assert db_equity.timestamp == timestamp


def test_get_writer_called_in_constructor():
    """Test that LiveRunner calls get_writer if no db_writer is provided."""
    with patch("algo_mvp.live.runner.get_writer") as mock_get_writer:
        mock_writer = MagicMock()
        mock_get_writer.return_value = mock_writer

        # Create a mock strategy_module and strategy
        with patch("importlib.import_module") as mock_import:
            mock_strategy_module = MagicMock()
            mock_import.return_value = mock_strategy_module

            # Create a mock broker_adapter
            mock_broker_adapter = MagicMock()

            # Create runner without db_writer
            runner = LiveRunner(
                strategy_path="tests.live.dummy_strategy:DummyStrategy",
                params={},
                broker_config={"provider": "mock", "adapter": mock_broker_adapter},
                datafeed_config={},
            )

            # Verify get_writer was called
            mock_get_writer.assert_called_once()

            # Verify db_writer was set
            assert runner.db_writer == mock_writer
