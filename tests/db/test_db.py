import pytest
from sqlalchemy import inspect
from sqlalchemy.exc import IntegrityError
import datetime
from sqlalchemy.sql import text
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from algo_mvp.db.models import Order, Fill, Equity, Log
from algo_mvp.db import get_engine  # Assuming SessionLocal might be used or tested


TABLE_NAMES = ["orders", "fills", "equity", "logs"]


def test_get_engine_default_url():
    """Test get_engine uses the default URL if none is provided and env var is not set."""
    # Ensure ALGO_DB_URL is not set for this test if it could interfere
    # For robustness, this might involve unsetting it temporarily if tests run in parallel with env changes
    engine = get_engine()
    assert str(engine.url) == "sqlite:///data/algo.db"


def test_get_engine_custom_url():
    """Test get_engine uses the provided URL."""
    custom_url = "sqlite:///:memory:custom"
    engine = get_engine(url=custom_url)
    assert str(engine.url) == custom_url


def test_get_engine_creates_directory():
    """Test get_engine creates parent directory for SQLite file URLs."""
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Construct a path to a non-existent nested directory
        nested_dir = Path(temp_dir) / "non" / "existent" / "directory"
        db_path = nested_dir / "test.db"

        # Create a SQLite URL with the path
        url = f"sqlite:///{db_path}"

        # Call get_engine which should create the directory
        engine = get_engine(url=url)

        # Check the parent directory exists
        assert nested_dir.exists(), "Parent directory was not created"

        # Cleanup by disposing the engine
        engine.dispose()


# Test migrations and table creation (using the migrated_memory_engine fixture from conftest)
def test_tables_created(migrated_memory_engine):
    """Test that all expected tables are created by migrations."""
    inspector = inspect(migrated_memory_engine)
    for table_name in TABLE_NAMES:
        assert inspector.has_table(table_name), (
            f"Table '{table_name}' not found after migration."
        )


# Test basic CRUD operations and relationships


def test_order_fill_relationship(db_session):
    """Test creating an order and a fill, and their relationship."""
    now = datetime.datetime.utcnow()
    new_order = Order(
        broker_order_id="test_broker_order_123",
        symbol="TEST_AAPL",
        side="BUY",
        order_type="LIMIT",
        qty=10.0,
        limit_price=150.0,
        status="NEW",
        created_at=now,
    )
    db_session.add(new_order)
    db_session.commit()

    retrieved_order = (
        db_session.query(Order)
        .filter_by(broker_order_id="test_broker_order_123")
        .first()
    )
    assert retrieved_order is not None
    assert retrieved_order.symbol == "TEST_AAPL"

    new_fill = Fill(
        order_id=retrieved_order.id,
        fill_qty=10.0,
        fill_price=150.0,
        commission=0.5,
        filled_at=now,
    )
    db_session.add(new_fill)
    db_session.commit()

    retrieved_fill = (
        db_session.query(Fill).filter_by(order_id=retrieved_order.id).first()
    )
    assert retrieved_fill is not None
    assert retrieved_fill.fill_qty == 10.0
    assert retrieved_fill.order == retrieved_order
    assert len(retrieved_order.fills) == 1
    assert retrieved_order.fills[0].fill_price == 150.0


def test_broker_order_id_unique_constraint(db_session):
    """Test the unique constraint on broker_order_id in the orders table."""
    now = datetime.datetime.utcnow()
    order1 = Order(
        broker_order_id="unique_broker_id_001",
        symbol="SYM1",
        side="BUY",
        order_type="MARKET",
        qty=1,
        status="FILLED",
        created_at=now,
    )
    db_session.add(order1)
    db_session.commit()

    order2 = Order(
        broker_order_id="unique_broker_id_001",  # Same broker_order_id
        symbol="SYM2",
        side="SELL",
        order_type="MARKET",
        qty=2,
        status="NEW",
        created_at=now,
    )
    db_session.add(order2)
    with pytest.raises(IntegrityError):
        db_session.commit()  # This should fail due to unique constraint


def test_equity_log_creation(db_session):
    """Test creating equity and log entries."""
    now = datetime.datetime.utcnow()
    new_equity = Equity(timestamp=now, equity=100000.0)
    db_session.add(new_equity)
    db_session.commit()

    retrieved_equity = db_session.query(Equity).first()
    assert retrieved_equity is not None
    assert retrieved_equity.equity == 100000.0

    new_log = Log(level="INFO", message="Test log entry", created_at=now)
    db_session.add(new_log)
    db_session.commit()

    retrieved_log = db_session.query(Log).first()
    assert retrieved_log is not None
    assert retrieved_log.message == "Test log entry"


def test_foreign_key_constraint_fills_orders(db_session):
    """Test that a fill cannot be created with a non-existent order_id."""
    now = datetime.datetime.utcnow()

    # Enable foreign key constraint enforcement explicitly
    db_session.execute(text("PRAGMA foreign_keys = ON;"))
    db_session.commit()

    fill_with_invalid_order_id = Fill(
        order_id=99999,  # Assuming 99999 is not a valid order_id
        fill_qty=5.0,
        fill_price=100.0,
        commission=0.1,
        filled_at=now,
    )
    db_session.add(fill_with_invalid_order_id)
    with pytest.raises(IntegrityError):
        db_session.commit()  # This should fail due to FK constraint


# Optional: Test nullable fields by omitting them
def test_order_nullable_fields(db_session):
    """Test creating an order with nullable fields (limit_price, stop_price) set to None."""
    now = datetime.datetime.utcnow()
    market_order = Order(
        broker_order_id="market_order_002",
        symbol="TEST_MSFT",
        side="BUY",
        order_type="MARKET",
        qty=5.0,
        status="NEW",
        created_at=now,
        limit_price=None,  # Explicitly None
        stop_price=None,  # Explicitly None
    )
    db_session.add(market_order)
    db_session.commit()

    retrieved_market_order = (
        db_session.query(Order).filter_by(broker_order_id="market_order_002").first()
    )
    assert retrieved_market_order is not None
    assert retrieved_market_order.limit_price is None
    assert retrieved_market_order.stop_price is None


def test_upgrade_db_older_sqlalchemy(tmp_path, monkeypatch):
    """Test upgrade_db with a mock engine emulating older SQLAlchemy."""
    # Mock engine with older attribute structure
    # Spec ensures hasattr(mock_engine_old, 'url') is False, forcing the 'else' branch
    mock_engine_old = MagicMock(spec=["engine", "begin"])
    mock_engine_old.engine = MagicMock()
    # Configure the url attribute to return the string directly
    mock_engine_old.engine.url = "sqlite:///test.db"
    mock_engine_old.begin = MagicMock()  # mock_engine_old.begin will be called, its return value (a MagicMock) will be the CM

    # Mock alembic command.upgrade
    with patch("algo_mvp.db.command.upgrade") as mock_upgrade:
        # Call upgrade_db with the mocked engine
        from algo_mvp.db import upgrade_db

        upgrade_db(mock_engine_old)

        # Assert that alembic command.upgrade was called
        mock_upgrade.assert_called_once()

        # Get the Alembic Config object passed to upgrade
        alembic_cfg_arg = mock_upgrade.call_args[0][0]

        # Check that sqlalchemy.url was set correctly in the config
        assert alembic_cfg_arg.get_main_option("sqlalchemy.url") == "sqlite:///test.db"


def test_get_sessionmaker_default_engine():
    """Test get_sessionmaker uses the default engine if none is provided."""
    from algo_mvp.db import get_sessionmaker, get_engine

    # Ensure ALGO_DB_URL is not set to avoid interference if necessary
    # We expect it to use the default engine, which in turn uses the default URL
    Session = get_sessionmaker()  # Call without an engine
    assert str(Session.kw["bind"].url) == str(
        get_engine().url
    )  # Compare engine URL strings


def test_get_writer_creates_instance():
    """Test get_writer creates a DBWriter instance when _writer_instance is None."""
    from algo_mvp.db import get_writer

    # Mock the DBWriter import to avoid actual DB operations
    with patch("algo_mvp.db.writer.DBWriter") as mock_db_writer:
        # Reset the _writer_instance global variable
        import algo_mvp.db

        algo_mvp.db._writer_instance = None

        # Call get_writer, which should create a new DBWriter
        writer = get_writer()

        # Verify DBWriter was instantiated
        mock_db_writer.assert_called_once()

        # Verify get_writer returns the mocked instance
        assert writer == mock_db_writer.return_value


def test_get_sessionmaker_with_provided_engine():
    """Test get_sessionmaker with a provided engine."""
    from algo_mvp.db import get_sessionmaker
    from sqlalchemy import create_engine

    # Create a test engine
    test_engine = create_engine("sqlite:///:memory:")

    # Get a sessionmaker with the test engine
    Session = get_sessionmaker(engine=test_engine)

    # Verify the sessionmaker is bound to our test engine
    assert Session.kw["bind"] == test_engine


def test_upgrade_db_newer_sqlalchemy(tmp_path, monkeypatch):
    """Test upgrade_db with a mock engine emulating newer SQLAlchemy (1.4+)."""
    # Mock engine with newer attribute structure
    mock_engine_new = MagicMock()
    mock_engine_new.url = (
        "sqlite:///test.db"  # Direct url attribute for SQLAlchemy 1.4+
    )
    mock_engine_new.begin = MagicMock()  # Mock begin method

    # Mock alembic command.upgrade
    with patch("algo_mvp.db.command.upgrade") as mock_upgrade:
        # Call upgrade_db with the mocked engine
        from algo_mvp.db import upgrade_db

        upgrade_db(mock_engine_new)

        # Assert that alembic command.upgrade was called
        mock_upgrade.assert_called_once()

        # Get the Alembic Config object passed to upgrade
        alembic_cfg_arg = mock_upgrade.call_args[0][0]

        # Check that sqlalchemy.url was set correctly in the config
        assert alembic_cfg_arg.get_main_option("sqlalchemy.url") == "sqlite:///test.db"
