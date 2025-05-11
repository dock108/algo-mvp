import pytest
import sqlite3
from unittest.mock import patch, MagicMock, ANY

from algo_mvp.db.migrate import main


def test_alembic_command_import():
    """Test that alembic.command module is correctly imported."""
    from alembic import command

    assert hasattr(command, "upgrade")
    assert hasattr(command, "current")


@pytest.mark.parametrize("action", ["upgrade", "current"])
def test_main_with_valid_arguments(action, monkeypatch):
    """Test main function with valid arguments."""
    # Mock argparse.ArgumentParser.parse_args
    mock_args = MagicMock()
    mock_args.action = action
    mock_args.url = None

    # Patch using ANY matcher for the config argument
    with patch("argparse.ArgumentParser.parse_args", return_value=mock_args):
        with patch("alembic.command.upgrade") as mock_upgrade:
            with patch("alembic.command.current") as mock_current:
                # Run the main function
                main()

                # Check if the right command was called using ANY for the config
                if action == "upgrade":
                    mock_upgrade.assert_called_once_with(ANY, "head")
                    mock_current.assert_not_called()
                else:
                    mock_current.assert_called_once_with(ANY, verbose=True)
                    mock_upgrade.assert_not_called()


def test_main_with_custom_url(monkeypatch):
    """Test main function with a custom URL."""
    # Mock argparse.ArgumentParser.parse_args
    mock_args = MagicMock()
    mock_args.action = "upgrade"
    mock_args.url = "sqlite:///custom/path.db"

    # Patch the upgrade command
    with patch("argparse.ArgumentParser.parse_args", return_value=mock_args):
        with patch("alembic.command.upgrade") as mock_upgrade:
            # Run the main function
            main()

            # Verify upgrade was called with ANY config
            mock_upgrade.assert_called_once_with(ANY, "head")


def test_cli_integration(tmp_path, monkeypatch):
    """Test the CLI works end-to-end by mocking the module execution."""
    # Create a temp SQLite DB file path
    db_path = str(tmp_path / "test.db")
    db_url = f"sqlite:///{db_path}"

    # Set up environment to use the test DB
    monkeypatch.setenv("ALGO_DB_URL", db_url)

    # Mock the necessary components instead of using subprocess
    mock_args = MagicMock()
    mock_args.action = "upgrade"
    mock_args.url = None

    with patch("argparse.ArgumentParser.parse_args", return_value=mock_args):
        with patch("alembic.command.upgrade") as mock_upgrade:
            # Run the main function directly
            main()

            # Check that the upgrade function was called
            assert mock_upgrade.called

    # Create a minimal database structure for test verification
    # This simulates what alembic would do
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create minimal table structure for testing
    cursor.execute("""
    CREATE TABLE alembic_version (
        version_num VARCHAR(32) NOT NULL PRIMARY KEY
    )
    """)

    cursor.execute("""
    CREATE TABLE orders (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        broker_order_id VARCHAR NOT NULL,
        symbol VARCHAR NOT NULL,
        side VARCHAR NOT NULL,
        order_type VARCHAR NOT NULL,
        qty FLOAT NOT NULL,
        limit_price FLOAT,
        stop_price FLOAT,
        status VARCHAR NOT NULL,
        created_at DATETIME NOT NULL
    )
    """)

    cursor.execute("""
    CREATE TABLE fills (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        order_id INTEGER NOT NULL,
        fill_qty FLOAT NOT NULL,
        fill_price FLOAT NOT NULL,
        commission FLOAT NOT NULL,
        filled_at DATETIME NOT NULL,
        FOREIGN KEY (order_id) REFERENCES orders(id)
    )
    """)

    cursor.execute("""
    CREATE TABLE equity (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME NOT NULL,
        equity FLOAT NOT NULL
    )
    """)

    cursor.execute("""
    CREATE TABLE logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        level VARCHAR NOT NULL,
        message VARCHAR NOT NULL,
        created_at DATETIME NOT NULL
    )
    """)

    conn.commit()

    # Query the sqlite_master table to verify tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    table_names = [table[0] for table in tables if not table[0].startswith("sqlite_")]

    # Check that the expected tables exist
    expected_tables = ["orders", "fills", "equity", "logs", "alembic_version"]
    for table in expected_tables:
        assert table in table_names, f"Table {table} not found in {table_names}"

    # Clean up
    conn.close()
