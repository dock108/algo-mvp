import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from algo_mvp.db.models import Base


@pytest.fixture
def db_engine():
    """Create a SQLite in-memory database engine with schema defined directly.

    This approach avoids the complexity of using Alembic migrations for tests.
    Instead, we create tables directly from the SQLAlchemy models.

    Function scope ensures each test gets a fresh database.
    """
    # Create an in-memory SQLite database
    engine = create_engine("sqlite:///:memory:", echo=False)

    # Create all tables directly from the models
    Base.metadata.create_all(engine)

    # Set pragmas for SQLite
    with engine.connect() as conn:
        conn.execute(text("PRAGMA foreign_keys = ON"))

    # Yield the configured engine
    yield engine

    # Clean up
    engine.dispose()


@pytest.fixture
def db_session(db_engine):
    """Create a session with transaction rollback for test isolation."""
    connection = db_engine.connect()
    transaction = connection.begin()

    Session = sessionmaker(bind=connection, expire_on_commit=False)
    session = Session()

    yield session

    # Rollback the transaction after the test completes
    session.close()
    transaction.rollback()
    connection.close()
