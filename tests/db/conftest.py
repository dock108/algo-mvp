import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from algo_mvp.db.models import Base


@pytest.fixture(scope="session")
def memory_db_url():
    # Using a sharable in-memory database URI
    return "sqlite:///file::memory:?cache=shared"


@pytest.fixture(scope="function")
def migrated_memory_engine(memory_db_url):
    """Creates an in-memory SQLite engine configured for multi-threading access
    within a single test function, and creates all tables directly using the pooled connection."""
    engine = create_engine(
        memory_db_url,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    # Establish the single persistent connection and create tables on it
    with engine.connect() as connection:
        Base.metadata.create_all(
            connection
        )  # Create tables using this specific connection
        connection.commit()  # Ensure schema changes are committed

    yield engine

    # Cleanup: Drop tables and dispose engine
    # Using a new connection for cleanup, as the original might be in an unknown state if tests failed
    with engine.connect() as connection:
        Base.metadata.drop_all(connection)  # Use connection for drop too
        connection.commit()  # Commit drop changes
    engine.dispose()


@pytest.fixture
def db_session(migrated_memory_engine):
    """Creates a new database session for a test."""
    Session = sessionmaker(bind=migrated_memory_engine)
    session = Session()
    yield session
    session.close()
