import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from algo_mvp.db.models import Base


@pytest.fixture(scope="session")
def memory_db_url():
    return "sqlite:///:memory:"


@pytest.fixture(scope="session")
def test_engine(memory_db_url):
    """Create an engine with a brand new in-memory SQLite database."""
    engine = create_engine(memory_db_url)
    return engine


@pytest.fixture(scope="session")
def migrated_memory_engine(memory_db_url):
    """Creates an in-memory SQLite engine and creates all tables directly."""
    # In-memory SQLite doesn't persist, so create tables directly from the models
    engine = create_engine(memory_db_url)
    Base.metadata.create_all(engine)
    return engine


@pytest.fixture
def db_session(migrated_memory_engine):
    """Creates a new database session for a test."""
    Session = sessionmaker(bind=migrated_memory_engine)
    session = Session()
    yield session
    session.close()
