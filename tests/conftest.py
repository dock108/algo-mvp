import pytest
from sqlalchemy.orm import sessionmaker

from algo_mvp.db import get_engine, upgrade_db


@pytest.fixture
def db_engine():
    engine = get_engine("sqlite:///:memory:")
    upgrade_db(engine)  # reuse migrate helper
    yield engine
    engine.dispose()


@pytest.fixture
def db_session(db_engine):
    connection = db_engine.connect()
    txn = connection.begin()
    Session = sessionmaker(bind=connection, expire_on_commit=False)
    session = Session()
    yield session
    session.close()
    txn.rollback()
    connection.close()
