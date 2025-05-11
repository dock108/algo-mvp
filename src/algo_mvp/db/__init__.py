import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


def get_engine(url=None):
    url = url or os.getenv("ALGO_DB_URL", "sqlite:///data/algo.db")
    return create_engine(url, echo=False, future=True)


SessionLocal = sessionmaker(bind=get_engine(), expire_on_commit=False)

# Singleton DBWriter instance
_writer_instance = None


def get_writer(engine=None, queue_max=1000):
    """Get or create a singleton DBWriter instance.

    Args:
        engine: Optional SQLAlchemy engine to use (if None, uses default from get_engine())
        queue_max: Maximum queue size for the writer's event queue

    Returns:
        A DBWriter instance (same instance for repeated calls)
    """
    global _writer_instance

    if _writer_instance is None:
        from algo_mvp.db.writer import DBWriter

        _writer_instance = DBWriter(engine=engine, queue_max=queue_max)

    return _writer_instance
