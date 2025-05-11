import os
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from alembic.config import Config
from alembic import command


def get_engine(url=None):
    url = url or os.getenv("ALGO_DB_URL", "sqlite:///data/algo.db")

    # For SQLite file URLs, ensure the parent directory exists
    if url.startswith("sqlite:///"):
        # Extract the file path part (skip sqlite:///)
        file_path = url[10:]
        # Only create directory if it's a file path, not memory or relative path
        if file_path and not file_path.startswith(":") and "/" in file_path:
            # Create parent directory if it doesn't exist
            dir_path = Path(file_path).parent
            os.makedirs(dir_path, exist_ok=True)

    return create_engine(url, echo=False, future=True)


def upgrade_db(engine):
    """Run the latest Alembic migration revision against a given engine.

    Args:
        engine: SQLAlchemy engine instance to run migrations on
    """
    # Get the path to the alembic.ini file
    script_dir = os.path.dirname(os.path.realpath(__file__))
    alembic_cfg_path = os.path.join(script_dir, "alembic.ini")

    alembic_cfg = Config(alembic_cfg_path)
    alembic_cfg.set_main_option("sqlalchemy.url", str(engine.url))

    # Run the upgrade to the latest revision
    command.upgrade(alembic_cfg, "head")


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
