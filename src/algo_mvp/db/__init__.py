import os
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from alembic.config import Config
from alembic import command

# Path to this package
package_dir = Path(os.path.dirname(os.path.abspath(__file__)))


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

    # Create and return the engine
    engine = create_engine(url)
    return engine


def get_sessionmaker(engine=None):
    """Get a SQLAlchemy sessionmaker.

    Args:
        engine: Optional SQLAlchemy engine to use. If not provided,
               a new engine will be created using get_engine().

    Returns:
        sqlalchemy.orm.sessionmaker: A sessionmaker bound to the engine.
    """
    if engine is None:
        engine = get_engine()
    Session = sessionmaker(bind=engine)
    return Session


def upgrade_db(engine):
    """Run database migrations to upgrade to the latest schema version.

    Args:
        engine: SQLAlchemy engine to use for the migration
    """
    # Create an Alembic config pointing to alembic.ini in this package
    alembic_cfg = Config()
    alembic_cfg.set_main_option("script_location", str(package_dir / "alembic"))

    # Set the database connection URL
    if hasattr(engine, "url"):  # SQLAlchemy 1.4+
        url = str(engine.url)
    else:  # Older SQLAlchemy
        url = str(engine.engine.url)

    alembic_cfg.set_main_option("sqlalchemy.url", url)

    # Run the migration to the latest version
    with engine.begin() as connection:
        alembic_cfg.attributes["connection"] = connection
        command.upgrade(alembic_cfg, "head")

    return True


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


# Explicitly export the upgrade_db helper function
__all__ = ["get_engine", "upgrade_db", "SessionLocal", "get_writer"]
