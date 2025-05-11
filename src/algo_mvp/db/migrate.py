import argparse
import os

from alembic.config import Config
from alembic import command


def main():
    parser = argparse.ArgumentParser(description="Alembic migration CLI wrapper.")
    parser.add_argument(
        "action", choices=["upgrade", "current"], help="Action to perform."
    )
    parser.add_argument(
        "--url", help="Database URL to override ALGO_DB_URL environment variable."
    )

    args = parser.parse_args()

    # Construct the path to alembic.ini relative to this script
    # This assumes migrate.py is in src/algo_mvp/db/ and alembic.ini is in src/algo_mvp/db/alembic.ini
    script_dir = os.path.dirname(os.path.realpath(__file__))
    alembic_cfg_path = os.path.join(script_dir, "alembic.ini")

    alembic_cfg = Config(alembic_cfg_path)

    # Get the database URL
    db_url = args.url or os.getenv("ALGO_DB_URL", "sqlite:///data/algo.db")
    alembic_cfg.set_main_option("sqlalchemy.url", db_url)

    # For SQLite file URLs, ensure the parent directory exists
    if db_url.startswith("sqlite:///"):
        # Extract the file path part (skip sqlite:///)
        file_path = db_url[10:]
        # Only create directory if it's a file path, not memory or relative path
        if file_path and not file_path.startswith(":") and "/" in file_path:
            # Create parent directory if it doesn't exist
            dir_path = os.path.dirname(file_path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)

    if args.action == "upgrade":
        command.upgrade(alembic_cfg, "head")
        print("Database upgraded to the latest revision.")
    elif args.action == "current":
        command.current(alembic_cfg, verbose=True)


if __name__ == "__main__":
    main()
