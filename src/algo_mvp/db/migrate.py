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

    if args.url:
        alembic_cfg.set_main_option("sqlalchemy.url", args.url)
    elif os.getenv("ALGO_DB_URL"):
        alembic_cfg.set_main_option("sqlalchemy.url", os.getenv("ALGO_DB_URL"))
    # If neither --url nor ALGO_DB_URL is set, Alembic will use the default from alembic.ini (via env.py)

    if args.action == "upgrade":
        command.upgrade(alembic_cfg, "head")
        print("Database upgraded to the latest revision.")
    elif args.action == "current":
        command.current(alembic_cfg, verbose=True)


if __name__ == "__main__":
    main()
