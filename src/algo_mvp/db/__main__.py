"""
CLI for database migrations.

Usage:
  python -m algo_mvp.db upgrade    # Run latest migration
  python -m algo_mvp.db current    # Print head revision
"""

from algo_mvp.db.migrate import main

if __name__ == "__main__":
    main()
