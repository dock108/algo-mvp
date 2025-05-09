#!/usr/bin/env python
"""Entry point for python -m algo_mvp.backtest"""

import os
import runpy
import sys


def main():
    """Run the backtest CLI module."""
    # Find the path to cli/run_backtest.py relative to this file
    # This file is src/algo_mvp/backtest.py
    # cli/run_backtest.py is ../../cli/run_backtest.py from its perspective
    cli_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "cli")
    )
    backtest_script_path = os.path.join(cli_dir, "run_backtest.py")

    # Add the project root to sys.path so that `cli.run_backtest` can import `algo_mvp`
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Store original sys.argv and replace it for the script run
    original_argv = sys.argv[:]
    # The first argument to run_path's script is the script name itself
    # Subsequent args are actual script arguments
    sys.argv = [backtest_script_path] + original_argv[1:]

    try:
        runpy.run_path(backtest_script_path, run_name="__main__")
    finally:
        # Restore original sys.argv
        sys.argv = original_argv


if __name__ == "__main__":
    main()
