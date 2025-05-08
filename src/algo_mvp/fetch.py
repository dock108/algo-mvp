# Entry point for python -m algo_mvp.fetch

import os
import runpy
import sys


def main():
    # This allows cli/fetch_data.py to be run as if it were a module
    # and correctly discover other modules within the project structure.
    # Construct path to cli/fetch_data.py relative to this file.
    # This file is src/algo_mvp/fetch.py
    # cli/fetch_data.py is ../../cli/fetch_data.py from its perspective.
    cli_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "cli")
    )
    fetch_script_path = os.path.join(cli_dir, "fetch_data.py")

    # Add the project root to sys.path so that `cli.fetch_data` can import `algo_mvp`
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # We need to ensure 'cli' directory is also discoverable if fetch_data.py uses `from . import ...` for helpers in `cli`
    # or if we want to be able to do `import cli.fetch_data` from elsewhere after this path modification.
    # However, runpy.run_path will execute fetch_script_path in its own module scope.

    # Store original sys.argv and replace it for the script run
    original_argv = sys.argv[:]
    # The first argument to run_path's script is the script name itself.
    # Subsequent args are actual script arguments.
    sys.argv = [fetch_script_path] + original_argv[1:]

    try:
        runpy.run_path(fetch_script_path, run_name="__main__")
    finally:
        # Restore original sys.argv
        sys.argv = original_argv


if __name__ == "__main__":
    main()
