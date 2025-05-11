"""Main entry point for running the dashboard via python -m algo_mvp.dashboard."""

import os
import subprocess
import sys


def main():
    """Run the Streamlit dashboard as a module."""
    # Get the path to the dashboard app
    current_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(current_dir, "app.py")

    # Verify the app file exists
    if not os.path.exists(app_path):
        print(f"Error: Dashboard app not found at {app_path}", file=sys.stderr)
        return 1

    try:
        # Run Streamlit with the dashboard app
        print(f"Starting Streamlit dashboard from {app_path}")

        # Run on 0.0.0.0 to make it accessible from other machines on the network
        cmd = ["streamlit", "run", app_path, "--server.address", "0.0.0.0"]

        # Execute the command
        process = subprocess.run(cmd, check=True)
        return process.returncode
    except subprocess.CalledProcessError as e:
        print(f"Error running Streamlit: {e}", file=sys.stderr)
        return 1
    except FileNotFoundError:
        print("Error: Streamlit not found. Make sure it's installed.", file=sys.stderr)
        print("You can install it with: pip install streamlit", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
