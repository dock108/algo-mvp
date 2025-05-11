import sys

# Ensure the CLI module can be found
# This adjusts the Python path to include the 'src' directory,
# allowing for `from algo_mvp.cli.run_orchestrator import main`
# This is often necessary when running a module with `python -m`
# from a different working directory or when `src` is not directly in PYTHONPATH.

# Assuming the package structure is src/algo_mvp/...
# and __main__.py is in src/algo_mvp/orchestrator/
# We need to go up two levels to reach 'src' and then access algo_mvp.cli
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

# A potentially more robust way if `algo_mvp` is installed or src is in PYTHONPATH:
try:
    from algo_mvp.cli.run_orchestrator import main
except ImportError:
    # Fallback for when `src` is not in PYTHONPATH and the module is run directly.
    # Adjust path to allow finding algo_mvp.cli from algo_mvp.orchestrator
    # current_dir = os.path.dirname(os.path.abspath(__file__)) # .../src/algo_mvp/orchestrator
    # parent_dir = os.path.dirname(current_dir) # .../src/algo_mvp
    # src_dir = os.path.dirname(parent_dir) # .../src
    # We need to make sure that 'src' is effectively treated as a source root for imports.
    # If `python -m algo_mvp.orchestrator` is run from the workspace root where `src` is,
    # then `from algo_mvp.cli...` should work if `src` is implicitly part of path resolution.

    # Let's assume standard Python module resolution will handle it if run correctly (e.g. `PYTHONPATH=.` or running from root)
    # If issues persist, the sys.path manipulation above might be needed, but it's best to avoid if possible.
    # For now, we rely on the standard import mechanism.
    # One common pattern is to ensure the top-level package directory (containing algo_mvp)
    # is in sys.path.

    # Simplified approach: The user is expected to have their PYTHONPATH set up correctly
    # or run `python -m` from a directory where `algo_mvp` is discoverable (e.g. from the root of the project, if `src` is the root for packages).
    # If `src` is a source root, then `algo_mvp.cli.run_orchestrator` is the canonical path.

    # The structure is /Users/michaelfuscoletti/Desktop/stocks/src/algo_mvp/orchestrator/__main__.py
    # We want to import from /Users/michaelfuscoletti/Desktop/stocks/src/algo_mvp/cli/run_orchestrator.py
    # If `python -m algo_mvp.orchestrator` is run from `/Users/michaelfuscoletti/Desktop/stocks/`
    # And PYTHONPATH includes `/Users/michaelfuscoletti/Desktop/stocks/src` or similar, it should work.
    # Or, if `src` itself is the current working directory for `python -m`, then `algo_mvp.cli...` is fine.
    # Let's ensure the structure is `src` and then `algo_mvp` package inside it.
    # Given the workspace is /Users/michaelfuscoletti/Desktop/stocks, and files are src/algo_mvp/...,
    # running `python -m algo_mvp.orchestrator` from `/Users/michaelfuscoletti/Desktop/stocks` means Python will look
    # for `algo_mvp` in `sys.path`. If `src` is not automatically added, we might need to add it.
    # However, `python -m` typically handles this by adding the parent of the top-level package it finds to sys.path.
    # So, if `algo_mvp.orchestrator` is found, its parent `algo_mvp` should be resolvable for other submodules.

    # Trying a relative import approach, assuming __main__.py is part of the algo_mvp package.
    # `..` goes from `orchestrator` to `algo_mvp`, then `.cli.run_orchestrator`
    try:
        from ..cli.run_orchestrator import main
    except ImportError as e:
        print("Error: Could not import the CLI main function for the orchestrator.")
        print(
            "Please ensure your PYTHONPATH is set up correctly or run from the project root."
        )
        print(f"Details: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
