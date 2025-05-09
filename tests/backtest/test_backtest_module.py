"""Tests for the algo_mvp.backtest module entry point."""

import importlib.util
import os
import sys
from unittest.mock import patch

# Load the src/algo_mvp/backtest.py file as a module to get its main function,
# bypassing the algo_mvp.backtest package.
# Assuming tests are run from the project root directory.
_path_to_backtest_py_file = os.path.abspath(
    os.path.join("src", "algo_mvp", "backtest.py")
)
_spec = importlib.util.spec_from_file_location(
    "algo_mvp_backtest_py_file", _path_to_backtest_py_file
)
_backtest_py_module = importlib.util.module_from_spec(_spec)
if _spec.loader:
    _spec.loader.exec_module(_backtest_py_module)
    backtest_main_entry_point = _backtest_py_module.main
else:
    # Fallback or error if loader is None, though spec_from_file_location should provide it
    raise ImportError(f"Could not load module from {_path_to_backtest_py_file}")


@patch("runpy.run_path")
def test_main_runs_cli_script(mock_run_path, monkeypatch):
    """Test that algo_mvp.backtest.main() executes the CLI script via runpy."""
    original_argv = sys.argv[:]
    test_script_name = "algo_mvp/backtest.py"  # This is what python -m would imply
    test_args = ["--config", "dummy.yaml"]
    monkeypatch.setattr(sys, "argv", [test_script_name] + test_args)

    expected_cli_script_path = os.path.abspath(os.path.join("cli", "run_backtest.py"))

    try:
        backtest_main_entry_point()
    finally:
        sys.argv = original_argv

    mock_run_path.assert_called_once()
    call_args = mock_run_path.call_args
    assert call_args[0][0] == expected_cli_script_path
    assert call_args[1]["run_name"] == "__main__"
    assert sys.argv == original_argv, "sys.argv was not restored correctly"

    captured_argv_at_runpy_call = []

    def runpy_side_effect(path, run_name):
        captured_argv_at_runpy_call.extend(sys.argv[:])  # Capture a copy
        return None  # Mock behavior

    mock_run_path.reset_mock()
    mock_run_path.side_effect = runpy_side_effect
    monkeypatch.setattr(sys, "argv", [test_script_name] + test_args)
    argv_for_this_run = sys.argv[:]

    try:
        backtest_main_entry_point()
    finally:
        sys.argv = argv_for_this_run

    mock_run_path.assert_called_once()
    assert captured_argv_at_runpy_call == [expected_cli_script_path] + test_args
