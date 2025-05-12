import subprocess
import time
import os
import sys
import pytest
import requests
import yaml
from typing import Generator, Tuple

# Define a test token
TEST_CLI_SUPERVISOR_TOKEN = "test_cli_shutdown_token_456"


@pytest.fixture(scope="module")
def supervisor_configs(tmp_path_factory) -> Generator[Tuple[str, str], None, None]:
    """Creates temporary supervisor and orchestrator config YAML files for CLI tests."""
    tmp_path = tmp_path_factory.mktemp("supervisor_cli_configs")

    # Create a dummy orchestrator config
    orch_config_content = {
        "runners": [{"name": "cli_runner1", "config": "dummy.yaml"}],
        "log_level": "DEBUG",
    }
    orchestrator_config_file = tmp_path / "orchestrator_cli_test_config.yaml"
    with open(orchestrator_config_file, "w") as f:
        yaml.dump(orch_config_content, f)

    # Create supervisor config pointing to the dummy orchestrator config
    supervisor_config_content = {
        "orchestrator_config": str(orchestrator_config_file),
        "host": "127.0.0.1",
        "port": 8889,  # Use a distinct port for CLI tests
        "log_level": "DEBUG",
    }
    supervisor_config_file = tmp_path / "supervisor_cli_test_config.yaml"
    with open(supervisor_config_file, "w") as f:
        yaml.dump(supervisor_config_content, f)

    yield str(supervisor_config_file), str(orchestrator_config_file)


def check_health(port: int, retries=5, delay=1) -> bool:
    """Polls the health endpoint."""
    for i in range(retries):
        try:
            response = requests.get(f"http://127.0.0.1:{port}/health", timeout=1)
            if (
                response.status_code == 200 or response.status_code == 503
            ):  # Allow 503 if orchestrator not fully up
                print(
                    f"Health check attempt {i + 1} successful with status {response.status_code}"
                )
                return True
        except requests.exceptions.ConnectionError:
            print(
                f"Health check attempt {i + 1} failed (ConnectionError), retrying in {delay}s..."
            )
            time.sleep(delay)
        except requests.exceptions.Timeout:
            print(
                f"Health check attempt {i + 1} failed (Timeout), retrying in {delay}s..."
            )
            time.sleep(delay)
    return False


def test_supervisor_cli_startup_and_shutdown(
    supervisor_configs: Tuple[str, str], monkeypatch
):
    """
    Tests basic supervisor startup via CLI and graceful shutdown via API.
    """
    supervisor_config_path, _ = supervisor_configs

    # Get supervisor port from config for requests
    with open(supervisor_config_path, "r") as f:
        sup_config_data = yaml.safe_load(f)
    port = sup_config_data.get("port", 8889)

    # Set environment variables for the subprocess
    env = os.environ.copy()
    env["SUPERVISOR_TOKEN"] = TEST_CLI_SUPERVISOR_TOKEN
    env.pop("PYTHONPATH", None)  # Clear PYTHONPATH
    env.pop("PYTHONHOME", None)  # Clear PYTHONHOME

    project_src_dir = os.path.join(os.getcwd(), "src")
    existing_python_path = env.get("PYTHONPATH")
    if existing_python_path:
        env["PYTHONPATH"] = f"{project_src_dir}{os.pathsep}{existing_python_path}"
    else:
        env["PYTHONPATH"] = project_src_dir

    # Command to run the supervisor CLI
    # Assumes tests are run from the workspace root where src/ is.
    # If src is not in PYTHONPATH for the subprocess, it might fail to find algo_mvp.
    # The `sys.executable` ensures we use the same Python interpreter.
    cmd = [
        sys.executable,
        "-m",
        "algo_mvp.cli.run_supervisor",
        "--config",
        supervisor_config_path,
    ]

    process = None
    try:
        print(f"Starting supervisor with command: {' '.join(cmd)}")
        print(f"Using PYTHONPATH: {env.get('PYTHONPATH')}")
        process = subprocess.Popen(
            cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        # Wait for the supervisor to start up
        assert check_health(port, retries=10, delay=1), (
            "Supervisor health endpoint not responding."
        )
        print("Supervisor health check passed.")

        # Attempt graceful shutdown
        shutdown_url = (
            f"http://127.0.0.1:{port}/shutdown?token={TEST_CLI_SUPERVISOR_TOKEN}"
        )
        print(f"Sending shutdown request to: {shutdown_url}")
        try:
            response = requests.post(shutdown_url, timeout=5)
            assert response.status_code == 200
            print(f"Shutdown request successful: {response.json().get('message')}")
        except requests.exceptions.RequestException as e:
            pytest.fail(f"Shutdown request failed: {e}")

        # Wait for the process to terminate
        try:
            process.wait(timeout=25)
        except subprocess.TimeoutExpired:
            print("Supervisor process did not terminate in time, killing...")
            process.kill()
            process.wait()
            pytest.fail(
                "Supervisor process did not terminate gracefully after shutdown request."
            )

        assert process.returncode is not None, "Process return code is None after wait."
        # Uvicorn's default exit code on graceful shutdown (e.g. via SIGINT or should_exit) is 0.
        # If it's killed, it might be different.
        print(f"Supervisor process terminated with return code: {process.returncode}")

        # A return code of 0 or a signal-induced negative value (like -SIGTERM) might be acceptable for graceful.
        # For now, let's be a bit flexible, but ideally it's 0.
        # On POSIX, exit due to SIGTERM (15) results in returncode -15 if not handled.
        # Uvicorn handles SIGTERM and tries to exit gracefully with 0.
        # If our `server.should_exit = True` works, Uvicorn should exit with 0.
        assert process.returncode == 0, (
            f"Supervisor process exited with code {process.returncode}, expected 0 for graceful shutdown."
        )

    finally:
        if process and process.poll() is None:
            print(
                "Cleaning up: Supervisor process still running, terminating forcefully."
            )
            process.kill()
            process.wait()

        if process:
            # Print stdout/stderr for debugging if the test fails
            stdout, stderr = (
                process.communicate()
            )  # Should have been read by wait() if ended, but just in case
            if stdout:
                print("Supervisor STDOUT:")
                print(stdout)
            if stderr:
                print("Supervisor STDERR:")
                print(stderr)


def test_supervisor_cli_missing_supervisor_config(tmp_path):
    """Tests that the supervisor CLI exits if the supervisor config file is missing."""
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)  # Clear PYTHONPATH
    env.pop("PYTHONHOME", None)  # Clear PYTHONHOME

    missing_config_path = tmp_path / "non_existent_supervisor_config.yaml"

    cmd = [
        sys.executable,
        "-m",
        "algo_mvp.cli.run_supervisor",
        "--config",
        str(missing_config_path),
    ]

    print(f"Attempting to start supervisor with missing config: {' '.join(cmd)}")
    process = subprocess.Popen(
        cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    try:
        stdout, stderr = process.communicate(timeout=10)  # CLI should exit quickly
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()
        pytest.fail("Supervisor CLI did not exit in time when config was missing.")

    print(f"CLI STDOUT: {stdout}")
    print(f"CLI STDERR: {stderr}")

    assert process.returncode != 0, (
        "CLI should exit with a non-zero code for missing supervisor config."
    )
    # run_supervisor.py explicitly logs to logger.error and sys.exit(1)
    assert (
        "Error: Supervisor configuration file not found" in stderr
        or "Error: Supervisor configuration file not found" in stdout
    )  # Depending on logger setup in CLI for early errors


def test_supervisor_cli_missing_orchestrator_config(tmp_path_factory):
    """Tests CLI failure if supervisor config points to a missing orchestrator config."""
    tmp_path = tmp_path_factory.mktemp("missing_orch_config")

    # Create a supervisor config file that points to a non-existent orchestrator config
    missing_orchestrator_config_path = tmp_path / "non_existent_orchestrator.yaml"
    supervisor_config_content = {
        "orchestrator_config": str(missing_orchestrator_config_path),
        "host": "127.0.0.1",
        "port": 8890,  # Yet another port
        "log_level": "DEBUG",
    }
    supervisor_config_file = tmp_path / "supervisor_for_missing_orch.yaml"
    with open(supervisor_config_file, "w") as f:
        yaml.dump(supervisor_config_content, f)

    env = os.environ.copy()
    env.pop("PYTHONPATH", None)  # Clear PYTHONPATH
    env.pop("PYTHONHOME", None)  # Clear PYTHONHOME

    cmd = [
        sys.executable,
        "-m",
        "algo_mvp.cli.run_supervisor",
        "--config",
        str(supervisor_config_file),
    ]

    print(f"Starting supervisor with missing orchestrator config: {' '.join(cmd)}")
    process = subprocess.Popen(
        cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    try:
        stdout, stderr = process.communicate(timeout=10)  # CLI should exit quickly
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()
        pytest.fail(
            "Supervisor CLI did not exit in time when orchestrator config was missing."
        )

    print(f"CLI STDOUT: {stdout}")
    print(f"CLI STDERR: {stderr}")

    assert process.returncode != 0, (
        "CLI should exit with non-zero code for missing orchestrator config."
    )
    # run_supervisor.py checks this and logs/exits.
    expected_error_msg_part = f"Error: Orchestrator config file '{str(missing_orchestrator_config_path)}' not found"
    assert expected_error_msg_part in stderr or expected_error_msg_part in stdout


def test_supervisor_cli_shutdown_token_validation(
    supervisor_configs: Tuple[str, str], monkeypatch
):
    """Tests shutdown token validation for a CLI-launched supervisor."""
    supervisor_config_path, _ = supervisor_configs

    with open(supervisor_config_path, "r") as f:
        sup_config_data = yaml.safe_load(f)
    port = sup_config_data.get(
        "port", 8889
    )  # Should match port in supervisor_configs fixture

    env = os.environ.copy()
    env["SUPERVISOR_TOKEN"] = TEST_CLI_SUPERVISOR_TOKEN  # Actual token
    env.pop("PYTHONPATH", None)  # Clear PYTHONPATH
    env.pop("PYTHONHOME", None)  # Clear PYTHONHOME

    cmd = [
        sys.executable,
        "-m",
        "algo_mvp.cli.run_supervisor",
        "--config",
        supervisor_config_path,
    ]

    process = None
    try:
        print(f"Starting supervisor for token validation test: {' '.join(cmd)}")
        process = subprocess.Popen(
            cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        assert check_health(port, retries=10, delay=1), (
            "Supervisor health endpoint not responding for token test."
        )
        print("Supervisor health check passed for token test.")

        # 1. Attempt shutdown with NO token
        shutdown_url_base = f"http://127.0.0.1:{port}/shutdown"
        print(f"Attempting shutdown with no token: POST {shutdown_url_base}")
        try:
            response_no_token = requests.post(shutdown_url_base, timeout=3)
            # Expect 401 Unauthorized if token is required but not provided via query param
            assert response_no_token.status_code == 401, (
                "Should be 401 for missing token"
            )
            print(
                f"Correctly received {response_no_token.status_code} for missing token."
            )
        except requests.exceptions.RequestException as e:
            pytest.fail(f"Shutdown request (no token) failed: {e}")

        # 2. Attempt shutdown with an INVALID token
        invalid_token = "this_is_not_the_correct_token"
        shutdown_url_invalid = f"{shutdown_url_base}?token={invalid_token}"
        print(f"Attempting shutdown with invalid token: POST {shutdown_url_invalid}")
        try:
            response_invalid_token = requests.post(shutdown_url_invalid, timeout=3)
            assert response_invalid_token.status_code == 403, (
                "Should be 403 for invalid token"
            )
            print(
                f"Correctly received {response_invalid_token.status_code} for invalid token."
            )
        except requests.exceptions.RequestException as e:
            pytest.fail(f"Shutdown request (invalid token) failed: {e}")

        # 3. Perform a VALID shutdown to clean up the process
        valid_shutdown_url = f"{shutdown_url_base}?token={TEST_CLI_SUPERVISOR_TOKEN}"
        print(f"Attempting valid shutdown: POST {valid_shutdown_url}")
        try:
            response_valid = requests.post(valid_shutdown_url, timeout=5)
            assert response_valid.status_code == 200
        except requests.exceptions.RequestException as e:
            # If this fails, the process might hang, so we must ensure it's killed in finally
            print(f"Valid shutdown request failed unexpectedly: {e}")
            pytest.fail(f"Valid shutdown for cleanup failed: {e}")

        try:
            process.wait(timeout=25)
        except subprocess.TimeoutExpired:
            print(
                "Supervisor (token test) did not terminate in time after valid shutdown, killing..."
            )
            process.kill()
            process.wait()
            pytest.fail(
                "Supervisor (token test) did not terminate gracefully after valid shutdown."
            )

        assert process.returncode == 0, (
            f"Supervisor (token test) exited with {process.returncode}, expected 0."
        )

    finally:
        if process and process.poll() is None:
            print(
                "Cleaning up (token test): Supervisor process still running, terminating forcefully."
            )
            process.kill()
            process.wait()
        if process:
            stdout, stderr = process.communicate()
            if stdout or stderr:
                print(f"Token Test STDOUT:\n{stdout}")
                print(f"Token Test STDERR:\n{stderr}")
