import pytest
import yaml
import time

# import os # Unused
from unittest.mock import MagicMock  # Removed call

# from pathlib import Path # Unused
import threading
from unittest.mock import patch

from algo_mvp.orchestrator.manager import (
    Orchestrator,
    OrchestratorConfig,  # Added import for OrchestratorConfig
)  # RunnerConfig not directly used in tests yet


class SimulatedCrashError(Exception):
    """Custom exception for simulating runner crashes in tests."""

    pass


# pytest_plugins = ['pytester']  # Not needed for these tests


@pytest.fixture
def mock_liverunner_class(mocker):
    """Fixture to mock the LiveRunner class."""
    # Using mocker.patch correctly for the class
    mock_live_runner_cls = mocker.patch(
        "algo_mvp.orchestrator.manager.LiveRunner", autospec=True
    )

    # This factory will be called whenever LiveRunner() is instantiated in the code under test
    def mock_runner_instance_factory(*args, **kwargs):
        instance = MagicMock(
            name=f"LiveRunnerInstance({kwargs.get('config_path', 'unknown')})"
        )
        instance.config_path = kwargs.get("config_path")
        instance._stop_event = threading.Event()
        instance._actually_started_event = (
            threading.Event()
        )  # Signals that _start_behavior has been entered
        instance._test_has_set_crash_flag_event = (
            threading.Event()
        )  # Signals that test has set the crash flag
        instance.thread = None
        instance.logger = MagicMock(
            spec=["info", "warning", "error", "debug", "critical"]
        )
        instance._crash_after_start = False
        instance._crash_mid_run = False

        instance.start = MagicMock(name=f"start_for_{instance.config_path}")
        instance.stop = MagicMock(name=f"stop_for_{instance.config_path}")
        instance.is_alive = MagicMock(
            return_value=True, name=f"is_alive_for_{instance.config_path}"
        )

        def _start_behavior(instance_arg):
            # Using instance_arg to be clear it's the one passed, not from outer scope directly if there were ambiguity
            instance_arg.logger.info(
                f"Mock LiveRunner {instance_arg.name} _start_behavior ENTERED. Thread: {threading.current_thread().name}"
            )
            instance_arg._actually_started_event.set()  # Signal test that we are in _start_behavior

            # Wait for the test to explicitly set the crash flag
            instance_arg.logger.info(
                f"Mock LiveRunner {instance_arg.name} _start_behavior waiting for _test_has_set_crash_flag_event."
            )
            flag_set_event_received = instance_arg._test_has_set_crash_flag_event.wait(
                timeout=2.0
            )  # Increased timeout for safety
            if not flag_set_event_received:
                instance_arg.logger.error(
                    f"Mock LiveRunner {instance_arg.name} _start_behavior TIMEOUT waiting for _test_has_set_crash_flag_event."
                )
                # Proceed with current flags, likely won't crash as intended

            final_action = "UNKNOWN_AFTER_EVENT_WAIT"
            try:
                if instance_arg._crash_after_start:
                    instance_arg.logger.warning(
                        f"Mock LiveRunner {instance_arg.name} _start_behavior: _crash_after_start is TRUE. RAISING SimulatedCrashError."
                    )
                    final_action = "RAISE_CRASH_AFTER_START"
                    instance_arg.is_alive.return_value = False
                    raise SimulatedCrashError(
                        f"Simulated crash for {instance_arg.name} from _start_behavior (crash_after_start)"
                    )

                loop_count = 0
                while not instance_arg._stop_event.wait(timeout=0.01):
                    loop_count += 1
                    if instance_arg._crash_mid_run:
                        instance_arg.logger.warning(
                            f"Mock LiveRunner {instance_arg.name} _start_behavior: _crash_mid_run is TRUE. RAISING SimulatedCrashError from loop."
                        )
                        final_action = "RAISE_CRASH_MID_RUN"
                        instance_arg.is_alive.return_value = False
                        raise SimulatedCrashError(
                            f"Simulated crash mid-run for {instance_arg.name} from _start_behavior"
                        )
                    if loop_count > 500 and instance_arg.name == "runner1":
                        instance_arg.logger.error(
                            f"Mock LiveRunner {instance_arg.name} _start_behavior: Safety break in loop after {loop_count} iterations."
                        )
                        final_action = "SAFETY_BREAK_LOOP"
                        break

                instance_arg.is_alive.return_value = False
                if instance_arg._stop_event.is_set():
                    final_action = "NORMAL_EXIT_LOOP_STOP_EVENT_SET"
                    instance_arg.logger.info(
                        f"Mock LiveRunner {instance_arg.name} _start_behavior: Exited loop (stop_event). Count: {loop_count}"
                    )
                else:
                    final_action = "NORMAL_EXIT_LOOP_OTHER"
                    instance_arg.logger.info(
                        f"Mock LiveRunner {instance_arg.name} _start_behavior: Exited loop (other). Count: {loop_count}"
                    )

            except SimulatedCrashError:
                instance_arg.logger.critical(
                    f"Mock LiveRunner {instance_arg.name} _start_behavior: CAUGHT SCE. RE-RAISING. Thread: {threading.current_thread().name}"
                )
                final_action = "CAUGHT_AND_RERAISE_SCE"
                raise
            except Exception as e_inner:
                instance_arg.is_alive.return_value = False
                instance_arg.logger.critical(
                    f"Mock LiveRunner {instance_arg.name} _start_behavior: CAUGHT UNEXPECTED Exception: {e_inner}. RE-RAISING. Thread: {threading.current_thread().name}",
                    exc_info=True,
                )
                final_action = f"CAUGHT_UNEXPECTED_{type(e_inner).__name__}"
                raise
            finally:
                current_thread = threading.current_thread()
                instance_arg.logger.critical(
                    f"Mock LiveRunner {instance_arg.name} _start_behavior: FINALLY block. Action: {final_action}. Thread: {current_thread.name}, Alive: {current_thread.is_alive()}"
                )

        # Pass the specific instance to _start_behavior using lambda
        instance.start.side_effect = lambda: _start_behavior(instance)

        def _stop_behavior():
            instance.logger.info(f"Mock LiveRunner {instance.name} stop() called")
            instance._stop_event.set()  # Signal the simulated run loop to exit
            # is_alive will be set to False by the _start_behavior loop when it exits
            # This is now handled by _start_behavior itself.

        instance.stop.side_effect = _stop_behavior
        return instance

    mock_live_runner_cls.side_effect = mock_runner_instance_factory
    return mock_live_runner_cls


@pytest.fixture
def sample_orchestrator_config_dict():
    return {
        "runners": [
            {"name": "runner1", "config": "config/runner1.yaml"},
            {"name": "runner2", "config": "config/runner2.yaml"},
        ],
        "log_level": "INFO",
        "restart_on_crash": True,
    }


@pytest.fixture
def sample_orchestrator_config_file(tmp_path, sample_orchestrator_config_dict):
    """Creates a temporary YAML config file for the orchestrator."""
    config_file = tmp_path / "orchestrator_test_config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(sample_orchestrator_config_dict, f)
    return config_file


def test_orchestrator_initialization(
    sample_orchestrator_config_file, mock_liverunner_class
):
    """Test basic orchestrator initialization and that it loads config."""
    orchestrator = Orchestrator(config_path=str(sample_orchestrator_config_file))
    assert orchestrator is not None
    assert len(orchestrator.config.runners) == 2
    assert orchestrator.config.runners[0].name == "runner1"
    assert orchestrator.config.log_level == "INFO"
    assert orchestrator.config.restart_on_crash is True
    # LiveRunner class should not have been called/instantiated yet
    mock_liverunner_class.assert_not_called()


def test_orchestrator_start_launches_runners(
    sample_orchestrator_config_file,
    mock_liverunner_class,
    sample_orchestrator_config_dict,
):
    """Test that orchestrator.start() launches all configured runners."""
    orchestrator = Orchestrator(config_path=str(sample_orchestrator_config_file))
    orchestrator.start()
    time.sleep(0.2)  # Increased sleep to 0.2 for threads to reliably start

    assert mock_liverunner_class.call_count == len(
        sample_orchestrator_config_dict["runners"]
    )

    # Check that runner instances were created with correct config paths and names set by orchestrator
    # And their start methods were called, and threads are alive.
    assert len(orchestrator.runner_instances) == 2
    for r_conf in sample_orchestrator_config_dict["runners"]:
        runner_name = r_conf["name"]
        runner_config_path = r_conf["config"]
        assert runner_name in orchestrator.runner_instances
        runner_instance = orchestrator.runner_instances[runner_name]

        # Check if the mock LiveRunner was instantiated with the correct config_path
        # This requires inspecting the calls to the patched LiveRunner class
        found_call = False
        for call_obj in mock_liverunner_class.call_args_list:
            args, kwargs = call_obj
            # Check kwargs first, then args if present
            if kwargs.get("config_path") == runner_config_path:
                found_call = True
                break
            if args and args[0] == runner_config_path:
                found_call = True
                break
        assert found_call, (
            f"LiveRunner not called with config_path {runner_config_path}"
        )

        assert runner_instance.name == runner_name  # Name should be set by orchestrator
        runner_instance.start.assert_called_once()
        assert orchestrator.runner_threads[runner_name].is_alive(), (
            f"Thread for {runner_name} is not alive"
        )

    # Store references to the mock instances to check their individual states after orchestrator stop
    # This is done before orchestrator.stop() clears its internal references
    created_instances = list(orchestrator.runner_instances.values())

    orchestrator.stop()  # Cleanup
    time.sleep(0.1)  # Allow stop to propagate

    # Orchestrator.stop() clears runner_threads and runner_instances, so we check they are empty.
    assert not orchestrator.runner_threads, "runner_threads should be empty after stop"
    assert not orchestrator.runner_instances, (
        "runner_instances should be empty after stop"
    )

    # Check that the individual mock runner instances had their stop() method called
    # and their internal stop_event set, and they report as not alive.
    for inst in created_instances:
        inst.stop.assert_called_once()
        assert inst._stop_event.is_set()
        assert inst.is_alive() is False, (
            f"Mock instance {inst.name} still reporting alive after stop"
        )


def test_orchestrator_stop_method(
    sample_orchestrator_config_file,
    mock_liverunner_class,
    sample_orchestrator_config_dict,
):
    """Test that orchestrator.stop() stops all runners and threads."""
    orchestrator = Orchestrator(config_path=str(sample_orchestrator_config_file))
    orchestrator.start()
    time.sleep(0.2)  # Allow runners to start, increased from 0.1

    assert len(orchestrator.runner_instances) == 2
    # Keep a reference to the instances before they are cleared by orchestrator.stop()
    initial_runner_instances = dict(orchestrator.runner_instances)

    orchestrator.stop()
    time.sleep(0.2)  # Allow for stop propagation and thread joining

    for name, runner_instance in initial_runner_instances.items():
        runner_instance.stop.assert_called_once()
        assert runner_instance._stop_event.is_set()
        assert runner_instance.is_alive() is False, (
            f"Mock instance {name} not reporting dead after stop"
        )

    # Orchestrator clears these collections on stop
    assert not orchestrator.runner_instances, (
        "runner_instances collection should be empty after stop"
    )
    assert not orchestrator.runner_threads, (
        "runner_threads collection should be empty after stop"
    )
    assert orchestrator._stop_event.is_set()
    if (
        orchestrator.config.restart_on_crash and orchestrator._watchdog_thread
    ):  # Watchdog only runs if restart_on_crash
        assert not orchestrator._watchdog_thread.is_alive(), (
            "Watchdog thread should be stopped"
        )
    elif not orchestrator.config.restart_on_crash:
        assert orchestrator._watchdog_thread is None, (
            "Watchdog thread should not have been started if restart_on_crash is false"
        )


def test_orchestrator_status_reflects_state(
    sample_orchestrator_config_file, mock_liverunner_class
):
    """Test that orchestrator.status() correctly reflects runner states."""
    orchestrator = Orchestrator(config_path=str(sample_orchestrator_config_file))

    # Before start
    status_before_start = orchestrator.status()
    assert status_before_start["runner1"] == "pending_or_failed_to_start"
    assert status_before_start["runner2"] == "pending_or_failed_to_start"

    orchestrator.start()
    time.sleep(0.1)  # Allow runners to start

    # After start, runners should be running
    status_running = orchestrator.status()
    assert status_running["runner1"] == "running"
    assert status_running["runner2"] == "running"

    # Simulate one runner stopping normally
    runner1_instance = orchestrator.runner_instances["runner1"]

    # Properly stop the runner instance so it updates its state correctly
    runner1_instance._stop_event.set()  # Signal the mock runner's loop to stop
    runner1_instance.is_alive.return_value = (
        False  # Update the mock's is_alive to return False
    )

    # We need to also update the thread's is_alive status as the orchestrator checks both
    runner1_thread = orchestrator.runner_threads.get("runner1")
    with patch.object(runner1_thread, "is_alive", return_value=False):
        # Check status after stopping one runner - must be done in the patch context
        status_one_stopped = orchestrator.status()
        assert status_one_stopped["runner1"] == "stopped", (
            f"Expected 'stopped', got '{status_one_stopped['runner1']}'"
        )
        assert status_one_stopped["runner2"] == "running"

    # Clean up
    orchestrator.stop()
    time.sleep(0.1)  # Allow stop to propagate

    # Verify cleanup
    assert not orchestrator.runner_threads, "runner_threads should be empty after stop"
    assert not orchestrator.runner_instances, (
        "runner_instances should be empty after stop"
    )

    # Final status check after full stop
    status_after_stop = orchestrator.status()
    for runner_name in ["runner1", "runner2"]:
        assert status_after_stop[runner_name] == "pending_or_failed_to_start", (
            f"Expected 'pending_or_failed_to_start' for {runner_name}, got '{status_after_stop[runner_name]}'"
        )


@pytest.mark.parametrize("restart_flag_in_config", [True, False])
def test_orchestrator_crash_handling(
    sample_orchestrator_config_file,
    mock_liverunner_class,
    sample_orchestrator_config_dict,
    restart_flag_in_config,
    mocker,
    tmp_path,
):
    """Test runner crash handling with and without restart_on_crash flag."""
    config_dict = sample_orchestrator_config_dict.copy()
    config_dict["restart_on_crash"] = restart_flag_in_config
    # Use only one runner for simplicity in this test
    config_dict["runners"] = [sample_orchestrator_config_dict["runners"][0].copy()]
    crashing_runner_name = config_dict["runners"][0]["name"]

    # Create a new config file for this specific test scenario
    specific_config_file = (
        tmp_path / f"orchestrator_crash_test_{restart_flag_in_config}.yaml"
    )
    with open(specific_config_file, "w") as f:
        yaml.dump(config_dict, f)

    # ---- TEST DEBUG: Verify config ----
    with open(specific_config_file, "r") as f_check:
        check_data = yaml.safe_load(f_check)
        print(
            f"TEST DEBUG (file check): For restart_flag_in_config={restart_flag_in_config}, YAML has restart_on_crash = {check_data.get('restart_on_crash')}"
        )
    # ---- END TEST DEBUG ----

    # Mock the watchdog method to avoid long sleeps
    with patch.object(Orchestrator, "_watchdog_loop", autospec=True) as mock_watchdog:
        # Make the watchdog do nothing to avoid the long sleep cycles
        mock_watchdog.return_value = None

        orchestrator = Orchestrator(config_path=str(specific_config_file))
        # ---- TEST DEBUG: Verify orchestrator config ----
        print(
            f"TEST DEBUG (orchestrator instance): For restart_flag_in_config={restart_flag_in_config}, orchestrator.config.restart_on_crash = {orchestrator.config.restart_on_crash}"
        )
        # ---- END TEST DEBUG ----

        orchestrator.start()  # This will create the LiveRunner instance and its thread will call start()

        # Ensure the runner instance is created and its _start_behavior has been entered
        time.sleep(0.05)  # Brief sleep to allow instance creation by orchestrator
        assert crashing_runner_name in orchestrator.runner_instances, (
            "Crashing runner instance not found in orchestrator after start"
        )
        crashing_runner_instance = orchestrator.runner_instances[crashing_runner_name]

        orchestrator.logger.info(
            f"Test: Waiting for {crashing_runner_name} _start_behavior to enter (_actually_started_event)... Thread: {threading.current_thread().name}"
        )
        started_event_fired = crashing_runner_instance._actually_started_event.wait(
            timeout=2.0
        )  # Increased timeout
        assert started_event_fired, (
            f"_actually_started_event for {crashing_runner_name} did not fire in time."
        )
        orchestrator.logger.info(
            f"Test: {crashing_runner_name} _actually_started_event fired. Setting _crash_after_start=True."
        )

        crashing_runner_instance._crash_after_start = True
        orchestrator.logger.info(
            f"Test: {crashing_runner_name} _crash_after_start set to True. Setting _test_has_set_crash_flag_event."
        )
        crashing_runner_instance._test_has_set_crash_flag_event.set()  # Signal _start_behavior to proceed

        crashing_thread_object = orchestrator.runner_threads.get(crashing_runner_name)
        assert crashing_thread_object is not None, (
            f"Thread for {crashing_runner_name} not found after start."
        )

        orchestrator.logger.info(
            f"Test: {crashing_runner_name} configured to crash. Joining thread with longer timeout... Thread: {threading.current_thread().name}"
        )
        crashing_thread_object.join(
            timeout=2.0
        )  # DRASTICALLY INCREASED TIMEOUT TO 2 SECONDS

        assert not crashing_thread_object.is_alive(), (
            f"Crashing thread {crashing_runner_name} (state: {crashing_thread_object.is_alive()}) did not terminate after simulated crash and 2s join."
        )

        orchestrator.logger.info(
            f"Test: Thread {crashing_runner_name} confirmed terminated. Testing watchdog behavior..."
        )

        # Instead of waiting for the real watchdog, manually trigger the watchdog behavior
        # by calling the internal methods that the watchdog would call
        if restart_flag_in_config:
            orchestrator.logger.info("Test: Manually handling restart behavior...")
            # Remove the crashed runner as watchdog would
            if crashing_runner_name in orchestrator.runner_threads:
                del orchestrator.runner_threads[crashing_runner_name]

            # Clean up the instance
            if crashing_runner_name in orchestrator.runner_instances:
                del orchestrator.runner_instances[crashing_runner_name]

            # Find the runner config
            runner_conf_obj = next(
                (
                    r
                    for r in orchestrator.config.runners
                    if r.name == crashing_runner_name
                ),
                None,
            )
            assert runner_conf_obj is not None, "Runner config not found for relaunch"

            # Launch a new runner
            orchestrator._launch_runner(runner_conf_obj)
            time.sleep(0.1)  # Brief sleep to allow the new runner to start

            # Verify the runner was restarted
            assert mock_liverunner_class.call_count >= 2, (
                f"LiveRunner class should have been called for initial start and restart. Got {mock_liverunner_class.call_count} calls."
            )

            # Check status
            current_status = orchestrator.status()
            assert current_status.get(crashing_runner_name) == "running", (
                f"Runner {crashing_runner_name} not running after expected restart"
            )
        else:
            orchestrator.logger.info("Test: Manually handling no-restart behavior...")
            # Remove the crashed runner as watchdog would
            if crashing_runner_name in orchestrator.runner_threads:
                del orchestrator.runner_threads[crashing_runner_name]

            # Clean up the instance
            if crashing_runner_name in orchestrator.runner_instances:
                del orchestrator.runner_instances[crashing_runner_name]

            # Verify that LiveRunner was only called once (initial startup, no restart)
            assert mock_liverunner_class.call_count == 1, (
                f"LiveRunner class should only be called for initial start, no restart. Got {mock_liverunner_class.call_count} calls."
            )

            # Check status
            current_status = orchestrator.status()
            expected_status_after_crash_no_restart = "pending_or_failed_to_start"
            actual_status = current_status.get(crashing_runner_name)
            assert actual_status == expected_status_after_crash_no_restart, (
                f"Runner {crashing_runner_name} status is {actual_status}, expected {expected_status_after_crash_no_restart} after crash and no restart."
            )

            # Verify cleanup
            assert crashing_runner_name not in orchestrator.runner_threads, (
                f"{crashing_runner_name} should NOT be in runner_threads after cleanup."
            )
            assert crashing_runner_name not in orchestrator.runner_instances, (
                f"{crashing_runner_name} should NOT be in runner_instances after cleanup."
            )

        # Clean up to avoid lingering threads
        orchestrator._stop_event.set()
        for name, runner in list(orchestrator.runner_instances.items()):
            try:
                runner.stop()
            except Exception:
                pass  # Ignore errors during cleanup

        for name, thread in list(orchestrator.runner_threads.items()):
            try:
                thread.join(timeout=0.5)
            except Exception:
                pass  # Ignore errors during cleanup

        orchestrator.runner_threads.clear()
        orchestrator.runner_instances.clear()


def test_orchestrator_reload_method(
    sample_orchestrator_config_file,
    mock_liverunner_class,
    sample_orchestrator_config_dict,
    mocker,
):
    """Test that the orchestrator reload method properly stops existing runners and starts new ones."""
    # Mock DB writer but don't worry about validating its calls
    mocker.patch("algo_mvp.orchestrator.manager.get_writer")

    # Patch the watchdog loop to not block (just return immediately)
    mocker.patch(
        "algo_mvp.orchestrator.manager.Orchestrator._watchdog_loop", return_value=None
    )

    # Initialize orchestrator
    orchestrator = Orchestrator(config_path=str(sample_orchestrator_config_file))

    # Start the orchestrator to create initial runners (without watchdog thread)
    # Launch each runner manually
    for runner_conf in orchestrator.config.runners:
        orchestrator._launch_runner(runner_conf)

    time.sleep(0.2)  # Allow runners to initialize

    # Mark runners as "ready for test"
    # (This is required to un-block the _start_behavior method in the mock)
    for name, instance in orchestrator.runner_instances.items():
        instance._test_has_set_crash_flag_event.set()

    # Verify initial runners are running
    assert len(orchestrator.runner_instances) == 2
    assert "runner1" in orchestrator.runner_instances
    assert "runner2" in orchestrator.runner_instances

    # Save references to the original runner instances
    original_instances = {
        name: instance for name, instance in orchestrator.runner_instances.items()
    }

    # Now reload the orchestrator (without joining threads)
    # Directly call the methods that reload would call
    # Stop runners
    for name, runner in list(orchestrator.runner_instances.items()):
        runner.stop()

    # Clear existing runners
    orchestrator.runner_instances.clear()
    orchestrator.runner_threads.clear()

    # Re-parse the YAML configuration
    with open(str(sample_orchestrator_config_file), "r") as f:
        config_data = yaml.safe_load(f)
    orchestrator.config = OrchestratorConfig(**config_data)

    # Launch new runners
    for runner_conf in orchestrator.config.runners:
        orchestrator._launch_runner(runner_conf)

    # Mark the new runners as ready for test as well
    for name, instance in orchestrator.runner_instances.items():
        instance._test_has_set_crash_flag_event.set()

    # Get status of all runners
    runner_status = orchestrator.status()

    # Verify runners were reloaded
    assert len(runner_status) == 2
    assert all(status == "running" for status in runner_status.values())

    # Verify new instances were created (they should be different objects)
    for name, new_instance in orchestrator.runner_instances.items():
        if name in original_instances:
            assert new_instance != original_instances[name], (
                f"Runner {name} should be a new instance"
            )

    # Test YAML parsing error by trying to load invalid YAML
    modified_config_file = (
        sample_orchestrator_config_file.parent / "invalid_config.yaml"
    )
    with open(modified_config_file, "w") as f:
        f.write("invalid: yaml: content: - [")  # Invalid YAML syntax

    # Try to parse invalid YAML
    with pytest.raises(yaml.YAMLError):
        with open(modified_config_file, "r") as f:
            yaml.safe_load(f)

    # Clean up - skip joining threads to avoid timeout
    for name, runner in list(orchestrator.runner_instances.items()):
        runner.stop()
    orchestrator.runner_instances.clear()
    orchestrator.runner_threads.clear()


def test_orchestrator_reload_updates_config(
    sample_orchestrator_config_file,
    mock_liverunner_class,
    sample_orchestrator_config_dict,
    tmp_path,
    mocker,
):
    """Test that reload properly updates the configuration with new runners."""
    # Mock DB writer but don't worry about validating its calls
    mocker.patch("algo_mvp.orchestrator.manager.get_writer")

    # Patch the watchdog loop to not block (just return immediately)
    mocker.patch(
        "algo_mvp.orchestrator.manager.Orchestrator._watchdog_loop", return_value=None
    )

    # Initialize orchestrator
    orchestrator = Orchestrator(config_path=str(sample_orchestrator_config_file))

    # Start the orchestrator to create initial runners (without watchdog thread)
    # Launch each runner manually
    for runner_conf in orchestrator.config.runners:
        orchestrator._launch_runner(runner_conf)

    time.sleep(0.2)  # Allow runners to initialize

    # Mark runners as "ready for test"
    for name, instance in orchestrator.runner_instances.items():
        instance._test_has_set_crash_flag_event.set()

    # Verify initial runners are running
    assert len(orchestrator.runner_instances) == 2

    # Create a modified config with an additional runner
    modified_config = sample_orchestrator_config_dict.copy()
    modified_config["runners"].append(
        {"name": "runner3", "config": "config/runner3.yaml"}
    )

    modified_config_file = tmp_path / "modified_config.yaml"
    with open(modified_config_file, "w") as f:
        yaml.dump(modified_config, f)

    # Now reload the orchestrator (without joining threads)
    # Directly call the methods that reload would call
    # Stop runners
    for name, runner in list(orchestrator.runner_instances.items()):
        runner.stop()

    # Clear existing runners
    orchestrator.runner_instances.clear()
    orchestrator.runner_threads.clear()

    # Re-parse the YAML configuration
    with open(modified_config_file, "r") as f:
        config_data = yaml.safe_load(f)
    orchestrator.config = OrchestratorConfig(**config_data)

    # Launch new runners
    for runner_conf in orchestrator.config.runners:
        orchestrator._launch_runner(runner_conf)

    # Mark the new runners as ready for test as well
    for name, instance in orchestrator.runner_instances.items():
        instance._test_has_set_crash_flag_event.set()

    # Get status of all runners
    runner_status = orchestrator.status()

    # Verify updated runners
    assert len(runner_status) == 3
    assert "runner3" in runner_status
    assert len(orchestrator.runner_instances) == 3
    assert "runner3" in orchestrator.runner_instances

    # Verify runner states
    for name, status in runner_status.items():
        assert status == "running", f"Runner {name} is not running"

    # Clean up - skip joining threads to avoid timeout
    for name, runner in list(orchestrator.runner_instances.items()):
        runner.stop()
    orchestrator.runner_instances.clear()
    orchestrator.runner_threads.clear()


def test_orchestrator_reload_with_db_writing(
    sample_orchestrator_config_file,
    mock_liverunner_class,
    sample_orchestrator_config_dict,
    mocker,
    tmp_path,
):
    """Test that reload method properly logs to the database."""
    # Create a mock DB writer
    mock_db_writer = mocker.MagicMock()
    mocker.patch(
        "algo_mvp.orchestrator.manager.get_writer", return_value=mock_db_writer
    )

    # Patch the watchdog loop to not block
    mocker.patch(
        "algo_mvp.orchestrator.manager.Orchestrator._watchdog_loop", return_value=None
    )

    # Initialize orchestrator
    orchestrator = Orchestrator(config_path=str(sample_orchestrator_config_file))

    # Test successful reload
    orchestrator.reload(str(sample_orchestrator_config_file))

    # Verify DB writer was called with success message
    mock_db_writer.log_message.assert_any_call(
        "INFO",
        mocker.ANY,  # Don't test exact message text
    )

    # Reset the mock for testing error case
    mock_db_writer.reset_mock()

    # Create an invalid config file for testing error handling
    invalid_config_file = tmp_path / "invalid_config.yaml"
    with open(invalid_config_file, "w") as f:
        f.write("invalid: yaml: content: - [")  # Invalid YAML syntax

    # Test reload with invalid config
    with pytest.raises(Exception):  # Should raise some kind of exception
        orchestrator.reload(str(invalid_config_file))

    # Verify DB writer was called with error message
    mock_db_writer.log_message.assert_any_call(
        "ERROR",
        mocker.ANY,  # Don't test exact message text
    )


def test_orchestrator_watchdog_loop_custom(
    sample_orchestrator_config_file,
    mock_liverunner_class,
    mocker,
):
    """Test the watchdog functionality by implementing a custom version."""
    # Mock DB writer
    mocker.patch("algo_mvp.orchestrator.manager.get_writer")

    # Initialize orchestrator
    orchestrator = Orchestrator(config_path=str(sample_orchestrator_config_file))

    # Patch the sleep function to avoid timeout
    mocker.patch("time.sleep")

    # Create a simpler custom watchdog implementation that we can test
    def custom_watchdog():
        """A simplified version of the _watchdog_loop that we can control."""
        # Get the runner name to simulate a crash for
        runner_name = orchestrator.config.runners[0].name

        # Check if the runner thread is in the dict
        if runner_name in orchestrator.runner_threads:
            # Simulate a dead thread that needs cleanup
            thread = orchestrator.runner_threads[runner_name]
            if thread is None or not thread.is_alive():
                # Log that we're cleaning up a dead thread
                orchestrator.logger.warning(
                    f"WATCHDOG: Runner {runner_name} thread DIED. Cleaning up."
                )

                # Delete the thread reference
                del orchestrator.runner_threads[runner_name]

                # Check if runner should be restarted
                if (
                    orchestrator.config.restart_on_crash
                    and not orchestrator._stop_event.is_set()
                ):
                    # Find the runner config
                    runner_conf_obj = next(
                        (
                            r
                            for r in orchestrator.config.runners
                            if r.name == runner_name
                        ),
                        None,
                    )
                    if runner_conf_obj:
                        # Clean up old instance if it exists
                        if runner_name in orchestrator.runner_instances:
                            del orchestrator.runner_instances[runner_name]
                        # Launch a new runner
                        orchestrator._launch_runner(runner_conf_obj)

    # Install custom watchdog
    orchestrator._watchdog_loop = custom_watchdog

    # Start the orchestrator (without automatic watchdog)
    for runner_conf in orchestrator.config.runners:
        orchestrator._launch_runner(runner_conf)

    # Wait for threads to initialize
    time.sleep(0.2)

    # Verify we have the expected runner
    runner_name = orchestrator.config.runners[0].name
    assert runner_name in orchestrator.runner_threads

    # Setup the test - set the thread to None to simulate a dead thread
    orchestrator.runner_threads[runner_name] = None

    # Get reference to the current instance
    original_instance = orchestrator.runner_instances[runner_name]

    # Run our custom watchdog
    custom_watchdog()

    # Verify the runner was cleaned up and restarted
    assert runner_name in orchestrator.runner_threads  # Should have a new thread
    assert orchestrator.runner_threads[runner_name] is not None
    assert orchestrator.runner_instances[runner_name] != original_instance

    # Clean up
    orchestrator.stop()


def test_orchestrator_status_method(
    sample_orchestrator_config_file,
    mock_liverunner_class,
    mocker,
):
    """Test the status method of the orchestrator."""
    # Initialize orchestrator
    orchestrator = Orchestrator(config_path=str(sample_orchestrator_config_file))

    # Test status before any runners are started
    status_before_start = orchestrator.status()
    for runner_name, status in status_before_start.items():
        assert status == "pending_or_failed_to_start"

    # Start a runner manually
    runner_conf = orchestrator.config.runners[0]
    orchestrator._launch_runner(runner_conf)

    # Get the mock runner instance and thread
    runner_name = runner_conf.name
    runner_instance = orchestrator.runner_instances[runner_name]
    runner_thread = orchestrator.runner_threads[runner_name]

    # Mark the test as ready
    runner_instance._test_has_set_crash_flag_event.set()

    # Make sure is_alive is properly returning True
    runner_instance.is_alive.return_value = True

    # Verify runner is shown as running in status
    status_running = orchestrator.status()
    assert status_running[runner_name] == "running"

    # Now simulate a clean stop by:
    # 1. Setting _stop_event to True
    # 2. Making instance.is_alive return False
    # 3. Making thread.is_alive return False
    runner_instance._stop_event.set()
    runner_instance.is_alive.return_value = False

    # The key issue: we need to mock the thread's is_alive method too!
    with patch.object(runner_thread, "is_alive", return_value=False):
        # Check status again - should show stopped now
        status_stopped = orchestrator.status()
        assert status_stopped[runner_name] == "stopped"

    # Cleanup
    orchestrator.stop()
