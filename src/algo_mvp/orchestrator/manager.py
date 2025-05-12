from pydantic import BaseModel
from typing import List, Literal, Dict, Optional
import yaml
import threading
import time
from rich.console import Console
from rich.logging import RichHandler
import logging

# Add import for database writer
from algo_mvp.db import get_writer


class RunnerConfig(BaseModel):
    name: str
    config: str


class OrchestratorConfig(BaseModel):
    runners: List[RunnerConfig]
    log_level: Literal["INFO", "DEBUG", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    restart_on_crash: bool = True


# Assuming LiveRunner is defined elsewhere and can be imported
# from algo_mvp.live_runner import LiveRunner # Placeholder


# Placeholder for LiveRunner for now, to be replaced with actual import
class LiveRunner:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.name = f"Runner_for_{config_path}"  # Placeholder name
        self._stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        # Ensure logger name is unique if multiple instances of this placeholder are created
        # For real LiveRunner, logger might be configured differently
        self.logger = logging.getLogger(f"LiveRunner.{self.name}.{id(self)}")
        self.logger.info(f"Initialized with config: {config_path}")

    def start(self) -> None:
        self.logger.info("Starting...")
        # Simulate work
        try:
            while not self._stop_event.is_set():
                self.logger.debug("Working...")
                time.sleep(1)
        except Exception as e:
            self.logger.error(f"Error during run: {e}", exc_info=True)
        finally:
            self.logger.info("Stopped.")

    def stop(self) -> None:
        self.logger.info("Stopping...")
        self._stop_event.set()

    def is_alive(self) -> bool:
        # This method in placeholder might not perfectly reflect real thread status
        # In a real scenario, it would check self.thread.is_alive()
        return self.thread is not None and self.thread.is_alive()


class Orchestrator:
    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)
        self.config = OrchestratorConfig(**config_data)
        self.runner_instances: Dict[
            str, LiveRunner
        ] = {}  # To store actual runner objects
        self.runner_threads: Dict[str, threading.Thread] = {}
        self._stop_event = threading.Event()
        self._watchdog_thread: Optional[threading.Thread] = None
        self.last_exception: Optional[Exception] = None

        self.console = Console()
        # Configure logging handler for the orchestrator
        # Individual runners should have their own logger instances if they log independently
        log_handler = RichHandler(
            console=self.console, rich_tracebacks=True, show_path=False
        )
        logging.basicConfig(
            level=self.config.log_level,
            format="%(name)s: %(message)s",  # Include logger name
            datefmt="[%X]",
            handlers=[log_handler],
        )
        self.logger = logging.getLogger("Orchestrator")
        self.logger.info(f"Initialized with config: {config_path}")

    def _runner_target_wrapper(self, runner: LiveRunner):
        """Wraps the runner's start method to catch and log exceptions."""
        try:
            runner.start()
        except Exception as e:
            self.logger.error(
                f"CRITICAL ERROR in runner {runner.name} thread: {e}", exc_info=True
            )
            # Optionally store the exception if needed elsewhere
            # self.last_runner_exception = e

    def _launch_runner(self, runner_conf: RunnerConfig) -> None:
        runner_name = runner_conf.name
        try:
            self.logger.info(
                f"Launching runner: {runner_name} with config {runner_conf.config}"
            )
            # runner = LiveRunner(config_path=runner_conf.config)  # Actual LiveRunner
            # Using the placeholder LiveRunner:
            runner = LiveRunner(config_path=runner_conf.config)
            runner.name = runner_name  # Set the name from orchestrator config to the runner instance

            # Actual LiveRunner instances should configure their own loggers.
            # The placeholder LiveRunner has a basic logger.
            # Example of how one might pass down/configure centrally if needed:
            # runner_logger = logging.getLogger(f"LiveRunner.{runner_name}")
            # runner_logger.handlers = self.logger.handlers # Inherit handlers
            # runner_logger.setLevel(self.logger.level) # Inherit level
            # runner.logger = runner_logger # Assign the configured logger

            self.runner_instances[runner_name] = runner

            # Use the wrapper for the thread target
            thread = threading.Thread(
                target=self._runner_target_wrapper, args=(runner,), name=runner_name
            )
            thread.daemon = True  # Allow main program to exit
            runner.thread = thread  # Store thread in runner for its is_alive
            self.runner_threads[runner_name] = thread
            thread.start()
            self.logger.info(f"Runner {runner_name} thread started.")
        except Exception as e:
            self.logger.error(
                f"Failed to launch runner {runner_name}: {e}", exc_info=True
            )
            # Store the exception that occurred during launch
            self.last_exception = e
            if runner_name in self.runner_instances:
                del self.runner_instances[
                    runner_name
                ]  # Clean up if launch failed mid-way

    def start(self) -> None:
        # Wrap the entire start logic in a try-except block
        try:
            self.logger.info("Starting orchestrator...")
            self._stop_event.clear()
            self.last_exception = None  # Clear previous exception
            for runner_conf in self.config.runners:
                self._launch_runner(runner_conf)
                # Check if launch failed immediately
                if self.last_exception:
                    # If _launch_runner stored an exception, re-raise it to stop the orchestrator start
                    raise self.last_exception

            # Always start watchdog to monitor runner threads for crashes/exits,
            # regardless of whether restart_on_crash is enabled. When restart_on_crash
            # is False the watchdog will still perform cleanup of internal state.
            if self._watchdog_thread is None or not self._watchdog_thread.is_alive():
                self._watchdog_thread = threading.Thread(
                    target=self._watchdog_loop, name="OrchestratorWatchdog"
                )
                self._watchdog_thread.daemon = True
                self._watchdog_thread.start()
            self.logger.info("Orchestrator start sequence initiated.")
        except Exception as e:
            # Catch any exception during the main start sequence and log it
            self.logger.error(
                f"CRITICAL ERROR during Orchestrator start: {e}", exc_info=True
            )
            self.last_exception = e
            # Ensure stop event is set if start fails critically, so watchdog exits cleanly
            self._stop_event.set()
            # No need to re-raise here, the thread will exit, and is_alive should reflect the state.

    def _watchdog_loop(self) -> None:
        self.logger.info("Watchdog started.")
        while not self._stop_event.is_set():
            time.sleep(5)  # Check every 5 seconds
            if self._stop_event.is_set():
                break  # Exit early if orchestrator is stopping

            for runner_name in list(
                self.runner_threads.keys()
            ):  # Iterate over a copy of names
                thread = self.runner_threads.get(runner_name)

                if thread is None:  # Should not happen if cleanup is correct
                    self.logger.warning(
                        f"Watchdog: Thread for {runner_name} not found, but was in key list. Skipping."
                    )
                    continue

                if not thread.is_alive():
                    # Check the actual restart_on_crash setting from the config
                    restart_flag = self.config.restart_on_crash
                    self.logger.warning(
                        f"WATCHDOG: Runner {runner_name} thread DIED. Configured restart_on_crash: {restart_flag}"
                    )

                    self.logger.info(
                        f"WATCHDOG: Pre-delete check for '{runner_name}' in runner_threads. Exists: {runner_name in self.runner_threads}"
                    )
                    del self.runner_threads[
                        runner_name
                    ]  # Clean up the dead thread reference
                    self.logger.info(
                        f"WATCHDOG: Post-delete check for '{runner_name}' in runner_threads. Exists: {runner_name in self.runner_threads}"
                    )

                    if restart_flag and not self._stop_event.is_set():
                        self.logger.info(
                            f"WATCHDOG: Restarting runner {runner_name} due to crash/exit."
                        )
                        runner_conf_obj = next(
                            (r for r in self.config.runners if r.name == runner_name),
                            None,
                        )
                        if runner_conf_obj:
                            if runner_name in self.runner_instances:
                                self.logger.info(
                                    f"WATCHDOG: Pre-delete check for '{runner_name}' in runner_instances. Exists: {runner_name in self.runner_instances}"
                                )
                                del self.runner_instances[
                                    runner_name
                                ]  # Clean up old instance
                                self.logger.info(
                                    f"WATCHDOG: Post-delete check for '{runner_name}' in runner_instances. Exists: {runner_name in self.runner_instances}"
                                )
                            self._launch_runner(runner_conf_obj)
                        else:
                            self.logger.error(
                                f"WATCHDOG: Could not find config for runner {runner_name} to restart."
                            )
                    elif not restart_flag:
                        self.logger.info(
                            f"WATCHDOG: Not restarting {runner_name} as restart_on_crash is False."
                        )
                        if runner_name in self.runner_instances:
                            self.logger.info(
                                f"WATCHDOG: Pre-delete check for '{runner_name}' in runner_instances. Exists: {runner_name in self.runner_instances}"
                            )
                            del self.runner_instances[
                                runner_name
                            ]  # Clean up instance if not restarting
                            self.logger.info(
                                f"WATCHDOG: Post-delete check for '{runner_name}' in runner_instances. Exists: {runner_name in self.runner_instances}"
                            )
                        else:
                            self.logger.warning(
                                f"WATCHDOG: {runner_name} was NOT in runner_instances when expected for deletion (no restart case)."
                            )
                    # If _stop_event was set, we don't restart while stopping
        self.logger.info("Watchdog stopped.")

    def stop(self) -> None:
        self.logger.info("Stopping orchestrator...")
        self._stop_event.set()  # Signal watchdog and runners to stop

        # Stop all runner instances
        for name, runner_instance in list(self.runner_instances.items()):
            try:
                self.logger.info(f"Requesting stop for runner {name}...")
                runner_instance.stop()  # This should signal the runner's internal loop to stop
            except Exception as e:
                self.logger.error(
                    f"Error signalling stop for runner {name}: {e}", exc_info=True
                )

        # Join all runner threads
        active_threads = list(
            self.runner_threads.items()
        )  # Get a list before iterating
        for name, thread in active_threads:
            self.logger.info(f"Joining thread for runner {name}...")
            thread.join(timeout=10)  # Wait for threads to finish
            if thread.is_alive():
                self.logger.warning(f"Thread for runner {name} did not stop in time.")
            # Remove after join attempt, regardless of outcome
            if name in self.runner_threads:
                del self.runner_threads[name]

        if self._watchdog_thread and self._watchdog_thread.is_alive():
            self.logger.info("Joining watchdog thread...")
            self._watchdog_thread.join(timeout=5)
            if self._watchdog_thread.is_alive():
                self.logger.warning("Watchdog thread did not stop in time.")
        self._watchdog_thread = None

        self.runner_instances.clear()
        # runner_threads should be clear by now, but ensure it
        self.runner_threads.clear()
        self.logger.info("Orchestrator stopped.")

    def status(self) -> Dict[str, str]:
        # Initialize status for all configured runners
        final_statuses: Dict[str, str] = {}
        for r_config in self.config.runners:
            runner_name = r_config.name
            thread = self.runner_threads.get(runner_name)
            instance = self.runner_instances.get(runner_name)

            # Check if the thread and instance exist and their states
            if thread and thread.is_alive():
                # Thread is alive, runner is running
                final_statuses[runner_name] = "running"
            elif instance and instance._stop_event.is_set():
                # Stop event was set, runner was intentionally stopped
                final_statuses[runner_name] = "stopped"
            elif thread and not thread.is_alive():
                # Thread existed but died
                final_statuses[runner_name] = "crashed_or_exited"
            elif instance:
                # Instance exists but thread is gone or never created
                # Check if is_alive on the runner instance (could be unreliable)
                if hasattr(instance, "is_alive") and callable(instance.is_alive):
                    if instance.is_alive():
                        final_statuses[runner_name] = "running"
                    else:
                        # If is_alive returns False and stop_event is set, it's stopped
                        if instance._stop_event.is_set():
                            final_statuses[runner_name] = "stopped"
                        else:
                            final_statuses[runner_name] = "error_or_crashed"
                else:
                    final_statuses[runner_name] = "error_or_crashed"
            else:
                # No instance, no thread
                final_statuses[runner_name] = "pending_or_failed_to_start"

        return final_statuses

    def reload(self, manifest_path: str) -> Dict[str, str]:
        """
        Reload the orchestrator configuration from the given manifest path.

        This method:
        1. Stops all existing runners
        2. Re-parses the YAML configuration
        3. Starts new runners based on the updated configuration

        Args:
            manifest_path: Path to the orchestrator YAML manifest

        Returns:
            Dict[str, str]: Status of each runner after reloading

        Raises:
            Exception: If YAML parsing or configuration loading fails
        """
        self.logger.info(f"Reloading orchestrator configuration from {manifest_path}")

        # First, stop all existing runners
        self.stop()

        # Get DB writer for logging (should never fail)
        db_writer = get_writer()

        # Re-parse the YAML configuration
        try:
            with open(manifest_path, "r") as f:
                config_data = yaml.safe_load(f)
            self.config = OrchestratorConfig(**config_data)
            self.logger.info(
                f"Configuration reloaded successfully. Starting {len(self.config.runners)} runners."
            )

            # Log success to database (always log)
            db_writer.log_message(
                "INFO",
                f"Orchestrator configuration reloaded successfully with {len(self.config.runners)} runners",
            )

        except Exception as e:
            error_msg = f"Failed to reload configuration: {e}"
            self.logger.error(error_msg, exc_info=True)

            # Log failure to database (always log)
            db_writer.log_message("ERROR", error_msg)

            # Re-raise the exception to propagate to the caller
            raise

        # Start the orchestrator with the new configuration
        self.start()

        # Return the status of all runners
        return self.status()

    def is_alive(self) -> bool:
        """Return True if every runner thread is alive and the watchdog is running."""
        if not hasattr(self, "_watchdog_thread") or self._watchdog_thread is None:
            return True
        all_runners_alive = all(t.is_alive() for t in self.runner_threads.values())
        watchdog_alive = self._watchdog_thread.is_alive()
        return all_runners_alive and watchdog_alive
