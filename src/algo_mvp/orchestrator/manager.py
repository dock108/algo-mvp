from pydantic import BaseModel
from typing import List, Literal, Dict, Optional
import yaml
import threading
import time
from rich.console import Console
from rich.logging import RichHandler
import logging


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

            thread = threading.Thread(target=runner.start, name=runner_name)
            thread.daemon = True  # Allow main program to exit
            runner.thread = thread  # Store thread in runner for its is_alive
            self.runner_threads[runner_name] = thread
            thread.start()
            self.logger.info(f"Runner {runner_name} thread started.")
        except Exception as e:
            self.logger.error(
                f"Failed to launch runner {runner_name}: {e}", exc_info=True
            )
            if runner_name in self.runner_instances:
                del self.runner_instances[
                    runner_name
                ]  # Clean up if launch failed mid-way

    def start(self) -> None:
        self.logger.info("Starting orchestrator...")
        self._stop_event.clear()
        for runner_conf in self.config.runners:
            self._launch_runner(runner_conf)

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
                    self.logger.warning(
                        f"WATCHDOG: Runner {runner_name} thread DIED. Configured restart_on_crash: {self.config.restart_on_crash}"
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

                    if self.config.restart_on_crash and not self._stop_event.is_set():
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
                    elif not self.config.restart_on_crash:
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
        statuses = {}
        # Initialize status from config to catch runners that never started
        for r_conf in self.config.runners:
            statuses[r_conf.name] = "pending_or_not_configured"

        # Update status for runners that have/had threads
        for name, thread in self.runner_threads.items():  # This uses the live threads
            runner_instance = self.runner_instances.get(name)
            if thread.is_alive():
                statuses[name] = "running"
            elif (
                runner_instance and runner_instance._stop_event.is_set()
            ):  # Was stopped intentionally
                statuses[name] = "stopped"
            else:  # Died or exited without intentional stop signal known to orchestrator here
                statuses[name] = "crashed_or_exited"

        # For runners in config but not in runner_threads (e.g., launch failed, or already cleaned up post-crash without restart)
        for r_conf in self.config.runners:
            if r_conf.name not in self.runner_threads:  # If no active thread
                if r_conf.name in self.runner_instances:  # But we have an instance
                    # This means it might have been stopped and thread joined, or crashed and not restarted
                    if self.runner_instances[r_conf.name]._stop_event.is_set():
                        statuses[r_conf.name] = "stopped"
                        # else:
                        # It might have crashed and been removed from runner_threads by watchdog
                        # If restart is off, it stays crashed. If restart is on, it should have been restarted or is in process.
                        # The "crashed_or_exited" from above loop might catch this if thread was briefly there.
                        # This part can be tricky to get perfect without more state.
                        # For now, if no thread, and not explicitly stopped, and was an instance -> assume issue
                        # However, the previous loop already handles threads that *were* there.
                        # This is more for runners that *never* got a thread or were cleaned up *completely*.
                        pass  # Let "pending_or_not_configured" or "stopped" (if set by instance check) stand

                elif (
                    statuses[r_conf.name] == "pending_or_not_configured"
                ):  # Truly never made it or fully gone
                    # Could refine this to "failed_to_start" if we track launch attempts better
                    pass

        # Ensure all configured runners have some status
        # current_runner_names = set(self.runner_instances.keys()) # This variable is unused and will be removed.
        for r_conf in self.config.runners:
            if r_conf.name not in statuses:  # Should not happen with initial population
                statuses[r_conf.name] = "unknown"  # Fallback
            # If a runner was in config, but no instance and no thread, it's likely "pending_or_failed_to_start"
            # or "crashed_and_cleaned" if restart is off.
            # The logic is getting complex; let's simplify.

        # Simplified status:
        # 1. Iterate through configured runners.
        # 2. Check thread status if a thread exists.
        # 3. Check instance's stop event if instance exists.

        final_statuses: Dict[str, str] = {}
        for r_config in self.config.runners:
            runner_name = r_config.name
            thread = self.runner_threads.get(runner_name)
            instance = self.runner_instances.get(runner_name)

            if thread and thread.is_alive():
                final_statuses[runner_name] = "running"
            elif instance and instance._stop_event.is_set():  # Intentionally stopped
                final_statuses[runner_name] = "stopped"
            elif thread and not thread.is_alive():  # Thread existed but died
                final_statuses[runner_name] = "crashed_or_exited"
            elif (
                instance
            ):  # Instance exists, but no thread or thread died, and not cleanly stopped
                # This could be a state where it crashed and the thread is gone.
                final_statuses[runner_name] = "error_or_crashed"
            else:  # No instance, no thread. Either never started, or failed very early.
                final_statuses[runner_name] = "pending_or_failed_to_start"
        return final_statuses
