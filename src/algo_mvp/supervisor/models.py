"""
Models for the supervisor module.
"""

import logging
import os
import threading
import time
import sys
from typing import List, Optional
from pydantic import BaseModel, FilePath

from algo_mvp.orchestrator.manager import Orchestrator

logger = logging.getLogger("supervisor")


class SupervisorConfig(BaseModel):
    """Configuration model for the Supervisor."""

    orchestrator_config: FilePath
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    shutdown_token: Optional[str] = None  # Loaded from env


class Supervisor:
    """
    Supervisor class that manages and monitors the Orchestrator.

    Responsibilities:
    - Starting and stopping the Orchestrator
    - Monitoring the Orchestrator's health
    - Automatically restarting the Orchestrator if it crashes
    """

    def __init__(self, config: SupervisorConfig):
        self.config = config
        self.orchestrator: Optional[Orchestrator] = None
        self.orchestrator_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self.restart_attempts = 0
        self.last_restart_timestamps: List[float] = []
        self.lock = threading.Lock()

        # Enhanced logging for debugging import issues
        logger.debug(f"Supervisor __init__: Current sys.path: {sys.path}")
        logger.debug(
            f"Supervisor __init__: Current PYTHONPATH: {os.getenv('PYTHONPATH')}"
        )
        logger.debug(f"Supervisor __init__: Current working directory: {os.getcwd()}")

        log_level_attr = getattr(logging, self.config.log_level.upper(), None)
        if log_level_attr is None:
            print(
                "Warning: Invalid log level '%s'. Defaulting to INFO."
                % self.config.log_level
            )
            log_level_attr = logging.INFO

        # Ensure clean logging setup
        # Remove all handlers associated with the root logger object.
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        # Basic config for the supervisor's own logger and potentially others
        logging.basicConfig(
            level=log_level_attr,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],  # Explicitly use sys.stdout
        )
        # Set level for the supervisor's specific logger instance
        logger.setLevel(log_level_attr)

        if not self.config.orchestrator_config.is_file():  # Use is_file() for FilePath
            actual_path = self.config.orchestrator_config.resolve()
            logger.error(
                "Orchestrator config file not found or is not a file: %s", actual_path
            )
            raise FileNotFoundError(
                "Orchestrator config file not found: %s" % actual_path
            )

        self.config.shutdown_token = os.getenv("SUPERVISOR_TOKEN")
        if not self.config.shutdown_token:
            logger.warning(
                "SUPERVISOR_TOKEN environment variable not set. Shutdown endpoint will be unprotected if accessed."
            )

    def _start_orchestrator_thread(self):
        """Start the orchestrator in a new thread if it's not already running."""
        with self.lock:
            if (
                self.orchestrator
                and hasattr(self.orchestrator, "is_alive")
                and self.orchestrator.is_alive()
            ):
                logger.info("Orchestrator already running and alive.")
                return

            logger.info(
                "Initializing Orchestrator with config: %s",
                self.config.orchestrator_config,
            )
            try:
                logger.debug("Supervisor: Attempting to instantiate Orchestrator...")
                self.orchestrator = Orchestrator(
                    config_path=str(self.config.orchestrator_config)
                )
                logger.debug(
                    f"Supervisor: Orchestrator instantiated. Type: {type(self.orchestrator)}"
                )
                if self.orchestrator:
                    has_is_alive = hasattr(self.orchestrator, "is_alive")
                    logger.debug(
                        f"Supervisor: Orchestrator instance has 'is_alive' attribute: {has_is_alive}"
                    )
                    if not has_is_alive:
                        logger.error(
                            "CRITICAL: Orchestrator instance was created BUT LACKS 'is_alive' method!"
                        )
                else:
                    logger.error(
                        "CRITICAL: Orchestrator instantiation resulted in None!"
                    )

            except Exception as e:
                logger.error(
                    "CRITICAL: Failed to initialize Orchestrator during instantiation: %s",
                    e,
                    exc_info=True,
                )
                self.orchestrator = None  # Ensure it's None if init fails
                return  # Don't attempt to start if init fails

            if not self.orchestrator:  # Double check after try-except
                logger.error(
                    "Supervisor: Orchestrator is None after instantiation attempt, cannot start thread."
                )
                return

            logger.info("Starting Orchestrator in a new thread...")
            self.orchestrator_thread = threading.Thread(
                target=self.orchestrator.start, name="OrchestratorRunner"
            )
            self.orchestrator_thread.daemon = True  # So it exits when supervisor exits
            self.orchestrator_thread.start()
            logger.info("Orchestrator thread started.")
            self.restart_attempts = 0  # Reset attempts on successful start

    def start(self):
        """Start the supervisor and the orchestrator."""
        logger.info("Supervisor starting...")
        self._stop_event.clear()
        self._start_orchestrator_thread()

        # Give the orchestrator thread a moment to initialize itself
        # time.sleep(0.5) # REMOVED - Potentially problematic with TestClient/asyncio

        # Start watchdog in a separate thread
        # Ensure watchdog doesn't start if orchestrator thread failed to init
        watchdog_thread = threading.Thread(
            target=self._watchdog_loop, name="SupervisorWatchdog"
        )
        watchdog_thread.daemon = True
        watchdog_thread.start()
        logger.info(
            "Supervisor started. Health check on http://%s:%s/health",
            self.config.host,
            self.config.port,
        )
        if not self.config.shutdown_token:
            logger.warning(
                "Shutdown token is not set. The /shutdown endpoint is currently UNPROTECTED."
            )
        else:
            logger.info("Shutdown endpoint /shutdown requires token authentication.")

    def _watchdog_loop(self):
        """Monitor the orchestrator's health and restart it if necessary."""
        logger.info("Supervisor watchdog started.")
        while not self._stop_event.is_set():
            time.sleep(5)  # Check orchestrator thread status every 5 seconds

            if self._stop_event.is_set():
                logger.debug("Watchdog: stop event received, exiting loop.")
                break

            orchestrator_healthy = False
            thread_alive = False
            instance_exists = False
            instance_alive = False

            if self.orchestrator and self.orchestrator_thread:
                instance_exists = True
                thread_alive = self.orchestrator_thread.is_alive()
                # Further check if orchestrator's own is_alive (if it has one that reflects internal health)
                if hasattr(self.orchestrator, "is_alive") and callable(
                    getattr(self.orchestrator, "is_alive")
                ):
                    instance_alive = self.orchestrator.is_alive()  # type: ignore
                    orchestrator_healthy = thread_alive and instance_alive
                else:  # No specific is_alive on orchestrator, rely on thread.
                    logger.warning(
                        "Watchdog: Orchestrator instance lacks is_alive method!"
                    )
                    orchestrator_healthy = thread_alive  # Degrade to just checking thread if method missing

            # Log detailed status before making decision
            logger.debug(
                f"Watchdog check: Instance={instance_exists}, Thread alive={thread_alive}, "
                f"Instance is_alive()={instance_alive}, Overall healthy={orchestrator_healthy}"
            )

            if not orchestrator_healthy:
                # Log the reason for unhealthiness
                reason = []
                if not instance_exists:
                    reason.append("No instance/thread")
                if instance_exists and not thread_alive:
                    reason.append("Thread DEAD")
                if instance_exists and thread_alive and not instance_alive:
                    reason.append("Instance is_alive()=False")
                logger.warning(
                    f"Watchdog: Orchestrator reported not healthy. Reason(s): {', '.join(reason)}"
                )
                if (
                    self._stop_event.is_set()
                ):  # Check again, in case stop was called during sleep
                    logger.info("Watchdog: Stop event set, not attempting restart.")
                    break

                # Implement restart logic with back-off and retry limit
                now = time.monotonic()
                # Remove timestamps older than 60 seconds
                self.last_restart_timestamps = [
                    ts for ts in self.last_restart_timestamps if now - ts <= 60
                ]

                if len(self.last_restart_timestamps) < 3:  # Max 3 retries per minute
                    logger.info(
                        "Attempting to restart Orchestrator (attempt %d)...",
                        self.restart_attempts + 1,
                    )
                    self.last_restart_timestamps.append(now)
                    self.restart_attempts += 1

                    if self.orchestrator:  # Try to stop it cleanly if it exists
                        try:
                            logger.info(
                                "Watchdog: Requesting stop on existing orchestrator before restart..."
                            )
                            self.orchestrator.stop()  # Signal orchestrator's own stop
                            if (
                                self.orchestrator_thread
                                and self.orchestrator_thread.is_alive()
                            ):
                                self.orchestrator_thread.join(
                                    timeout=5
                                )  # Give it a moment to stop
                        except Exception as e:
                            logger.error(
                                "Watchdog: Error stopping orchestrator during restart attempt: %s",
                                e,
                                exc_info=True,
                            )

                    time.sleep(3)  # 3-second back-off
                    if (
                        not self._stop_event.is_set()
                    ):  # Re-check stop event before starting
                        self._start_orchestrator_thread()
                    else:
                        logger.info(
                            "Watchdog: Stop event set during backoff, restart aborted."
                        )
                        break
                else:
                    logger.error(
                        "Orchestrator restart limit reached (3 retries in 60s). Not attempting further restarts automatically."
                    )
                    # Consider what to do here - maybe a longer cooldown or require manual intervention.
                    # For now, it will just stop trying until the 60s window passes.
                    time.sleep(
                        10
                    )  # Wait longer before checking again to avoid busy-looping on this error message.
        logger.info("Supervisor watchdog stopped.")

    def stop(self, from_signal: bool = False):
        """Stop the supervisor and orchestrator."""
        logger.info("Supervisor stopping... (Signal: %s)", from_signal)
        if self._stop_event.is_set():
            logger.info("Supervisor already stopping.")
            return
        self._stop_event.set()

        if self.orchestrator:
            logger.info("Stopping Orchestrator...")
            try:
                self.orchestrator.stop()
            except Exception as e:
                logger.error("Error stopping orchestrator: %s", e, exc_info=True)

        if self.orchestrator_thread and self.orchestrator_thread.is_alive():
            logger.info("Joining Orchestrator thread...")
            self.orchestrator_thread.join(timeout=10)
            if self.orchestrator_thread.is_alive():
                logger.warning("Orchestrator thread did not stop in time.")

        logger.info("Supervisor stopped.")
