# FastAPI app + Supervisor class will be implemented here.

import logging
import os
import threading
import time
import sys
from typing import List, Optional, Literal

import yaml
from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, FilePath

# Use the actual Orchestrator and OrchestratorConfig
from algo_mvp.orchestrator.manager import Orchestrator


class SupervisorConfig(BaseModel):
    orchestrator_config: FilePath
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    shutdown_token: Optional[str] = None  # Loaded from env


class HealthResponseRunner(BaseModel):
    name: str
    status: str


class HealthResponse(BaseModel):
    status: Literal["ok", "error"]
    runners: List[HealthResponseRunner]


logger = logging.getLogger("supervisor")
app = FastAPI()

# This will hold the global supervisor instance
_supervisor_instance: Optional["Supervisor"] = None


class Supervisor:
    def __init__(self, config: SupervisorConfig):
        self.config = config
        self.orchestrator: Optional[Orchestrator] = None
        self.orchestrator_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self.restart_attempts = 0
        self.last_restart_timestamps: List[float] = []
        self.lock = threading.Lock()

        log_level_attr = getattr(logging, self.config.log_level.upper(), None)
        if log_level_attr is None:
            print(
                f"Warning: Invalid log level '{self.config.log_level}'. Defaulting to INFO."
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
                f"Orchestrator config file not found or is not a file: {actual_path}"
            )
            raise FileNotFoundError(
                f"Orchestrator config file not found: {actual_path}"
            )

        self.config.shutdown_token = os.getenv("SUPERVISOR_TOKEN")
        if not self.config.shutdown_token:
            logger.warning(
                "SUPERVISOR_TOKEN environment variable not set. Shutdown endpoint will be unprotected if accessed."
            )
            # In a real scenario, you might want to exit or disable the endpoint.
            # For this exercise, we allow it but log a warning.

    def _start_orchestrator_thread(self):
        with self.lock:
            if self.orchestrator and self.orchestrator.is_alive():
                logger.info("Orchestrator already running.")
                return

            logger.info(
                f"Initializing Orchestrator with config: {self.config.orchestrator_config}"
            )
            try:
                # Replace with actual Orchestrator once available
                self.orchestrator = Orchestrator(
                    config_path=str(self.config.orchestrator_config)
                )
            except Exception as e:
                logger.error(f"Failed to initialize Orchestrator: {e}", exc_info=True)
                return  # Don't attempt to start if init fails

            logger.info("Starting Orchestrator in a new thread...")
            self.orchestrator_thread = threading.Thread(
                target=self.orchestrator.start, name="OrchestratorRunner"
            )
            self.orchestrator_thread.daemon = True  # So it exits when supervisor exits
            self.orchestrator_thread.start()
            logger.info("Orchestrator thread started.")
            self.restart_attempts = 0  # Reset attempts on successful start

    def start(self):
        logger.info("Supervisor starting...")
        self._stop_event.clear()
        self._start_orchestrator_thread()

        # Start watchdog in a separate thread
        watchdog_thread = threading.Thread(
            target=self._watchdog_loop, name="SupervisorWatchdog"
        )
        watchdog_thread.daemon = True
        watchdog_thread.start()
        logger.info(
            f"Supervisor started. Health check on http://{self.config.host}:{self.config.port}/health"
        )
        if not self.config.shutdown_token:
            logger.warning(
                "Shutdown token is not set. The /shutdown endpoint is currently UNPROTECTED."
            )
        else:
            logger.info(
                "Shutdown endpoint /shutdown requires a token (SUPERVISOR_TOKEN)."
            )

    def _watchdog_loop(self):
        logger.info("Supervisor watchdog started.")
        while not self._stop_event.is_set():
            time.sleep(1)  # Check orchestrator thread status every second

            if self._stop_event.is_set():
                logger.debug("Watchdog: stop event received, exiting loop.")
                break

            orchestrator_healthy = False
            if (
                self.orchestrator
                and self.orchestrator_thread
                and self.orchestrator_thread.is_alive()
            ):
                # Further check if orchestrator's own is_alive (if it has one that reflects internal health)
                if hasattr(self.orchestrator, "is_alive") and callable(
                    getattr(self.orchestrator, "is_alive")
                ):
                    if self.orchestrator.is_alive():  # type: ignore
                        orchestrator_healthy = True
                    else:  # Orchestrator's internal mechanism says it's not alive, even if thread is.
                        logger.warning(
                            "Watchdog: Orchestrator thread is alive, but Orchestrator.is_alive() is false."
                        )
                else:  # No specific is_alive on orchestrator, rely on thread.
                    orchestrator_healthy = True

            if not orchestrator_healthy:
                logger.warning(
                    "Watchdog: Orchestrator thread is not alive or Orchestrator reported not healthy."
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
                        f"Attempting to restart Orchestrator (attempt {self.restart_attempts + 1})..."
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
                                f"Watchdog: Error stopping orchestrator during restart attempt: {e}",
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
        logger.info(f"Supervisor stopping... (Signal: {from_signal})")
        if self._stop_event.is_set():
            logger.info("Supervisor already stopping.")
            return
        self._stop_event.set()

        if self.orchestrator:
            logger.info("Stopping Orchestrator...")
            try:
                self.orchestrator.stop()
            except Exception as e:
                logger.error(f"Error stopping orchestrator: {e}", exc_info=True)

        if self.orchestrator_thread and self.orchestrator_thread.is_alive():
            logger.info("Joining Orchestrator thread...")
            self.orchestrator_thread.join(timeout=10)
            if self.orchestrator_thread.is_alive():
                logger.warning("Orchestrator thread did not stop in time.")

        logger.info("Supervisor stopped.")
        # For FastAPI/uvicorn, actual server shutdown is handled by uvicorn.Server.should_exit
        # This method ensures our managed threads are stopped.


def get_supervisor() -> Supervisor:
    global _supervisor_instance
    if _supervisor_instance is None:
        # This should not happen in normal FastAPI flow if startup event is used
        raise RuntimeError(
            "Supervisor not initialized. Ensure it's set up in FastAPI startup."
        )
    return _supervisor_instance


@app.on_event("startup")
async def startup_event():
    global _supervisor_instance
    print(
        "STARTUP_EVENT: Entered startup_event.", file=sys.stderr
    )  # Ensure it flushes / is visible

    config_path_str = os.getenv(
        "SUPERVISOR_CONFIG_PATH", "configs/supervisor_sample.yaml"
    )
    print(
        f"STARTUP_EVENT: Attempting to load supervisor config from: {config_path_str}",
        file=sys.stderr,
    )

    try:
        print(
            f"STARTUP_EVENT: Step 1: Checking if config file exists at '{config_path_str}'",
            file=sys.stderr,
        )
        if not os.path.exists(config_path_str):
            print(
                f"STARTUP_EVENT ERROR: Config file '{config_path_str}' does not exist (os.path.exists failed).",
                file=sys.stderr,
            )
            raise FileNotFoundError(
                f"Explicit check failed: Supervisor configuration file not found at {config_path_str}"
            )
        print("STARTUP_EVENT: Step 1: Config file exists.", file=sys.stderr)

        print(
            "STARTUP_EVENT: Step 2: Attempting to open and read config file.",
            file=sys.stderr,
        )
        with open(config_path_str, "r") as f:
            config_data = yaml.safe_load(f)
        print(
            f"STARTUP_EVENT: Step 2: Config file opened and YAML loaded. Data: {config_data}",
            file=sys.stderr,
        )

        print(
            "STARTUP_EVENT: Step 3: Attempting to validate SupervisorConfig with Pydantic.",
            file=sys.stderr,
        )
        supervisor_config = SupervisorConfig(**config_data)
        print(
            f"STARTUP_EVENT: Step 3: SupervisorConfig validated. Orch config path: {supervisor_config.orchestrator_config}",
            file=sys.stderr,
        )

        print(
            f"STARTUP_EVENT: Step 3.5: Checking orchestrator config file: {supervisor_config.orchestrator_config}",
            file=sys.stderr,
        )
        if not supervisor_config.orchestrator_config.is_file():
            print(
                f"STARTUP_EVENT ERROR: Orch config '{supervisor_config.orchestrator_config}' not a file.",
                file=sys.stderr,
            )
            raise FileNotFoundError(
                f"Orchestrator config file '{supervisor_config.orchestrator_config}' not found during startup check."
            )
        print(
            "STARTUP_EVENT: Step 3.5: Orchestrator config file confirmed to exist.",
            file=sys.stderr,
        )

        print(
            "STARTUP_EVENT: Step 4: Attempting to initialize Supervisor instance.",
            file=sys.stderr,
        )
        _supervisor_instance = Supervisor(config=supervisor_config)
        print(
            "STARTUP_EVENT: Step 4: Supervisor instance initialized.", file=sys.stderr
        )

        print(
            "STARTUP_EVENT: Step 5: Attempting to start supervisor instance.",
            file=sys.stderr,
        )
        _supervisor_instance.start()
        print(
            "STARTUP_EVENT: Step 5: Supervisor instance started successfully.",
            file=sys.stderr,
        )

    except FileNotFoundError as e:
        print(f"STARTUP_EVENT ERROR (FileNotFoundError): {e}", file=sys.stderr)
        _supervisor_instance = None
        return
    except Exception as e:
        print(f"STARTUP_EVENT ERROR (General Exception): {e}", file=sys.stderr)
        _supervisor_instance = None
        return


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("FastAPI shutdown event triggered.")
    supervisor = get_supervisor()
    if supervisor:
        supervisor.stop()
    # Uvicorn handles its own server stop


@app.get("/health", response_model=HealthResponse)
async def health_check(request: Request):
    supervisor = get_supervisor()
    if not supervisor or not supervisor.orchestrator:
        logger.error("/health: Supervisor or Orchestrator not initialized.")
        # Attempt to return a 503 if basic components are missing
        from fastapi.responses import JSONResponse

        return JSONResponse(
            content={
                "status": "error",
                "runners": [{"name": "supervisor_status", "status": "uninitialized"}],
            },
            status_code=503,
        )

    try:
        orchestrator_status_dict = supervisor.orchestrator.status()
        logger.debug(f"/health: Orchestrator status: {orchestrator_status_dict}")
    except Exception as e:
        logger.error(f"/health: Error getting orchestrator status: {e}", exc_info=True)
        # This means the orchestrator itself might be broken or in a bad state
        # Return a 503 if we can't even get status
        from fastapi.responses import JSONResponse

        return JSONResponse(
            content={
                "status": "error",
                "runners": [{"name": "orchestrator_communication", "status": "error"}],
            },
            status_code=503,
        )

    runners_health: List[HealthResponseRunner] = []
    all_ok = True
    if not orchestrator_status_dict:  # No runners configured or reported
        # If orchestrator is up but has no runners, is that "ok" or an "error"?
        # Assuming "ok" if orchestrator itself is fine and just has no tasks.
        pass  # all_ok remains true, runners_health list remains empty

    for name, stat_val in orchestrator_status_dict.items():
        runners_health.append(HealthResponseRunner(name=name, status=str(stat_val)))
        # Define what constitutes "not ok".
        # "running" is clearly ok. "stopped" might be ok if intentional.
        # "crashed", "error", "pending_or_failed_to_start" are not ok.
        # Let's assume only "running" (and maybe "starting" if that's a state) are truly "ok" for health.
        # For simplicity, if any runner is not in a state considered "fully operational", overall is "error".
        # The prompt says "HTTP 200 when all runners running, else 503."
        # So, any status other than "running" (or a similar positive active state) means not all are "running".
        if not isinstance(stat_val, str) or "running" not in stat_val.lower():
            all_ok = False

    # Check orchestrator's main thread health as well
    # If orchestrator thread died but status somehow reports old data, that's also an error.
    if supervisor.orchestrator_thread and not supervisor.orchestrator_thread.is_alive():
        logger.warning("/health: Orchestrator main thread is not alive.")
        all_ok = False  # If orchestrator thread is dead, system is not ok.
        # Add a pseudo-runner for orchestrator's own health if it's not part of its status()
        if not any(r.name == "_orchestrator_thread" for r in runners_health):
            runners_health.append(
                HealthResponseRunner(name="_orchestrator_thread", status="DEAD")
            )

    current_status: Literal["ok", "error"] = "ok" if all_ok else "error"
    http_status_code = 200 if all_ok else 503

    # To return a 503, we need to raise HTTPException or return a Starlette Response.
    if http_status_code == 503:
        from fastapi.responses import JSONResponse

        return JSONResponse(
            content={
                "status": current_status,
                "runners": [r.model_dump() for r in runners_health],
            },
            status_code=http_status_code,
        )

    # If 200, FastAPI will handle it correctly with the response_model
    return HealthResponse(status=current_status, runners=runners_health)


@app.post("/shutdown")
async def shutdown_server(
    request: Request,
    token: Optional[str] = Query(None),  # Token from query parameter
):
    supervisor = get_supervisor()

    # Token validation
    if supervisor.config.shutdown_token:  # If token is configured
        if not token:
            logger.warning(
                "/shutdown: Attempted access without token when token is required."
            )
            raise HTTPException(status_code=401, detail="Shutdown token required.")
        if token != supervisor.config.shutdown_token:
            logger.warning(
                f"/shutdown: Invalid token received: {token[:5]}..."
            )  # Log only a snippet
            raise HTTPException(status_code=403, detail="Invalid shutdown token.")

    logger.info("/shutdown endpoint called. Requesting supervisor stop.")
    supervisor.stop(from_signal=False)  # Indicate it's an API-triggered stop

    # Signal uvicorn to shutdown
    # We need to get the Uvicorn server instance from app.state
    server = getattr(request.app.state, "uvicorn_server", None)
    if server and hasattr(server, "should_exit"):
        logger.info("Signaling Uvicorn server to exit.")
        server.should_exit = True
    else:
        logger.warning(
            "Uvicorn server instance not found in app.state.uvicorn_server. "
            "Cannot set should_exit. Orchestrator will stop, but Uvicorn might not exit gracefully without external signal."
        )

    return {
        "message": "Shutdown initiated. Orchestrator stopping. Uvicorn server will exit if configured."
    }
