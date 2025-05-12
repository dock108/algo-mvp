"""
Supervisor module for managing the Orchestrator service.

This module provides a FastAPI application with endpoints for health checking and controlled shutdown,
as well as a Supervisor class that monitors and auto-restarts the Orchestrator if it crashes.
"""

import logging
import os
import yaml
from typing import List, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query, Request, status as http_status
from pydantic import BaseModel, Field
from fastapi.responses import JSONResponse

from algo_mvp.supervisor.models import Supervisor, SupervisorConfig
# Orchestrator import removed as it's only used for type hinting in models.py now

logger = logging.getLogger("supervisor")

# Global supervisor instance REMOVED


# --- Pydantic Models for API Responses ---
class HealthResponseRunner(BaseModel):
    """Model representing the health status of a runner."""

    name: str
    status: str


class HealthResponse(BaseModel):
    """Model for the /health endpoint response."""

    status: str  # 'ok' or 'error'
    runners: List[HealthResponseRunner] = Field(default_factory=list)


# --- FastAPI Lifespan Management ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles application startup and shutdown events."""
    # Use print for TestClient visibility, logger might not be configured early enough
    print("LIFESPAN: Startup sequence initiated.")
    # Initialize startup error state
    app.state.supervisor_startup_error = None
    app.state.supervisor = None

    config_path_str = os.getenv("SUPERVISOR_CONFIG_PATH")
    print(f"LIFESPAN: Env SUPERVISOR_CONFIG_PATH = {config_path_str}")

    if not config_path_str:
        err_msg = "SUPERVISOR_CONFIG_PATH environment variable not set."
        print(f"LIFESPAN: CRITICAL - {err_msg}")
        app.state.supervisor_startup_error = err_msg
        raise RuntimeError(err_msg)

    if not os.path.isfile(config_path_str):
        err_msg = f"Supervisor configuration file not found: {config_path_str}"
        print(f"LIFESPAN: CRITICAL - {err_msg}")
        app.state.supervisor_startup_error = err_msg
        raise FileNotFoundError(err_msg)

    try:
        print(f"LIFESPAN: Loading config from {config_path_str}...")
        with open(config_path_str, "r") as f:
            config_data = yaml.safe_load(f)
        supervisor_config = SupervisorConfig(**config_data)
        print("LIFESPAN: Config loaded successfully.")

        # Explicit check if the orchestrator config file path from the loaded config actually exists
        if not supervisor_config.orchestrator_config.is_file():
            err_msg = f"Orchestrator config file specified in supervisor config does not exist: {supervisor_config.orchestrator_config}"
            print(f"LIFESPAN: CRITICAL - {err_msg}")
            app.state.supervisor_startup_error = err_msg
            raise FileNotFoundError(err_msg)
        else:
            print(
                f"LIFESPAN: Orchestrator config path confirmed exists: {supervisor_config.orchestrator_config}"
            )

        print("LIFESPAN: Attempting to instantiate Supervisor...")
        supervisor_instance = Supervisor(config=supervisor_config)
        print(f"LIFESPAN: Supervisor instantiated: {type(supervisor_instance)}")

        print("LIFESPAN: Assigning supervisor instance to app.state.supervisor...")
        app.state.supervisor = supervisor_instance
        print(
            f"LIFESPAN: Assigned. Checking app.state.supervisor: {getattr(app.state, 'supervisor', 'NOT FOUND')}"
        )

        print("LIFESPAN: Attempting supervisor.start()...")
        app.state.supervisor.start()
        print("LIFESPAN: supervisor.start() completed.")
        # logger.info("Supervisor initialized and started via lifespan.")

    except Exception as e:
        err_msg = f"Supervisor initialization failed: {e}"
        print(f"LIFESPAN: CRITICAL - Exception during startup: {err_msg}")
        app.state.supervisor_startup_error = err_msg
        app.state.supervisor = None  # Ensure state reflects failure
        raise RuntimeError(err_msg) from e

    print("LIFESPAN: Startup yield.")
    yield  # Application runs here

    # --- Shutdown ---
    print("LIFESPAN: Shutdown sequence initiated.")
    # logger.info(...)
    supervisor: Optional[Supervisor] = getattr(app.state, "supervisor", None)
    if supervisor:
        print("LIFESPAN: Stopping supervisor instance...")
        # logger.info(...)
        supervisor.stop()
    else:
        print("LIFESPAN: No supervisor instance found in app.state during shutdown.")
        # logger.warning(...)
    print("LIFESPAN: Shutdown complete.")
    # logger.info(...)


# --- FastAPI App Initialization ---
app = FastAPI(
    title="Algo MVP Supervisor",
    description="Manages and monitors the Algo MVP Orchestrator.",
    version="1.0.0",
    lifespan=lifespan,
)


# --- API Endpoints ---
@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Check system health",
    description="Returns the status of the supervisor and the orchestrator's runners.",
    responses={
        200: {"description": "Service is healthy"},
        503: {"description": "Service is unhealthy or orchestrator is unavailable"},
    },
)
async def get_health(request: Request):
    """Check the health of the supervisor and the orchestrator."""
    supervisor: Optional[Supervisor] = getattr(request.app.state, "supervisor", None)
    if not supervisor:
        raise HTTPException(
            status_code=http_status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Supervisor not initialized",
        )

    orchestrator = supervisor.orchestrator
    status_data = {"status": "unknown", "runners": []}
    overall_status = http_status.HTTP_503_SERVICE_UNAVAILABLE  # Default to unhealthy

    # Check supervisor's orchestrator thread first
    orchestrator_thread_alive = (
        supervisor.orchestrator_thread is not None
        and supervisor.orchestrator_thread.is_alive()
    )

    # Fetch runner status early IF orchestrator exists, but don't fail the whole check yet
    # Initialize runner_status related vars
    runner_status_data = []
    runners_ok = True  # Assume ok unless proven otherwise
    status_fetch_error = None
    orchestrator_alive_internal = True  # Assume ok unless proven otherwise

    if orchestrator:
        try:
            # Check orchestrator's internal health *first*
            if hasattr(orchestrator, "is_alive") and callable(
                getattr(orchestrator, "is_alive")
            ):
                orchestrator_alive_internal = orchestrator.is_alive()

            # Attempt to get runner status regardless of internal health (might still be possible)
            raw_runner_status = orchestrator.status()
            for name, state in raw_runner_status.items():
                runner_status_data.append({"name": name, "status": state})
                if state.lower() != "running":
                    runners_ok = False
                    logger.warning(
                        f"Health check: Runner '{name}' is not running (state: {state})"
                    )

        except Exception as e:
            logger.error(
                f"Error getting orchestrator status for health check: {e}",
                exc_info=True,
            )
            status_fetch_error = e
            runners_ok = False  # Cannot confirm runners are ok if status fails

    # --- Determine Overall Status ---

    if not orchestrator_thread_alive:
        status_data["status"] = "error"
        status_data["detail"] = "Orchestrator thread is not alive."
        # Add placeholder if runner status couldn't be fetched
        if not runner_status_data and not status_fetch_error:
            runner_status_data.append(
                {"name": "_orchestrator_thread", "status": "DEAD"}
            )
        overall_status = http_status.HTTP_503_SERVICE_UNAVAILABLE
        logger.error("Health check failed: Orchestrator thread is not alive.")

    elif not orchestrator:
        status_data["status"] = "error"
        status_data["detail"] = "Orchestrator instance not found in supervisor."
        overall_status = http_status.HTTP_503_SERVICE_UNAVAILABLE
        logger.error("Health check failed: Orchestrator instance not found.")

    elif status_fetch_error:
        status_data["status"] = "error"
        status_data["detail"] = (
            f"Failed to get orchestrator status: {status_fetch_error}"
        )
        # Add placeholder indicating communication error
        runner_status_data.append(
            {"name": "orchestrator_communication", "status": "error"}
        )
        overall_status = http_status.HTTP_500_INTERNAL_SERVER_ERROR  # Internal error
        logger.error(f"Health check failed: Status fetch error: {status_fetch_error}")

    elif not orchestrator_alive_internal:
        status_data["status"] = "error"
        status_data["detail"] = "Orchestrator internal state reports not alive."
        # Add placeholder indicating internal error
        runner_status_data.append(
            {"name": "orchestrator_internal", "status": "unhealthy"}
        )
        overall_status = http_status.HTTP_503_SERVICE_UNAVAILABLE
        logger.error(
            "Health check failed: Orchestrator internal state reports not alive."
        )

    elif not runners_ok:
        status_data["status"] = "error"
        status_data["detail"] = "One or more runners are not running."
        overall_status = http_status.HTTP_503_SERVICE_UNAVAILABLE
        logger.warning("Health check failed: One or more runners not running.")

    else:  # All checks passed
        status_data["status"] = "ok"
        overall_status = http_status.HTTP_200_OK
        logger.debug("Health check: Status OK.")

    # Populate runners list, ensuring it's always included
    status_data["runners"] = runner_status_data

    # Use JSONResponse to ensure the status code is set correctly based on overall_status
    return JSONResponse(content=status_data, status_code=overall_status)


@app.post(
    "/shutdown",
    summary="Stop the orchestrator and shut down the server",
    description="Requires a valid token if SUPERVISOR_TOKEN is set.",
)
async def shutdown_server(request: Request, token: Optional[str] = Query(None)):
    supervisor: Optional[Supervisor] = getattr(request.app.state, "supervisor", None)

    if not supervisor:
        raise HTTPException(
            status_code=http_status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Supervisor not initialized, cannot process shutdown.",
        )

    # Validate token if configured
    if supervisor.config.shutdown_token:
        if not token:
            raise HTTPException(
                status_code=http_status.HTTP_401_UNAUTHORIZED,
                detail="Shutdown token required.",
            )
        if token != supervisor.config.shutdown_token:
            raise HTTPException(
                status_code=http_status.HTTP_403_FORBIDDEN,
                detail="Invalid shutdown token.",
            )

    logger.info("Shutdown requested via API.")

    # Signal supervisor to stop its components (orchestrator, watchdog)
    # Supervisor's stop method handles orchestrator stop and thread joins
    supervisor.stop()

    # Additionally, signal uvicorn to exit gracefully if running under uvicorn directly
    # This relies on uvicorn setting 'server' in app.state, which might not always happen
    uvicorn_server = getattr(request.app.state, "uvicorn_server", None)
    if uvicorn_server and hasattr(uvicorn_server, "handle_exit"):
        logger.info("Signaling Uvicorn server to exit.")
        # uvicorn_server.should_exit = True # Old way
        import signal

        uvicorn_server.handle_exit(sig=signal.SIGINT, frame=None)  # More reliable way
    else:
        logger.warning(
            "Uvicorn server instance not found in app.state or does not have 'handle_exit'. Cannot signal programmatic exit."
        )
        logger.warning(
            "Supervisor and Orchestrator stopping, but Uvicorn process might need manual stop (e.g., CTRL+C or external signal)."
        )

    return {"message": "Shutdown initiated. Supervisor and Orchestrator stopping."}


# --- Example Actions (Placeholder, adapt as needed) ---


@app.post(
    "/action/flatten_all",
    summary="[Example] Flatten all positions across all runners",
    description="Requires token. Tells all runners to close open positions.",
)
async def action_flatten_all(request: Request, token: Optional[str] = Query(None)):
    supervisor: Optional[Supervisor] = getattr(request.app.state, "supervisor", None)
    if not supervisor or not supervisor.orchestrator:
        raise HTTPException(
            status_code=http_status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Supervisor or Orchestrator not initialized.",
        )

    if supervisor.config.shutdown_token:  # Use same token for actions for simplicity
        if not token:
            raise HTTPException(
                status_code=401, detail="Authentication token required."
            )
        if token != supervisor.config.shutdown_token:
            raise HTTPException(status_code=403, detail="Invalid authentication token.")

    logger.info("Flatten all action requested via API.")
    # Access orchestrator runners safely
    runners = getattr(supervisor.orchestrator, "runners", {})
    for runner_name, runner_instance in runners.items():
        # Check if runner has the necessary method (adapt based on actual runner implementation)
        if hasattr(runner_instance, "adapter") and hasattr(
            runner_instance.adapter, "close_all_positions"
        ):
            try:
                logger.info(f"Requesting flatten for runner: {runner_name}")
                # Assuming close_all_positions is synchronous for now
                runner_instance.adapter.close_all_positions()
            except Exception as e:
                logger.error(
                    f"Error calling close_all_positions for runner {runner_name}: {e}",
                    exc_info=True,
                )
        else:
            logger.warning(
                f"Runner {runner_name} does not support flatten action (close_all_positions)."
            )

    return {"message": "Flatten all action initiated for capable runners."}


@app.post(
    "/action/pause",
    summary="[Example] Pause/unpause a specific runner",
    description="Requires token. Toggles the paused state of a runner.",
)
async def action_pause_runner(
    request: Request, runner: str = Query(...), token: Optional[str] = Query(None)
):
    supervisor: Optional[Supervisor] = getattr(request.app.state, "supervisor", None)
    if not supervisor or not supervisor.orchestrator:
        raise HTTPException(
            status_code=http_status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Supervisor or Orchestrator not initialized.",
        )

    if supervisor.config.shutdown_token:
        if not token:
            raise HTTPException(
                status_code=401, detail="Authentication token required."
            )
        if token != supervisor.config.shutdown_token:
            raise HTTPException(status_code=403, detail="Invalid authentication token.")

    logger.info(f"Pause/unpause action requested for runner '{runner}' via API.")
    runners = getattr(supervisor.orchestrator, "runners", {})
    target_runner = runners.get(runner)

    if not target_runner:
        raise HTTPException(status_code=404, detail=f"Runner '{runner}' not found.")

    # Example: Assume runners have a 'paused' boolean attribute
    if not hasattr(target_runner, "paused"):
        # Initialize if doesn't exist? Or raise error? Let's initialize for flexibility.
        logger.warning(
            f"Runner {runner} lacks 'paused' attribute, initializing to False."
        )
        target_runner.paused = False

    # Toggle paused state
    target_runner.paused = not target_runner.paused
    new_state = "paused" if target_runner.paused else "running"
    logger.info(f"Runner {runner} state changed to: {new_state}")

    return {"runner": runner, "paused": target_runner.paused}


@app.post(
    "/action/reload_config",
    summary="[Example] Reload orchestrator config",
    description="Requires token. Attempts to reload the orchestrator's configuration.",
)
async def action_reload_config(request: Request, token: Optional[str] = Query(None)):
    supervisor: Optional[Supervisor] = getattr(request.app.state, "supervisor", None)
    if not supervisor or not supervisor.orchestrator:
        raise HTTPException(
            status_code=http_status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Supervisor or Orchestrator not initialized.",
        )

    if supervisor.config.shutdown_token:
        if not token:
            raise HTTPException(
                status_code=401, detail="Authentication token required."
            )
        if token != supervisor.config.shutdown_token:
            raise HTTPException(status_code=403, detail="Invalid authentication token.")

    logger.info("Reload config action requested via API.")
    try:
        # Assuming orchestrator has a reload method that returns new runner status
        if hasattr(supervisor.orchestrator, "reload") and callable(
            getattr(supervisor.orchestrator, "reload")
        ):
            reload_result = supervisor.orchestrator.reload()
            logger.info(
                f"Orchestrator config reload attempted. Result: {reload_result}"
            )
            return {"reloaded": True, "runners": reload_result}
        else:
            logger.error(
                "Orchestrator instance does not have a callable 'reload' method."
            )
            raise HTTPException(
                status_code=501,
                detail="Reload config action not implemented by Orchestrator.",
            )

    except yaml.YAMLError as e:
        logger.error(f"YAML parsing error during config reload: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to reload config: YAML parsing error: {e}"
        )
    except Exception as e:
        logger.error(f"Error during orchestrator config reload: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to reload config: {e}")
