import logging
from typing import Optional

# from fastapi.responses import JSONResponse # Unused import
from fastapi import FastAPI, Request, Query, HTTPException
from fastapi import status as http_status
import yaml

# Assuming models.py is one level up relative to server.py
from ..models import Supervisor

# Initialize logger and app - Adjust imports/paths as needed for your project structure
logger = logging.getLogger(__name__)
app = FastAPI()  # Assuming app is initialized here or imported


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
                status_code=http_status.HTTP_401_UNAUTHORIZED,
                detail="Authentication token required.",
            )
        if token != supervisor.config.shutdown_token:
            raise HTTPException(
                status_code=http_status.HTTP_403_FORBIDDEN,
                detail="Invalid authentication token.",
            )

    logger.info("Reload config action requested via API.")

    # Check for reload capability first
    if not (
        hasattr(supervisor.orchestrator, "reload")
        and callable(getattr(supervisor.orchestrator, "reload"))
    ):
        logger.error("Orchestrator instance does not have a callable 'reload' method.")
        raise HTTPException(
            status_code=http_status.HTTP_501_NOT_IMPLEMENTED,
            detail="Reload config action not implemented by Orchestrator.",
        )

    try:
        # If we reach here, orchestrator.reload exists and is callable
        reload_result = supervisor.orchestrator.reload()
        logger.info(f"Orchestrator config reload attempted. Result: {reload_result}")
        return {"reloaded": True, "runners": reload_result}
    except yaml.YAMLError as e:
        logger.error(f"YAML parsing error during config reload: {e}", exc_info=True)
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reload config: YAML parsing error: {e}",
        )
    except Exception as e:
        logger.error(f"Error during orchestrator config reload: {e}", exc_info=True)
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reload config: {e}",
        )
