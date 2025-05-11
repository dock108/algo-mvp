# CLI for running the supervisor will be implemented here.

import argparse
import logging
import os
import sys

import uvicorn
import yaml

from algo_mvp.supervisor.server import app, SupervisorConfig

logger = logging.getLogger("supervisor_cli")


def main():
    parser = argparse.ArgumentParser(description="Run the Algo MVP Supervisor.")
    parser.add_argument(
        "--config",
        type=str,
        default=os.getenv("SUPERVISOR_CONFIG_PATH", "configs/supervisor_sample.yaml"),
        help="Path to the supervisor YAML configuration file.",
    )
    args = parser.parse_args()

    # Validate config path before Uvicorn tries to load the app (which loads the config)
    if not os.path.exists(args.config):
        logger.error(f"Error: Supervisor configuration file not found at {args.config}")
        sys.exit(1)

    try:
        with open(args.config, "r") as f:
            config_data = yaml.safe_load(f)
        # Validate basic structure (SupervisorConfig will do more detailed validation)
        if "orchestrator_config" not in config_data:
            logger.error(
                f"Error: 'orchestrator_config' missing in supervisor config {args.config}"
            )
            sys.exit(1)
        if not os.path.exists(config_data["orchestrator_config"]):
            logger.error(
                f"Error: Orchestrator config file '{config_data['orchestrator_config']}' not found (referenced in {args.config})."
            )
            sys.exit(1)

        sup_config = SupervisorConfig(**config_data)  # Validate full config

    except FileNotFoundError as e:
        logger.error(f"Error: Configuration file issue: {e}")
        sys.exit(1)
    except yaml.YAMLError as e:
        logger.error(
            f"Error parsing supervisor YAML configuration at {args.config}: {e}"
        )
        sys.exit(1)
    except Exception as e:  # Catch Pydantic validation errors or other issues
        logger.error(
            f"Error loading or validating supervisor configuration from {args.config}: {e}"
        )
        sys.exit(1)

    # Set environment variable for SUPERVISOR_CONFIG_PATH so FastAPI app can load it
    # This is crucial for the app.on_event("startup") to find the config
    os.environ["SUPERVISOR_CONFIG_PATH"] = args.config

    token = os.getenv("SUPERVISOR_TOKEN")
    if not token:
        print("Warning: SUPERVISOR_TOKEN environment variable is not set.")
        print("The /shutdown endpoint will be unprotected.")
    else:
        print("Supervisor token is set. Use this token for the /shutdown endpoint.")
        # Avoid printing the token itself unless for very specific debug scenarios.
        # print(f"  -> Token: {token}")

    print(
        f"Starting Supervisor HTTP server on http://{sup_config.host}:{sup_config.port}"
    )
    print(f"Health endpoint: http://{sup_config.host}:{sup_config.port}/health")
    print(
        f"Shutdown endpoint: POST http://{sup_config.host}:{sup_config.port}/shutdown?token=<YOUR_TOKEN>"
    )

    # Programmatic Uvicorn run to allow passing server instance to app state for graceful shutdown
    uvicorn_config = uvicorn.Config(
        app=app,  # Can be the app instance or "algo_mvp.supervisor.server:app"
        host=sup_config.host,
        port=sup_config.port,
        log_level=sup_config.log_level.lower(),
        # lifespan="on"  # Lifespan events are handled by FastAPI @app.on_event
    )
    server = uvicorn.Server(config=uvicorn_config)

    # Make the server instance available to the FastAPI app state
    # This allows the /shutdown endpoint to call server.should_exit = True
    # Ensure the app object is the actual FastAPI instance, not an import string, if modifying its state directly.
    # If 'app' is the instance (as imported), this should work.
    if hasattr(app, "state"):
        app.state.uvicorn_server = server
    else:
        logger.warning(
            "FastAPI app instance does not have 'state' attribute. Cannot set uvicorn_server for graceful shutdown."
        )
        logger.warning(
            "This might happen if 'app' is an import string in uvicorn.Config and not the direct instance."
        )
        logger.warning(
            "Consider using an app factory pattern if direct state modification is needed and uvicorn is passed a string."
        )

    # It's important that signal handlers are set up correctly by Uvicorn
    # so that Ctrl+C also triggers the FastAPI shutdown event.
    try:
        server.run()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down supervisor CLI.")
        # Uvicorn should handle its own shutdown. FastAPI shutdown events should fire.
    finally:
        logger.info("Supervisor CLI process exiting.")


if __name__ == "__main__":
    main()
