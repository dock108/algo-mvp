import argparse
import signal
import sys
import time
import os
import logging

from rich.live import Live
from rich.table import Table
from rich.console import Console

# Adjust import path based on your project structure
# This assumes 'src' is in PYTHONPATH or the command is run from the root directory
try:
    from algo_mvp.orchestrator.manager import Orchestrator
except ImportError:
    # This is a fallback for local development if PYTHONPATH isn't set
    # You might need to adjust this depending on how you run the script
    sys.path.insert(
        0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    )
    from algo_mvp.orchestrator.manager import Orchestrator

console = Console()


def generate_status_table(orchestrator: Orchestrator) -> Table:
    """Generates a Rich Table displaying the status of runners."""
    table = Table(title="Runner Status")
    table.add_column("Runner Name", style="cyan", no_wrap=True)
    table.add_column("Status", style="magenta")

    statuses = orchestrator.status()
    if not statuses:
        table.add_row("[dim]No runners configured or status unavailable.[/dim]", "")
        return table

    for name, status_value in statuses.items():
        style = "green"
        if "running" not in status_value.lower():
            style = "red"
        if "pending" in status_value.lower() or "stopped" in status_value.lower():
            style = "yellow"
        table.add_row(name, f"[{style}]{status_value}[/{style}]")
    return table


def main():
    parser = argparse.ArgumentParser(
        description="Run the Orchestrator for multiple trading strategies."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the orchestrator YAML configuration file (e.g., configs/orchestrator_sample.yaml)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging (sets log level to DEBUG).",
    )
    parser.add_argument(
        "--no-restart",
        action="store_false",
        dest="restart_on_crash",
        default=None,  # Will be handled by OrchestratorConfig default or YAML
        help="Disable automatic restarting of crashed runners (overrides YAML if set).",
    )

    args = parser.parse_args()

    # Initialize Orchestrator
    # Note: The Orchestrator itself reads log_level and restart_on_crash from its YAML.
    # CLI flags for these could override or supplement the YAML.
    try:
        orchestrator = Orchestrator(config_path=args.config)
    except FileNotFoundError:
        console.print(
            f"[bold red]Error: Configuration file not found at {args.config}[/bold red]"
        )
        sys.exit(1)
    except Exception as e:
        console.print(f"[bold red]Error initializing orchestrator: {e}[/bold red]")
        sys.exit(1)

    # Override restart_on_crash from CLI if provided
    if args.restart_on_crash is False:  # Explicitly --no-restart
        orchestrator.config.restart_on_crash = False
        orchestrator.logger.info("Runner auto-restart disabled via CLI flag.")

    # Override log_level if --verbose is set
    if args.verbose:
        orchestrator.config.log_level = "DEBUG"
        orchestrator.logger.setLevel("DEBUG")  # Set orchestrator's own logger level
        for handler in orchestrator.logger.handlers:  # And its handlers
            handler.setLevel("DEBUG")
        logging.getLogger().setLevel("DEBUG")  # Set root logger level
        orchestrator.logger.info("Verbose logging enabled via CLI flag (DEBUG level).")

    def signal_handler(sig, frame):
        console.print(
            "\n[bold yellow]Ctrl+C detected. Shutting down gracefully...[/bold yellow]"
        )
        orchestrator.stop()
        time.sleep(2)  # Allow some time for threads to join
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    orchestrator.start()

    try:
        with Live(
            generate_status_table(orchestrator), refresh_per_second=0.1, console=console
        ) as live:
            while True:
                time.sleep(10)  # Refresh interval for the table
                live.update(generate_status_table(orchestrator))
    except Exception as e:
        console.print(
            f"[bold red]An unexpected error occurred in the CLI: {e}[/bold red]",
            exc_info=True,
        )
    finally:
        console.print(
            "[bold blue]CLI exiting. Ensuring orchestrator shutdown...[/bold blue]"
        )
        orchestrator.stop()
        console.print(
            "[bold blue]Orchestrator shutdown requested. Exiting CLI.[/bold blue]"
        )


if __name__ == "__main__":
    # This allows running the script directly for development,
    # but the primary entry point should be `python -m algo_mvp.cli.run_orchestrator`
    # To make `python -m` work, ensure `src/algo_mvp/cli/__main__.py` or `src/algo_mvp/__main__.py` exists
    # and calls this main, or this script is runnable as part of the module.
    # For `python -m algo_mvp.orchestrator ...` as per prompt, we'll need an __main__.py in orchestrator.
    main()
