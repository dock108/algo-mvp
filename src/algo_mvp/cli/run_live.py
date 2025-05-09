# Placeholder for CLI

import argparse
import asyncio
import importlib  # For loading strategy by string
import signal
import sys
import time
from pathlib import Path
from typing import Any, List, Type  # Removed Dict

import backtrader as bt  # Required for strategy type hint if used directly
import yaml
from rich.console import Console
from rich.live import Live  # For dynamic status table
from rich.table import Table
from rich.text import Text  # Import for status table cell styling

from algo_mvp.live.adapters.mock import MockBrokerAdapter
from algo_mvp.live.runner import LiveRunner, RunnerStatus
from algo_mvp.models import LiveTradingConfig, RunnerConfig  # YAML config models

# Import other adapters as they are created, e.g.:
# from algo_mvp.live.adapters.alpaca import AlpacaBrokerAdapter
# from algo_mvp.live.adapters.tradovate import TradovateBrokerAdapter

console = Console()

# Global list to keep track of active runners for shutdown
active_runners: List[LiveRunner] = []
shutdown_event = asyncio.Event()


def _get_strategy_class(dotted_path: str) -> Type[bt.Strategy]:
    """Dynamically imports and returns a strategy class from a dotted path string."""
    try:
        module_path, class_name = dotted_path.split(":")
        module = importlib.import_module(module_path)
        strategy_class = getattr(module, class_name)
        if not issubclass(strategy_class, bt.Strategy):
            raise TypeError(
                f"Strategy class {class_name} must be a subclass of backtrader.Strategy"
            )
        return strategy_class
    except (ImportError, AttributeError, ValueError, TypeError) as e:
        console.print(
            f"[bold red]Error loading strategy class '{dotted_path}': {e}[/bold red]"
        )
        raise


def _get_broker_adapter(config: RunnerConfig, loop: asyncio.AbstractEventLoop) -> Any:
    """Instantiates and returns a broker adapter based on provider config."""
    provider = config.provider.lower()
    if provider == "mock":
        console.print(
            f"[grey50]Instantiating MockBrokerAdapter for runner '{config.name}'[/grey50]"
        )
        return MockBrokerAdapter(event_loop=loop)
    # elif provider == 'alpaca':
    #     # TODO: Load credentials securely
    #     return AlpacaBrokerAdapter(api_key='...', secret_key='...', paper=True, event_loop=loop)
    # elif provider == 'tradovate':
    #     # TODO: Load credentials
    #     return TradovateBrokerAdapter(username='...', password='...', ..., event_loop=loop)
    else:
        raise ValueError(f"Unsupported broker provider: {config.provider}")


async def manage_runners(runners_to_manage: List[LiveRunner]):
    """Manages the lifecycle of runners, prints status, and handles shutdown."""
    global active_runners
    active_runners = runners_to_manage

    for runner in active_runners:
        console.print(
            f"[cyan]Starting runner for {runner.data_feed_config.get('symbol')} ({runner._strategy_name})...[/cyan]"
        )
        try:
            runner.start()  # This initializes and starts cerebro in a thread
            if runner.status() == RunnerStatus.ERROR.value:
                console.print(
                    f"[bold red]Runner '{runner._log_prefix}' failed to start and is in ERROR state.[/bold red]"
                )
            elif runner.status() == RunnerStatus.RUNNING.value:
                console.print(
                    f"[green]Runner '{runner._log_prefix}' started successfully.[/green]"
                )
            else:
                console.print(
                    f"[yellow]Runner '{runner._log_prefix}' start initiated, current status: {runner.status()}[/yellow]"
                )

        except Exception as e_runner_start:
            console.print(
                f"[bold red]Failed to start runner '{runner._log_prefix}': {e_runner_start}[/bold red]"
            )
            # Runner's internal error handling should set its status to ERROR.

    last_status_check_time = 0
    status_update_interval = 10  # seconds

    with Live(console=console, refresh_per_second=1) as live_display:
        while not shutdown_event.is_set():
            current_time = time.monotonic()
            if current_time - last_status_check_time >= status_update_interval:
                table = Table(
                    title="Live Runner Status",
                    title_style="bold blue",
                    show_header=True,
                    header_style="bold magenta",
                )
                table.add_column("Name", style="dim cyan", width=20)
                table.add_column("Symbol", width=12)
                table.add_column("Strategy", width=30)
                table.add_column("Status", width=12)
                table.add_column("Broker Adapter", width=20)

                all_stopped_or_error = True
                for (
                    r_status_check
                ) in active_runners:  # Renamed loop variable to avoid conflict
                    status_val = r_status_check.status()
                    status_style = (
                        "green"
                        if status_val == RunnerStatus.RUNNING.value
                        else (
                            "red"
                            if status_val == RunnerStatus.ERROR.value
                            else "yellow"
                        )
                    )
                    table.add_row(
                        (
                            r_status_check.broker_adapter.config.name
                            if hasattr(r_status_check.broker_adapter, "config")
                            and hasattr(r_status_check.broker_adapter.config, "name")
                            else r_status_check._log_prefix.split("-")[1]
                        ),  # Runner name from config
                        r_status_check.data_feed_config.get("symbol", "N/A"),
                        r_status_check._strategy_name,
                        Text(status_val, style=status_style),
                        type(r_status_check.broker_adapter).__name__,
                    )
                    if status_val == RunnerStatus.RUNNING.value:
                        all_stopped_or_error = False

                live_display.update(table)
                last_status_check_time = current_time

                if (
                    all_stopped_or_error and active_runners
                ):  # If all runners stopped/errored out and we have runners
                    console.print(
                        "[yellow]All active runners have stopped or are in an error state. Initiating shutdown.[/yellow]"
                    )
                    shutdown_event.set()  # Trigger shutdown
                    break  # Exit status loop

            try:
                await asyncio.sleep(0.5)  # Check shutdown_event periodically
            except asyncio.CancelledError:
                console.print("[yellow]Status update loop cancelled.[/yellow]")
                break

    console.print(
        "[bold yellow]Shutdown signal received. Stopping all runners...[/bold yellow]"
    )
    for runner_to_stop in active_runners:  # Renamed loop variable
        if runner_to_stop.status() == RunnerStatus.RUNNING.value:
            console.print(
                f"[cyan]Stopping runner for {runner_to_stop.data_feed_config.get('symbol')}...[/cyan]"
            )
            runner_to_stop.stop()
            console.print(
                f"[green]Runner for {runner_to_stop.data_feed_config.get('symbol')} stop sequence initiated. Final status: {runner_to_stop.status()}[/green]"
            )
        else:
            console.print(
                f"[grey50]Runner for {runner_to_stop.data_feed_config.get('symbol')} already stopped (status: {runner_to_stop.status()}).[/grey50]"
            )

    console.print(
        "[bold green]All runners processed for shutdown. Exiting.[/bold green]"
    )


def signal_handler(sig, frame):
    console.print(
        f"[bold red]Signal {signal.Signals(sig).name} received. Initiating graceful shutdown...[/bold red]"
    )
    shutdown_event.set()


async def main():
    parser = argparse.ArgumentParser(description="AlgoMVP Live Trading Engine CLI")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the YAML configuration file for live runners.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging (currently sets LiveRunner internal logging to debug if implemented).",
    )
    args = parser.parse_args()

    if args.verbose:
        # This is a placeholder. Verbosity should ideally be passed to LiveRunner's logger.
        # For now, rich console output is the primary feedback.
        console.print("[dim]Verbose mode activated (placeholder).[/dim]")

    # Load YAML configuration
    try:
        with open(args.config, "r") as f_yaml:
            raw_config = yaml.safe_load(f_yaml)
        live_config = LiveTradingConfig(**raw_config)
        console.print(
            f"[green]Successfully loaded and validated configuration from: {args.config}[/green]"
        )
    except FileNotFoundError:
        console.print(
            f"[bold red]Error: Configuration file not found at '{args.config}'[/bold red]"
        )
        sys.exit(1)
    except yaml.YAMLError as e_yaml:
        console.print(
            f"[bold red]Error parsing YAML configuration file '{args.config}': {e_yaml}[/bold red]"
        )
        sys.exit(1)
    except Exception as e_config_val:  # Catches Pydantic validation errors
        console.print(
            f"[bold red]Error validating configuration from '{args.config}': {e_config_val}[/bold red]"
        )
        sys.exit(1)

    # Setup signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # kill command

    loop = asyncio.get_event_loop()
    runners_to_start: List[LiveRunner] = []

    for runner_cfg_item in live_config.runners:  # Renamed loop variable
        console.print(f"[blue]Preparing runner: {runner_cfg_item.name}[/blue]")
        try:
            strategy_class = _get_strategy_class(runner_cfg_item.strategy)
            broker_adapter = _get_broker_adapter(runner_cfg_item, loop)

            # Add the original RunnerConfig to the adapter if it doesn't have it for reference
            if not hasattr(broker_adapter, "config"):
                broker_adapter.config = runner_cfg_item  # For status display

            live_runner_instance = LiveRunner(  # Renamed variable
                strategy_source=strategy_class,  # Pass the loaded class
                strategy_params=runner_cfg_item.params,
                broker_adapter=broker_adapter,
                data_feed_config={
                    "symbol": runner_cfg_item.symbol,
                    "timeframe": runner_cfg_item.timeframe,
                    # Add other data_feed related params from runner_cfg_item if needed by LiveRunner
                },
                event_loop=loop,
                # on_trade= (optional callback function),
                # on_error= (optional callback function),
            )
            runners_to_start.append(live_runner_instance)
            console.print(
                f"  [green]Runner '{runner_cfg_item.name}' configured for {runner_cfg_item.symbol} with {strategy_class.__name__}.[/green]"
            )
        except Exception as e_runner_setup:  # Renamed exception variable
            console.print(
                f"[bold red]Error setting up runner '{runner_cfg_item.name}': {e_runner_setup}[/bold red]"
            )
            # Optionally, decide if one runner failing setup should stop all, or just skip this one.
            # For now, we skip and try to run others.

    if not runners_to_start:
        console.print(
            "[bold yellow]No runners configured or all failed to set up. Exiting.[/bold yellow]"
        )
        sys.exit(0)

    try:
        await manage_runners(runners_to_start)
    except asyncio.CancelledError:
        console.print(
            "[yellow]Main task cancelled. Ensuring runners are stopped.[/yellow]"
        )
    finally:
        # Final cleanup, though manage_runners should handle stopping runners.
        # This ensures even if manage_runners itself is cancelled, we attempt a stop.
        console.print(
            "[cyan]CLI main finally block. Forcing shutdown signal if not already set.[/cyan]"
        )
        shutdown_event.set()  # Ensure it's set so any lingering tasks know to stop
        # Give a moment for any final stop actions triggered by shutdown_event
        # This is a bit of a safety net. Ideally, manage_runners handles all.
        await asyncio.sleep(0.1)
        console.print("[bold blue]CLI execution finished.[/bold blue]")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print(
            "[bold yellow]\nKeyboardInterrupt caught in __main__. Exiting gracefully.[/bold yellow]"
        )
        # Shutdown event should have been set by signal_handler.
        # Allow any cleanup in main's finally block to run if asyncio.run was interrupted.
    except Exception as e_global_exc:  # Renamed exception variable
        console.print(
            f"[bold red]Unhandled global exception in CLI: {e_global_exc}[/bold red]"
        )
        sys.exit(1)
