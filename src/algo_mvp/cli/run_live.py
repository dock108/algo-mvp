import argparse
import signal
import sys
import time

import yaml
from rich.console import Console
from rich.table import Table

from algo_mvp.live.adapters.mock import MockBrokerAdapter
from algo_mvp.live.config import LiveConfig
from algo_mvp.live.runner import LiveRunner

console = Console()
runners = []


def load_config(config_path: str) -> LiveConfig:
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)
    return LiveConfig(**config_data)


def main():
    parser = argparse.ArgumentParser(description="AlgoMVP Live Trading Engine")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the YAML configuration file."
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging."
    )
    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except Exception as e:
        console.print(f"Error loading config: {e}", style="bold red")
        sys.exit(1)

    # Placeholder for actual broker adapter loading based on config.provider
    # For now, only MockBrokerAdapter is used.

    for runner_config in config.runners:
        if runner_config.provider == "mock":
            broker_adapter = MockBrokerAdapter()
        else:
            console.print(
                f"Unsupported provider: {runner_config.provider}", style="bold red"
            )
            continue  # Or sys.exit(1) depending on desired strictness

        runner = LiveRunner(
            strategy_path=runner_config.strategy,
            params=runner_config.params,
            broker_adapter=broker_adapter,
            datafeed_config={
                "symbol": runner_config.symbol,
                "timeframe": runner_config.timeframe,
            },
        )
        runners.append(runner)
        runner.start()

    def signal_handler(sig, frame):
        console.print("\nCtrl-C detected. Shutting down runners...", style="yellow")
        for runner in runners:
            runner.stop()
        # Allow time for threads to join
        time.sleep(1)  # Adjust as needed, or implement more robust thread joining wait
        console.print("All runners stopped. Exiting.", style="green")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        while True:
            table = Table(title="Live Runner Status")
            table.add_column("Runner Name", style="cyan")
            table.add_column("Status", style="magenta")
            table.add_column("Symbol", style="green")
            table.add_column("Timeframe", style="blue")

            for i, runner in enumerate(runners):
                rc = config.runners[i]  # Assuming order is maintained
                table.add_row(rc.name, runner.status(), rc.symbol, rc.timeframe)

            console.print(table)
            time.sleep(10)
    except KeyboardInterrupt:
        # This handles the case where the loop is interrupted directly,
        # though the signal_handler should ideally catch it first.
        console.print("KeyboardInterrupt in main loop. Exiting.", style="yellow")
    finally:
        # Ensure cleanup even if loop exits unexpectedly
        for runner in runners:
            if (
                runner.status() == "running"
            ):  # Check if not already stopped by signal handler
                runner.stop()
        console.print("CLI cleanup complete.", style="dim")


if __name__ == "__main__":
    main()
