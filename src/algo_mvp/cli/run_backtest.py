#!/usr/bin/env python
"""CLI for running vectorbt backtests."""

import argparse
import sys
from pathlib import Path

from rich.console import Console

from algo_mvp.backtest.engine import BacktestEngine


def main():
    """Run the backtest CLI."""
    parser = argparse.ArgumentParser(
        description="Run vectorbt backtest based on YAML config."
    )

    parser.add_argument(
        "--config", type=str, required=True, help="Path to the YAML configuration file."
    )

    parser.add_argument(
        "--output-dir", type=str, help="Custom output directory for backtest results."
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose output.")

    args = parser.parse_args()
    console = Console()

    try:
        # Check if config file exists
        config_path = Path(args.config)
        if not config_path.exists():
            console.print(
                f"[red]Error: Configuration file not found at {args.config}[/red]"
            )
            sys.exit(1)

        # Initialize backtest engine
        engine = BacktestEngine(
            config_file=args.config, output_dir=args.output_dir, verbose=args.verbose
        )

        # Run the backtest
        success = engine.run()

        if success:
            sys.exit(0)
        else:
            sys.exit(1)

    except Exception as e:
        console.print(f"[red]Error running backtest:[/red] {str(e)}")
        if args.verbose:
            import traceback

            console.print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
