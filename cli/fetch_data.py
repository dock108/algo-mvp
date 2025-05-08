#!/usr/bin/env python
"""CLI for fetching historical market data."""

import argparse
import sys
import yaml
from pydantic import ValidationError

# The src/algo_mvp/fetch.py script should handle adding the project root to sys.path
# so that algo_mvp can be imported directly.
from algo_mvp.models import AlpacaConfig, TradovateConfig, BaseConfig  # BaseConfig for type hint
from algo_mvp.data.fetcher import DataFetcher

def main():
    parser = argparse.ArgumentParser(description="Fetch historical market data.")
    parser.add_argument(
        "--config", 
        type=str, 
        required=True, 
        help="Path to the YAML configuration file."
    )
    parser.add_argument(
        "--force", 
        action="store_true", 
        help="Force download even if data already exists."
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Enable verbose output."
    )

    args = parser.parse_args()

    if args.verbose:
        print(f"Configuration file: {args.config}")
        print(f"Force download: {args.force}")
        print(f"Verbose output: {args.verbose}")

    try:
        with open(args.config, 'r') as f:
            raw_config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {args.config}", file=sys.stderr)
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error: Could not parse YAML configuration file: {e}", file=sys.stderr)
        sys.exit(1)

    if not isinstance(raw_config, dict) or "provider" not in raw_config:
        print("Error: Configuration must be a dictionary and include a 'provider' field.", file=sys.stderr)
        sys.exit(1)

    provider = raw_config.get("provider")
    config_model: BaseConfig

    try:
        if provider == 'alpaca':
            config_model = AlpacaConfig(**raw_config)
        elif provider == 'tradovate':
            config_model = TradovateConfig(**raw_config)
        else:
            print(f"Error: Unsupported provider '{provider}'. Supported: alpaca, tradovate.", file=sys.stderr)
            sys.exit(1)
        
        if args.verbose:
            print(f"Validated configuration: {config_model.model_dump_json(indent=2)}")

    except ValidationError as e:
        print(f"Error: Invalid configuration for provider '{provider}':\n{e}", file=sys.stderr)
        sys.exit(1)
    
    # Initialize and run the main data fetcher
    try:
        fetcher_instance = DataFetcher(config=config_model, verbose=args.verbose)
        success = fetcher_instance.fetch_data(force=args.force)
    except Exception as e:
        # Catch-all for unexpected errors during DataFetcher instantiation or fetching
        print(f"An critical error occurred during data fetching process: {e}", file=sys.stderr)
        # import traceback; traceback.print_exc() # For debugging
        success = False

    if success:
        if args.verbose:
            # DataFetcher now handles its own success/skip messages, so CLI can be simpler
            print("CLI: Data fetching process concluded.") 
        sys.exit(0)
    else:
        print("CLI: Data fetching process failed or encountered an error.", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main() 