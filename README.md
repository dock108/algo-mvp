# algo-mvp

Lean backtest + paper-trade stack for Mike (Dock108).

## Quick Start

1.  **Install dependencies:**
    ```bash
    poetry install
    ```

2.  **Run tests:**
    ```bash
    poetry run pytest
    ```

3.  **Launch Streamlit app (placeholder):**
    ```bash
    poetry run streamlit hello
    ```

## Database setup

The project uses SQLite for storing trade history, logs, and equity snapshots. Alembic is used for database migrations.

To create or upgrade the database to the latest version, run:

```bash
poetry run python -m algo_mvp.db upgrade
```

This command will create `data/algo.db` if it doesn't exist and apply any pending migrations.

## Historical Data Fetching

This project includes a module to download historical market data for stocks/ETFs (via Alpaca) and micro-futures (via Tradovate demo).

### Configuration
Data fetching is configured using YAML files. Sample configurations can be found in the `configs/` directory:
- `configs/sample_equity.yaml`
- `configs/sample_futures.yaml`

Key configuration options include:
- `provider`: `alpaca` or `tradovate`
- `symbol`: The ticker symbol (e.g., `AAPL` for Alpaca, `MESM25` for Tradovate).
- `timeframe`: Data granularity (e.g., `1Min`, `1Day` for Alpaca; `1Min` for Tradovate which resamples ticks).
- `start`: Start date (YYYY-MM-DD).
- `end`: End date (YYYY-MM-DD).
- `adjust`: (Alpaca only) `true` or `false` for split/dividend adjustment.

### CLI Usage

To fetch data, use the following command:

```bash
python -m algo_mvp.fetch --config path/to/your_config.yaml
```

**Example:**
```bash
python -m algo_mvp.fetch --config configs/sample_equity.yaml
```

**Optional flags:**
- `--force`: Force download even if data for the requested range seems to exist locally. This overwrites any existing data for the specified timeframe and range.
- `--verbose`: Enable verbose output for more details on the fetching process.

Fetched data is stored in Parquet files under the `data/` directory, organized by provider, symbol, and timeframe (e.g., `data/alpaca/AAPL/1Day.parquet` or `data/alpaca/AAPL/1Day/2023.parquet` if partitioned by year due to size).

### Environment Variables

API credentials and service endpoints should be set as environment variables.

**Alpaca:**
- `ALPACA_KEY_ID`: Your Alpaca API Key ID.
- `ALPACA_SECRET_KEY`: Your Alpaca Secret Key.
- `ALPACA_BASE_URL`: The base URL for the Alpaca API. For paper trading with `AlpacaBrokerAdapter`, this defaults to `https://paper-api.alpaca.markets`. If not set, the adapter will use this paper trading URL. For live trading, you would set this to the live API endpoint.

> **Note:** This project uses the modern `alpaca-py` SDK (^0.40.0) instead of the deprecated `alpaca-trade-api`. The `AlpacaBrokerAdapter` and data fetching modules have been updated to use this newer SDK, which provides better compatibility with recent Python versions and more reliable operation.

**Tradovate (Demo):**
- `TRADOVATE_CLIENT_ID`: Your Tradovate demo client ID (optional, might depend on specific demo setup).
- `TRADOVATE_ACCESS_TOKEN`: Your Tradovate demo access token.

**Note:** Never commit your API keys or secrets to the repository. Use a `.env` file (add it to `.gitignore`) or manage environment variables through your system for local development.

### Supported Providers & Granularities

| Provider  | Granularities (Examples)        | Notes                                         |
|-----------|---------------------------------|-----------------------------------------------|
| Alpaca    | 1Min, 5Min, 15Min, 1H, 1Day     | Uses SIP feed (free tier for IEX data)        |
| Tradovate | 1Min, 5Min, 1H (from ticks)     | Uses demo env; resamples tick data to OHLCV   |

## Live Trading (skeleton)

This project provides a basic live trading engine skeleton. It can load Backtrader strategies, connect to a broker adapter, and be controlled via a CLI.
The `AlpacaBrokerAdapter` connects to Alpaca for paper trading (see Environment Variables section for required API keys and URL configuration).
The adapter uses asynchronous APIs for all broker operations to ensure efficient and non-blocking execution.

**CLI Demo:**
```bash
python -m algo_mvp.live --config configs/live_sample.yaml
```

## Live Trading (Experimental)

This project includes an experimental live trading bridge using Backtrader and a custom integration with Alpaca built on the modern `alpaca-py` SDK. It supports paper and live trading.

### Configuration

Live trading configuration is done via YAML files, similar to data fetching and backtesting. See `configs/live_spy.yaml` for an example.

Key configuration options include:
- `provider`: Currently only `alpaca` is supported by the custom bridge.
- `symbol`: Ticker symbol for trading.
- `timeframe`: Data timeframe (currently `1Min` for the custom Alpaca bridge).
- `strategy`: Strategy name (must be compatible with Backtrader).
- `strategy_params`: Parameters for the chosen strategy.
- `runtime`: Runtime settings like `max_daily_loss_pct`, `session_start`, `session_end`, `kill_time`, `equity_snapshot_freq_sec`.

### CLI Usage

```bash
poetry run python -m algo_mvp.live --config configs/live_spy.yaml [--paper | --live] [--db <path>] [--kill-time HH:MM] [--verbose]
```

**Arguments:**
- `--config`: Path to the live config file (required).
- `--paper`: Run in paper trading mode (default).
- `--live`: Run in live trading mode.
- `--db`: Path to the SQLite database for logging trades, orders, and equity (default: `./databases/algo.db`).
- `--kill-time`: Optional time (HH:MM) to stop the runner early.
- `--verbose`: Enable detailed logging.

### Trade Logging

Live trading activity (orders, trades, equity snapshots, daily summaries) is logged to a SQLite database specified by the `--db` argument. The schema is defined in `databases/schema.sql`.

### Important Notes
- **Credentials:** Ensure `ALPACA_KEY_ID`, `ALPACA_SECRET_KEY`, and optionally `ALPACA_BASE_URL` are set as environment variables for the `AlpacaBrokerAdapter`.
- **Market Data:** The custom `AlpacaData` feed uses the `StockDataStream` from `alpaca-py`. Ensure your Alpaca account has the necessary market data subscriptions (e.g., IEX is free, SIP may require paid plans).
- **Brokerage:** The custom `AlpacaBroker` interacts with the Alpaca Trading API via `alpaca-py`.
- **Risk:** Live trading involves real financial risk. Use with extreme caution, especially with real money (`--live` flag). Start with paper trading (`--paper`) extensively.

## Running Multiple Strategies (Orchestrator)

This project includes an orchestrator module to launch and manage multiple `LiveRunner` instances simultaneously, each potentially running a different strategy or configuration. This is useful for deploying several trading bots from a single control point.

### Configuration

The orchestrator is configured using a YAML manifest file. A sample manifest is provided in `configs/orchestrator_sample.yaml`.

**Example `configs/orchestrator_sample.yaml`:**
```yaml
runners:
  - name: mes_scalp_bot
    config: configs/live_tradovate_mes.yaml  # Path to a LiveRunner YAML config
  - name: spy_momo_bot
    config: configs/live_alpaca_spy.yaml    # Path to another LiveRunner YAML config
log_level: INFO  # Overall log level for the orchestrator (DEBUG, INFO, WARNING, ERROR)
restart_on_crash: true  # If true, automatically restarts a runner if it crashes
```

Each item under `runners` defines a `LiveRunner` instance to be managed:
- `name`: A unique name for this runner instance (used for logging and status display).
- `config`: The file path to the standard YAML configuration file for that `LiveRunner` (e.g., `configs/live_sample.yaml`, `configs/live_spy.yaml`).

### CLI Usage

To start the orchestrator and launch all configured runners, use the following command:

```bash
python -m algo_mvp.orchestrator --config configs/orchestrator_sample.yaml
```

**Optional flags:**
- `--verbose`: Enable verbose logging for the orchestrator and potentially its runners (sets orchestrator log level to DEBUG, overriding the YAML `log_level`).
- `--no-restart`: Disable automatic restarting of crashed runners, even if `restart_on_crash: true` is set in the YAML manifest.

The CLI will display a live table with the status of each runner, refreshing periodically. Press `Ctrl+C` to gracefully shut down the orchestrator and all its runners.

## Supervisor & Health Check

A lightweight supervisor provides an HTTP interface for monitoring and controlling the Orchestrator.

### Configuration

The supervisor is configured using a YAML file. A sample configuration is provided in `configs/supervisor_sample.yaml`.

**Example `configs/supervisor_sample.yaml`:**
```yaml
orchestrator_config: configs/orchestrator_sample.yaml
host: 0.0.0.0
port: 8000
log_level: INFO
```

- `orchestrator_config`: Path to the Orchestrator's manifest YAML file.
- `host`: Host address for the supervisor's HTTP server.
- `port`: Port for the supervisor's HTTP server.
- `log_level`: Log level for the supervisor.

### CLI Usage

To start the supervisor, use the following command:

```bash
python -m algo_mvp.cli.run_supervisor --config configs/supervisor_sample.yaml
```

The CLI will print the health endpoint URL and instructions for the shutdown token.

### Environment Variables

- `SUPERVISOR_TOKEN`: A shared secret token required for the `/shutdown` endpoint.

### API Endpoints

- **`GET /health`**
  - Returns the health status of the Orchestrator and its runners.
  - Response (200 OK if all runners healthy):
    ```json
    {
      "status": "ok",
      "runners": [
        {"name": "runner1", "status": "RUNNING"},
        {"name": "runner2", "status": "RUNNING"}
      ]
    }
    ```
  - Response (503 Service Unavailable if any runner is not healthy):
    ```json
    {
      "status": "error",
      "runners": [
        {"name": "runner1", "status": "RUNNING"},
        {"name": "runner2", "status": "CRASHED"}
      ]
    }
    ```
  - **Curl Example:**
    ```bash
    curl http://localhost:8000/health
    ```

- **`POST /shutdown?token=<TOKEN>`**
  - Gracefully shuts down the Orchestrator and the supervisor process.
  - Requires a valid `token` query parameter matching the `SUPERVISOR_TOKEN` environment variable.
  - **Curl Example (replace `<YOUR_TOKEN>`):**
    ```bash
    curl -X POST "http://localhost:8000/shutdown?token=<YOUR_TOKEN>"
    ```

## Back-testing

This project includes a vectorbt-powered backtesting engine that can run single or grid parameter sweeps based on YAML configuration files.

### Configuration

Backtests are configured using YAML files. A sample configuration can be found in the `configs/` directory:
- `configs/sample_sweep.yaml`

Key configuration options include:
- `provider`: `alpaca` or `tradovate` (data source)
- `symbol`: The ticker symbol to backtest
- `timeframe`: Data granularity (e.g., `1Min`, `1Day`)
- `strategy`: Strategy name (e.g., `vwap_atr`, `three_day_momo`)
- `params`: Strategy parameters (scalar values for single runs, lists for grid sweeps)
- `cash`: Initial cash amount
- `commission`: Commission per trade
- `metrics`: Performance metrics to calculate (e.g., `sharpe`, `max_drawdown`, `cagr`)

### CLI Usage

To run a backtest, use the following command:

```bash
poetry run python -m algo_mvp.backtest --config configs/sample_sweep.yaml
```

**Optional flags:**
- `--output-dir`: Custom output directory for backtest results
- `--verbose`: Enable verbose output for more details on the backtesting process

### Strategies

The following strategies are included:

1. **VWAP-ATR** (`vwap_atr`): Uses Volume Weighted Average Price (VWAP) and Average True Range (ATR) to generate trading signals.
   - Parameters:
     - `band_mult`: Multiplier for ATR bands around VWAP
     - `atr_len`: Period for ATR calculation

2. **Three Day Momentum** (`three_day_momo`): Looks for strong directional moves over a three-day period and enters in the direction of the momentum with ATR-based stops.
   - Parameters:
     - `band_mult`: Multiplier for ATR stop bands
     - `atr_len`: Period for ATR calculation

### Backtest Results

Backtest results are stored in the `backtests/` directory, organized by strategy, symbol, and timestamp. Each backtest run produces:
- `metrics.csv`: Summary of performance metrics for all parameter combinations
- `equity_{run_id}.csv`: Equity curve for each run
- `plot_{run_id}.html`: Interactive vectorbt plot for each run
- A copy of the configuration file used for the backtest
