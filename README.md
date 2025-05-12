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

## Docker Quick-Start

To run the entire stack (supervisor, orchestrator, runners, dashboard, and database) using Docker Compose:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/algo-mvp.git
    cd algo-mvp
    ```

2.  **Build and run the services:**
    ```bash
    docker compose up --build
    ```
    This command will build the Docker image and start all services. The backend (supervisor and orchestrator) will be available on port `8000`, and the dashboard will be on port `8501`.

3.  **Access the dashboard:**
    Open your browser and navigate to `http://localhost:8501`.

### Environment Variable Overrides

You can override the default `SUPERVISOR_TOKEN` and `DASHBOARD_PASSWORD` by setting them in your environment before running `docker compose up` or by creating a `.env` file in the project root:

```env
# .env
SUPERVISOR_TOKEN=your_custom_secure_token
DASHBOARD_PASSWORD=your_strong_dashboard_password
```

Data will be persisted to the `./data` directory on your host machine.

## Database setup

The project uses SQLite for storing trade history, logs, and equity snapshots. Alembic is used for database migrations.

To create or upgrade the database to the latest version, run:

```bash
poetry run python -m algo_mvp.db upgrade
```

This command will create `data/algo.db` if it doesn't exist and apply any pending migrations.

## Real-time DB Logging

The project includes a thread-safe DBWriter middleware that automatically logs all trading activity to the SQLite database. All live trading orders, fills, equity snapshots, and error logs are persisted in real-time, providing a complete record of your trading system's activity.

The DBWriter uses a background worker thread with a queue to ensure that database writes don't block the main trading threads, maintaining system responsiveness even during high-frequency trading.

### Features

- Thread-safe, queue-based architecture for non-blocking database writes
- Automatic logging of orders, fills, equity snapshots, and critical messages
- Integration with LiveRunner for seamless recording of all trading activity
- Graceful shutdown with proper queue flushing to prevent data loss

No additional configuration is required - the LiveRunner automatically sets up the DBWriter when it starts.

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

## Dashboard

The project includes a Streamlit-based dashboard for monitoring live trading. The dashboard displays:

- Equity curve chart
- Performance metrics (Net P&L, CAGR, Max Drawdown, Hit Rate)
- Open positions
- Recent trades

To launch the dashboard, run:

```bash
python -m algo_mvp.dashboard
```

The dashboard will be available at http://localhost:8501 by default and features:

- Dark/light theme toggle (🌗) in the sidebar
- Responsive layout that adapts to mobile screens (down to 390px width)
- Manual refresh control to pause/resume the 5-second auto-refresh
- Styled metric cards and tables
- Admin controls for emergency actions (token-protected)
- Password-protected access

![Dashboard Screenshot](assets/dashboard_dark.png)

### Admin Controls

The dashboard includes protected admin controls for emergency actions and configuration changes:

- **Flatten All** - Closes all open positions across all running strategies
- **Pause Strategy** - Toggles a specific runner between active and paused state
- **Reload Config** - Hot-reloads the orchestrator YAML config without restarting

All admin actions require a valid supervisor token to execute.

### Security

The dashboard and API endpoints include several security features:

1. **Password Protection**:
   - The dashboard is protected by a password gate
   - Set via environment variable: `DASHBOARD_PASSWORD=your-secure-password`
   - Or in `.streamlit/secrets.toml`: `dashboard_password = "your-secure-password"`
   - Default password is "password" if not configured

2. **Supervisor Token**:
   - All action endpoints (/action/flatten_all, /action/pause, /action/reload_config) are protected
   - Set via environment variable: `SUPERVISOR_TOKEN=your-secure-token`
   - Required for all sensitive operations

3. **Config Hot-Reload**:
   - The `/action/reload_config` endpoint allows updating runner configurations without downtime
   - Example usage:
     ```bash
     curl -X POST "http://localhost:8000/action/reload_config?token=your-secure-token"
     ```
   - Returns status of all runners after reload

For production deployments, always set strong passwords/tokens and consider using HTTPS.

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

- **`POST /action/flatten_all?token=<TOKEN>`**
  - Closes all open positions across all running LiveRunners.
  - Requires a valid `token` query parameter matching the `SUPERVISOR_TOKEN` environment variable.
  - Response (200 OK):
    ```json
    {
      "message": "Flatten all initiated. Closing positions for 2 runner(s)."
    }
    ```
  - **Curl Example (replace `<YOUR_TOKEN>`):**
    ```bash
    curl -X POST "http://localhost:8000/action/flatten_all?token=<YOUR_TOKEN>"
    ```

- **`POST /action/pause?runner=<RUNNER_NAME>&token=<TOKEN>`**
  - Toggles a runner between active and paused states. When paused, new orders are suppressed.
  - Parameters:
    - `runner`: Name of the runner to toggle (required)
    - `token`: Supervisor authentication token (required)
  - Response (200 OK):
    ```json
    {
      "paused": true  // true if paused, false if active
    }
    ```
  - **Curl Example (replace `<RUNNER_NAME>` and `<YOUR_TOKEN>`):**
    ```bash
    curl -X POST "http://localhost:8000/action/pause?runner=<RUNNER_NAME>&token=<YOUR_TOKEN>"
    ```

## Back-testing

This project includes a vectorbt-powered backtesting engine that can run single or grid parameter sweeps based on YAML configuration files.

### Configuration

Backtests are configured using YAML files. A sample configuration can be found in the `configs/` directory:
- `configs/sample_sweep.yaml`

Key configuration options include:
- `provider`: `alpaca` or `tradovate`

## One-command paper run

This project includes a streamlined workflow to go from backtesting recent data to launching a 30-day paper trading session with the best-performing parameters. This is orchestrated via a single `make` command.

### Prerequisites

- Ensure all dependencies for data fetching, backtesting, and live trading are installed.
- Your trading supervisor (e.g., Docker-based) should be configurable and support hot-reloading or restarting with new configurations.
- `jinja2-cli` and `jq` must be installed and accessible in your PATH for the helper scripts to function correctly.
  - Install `jinja2-cli`: `pip install jinja2-cli`
  - Install `jq`: (Consult your OS package manager, e.g., `sudo apt-get install jq` or `brew install jq`)
- The scripts `./scripts/backtest_last45.sh`, `./scripts/choose_best.py`, and `./scripts/start_paper.sh` must be executable (`chmod +x <script_name>`). The `make ensure_scripts_executable` command can do this for you.

### How it Works

The `make paper` command executes the following steps:

1.  **`clean_paper_artifacts` (Optional Cleanup):** Removes artifacts from previous paper runs (e.g., `best_params.json`, `generated_configs/`).
2.  **`fetch_and_backtest` (`./scripts/backtest_last45.sh`):**
    *   Fetches the last 45 days of market data (you need to configure the actual data fetching command within the script).
    *   Runs parameter sweeps on your default trading strategies using this data (you need to configure the actual backtesting command).
    *   *Expected output*: A metrics file (e.g., `./backtest_results/metrics.csv`) summarizing the performance of different parameter sets.
3.  **`choose_parameters` (`./scripts/choose_best.py`):**
    *   Reads the metrics file generated in the previous step.
    *   Selects the best parameter set based on predefined criteria (e.g., Sharpe ratio, total return, max drawdown).
    *   *Output*: Saves the best parameters to a JSON file (e.g., `./best_params.json`).
4.  **`start_paper_session` (`./scripts/start_paper.sh`):**
    *   Reads the `best_params.json` file.
    *   Uses Jinja2 templates (`workflows/templates/live_paper_template.yaml.jinja` and `workflows/templates/orchestrator_paper.yaml.jinja`) to generate live runner and orchestrator YAML configuration files for a 30-day paper trading session.
    *   The generated configuration files are saved in `./generated_configs/`.
    *   Hot-reloads or updates your trading supervisor with the new configurations to start the paper trading session (you need to configure the actual supervisor command).

### Running the Workflow

1.  **Ensure Prerequisites:** Verify all prerequisites mentioned above are met.
2.  **Configure Scripts:** You **must** customize the placeholder commands within:
    *   `./scripts/backtest_last45.sh` (for data fetching and backtesting).
    *   `./scripts/start_paper.sh` (for hot-reloading your supervisor).
    *   Also, ensure the metrics output path in `backtest_last45.sh` matches the input path in the `Makefile` for `choose_best.py` (default: `./backtest_results/metrics.csv`).
3.  **Make Scripts Executable (if not already):**
    ```bash
    make ensure_scripts_executable
    ```
4.  **Run the Paper Workflow:**
    ```bash
    make paper
    ```

This command will automate the entire process. Monitor the output for any errors. If successful, your paper trading session should be initiated with the optimized parameters.

### Customization

-   **Parameter Selection Logic:** Modify `scripts/choose_best.py` to change the metrics or logic for selecting the best parameters.
-   **Configuration Templates:** Adjust the Jinja2 templates in `workflows/templates/` to match the exact configuration schema required by your live runners and orchestrator.
-   **File Paths:** If you change default file paths for metrics or parameters, update them in the `Makefile` and relevant scripts.
