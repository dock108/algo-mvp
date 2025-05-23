# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.15.0] – 2025-05-11
### Added
- Production-ready Dockerfile and docker-compose.yml for one-command deploy.
- Optional Makefile tasks for build / up / down.
- CI smoke test spins compose and checks health endpoints.

## [0.14.0] – 2025-05-11
### Added
- End-to-end integration test covering adapter→DB→analytics→dashboard.
### Changed
- CI coverage threshold raised to 90%.

## [0.13.0] – 2025-05-11
### Added
- POST /action/reload_config to Supervisor for zero-downtime YAML hot-reload.
- Password-protected dashboard (`DASHBOARD_PASSWORD`).
- Reload Config button in dashboard admin controls.
- Enhanced security documentation in README.

## [0.12.0] – 2025-05-11
### Added
- Flatten All and Pause Strategy controls in dashboard, secured via Supervisor token.
- LiveRunner pause logic and Supervisor action endpoints.
- Admin Controls section in dashboard sidebar with token-based security.
- Documentation for API endpoints and configuration in README.

## [0.11.1] – 2025-05-11
### Added
- Custom Streamlit theme, dark/light toggle, metric cards, responsive layout.
- Playwright mobile-view smoke test.

## [0.9.1] – 2025-05-11
### Added
- Transaction-rollback fixtures and high-coverage edge-case tests for DBWriter.
### Changed
- CI coverage threshold raised to 85%.

## [0.9.0] - 2025-05-01
### Added
- Initial version with core functionality
- Basic trading engine
- Backtesting framework
- Database handling for orders and fills

## [0.1.0] – 2025-05-08
### Added
- Added LiveRunner with asynchronous order execution
- Broker adapter system for connecting to trading venues
- Error recovery with automatic reconnect
- Configurable logging levels
### Changed
- Refactored core trading logic for better separation of concerns
### Fixed
- Fixed potential thread/resource leaks when shutting down the LiveRunner.

## [0.2.0] – 2025-05-08
### Added
- Historical data fetching module (`algo_mvp.data`) supporting Alpaca (stocks/ETFs) and Tradovate demo (micro-futures).
- CLI for data fetching: `python -m algo_mvp.fetch --config <config_path> [--force] [--verbose]`.
- YAML configuration schema for data fetching, validated with Pydantic (`src/algo_mvp/models.py`).
- Sample configuration files in `configs/` (`sample_equity.yaml`, `sample_futures.yaml`).
- Parquet file output for historical data under `data/{provider}/{symbol}/{timeframe}/` with metadata (provider, download time, API version, date ranges).
- Logic to skip downloads if data for the requested range already exists (unless `--force` is used).
- Conditional year-based partitioning for large multi-year datasets.
- Tenacity-based retries for API calls in data fetchers.
- API client implementations for Alpaca (`src/algo_mvp/data/alpaca.py`) and Tradovate (`src/algo_mvp/data/tradovate.py`).
- Unit test structure initiated in `tests/data/` (actual test implementation to follow for full coverage).

### Changed
- Project version bumped to 0.2.0 in `pyproject.toml`.
- Added dependencies: `pydantic`, `tenacity`, `pyarrow`, `pendulum`, `requests`, `PyYAML`.
- Added dev dependency: `responses` (for testing HTTP requests).
- Updated `README.md` with data fetching instructions, CLI usage, and environment variable details.
- Updated `src/algo_mvp/fetch.py` to correctly run `cli/fetch_data.py`.

## [0.3.0] – 2025-05-08
### Added
- Vectorbt back-test engine with YAML-driven parameter sweeps.
- CLI `python -m algo_mvp.backtest --config <config_path> [--output-dir <path>] [--verbose]`.
- Two reference strategies (`vwap_atr`, `three_day_momo`).
- Parameter grid expansion for efficient backtesting of multiple parameter combinations.
- Rich output formatting with progress bars and summary tables.
- Comprehensive performance metrics (Sharpe, CAGR, Max Drawdown, Win Rate, Expectancy).
- Artifact storage including equity curves, interactive plots, and metrics summaries.
- Pydantic validation for backtest configuration.
- Unit tests for engine, strategies, and CLI with >80% coverage.

### Changed
- Project version bumped to 0.3.0 in `pyproject.toml`.
- Added dependencies: `vectorbt ~=0.26.0`, `plotly`, `rich`, `tqdm`.
- Updated `README.md` with backtesting instructions and strategy documentation.
- Updated `.pre-commit-config.yaml` with end-of-file-fixer hook.

## [0.4.0] – 2025-05-08
### Added
- LiveRunner skeleton with mock broker adapter.
- CLI `python -m algo_mvp.live`.
- Sample YAML config `configs/live_sample.yaml`.
- Pydantic models for live configuration in `src/algo_mvp/live/config.py`.
- Basic unit tests for `LiveRunner` and `MockBrokerAdapter`.

### Changed
- Project version bumped to 0.4.0 in `pyproject.toml`.

## [0.3.2] – 2025-05-08
### Added
- Custom AlpacaStore / AlpacaData / AlpacaBroker built on alpaca-py.
- Full async tests with mocked REST & WebSocket streams.

## [0.3.1] – 2025-05-08
### Added
- Backtrader live bridge for Alpaca (paper/live via `AlpacaStore`).
- SQLite trade logger (`algo_mvp.live.logger`) with `orders`, `trades`, and `equity_curve` tables.
- Equity snapshot logging via `TradeLogger.log_equity_snapshot`.
- Session summary logging (`TradeLogger.on_session_summary`) for daily metrics.
- Live runner orchestrator (`algo_mvp.live.runner`) with heartbeat, session timing, kill-time, and max daily loss checks.
- CLI `python -m algo_mvp.live` updated with `--kill-time` flag.
- Placeholder metrics calculation module (`algo_mvp.live.metrics`).
- Sample Alpaca live config (`configs/live_spy.yaml`).

### Changed
- Refactored broker logic into `algo_mvp.live.alpaca`.
- Renamed `TradeLogger` methods (`on_order` -> `on_order_notify`, etc.).
- Updated dependencies: added `alpaca-py`.
- Updated `.pre-commit-config.yaml` with `sqlfmt` hook.
- Updated `.github/workflows/ci.yml` with `SQLITE_SYNCHRONOUS=OFF` env var.

### Removed
- Generic `broker.py` (replaced by `alpaca.py`, Tradovate logic deferred to Issue 4b).

### Fixed
- Persistent linting errors in various files.

## [0.5.0] – 2025-05-09
### Added
- AlpacaBrokerAdapter (paper) with websocket fills + REST orders.
- Fixed failing tests in data/alpaca and data/tradovate modules.

## [0.5.1] – 2025-05-09
### Added
- Proper resource cleanup with close() method in BrokerAdapterBase and implementations.
- LiveRunner now properly cleans up adapter resources when stopping.

### Changed
- Converted AlpacaBrokerAdapter REST calls to async for consistent IO paradigm with WebSocket streaming.
- Improved error handling with async retries for all API calls.
- Updated MockBrokerAdapter to support the async interface.

### Fixed
- Fixed potential thread/resource leaks when shutting down the LiveRunner.

## [Unreleased] - YYYY-MM-DD

### Fixed
- Resolved `ModuleNotFoundError` in GitHub Actions for `algo_mvp` by ensuring root package installation and correct `pyproject.toml` `packages` directive for `src` layout.
- Addressed hanging tests in `tests/live/adapters/test_alpaca_adapter.py` by:
    - Converting synchronous test functions calling async methods to `async def` with `@pytest.mark.asyncio`.
    - Ensuring mocked client methods (e.g., `submit_order_async`) are `AsyncMock`.
    - Refining `test_close_method_stops_stream_and_joins_thread` to correctly mock and assert thread termination.
- Fixed hanging tests and assertion errors in `tests/live/adapters/test_mock.py` by:
    - Correcting assertions for `MockBrokerAdapter` methods (`get_cash`, `get_positions`) to match actual return types (dict for cash, List[Position] for positions).
    - Updating `test_mock_adapter_cancel_order` to assert boolean return from `cancel_order`.
- Enhanced `get_engine()` to automatically create parent directories for SQLite database files, preventing failures when ALGO_DB_URL points to a non-existent directory.

### Changed
- Migrated Pydantic from V1 to V2:
    - Replaced `validator` with `field_validator` and `model_validator`.
    - Updated validator signatures (e.g., `@classmethod`, `self` for `model_validator`).
    - Replaced `model.dict()` with `model.model_dump()`.
    - Applied to `src/algo_mvp/live/config.py`, `src/algo_mvp/models.py`, `src/algo_mvp/backtest/engine.py`.
- Updated pandas resampling rule in `src/algo_mvp/data/tradovate.py` from `"T"` to `"min"` to resolve `FutureWarning`.

### Removed
- N/A

### Added
- N/A

## [0.6.0] – 2025-05-09
### Added
- TradovateBrokerAdapter (demo) with REST + WebSocket support.
- Fixed & expanded data/tradovate tests.

## [0.7.0] – 2025-05-11
### Added
- Orchestrator module (`algo_mvp.orchestrator`) to launch and manage multiple LiveRunner instances based on a YAML manifest.
- CLI for the orchestrator: `python -m algo_mvp.orchestrator --config <manifest_path> [--verbose] [--no-restart]`.
- Rich-based live status table in the CLI displaying runner states, updated periodically.
- Graceful shutdown (Ctrl-C) for the orchestrator CLI.
- Automatic restart of crashed runners (configurable via YAML and CLI).
- Pydantic models for orchestrator configuration validation.
- Unit tests for the orchestrator module with mocked LiveRunners, covering start, stop, status, and crash handling (restart true/false).
- Sample orchestrator manifest `configs/orchestrator_sample.yaml`.

## [0.7.1] – 2025-05-11
### Added
- FastAPI-based Supervisor with /health and /shutdown routes.
- Env `SUPERVISOR_TOKEN` for shutdown auth.

## [0.8.0] – 2025-05-11
### Added
- SQLite schema (orders, fills, equity, logs) and Alembic migration 0001.
- CLI `python -m algo_mvp.db` for upgrade / current.

## [0.9.0] – 2025-05-11
### Added
- Thread-safe DBWriter middleware that records orders, fills, equity, and logs to SQLite.
- LiveRunner now logs to DB automatically.
- Real-time queryable P&L history in SQLite while trading is live.
- Comprehensive unit tests for the writer and LiveRunner integration.

## [0.10.0] – 2025-05-11
### Added
- Analytics API for P&L curves, trade logs, summary stats, drawdowns, open positions.

## [0.11.0] – 2025-05-11
### Added
- Read-only Streamlit dashboard: equity curve, stats, open positions, last trades.
- Auto-refresh every 5 seconds to display real-time updates.
- CLI command for launching the dashboard: `python -m algo_mvp.dashboard`

## [0.16.0] - YYYY-MM-DD
### Added
- Implemented a "Backtest-to-Paper" workflow, enabling a single command (`make paper`) to:
  - Fetch recent market data (last 45 days, placeholder).
  - Run parameter sweeps on trading strategies (placeholder).
  - Select the best performing parameters based on configurable metrics.
  - Generate live runner and orchestrator YAML configuration files using Jinja2 templates.
  - Hot-reload a trading supervisor to start a 30-day paper trading session (placeholder for supervisor command).
- New scripts:
  - `scripts/backtest_last45.sh`: Orchestrates data fetching and backtesting.
  - `scripts/choose_best.py`: Selects best parameters from backtest metrics (supports CSV/JSON input, outputs JSON).
  - `scripts/start_paper.sh`: Generates live configurations using `jinja2-cli` and reloads supervisor.
- New Jinja2 templates for configuration generation:
  - `workflows/templates/fetch_last45.yaml`: Example config for data fetching.
  - `workflows/templates/live_paper_template.yaml.jinja`: For individual live runner paper trading configs.
  - `workflows/templates/orchestrator_paper.yaml.jinja`: For the paper trading orchestrator config.
- New Makefile targets: `paper`, `fetch_and_backtest`, `choose_parameters`, `start_paper_session`, `clean_paper_artifacts`, `ensure_scripts_executable`.
- Unit tests for `scripts/choose_best.py` (`tests/workflows/test_choose_best.py`).
- Integration test structure for the `make paper` workflow (`tests/workflows/test_paper_workflow_integration.py`), skippable if Docker is unavailable.
- New section "One-command paper run" in `README.md` documenting the feature.

### Changed
- (If any existing components were modified to support this, list them here)

### Fixed
- (If this feature fixes any existing bugs, list them here)
