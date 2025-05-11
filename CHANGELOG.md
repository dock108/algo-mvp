## [0.1.0] – 2025-05-08
### Added
- Initial repo skeleton, Poetry config, CI, pre-commit hooks, smoke test.

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

## [0.9.1] – 2025-05-11
### Added
- Transaction-rollback fixtures and high-coverage edge-case tests for DBWriter.
### Changed
- CI coverage threshold raised to 85 %.

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
