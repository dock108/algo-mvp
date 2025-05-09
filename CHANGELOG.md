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
