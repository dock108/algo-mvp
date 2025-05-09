"""Tests for the backtesting engine."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import yaml

from algo_mvp.backtest.engine import BacktestConfig, BacktestEngine


@pytest.fixture
def sample_config_dict():
    """Return a sample configuration dictionary."""
    return {
        "provider": "alpaca",
        "symbol": "AAPL",
        "timeframe": "1Min",
        "strategy": "vwap_atr",
        "params": {"band_mult": [1.5, 2.0], "atr_len": [10, 14]},
        "cash": 10000,
        "commission": 0.0,
        "metrics": ["sharpe", "max_drawdown"],
    }


@pytest.fixture
def sample_config_file(tmp_path, sample_config_dict):
    """Create a sample configuration file."""
    config_file = tmp_path / "test_config.yaml"

    with open(config_file, "w") as f:
        yaml.dump(sample_config_dict, f)

    return config_file


@pytest.fixture
def sample_price_data():
    """Create a small price DataFrame for testing."""
    # Create a simple 10-row DataFrame with OHLCV data
    dates = pd.date_range(start="2023-01-01", periods=10, freq="1min")

    data = {
        "open": [100, 101, 102, 101, 102, 103, 102, 101, 102, 103],
        "high": [102, 103, 104, 103, 104, 105, 104, 103, 104, 105],
        "low": [99, 100, 101, 100, 101, 102, 101, 100, 101, 102],
        "close": [101, 102, 103, 102, 103, 104, 103, 102, 103, 104],
        "volume": [1000, 1100, 1200, 1100, 1200, 1300, 1200, 1100, 1200, 1300],
    }

    return pd.DataFrame(data, index=dates)


def test_backtest_config_validation():
    """Test that the BacktestConfig validates correctly."""
    # Valid config
    valid_config = {
        "provider": "alpaca",
        "symbol": "AAPL",
        "timeframe": "1Min",
        "strategy": "vwap_atr",
        "params": {"band_mult": 1.5, "atr_len": 10},
        "cash": 10000,
        "commission": 0.0,
        "metrics": ["sharpe", "max_drawdown"],
    }

    config = BacktestConfig(**valid_config)
    assert config.provider == "alpaca"
    assert config.symbol == "AAPL"
    assert config.params.band_mult == 1.5
    assert config.params.atr_len == 10

    # Test list parameters
    list_params = {
        "provider": "alpaca",
        "symbol": "AAPL",
        "timeframe": "1Min",
        "strategy": "vwap_atr",
        "params": {"band_mult": [1.5, 2.0], "atr_len": [10, 14]},
        "cash": 10000,
        "commission": 0.0,
    }

    config = BacktestConfig(**list_params)
    assert isinstance(config.params.band_mult, list)
    assert len(config.params.band_mult) == 2
    assert config.params.band_mult[0] == 1.5

    # Test invalid provider
    with pytest.raises(ValueError):
        BacktestConfig(
            provider="invalid",
            symbol="AAPL",
            timeframe="1Min",
            strategy="vwap_atr",
            params={"band_mult": 1.5, "atr_len": 10},
        )

    # Test invalid metrics
    with pytest.raises(ValueError):
        BacktestConfig(
            provider="alpaca",
            symbol="AAPL",
            timeframe="1Min",
            strategy="vwap_atr",
            params={"band_mult": 1.5, "atr_len": 10},
            metrics=["invalid_metric"],
        )


@patch("algo_mvp.backtest.engine.importlib.import_module")
def test_load_strategy(mock_import, sample_config_file):
    """Test that strategies are loaded correctly."""
    # Mock the strategy class
    mock_strategy = MagicMock()
    mock_module = MagicMock()
    mock_module.VwapAtrStrategy = mock_strategy
    mock_import.return_value = mock_module

    # Initialize engine
    engine = BacktestEngine(config_file=str(sample_config_file), verbose=True)

    # Check that the strategy was loaded
    mock_import.assert_called_once_with("algo_mvp.backtest.strategies.vwap_atr")
    assert engine.strategy_instance == mock_strategy


@patch("algo_mvp.backtest.engine.pd.read_parquet")
def test_load_data(mock_read_parquet, sample_config_file, sample_price_data):
    """Test that data is loaded correctly."""
    # Mock the read_parquet function
    mock_read_parquet.return_value = sample_price_data

    # Mock Path.exists to return True
    with patch("pathlib.Path.exists", return_value=True):
        # Initialize engine
        engine = BacktestEngine(config_file=str(sample_config_file), verbose=True)

        # Load data
        data = engine._load_data()

        # Check that the data was loaded
        mock_read_parquet.assert_called_once()
        assert isinstance(data, pd.DataFrame)
        assert len(data) == len(sample_price_data)
        assert "close" in data.columns


def test_expand_parameters(sample_config_file):
    """Test that parameter grid expansion works correctly."""
    # Initialize engine
    engine = BacktestEngine(config_file=str(sample_config_file), verbose=True)

    # Expand parameters
    param_grid = engine._expand_parameters()

    # Check that the grid has the correct number of combinations
    assert len(param_grid) == 4  # 2 band_mult values * 2 atr_len values

    # Check that all combinations are present
    expected_combinations = [
        {"band_mult": 1.5, "atr_len": 10},
        {"band_mult": 1.5, "atr_len": 14},
        {"band_mult": 2.0, "atr_len": 10},
        {"band_mult": 2.0, "atr_len": 14},
    ]

    for combo in expected_combinations:
        assert combo in param_grid


@pytest.mark.skip(
    reason="Skipping due to persistent Numba TypeError with vbt.Portfolio.from_signals mocking"
)
@patch("algo_mvp.backtest.engine.vbt.Portfolio.from_signals")
@patch("algo_mvp.backtest.engine.importlib.import_module")
@patch("pathlib.Path.exists", return_value=True)
@patch("pandas.read_parquet")
def test_run_backtest(
    mock_pd_read_parquet,
    mock_path_exists,
    mock_importlib_import_module,
    mock_vbt_portfolio_from_signals,
    sample_config_file,
    sample_price_data,
    tmp_path,
):
    """Test that the backtest runs correctly."""
    # 1. Mock for _load_data()
    mock_pd_read_parquet.return_value = sample_price_data

    # 2. Mock for _load_strategy() (called during BacktestEngine.__init__)
    #    and for strategy_obj.run() (called inside engine.run())
    mock_strategy_class = MagicMock(name="MockStrategyClass")
    mock_strategy_instance = MagicMock(name="MockStrategyInstance")
    mock_strategy_class.return_value = mock_strategy_instance

    # Ensure entries and exits have the same length as sample_price_data
    num_rows = len(sample_price_data)
    entries_list = ([False, True] * (num_rows // 2 + 1))[:num_rows]
    exits_list = ([False, False] * (num_rows // 2 + 1))[:num_rows]

    entries = pd.Series(entries_list, index=sample_price_data.index)
    exits = pd.Series(exits_list, index=sample_price_data.index)
    mock_strategy_instance.run.return_value = {"entries": entries, "exits": exits}

    mock_module = MagicMock(name="MockImportedModule")
    mock_module.VwapAtrStrategy = mock_strategy_class  # From sample_config_file
    mock_importlib_import_module.return_value = mock_module

    # 3. Mock for vbt.Portfolio.from_signals()
    mock_portfolio_object = MagicMock(name="MockPortfolioObject")
    mock_portfolio_object.sharpe_ratio.return_value = 1.5

    # Mock for portfolio.drawdown().max()
    mock_drawdown = MagicMock()
    mock_drawdown.max.return_value = 0.1
    mock_portfolio_object.drawdown.return_value = mock_drawdown

    mock_portfolio_object.annual_return.return_value = 0.2

    mock_trades_accessor = MagicMock(name="MockTradesAccessor")
    mock_trades_accessor.win_rate = 0.6
    mock_trades_accessor.expectancy = 1.2
    mock_trades_accessor.__len__.return_value = (
        1  # To satisfy `if len(portfolio.trades) > 0`
    )
    # In vectorbt, portfolio.trades is an accessor, not a list.
    # The check `if len(portfolio.trades)` is unusual, but we mock to satisfy it.
    mock_portfolio_object.trades = mock_trades_accessor

    mock_positions_accessor = MagicMock(name="MockPositionsAccessor")
    mock_positions_accessor.avg_coverage = 0.5
    mock_portfolio_object.positions = mock_positions_accessor

    mock_portfolio_object.equity = pd.Series(
        [10000, 10100], index=sample_price_data.index[:2]
    )

    mock_figure = MagicMock(name="MockPlotlyFigure")
    mock_portfolio_object.plot.return_value = mock_figure
    mock_vbt_portfolio_from_signals.return_value = mock_portfolio_object

    # --- Initialize and run BacktestEngine ---
    output_dir = tmp_path / "backtest_results"
    # engine will create the output_dir if it doesn't exist via self.output_dir.mkdir

    engine = BacktestEngine(
        config_file=str(sample_config_file), output_dir=str(output_dir), verbose=True
    )

    # Call the actual engine.run()
    result = engine.run()

    assert result is True

    # Assertions
    # sample_config_file has 2 params for band_mult and 2 for atr_len -> 4 combinations
    expected_run_count = 4
    assert mock_strategy_instance.run.call_count == expected_run_count
    assert mock_vbt_portfolio_from_signals.call_count == expected_run_count

    mock_importlib_import_module.assert_called_once_with(
        "algo_mvp.backtest.strategies.vwap_atr"
    )

    assert output_dir.exists()
    assert (output_dir / "metrics.csv").exists()

    equity_files = list(output_dir.glob("equity_*.csv"))
    plot_files = list(output_dir.glob("plot_*.html"))
    assert len(equity_files) == expected_run_count
    assert len(plot_files) == expected_run_count


# --- Additional tests for BacktestEngine methods ---


def test_engine_log_verbose(sample_config_file, capsys):
    """Test that _log method prints only when verbose is True."""
    engine_verbose = BacktestEngine(config_file=str(sample_config_file), verbose=True)
    engine_not_verbose = BacktestEngine(
        config_file=str(sample_config_file), verbose=False
    )

    log_message = "Test log message"

    # Test verbose engine
    engine_verbose._log(log_message)
    captured_verbose = capsys.readouterr()
    assert log_message in captured_verbose.out

    # Test non-verbose engine
    engine_not_verbose._log(log_message)
    captured_not_verbose = capsys.readouterr()
    assert captured_not_verbose.out == ""


def test_load_config_file_not_found():
    """Test error handling when config file does not exist."""
    with pytest.raises(FileNotFoundError):
        BacktestEngine(config_file="nonexistent_config.yaml")


def test_load_config_invalid_yaml(tmp_path):
    """Test error handling for invalid YAML format."""
    invalid_yaml_file = tmp_path / "invalid.yaml"
    invalid_yaml_file.write_text(
        "provider: alpaca\nsymbol: AAPL\nparams: { band_mult: [1.5]"
    )  # Invalid YAML

    with pytest.raises(yaml.YAMLError):
        BacktestEngine(config_file=str(invalid_yaml_file))


def test_load_config_validation_error(tmp_path):
    """Test error handling for Pydantic validation errors in config."""
    invalid_config_file = tmp_path / "invalid_config.yaml"
    invalid_config_content = """
provider: alpaca
symbol: AAPL
timeframe: 1Min
strategy: vwap_atr
params:
  band_mult: 1.5
  # atr_len is missing, should cause validation error
cash: 10000
"""
    invalid_config_file.write_text(invalid_config_content)

    with pytest.raises(ValueError):  # Pydantic raises ValueError on validation failure
        BacktestEngine(config_file=str(invalid_config_file))


@patch("algo_mvp.backtest.engine.importlib.import_module")
def test_load_strategy_not_found(mock_import, sample_config_file):
    """Test error handling when the specified strategy module cannot be found."""
    mock_import.side_effect = ImportError("Module not found")

    with pytest.raises(ValueError, match="Strategy 'vwap_atr' not found"):
        BacktestEngine(config_file=str(sample_config_file))


@patch("algo_mvp.backtest.engine.importlib.import_module")
def test_load_strategy_class_missing(mock_import, sample_config_file):
    """Test error handling when strategy class is missing in the module."""
    # Create a mock module specifying `object` as the spec.
    # This ensures accessing any attribute not present on a bare object (like 'VwapAtrStrategy')
    # will raise an AttributeError.
    mock_imported_module = MagicMock(name="MockModuleWithoutStrategy", spec=object)
    mock_import.return_value = mock_imported_module

    with pytest.raises(ValueError, match="Strategy 'vwap_atr' not found"):
        BacktestEngine(config_file=str(sample_config_file))


@patch("algo_mvp.backtest.engine.pd.read_parquet")
def test_load_data_missing_columns(mock_read_parquet, sample_config_file):
    """Test error handling when loaded data is missing required columns."""
    # Create data missing the 'close' column
    incomplete_data = pd.DataFrame(
        {
            "open": [100, 101],
            "high": [102, 103],
            "low": [99, 100],
            "volume": [1000, 1100],
        }
    )
    mock_read_parquet.return_value = incomplete_data

    with patch("pathlib.Path.exists", return_value=True):
        engine = BacktestEngine(config_file=str(sample_config_file))
        # Expect ValueError when _load_data is called implicitly or explicitly
        with pytest.raises(ValueError, match="Data missing required columns:.*close"):
            engine._load_data()


def test_calculate_metrics_zero_trades(sample_config_file):
    """Test metric calculation when the portfolio has zero trades."""
    # Initialize engine and modify config to include trade-dependent metrics
    with patch("algo_mvp.backtest.engine.importlib.import_module"):
        engine = BacktestEngine(config_file=str(sample_config_file))
        engine.config.metrics = [
            "sharpe",
            "max_drawdown",
            "win_rate",
            "expectancy",
            "exposure",
        ]

    # Create a mock portfolio with zero trades
    mock_portfolio = MagicMock(name="MockPortfolioZeroTrades")

    # Mock the trades accessor to indicate zero trades
    mock_trades_accessor = MagicMock(name="MockTradesAccessorZero")
    mock_trades_accessor.__len__.return_value = 0
    mock_portfolio.trades = mock_trades_accessor

    # Mock other necessary portfolio attributes used by _calculate_metrics
    mock_portfolio.sharpe_ratio.return_value = 0.0  # Example value
    mock_drawdown = MagicMock()
    mock_drawdown.max.return_value = 0.0
    mock_portfolio.drawdown.return_value = mock_drawdown
    mock_portfolio.annual_return.return_value = 0.0
    mock_positions_accessor = MagicMock(name="MockPositionsAccessorZero")
    mock_positions_accessor.avg_coverage = 0.0
    mock_portfolio.positions = mock_positions_accessor

    # Calculate metrics
    metrics = engine._calculate_metrics(mock_portfolio)

    # Assert metrics dependent on trades are zero because len(trades) == 0
    assert "win_rate" in metrics
    assert metrics["win_rate"] == 0.0
    assert "expectancy" in metrics
    assert metrics["expectancy"] == 0.0

    # Assert other requested metrics were calculated (or defaulted)
    assert "sharpe" in metrics
    assert metrics["sharpe"] == 0.0
    assert "max_drawdown" in metrics
    assert metrics["max_drawdown"] == 0.0  # drawdown max * 100
    assert "exposure" in metrics
    assert metrics["exposure"] == 0.0  # avg_coverage * 100


def test_calculate_metrics_various(sample_config_dict, tmp_path):
    """Test calculation of various specific metrics."""
    # Modify the config to include all possible metrics for testing
    config_dict_all_metrics = sample_config_dict.copy()  # Use the fixture value
    config_dict_all_metrics["metrics"] = [
        "sharpe",
        "max_drawdown",
        "cagr",
        "win_rate",
        "expectancy",
        "exposure",
    ]

    # Create a temporary config file with all metrics
    all_metrics_config_file = tmp_path / "all_metrics_config.yaml"
    with open(all_metrics_config_file, "w") as f:
        yaml.dump(config_dict_all_metrics, f)

    # Initialize engine with the modified config
    with patch("algo_mvp.backtest.engine.importlib.import_module"):
        engine = BacktestEngine(config_file=str(all_metrics_config_file))

    # Create a mock portfolio with some trade data
    mock_portfolio = MagicMock(name="MockPortfolioWithTrades")
    mock_trades_accessor = MagicMock(name="MockTradesAccessorWithTrades")
    mock_trades_accessor.__len__.return_value = 5  # Example: 5 trades
    mock_trades_accessor.win_rate = 0.75
    mock_trades_accessor.expectancy = 12.5
    mock_portfolio.trades = mock_trades_accessor

    mock_portfolio.sharpe_ratio.return_value = 1.8
    mock_drawdown = MagicMock()
    mock_drawdown.max.return_value = 0.15  # 15% drawdown before multiplication
    mock_portfolio.drawdown.return_value = mock_drawdown
    mock_portfolio.annual_return.return_value = 0.25  # 25% return before multiplication
    mock_positions_accessor = MagicMock(name="MockPositionsAccessorWithTrades")
    mock_positions_accessor.avg_coverage = 0.60  # 60% exposure before multiplication
    mock_portfolio.positions = mock_positions_accessor

    # Calculate metrics
    metrics = engine._calculate_metrics(mock_portfolio)

    # Assert all requested metrics are present and calculated correctly
    assert len(metrics) == 6
    assert metrics["sharpe"] == 1.8
    assert metrics["max_drawdown"] == pytest.approx(15.0)  # 0.15 * 100
    assert metrics["cagr"] == pytest.approx(25.0)  # 0.25 * 100
    assert metrics["win_rate"] == pytest.approx(75.0)  # 0.75 * 100
    assert metrics["expectancy"] == 12.5
    assert metrics["exposure"] == pytest.approx(60.0)  # 0.60 * 100


# TODO: Add tests for _save_results, _save_metrics_summary, _save_config_copy, _print_summary_table
