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


@patch("algo_mvp.backtest.engine.vbt.Portfolio.from_signals")
@patch("algo_mvp.backtest.engine.BacktestEngine._load_data")
@patch("algo_mvp.backtest.engine.importlib.import_module")
def test_run_backtest(
    mock_import,
    mock_load_data,
    mock_portfolio,
    sample_config_file,
    sample_price_data,
    tmp_path,
):
    """Test that the backtest runs correctly."""
    # Mock strategy
    mock_strategy_class = MagicMock()
    mock_strategy_instance = MagicMock()
    mock_strategy_class.return_value = mock_strategy_instance

    # Mock the strategy run method to return entry/exit signals
    entries = pd.Series(
        [False, True, False, False, True, False, False, False, False, False],
        index=sample_price_data.index,
    )
    exits = pd.Series(
        [False, False, True, False, False, True, False, False, False, False],
        index=sample_price_data.index,
    )
    mock_strategy_instance.run.return_value = {"entries": entries, "exits": exits}

    # Mock the module import
    mock_module = MagicMock()
    mock_module.VwapAtrStrategy = mock_strategy_class
    mock_import.return_value = mock_module

    # Mock data loading
    mock_load_data.return_value = sample_price_data

    # Mock portfolio
    mock_portfolio_instance = MagicMock()
    mock_portfolio_instance.sharpe_ratio.return_value = 1.5
    mock_portfolio_instance.drawdown.return_value.max.return_value = 0.1
    mock_portfolio_instance.annual_return.return_value = 0.2
    mock_portfolio_instance.trades = MagicMock()
    mock_portfolio_instance.trades.win_rate = 0.6
    mock_portfolio_instance.trades.expectancy = 1.2
    mock_portfolio_instance.positions = MagicMock()
    mock_portfolio_instance.positions.avg_coverage = 0.5
    mock_portfolio_instance.equity = pd.Series(
        [10000, 10100, 10200, 10300], index=sample_price_data.index[:4]
    )
    mock_portfolio.return_value = mock_portfolio_instance

    # Initialize engine with custom output directory
    output_dir = tmp_path / "backtest_results"
    engine = BacktestEngine(
        config_file=str(sample_config_file), output_dir=str(output_dir), verbose=True
    )

    # Run backtest
    result = engine.run()

    # Check that the backtest ran successfully
    assert result is True

    # Check that the strategy was called with the correct parameters
    assert mock_strategy_class.call_count > 0

    # Check that the portfolio was created
    assert mock_portfolio.call_count > 0

    # Check that metrics were calculated
    assert len(engine.results) > 0

    # Check that output files were created
    assert output_dir.exists()
    assert (output_dir / "metrics.csv").exists()

    # Check that at least one equity and plot file was created
    equity_files = list(output_dir.glob("equity_*.csv"))
    plot_files = list(output_dir.glob("plot_*.html"))
    assert len(equity_files) > 0
    assert len(plot_files) > 0
