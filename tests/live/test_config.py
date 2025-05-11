import pytest
from pydantic import ValidationError

from algo_mvp.live.config import LiveConfig, RunnerConfig


class TestRunnerConfig:
    def test_runner_config_valid(self):
        config_data = {
            "name": "test_runner",
            "provider": "mock_provider",
            "strategy": "my_module.strategies:MyStrategy",
            "params": {"param1": "value1"},
            "symbol": "AAPL",
            "timeframe": "1D",
        }
        config = RunnerConfig(**config_data)
        assert config.name == "test_runner"
        assert config.provider == "mock_provider"
        assert config.strategy == "my_module.strategies:MyStrategy"
        assert config.params == {"param1": "value1"}
        assert config.symbol == "AAPL"
        assert config.timeframe == "1D"

    def test_runner_config_invalid_strategy_format(self):
        with pytest.raises(ValidationError) as excinfo:
            RunnerConfig(
                name="test_runner",
                provider="mock_provider",
                strategy="invalid_path_format_no_colon",  # Invalid format
                symbol="AAPL",
                timeframe="1D",
            )
        assert "strategy must be in format module.path:ClassName" in str(excinfo.value)

    def test_runner_config_missing_required_fields(self):
        with pytest.raises(ValidationError) as excinfo:
            # Missing 'provider', 'strategy', 'symbol', 'timeframe'
            RunnerConfig(name="test_runner")
        error_str = str(excinfo.value)
        # Check that the names of missing fields are in the error message
        assert "provider" in error_str
        assert "strategy" in error_str
        assert "symbol" in error_str
        assert "timeframe" in error_str
        assert "Field required" in error_str  # General check for the type of error

    def test_runner_config_optional_params(self):
        config_data = {
            "name": "test_runner_no_params",
            "provider": "mock_provider",
            "strategy": "my_module.strategies:AnotherStrategy",
            "symbol": "MSFT",
            "timeframe": "1H",
            # params is optional and should default to {}
        }
        config = RunnerConfig(**config_data)
        assert config.params == {}

    def test_runner_config_extra_fields_allowed(self):
        # Pydantic's default is to ignore extra fields if not configured otherwise
        config_data = {
            "name": "test_runner_extra",
            "provider": "mock_provider",
            "strategy": "my_module.strategies:ExtraStrategy",
            "symbol": "GOOG",
            "timeframe": "5Min",
            "extra_field_1": "some_value",
            "another_extra": 123,
        }
        try:
            config = RunnerConfig(**config_data)
            # Check a few known fields to ensure parsing still worked
            assert config.name == "test_runner_extra"
            assert config.symbol == "GOOG"
        except ValidationError as e:
            pytest.fail(f"Extra fields should be allowed by default: {e}")


class TestLiveConfig:
    def test_live_config_valid(self):
        config_data = {
            "runners": [
                {
                    "name": "runner1",
                    "provider": "mock",
                    "strategy": "my_module.strategies:StrategyA",
                    "params": {"key": "val"},
                    "symbol": "SPY",
                    "timeframe": "1D",
                },
                {
                    "name": "runner2",
                    "provider": "paper",
                    "strategy": "another.module:StrategyB",
                    "params": {"fast": 10, "slow": 20},
                    "symbol": "BTCUSD",
                    "timeframe": "1H",
                },
            ]
        }
        live_config = LiveConfig(**config_data)
        assert len(live_config.runners) == 2
        assert live_config.runners[0].name == "runner1"
        assert live_config.runners[0].strategy == "my_module.strategies:StrategyA"
        assert live_config.runners[0].params == {"key": "val"}
        assert live_config.runners[1].name == "runner2"
        assert live_config.runners[1].params == {"fast": 10, "slow": 20}

    def test_live_config_empty_runners(self):
        config_data = {"runners": []}
        live_config = LiveConfig(**config_data)
        assert len(live_config.runners) == 0

    def test_live_config_invalid_runner_config_in_list(self):
        config_data = {
            "runners": [
                {
                    "name": "runner1",
                    "provider": "mock",
                    "strategy": "invalid_format_for_runner",  # Invalid strategy format
                    "symbol": "ETHUSD",
                    "timeframe": "1Min",
                }
            ]
        }
        with pytest.raises(ValidationError) as excinfo:
            LiveConfig(**config_data)
        # Check that the error message points to the specific runner and field
        assert "runners.0.strategy" in str(excinfo.value)
        assert "strategy must be in format module.path:ClassName" in str(excinfo.value)

    def test_live_config_missing_runners_field_uses_default(self):
        # LiveConfig now defines runners: List[RunnerConfig] = Field(default_factory=list)
        # So, if 'runners' is missing, it should default to an empty list without error.
        config_data = {}  # Missing 'runners'
        try:
            live_config = LiveConfig(**config_data)
            assert live_config.runners == []
        except ValidationError as e:
            pytest.fail(f"LiveConfig should default 'runners' to [] if missing: {e}")

    def test_live_config_empty(self):
        """Test LiveConfig initialization with no runners."""
        config = LiveConfig()
        assert config.runners == []

    def test_live_config_with_runners(self):
        """Test LiveConfig initialization with multiple runners."""
        runner1_data = {
            "name": "runner1",
            "provider": "mock_provider",
            "strategy": "my_module.strategies:Strategy1",
            "symbol": "AAPL",
            "timeframe": "1D",
        }
        runner2_data = {
            "name": "runner2",
            "provider": "mock_provider",
            "strategy": "my_module.strategies:Strategy2",
            "symbol": "MSFT",
            "timeframe": "1H",
        }

        config = LiveConfig(
            runners=[RunnerConfig(**runner1_data), RunnerConfig(**runner2_data)]
        )

        assert len(config.runners) == 2
        assert config.runners[0].name == "runner1"
        assert config.runners[1].name == "runner2"

    def test_live_config_add_runner(self):
        """Test adding a runner to LiveConfig."""
        config = LiveConfig()
        runner_data = {
            "name": "new_runner",
            "provider": "mock_provider",
            "strategy": "my_module.strategies:Strategy",
            "symbol": "GOOG",
            "timeframe": "5Min",
        }

        config.runners.append(RunnerConfig(**runner_data))

        assert len(config.runners) == 1
        assert config.runners[0].name == "new_runner"

    def test_live_config_validation(self):
        """Test LiveConfig validation with invalid runner data."""
        runner_data = {
            "name": "invalid_runner",
            "provider": "mock_provider",
            "strategy": "invalid_strategy_format",  # Invalid format
            "symbol": "AAPL",
            "timeframe": "1D",
        }

        with pytest.raises(ValidationError) as excinfo:
            LiveConfig(runners=[RunnerConfig(**runner_data)])

        assert "strategy must be in format module.path:ClassName" in str(excinfo.value)
