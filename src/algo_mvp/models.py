from typing import Any, Dict, List, Literal

import pendulum
from pydantic import BaseModel, Field, validator


class BaseConfig(BaseModel):
    symbol: str
    timeframe: str  # Later, this could be an Enum for better validation per provider
    start: str
    end: str

    @validator("start", "end")
    def ensure_date_format(cls, v):
        try:
            pendulum.parse(v)
        except Exception:
            raise ValueError(
                "Dates must be in a recognizable format (e.g., YYYY-MM-DD)"
            )
        return v

    @validator("end")
    def start_must_be_before_end(cls, v, values):
        if "start" in values and values["start"]:
            # Ensure 'start' is parsed if it's valid, otherwise this validator might error
            # on an already invalid 'start' date before start.ensure_date_format runs for it.
            # Pydantic runs validators in order of field definition then validator definition.
            try:
                start_dt = pendulum.parse(values["start"])
                end_dt = pendulum.parse(
                    v
                )  # v is already validated by ensure_date_format for 'end'
                if start_dt >= end_dt:
                    raise ValueError("Start date must be before end date")
            except (
                ValueError
            ):  # Catches parsing errors from pendulum.parse if 'start' was bad
                # If start date itself is invalid, let its own validator handle it.
                # This validator should only care about the relationship if both dates are validly formatted.
                pass
        return v


class AlpacaConfig(BaseConfig):
    provider: Literal["alpaca"] = "alpaca"
    adjust: bool = True
    # Example Alpaca timeframes: 1Min, 5Min, 15Min, 1H, 1D
    # We could add specific validation for Alpaca timeframes here if needed


class TradovateConfig(BaseConfig):
    provider: Literal["tradovate"] = "tradovate"
    # For Tradovate, timeframe represents the target resampling interval from ticks
    # e.g., "1Min", "5Min"
    # `adjust` is not applicable here


# A union type for config loading, or the CLI can load the appropriate one based on 'provider' field.
# For now, the CLI will inspect the 'provider' field first.


class RunnerConfig(BaseModel):
    name: str
    provider: str  # mock, alpaca, tradovate, etc.
    strategy: str  # Dotted path: module.submodule:ClassName
    params: Dict[str, Any] = Field(default_factory=dict)
    symbol: str
    timeframe: str  # e.g., 1Min, 5Min, 1H, 1D

    @validator("strategy")
    def strategy_format(cls, value):
        if ":" not in value or "." not in value.split(":")[0]:
            raise ValueError("Strategy must be in 'module.submodule:ClassName' format.")
        return value


class LiveTradingConfig(BaseModel):
    runners: List[RunnerConfig]


# Example usage:
# if __name__ == '__main__':
#     yaml_data = """
# runners:
#   - name: mes_demo
#     provider: mock
#     strategy: algo_mvp.backtest.strategies.vwap_atr:VWAPATRStrategy
#     params:
#       band_mult: 2
#       atr_len: 14
#     symbol: MESM25
#     timeframe: 1Min
#   - name: cl_test
#     provider: mock
#     strategy: some.other.strategy:AnotherStrategy
#     params:
#       lookback: 50
#     symbol: CLZ25
#     timeframe: 5Min
# """
#     import yaml
#     data = yaml.safe_load(yaml_data)
#     config = LiveTradingConfig(**data)
#     print(config.model_dump_json(indent=2))
#     print(f"Strategy for first runner: {config.runners[0].strategy}")
