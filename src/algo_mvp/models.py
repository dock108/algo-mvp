from typing import Literal

import pendulum
from pydantic import BaseModel, field_validator, model_validator
from typing_extensions import Self


class BaseConfig(BaseModel):
    symbol: str
    timeframe: str  # Later, this could be an Enum for better validation per provider
    start: str
    end: str

    @field_validator("start", "end")
    @classmethod
    def ensure_date_format(cls, v: str) -> str:
        try:
            pendulum.parse(v)
        except Exception as e:
            raise ValueError(
                f"Dates must be in a recognizable format (e.g., YYYY-MM-DD). Error: {e}"
            )
        return v

    @model_validator(mode="after")
    def start_must_be_before_end(self) -> Self:
        # This validator runs after individual field validation,
        # so self.start and self.end are expected to be valid date strings.
        start_dt = pendulum.parse(self.start)
        end_dt = pendulum.parse(self.end)
        if start_dt >= end_dt:
            raise ValueError("Start date must be before end date")
        return self


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
