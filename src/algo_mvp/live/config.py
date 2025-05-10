from typing import Any, Dict, List

from pydantic import BaseModel, Field, field_validator


class RunnerConfig(BaseModel):
    name: str
    provider: str
    strategy: str
    params: Dict[str, Any] = {}
    symbol: str
    timeframe: str

    @field_validator("strategy")
    @classmethod
    def strategy_must_contain_colon(cls, value: str) -> str:
        if ":" not in value:
            raise ValueError("strategy must be in format module.path:ClassName")
        return value


class LiveConfig(BaseModel):
    runners: List[RunnerConfig] = Field(default_factory=list)
