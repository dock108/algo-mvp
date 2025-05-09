from typing import Any, Dict, List

from pydantic import BaseModel, validator


class RunnerConfig(BaseModel):
    name: str
    provider: str
    strategy: str
    params: Dict[str, Any]
    symbol: str
    timeframe: str

    @validator("strategy")
    def strategy_must_contain_colon(cls, value):
        if ":" not in value:
            raise ValueError("strategy must be in format module.path:ClassName")
        return value


class LiveConfig(BaseModel):
    runners: List[RunnerConfig]
