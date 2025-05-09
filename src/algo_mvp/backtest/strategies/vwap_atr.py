"""VWAP-ATR strategy implementation.

This strategy uses Volume Weighted Average Price (VWAP) and Average True Range (ATR)
to generate trading signals.

Parameters:
    band_mult: Multiplier for ATR bands around VWAP
    atr_len: Period for ATR calculation

Required columns:
    open, high, low, close, volume
"""

from typing import Any, Dict

import pandas as pd
import vectorbt as vbt

from algo_mvp.backtest.engine import Strategy


class VwapAtrStrategy(Strategy):
    """VWAP-ATR strategy for vectorbt.

    This strategy calculates VWAP and places ATR-width bands around it,
    generating long entries when price crosses above the upper band and
    exits when price crosses below VWAP.
    """

    def __init__(self, params: Dict[str, Any]):
        """Initialize with strategy parameters."""
        super().__init__(params)
        self.band_mult = params["band_mult"]
        self.atr_len = params["atr_len"]

    def run(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Run strategy on data and return entry/exit signals.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            Dict with 'entries' and 'exits' Series
        """
        # Ensure we have the required columns
        required_columns = ["open", "high", "low", "close", "volume"]
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Required column '{col}' not found in data")

        # Calculate VWAP
        data = data.copy()
        data["typical_price"] = (data["high"] + data["low"] + data["close"]) / 3
        data["vwap_numerator"] = data["typical_price"] * data["volume"]
        data["volume_cumsum"] = data["volume"].cumsum()
        data["vwap_numerator_cumsum"] = data["vwap_numerator"].cumsum()
        data["vwap"] = data["vwap_numerator_cumsum"] / data["volume_cumsum"]

        # Calculate ATR
        high_low = data["high"] - data["low"]
        high_close_prev = (data["high"] - data["close"].shift(1)).abs()
        low_close_prev = (data["low"] - data["close"].shift(1)).abs()
        tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        atr = tr.rolling(window=self.atr_len).mean()

        # Calculate bands
        data["upper_band"] = data["vwap"] + (atr * self.band_mult)
        data["lower_band"] = data["vwap"] - (atr * self.band_mult)

        # Generate signals
        entries = (data["close"] > data["upper_band"]) & (
            data["close"].shift() <= data["upper_band"].shift()
        )
        exits = (data["close"] < data["vwap"]) & (
            data["close"].shift() >= data["vwap"].shift()
        )

        # Clean up signals (no entry after entry without exit)
        entries, exits = vbt.signals.nb.clean_enex_1d(entries.values, exits.values)

        return {
            "entries": pd.Series(entries, index=data.index),
            "exits": pd.Series(exits, index=data.index),
        }
