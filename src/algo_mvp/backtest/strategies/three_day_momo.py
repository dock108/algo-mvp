"""Three Day Momentum strategy implementation.

This strategy looks for strong directional moves over a three-day period and
enters in the direction of the momentum with ATR-based stops.

Parameters:
    band_mult: Multiplier for ATR stop bands
    atr_len: Period for ATR calculation

Required columns:
    open, high, low, close, volume
"""

from typing import Any, Dict

import pandas as pd
import vectorbt as vbt

from algo_mvp.backtest.engine import Strategy


class ThreeDayMomoStrategy(Strategy):
    """Three Day Momentum strategy for vectorbt.

    Looks for three consecutive days moving in the same direction
    and enters on the close of the third day, with an ATR-based stop.
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

        # Copy data to avoid modifying the original
        data = data.copy()

        # Calculate daily returns
        data["returns"] = data["close"].pct_change()

        # Identify three consecutive up days
        data["up_day"] = data["returns"] > 0
        data["three_up"] = (
            data["up_day"] & data["up_day"].shift(1) & data["up_day"].shift(2)
        )

        # Identify three consecutive down days
        data["down_day"] = data["returns"] < 0
        data["three_down"] = (
            data["down_day"] & data["down_day"].shift(1) & data["down_day"].shift(2)
        )

        # Calculate ATR for stop placement
        high_low = data["high"] - data["low"]
        high_close_prev = (data["high"] - data["close"].shift(1)).abs()
        low_close_prev = (data["low"] - data["close"].shift(1)).abs()
        tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        data["atr"] = tr.rolling(window=self.atr_len).mean()

        # Generate entry signals (on close of third day)
        entries = data["three_up"] | data["three_down"]

        # For exit signals, use ATR-based stops
        # For longs (three up days), exit when price falls below entry - ATR*mult
        # For shorts (three down days), exit when price rises above entry + ATR*mult

        # First create arrays for our stops
        stops = pd.Series(index=data.index, dtype=float)

        # Iterate through entries to set appropriate stops
        for i, (idx, entry) in enumerate(entries[entries].items()):
            entry_price = data.loc[idx, "close"]
            atr_value = data.loc[idx, "atr"]

            if data.loc[idx, "three_up"]:
                # Long position - stop is below
                stop_price = entry_price - (atr_value * self.band_mult)
            else:
                # Short position - stop is above
                stop_price = entry_price + (atr_value * self.band_mult)

            stops.loc[idx] = stop_price

        # Now calculate exits based on stops
        exits = pd.Series(False, index=data.index)

        for i in range(len(data) - 1):
            if not entries.iloc[i]:
                continue

            entry_idx = data.index[i]
            stop_price = stops.loc[entry_idx]

            # Determine if long or short
            is_long = data.loc[entry_idx, "three_up"]

            # Look for stop hit in future bars
            for j in range(i + 1, len(data)):
                future_idx = data.index[j]

                if is_long:
                    # For longs, exit if low price hits stop
                    if data.loc[future_idx, "low"] <= stop_price:
                        exits.loc[future_idx] = True
                        break
                else:
                    # For shorts, exit if high price hits stop
                    if data.loc[future_idx, "high"] >= stop_price:
                        exits.loc[future_idx] = True
                        break

        # Clean up signals (no entry after entry without exit)
        entries_array, exits_array = vbt.signals.nb.clean_enex_1d(
            entries.values, exits.values
        )

        return {
            "entries": pd.Series(entries_array, index=data.index),
            "exits": pd.Series(exits_array, index=data.index),
        }
