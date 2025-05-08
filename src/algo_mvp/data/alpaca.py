# Alpaca data fetching implementation
import os
import sys

import pandas as pd
import pendulum
from alpaca_trade_api.rest import REST, APIError, TimeFrame, TimeFrameUnit
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


# Define a more specific retry condition for typical Alpaca API errors
def is_retryable_alpaca_error(exception):
    """Determines if an Alpaca APIError is retryable (e.g., rate limit, server-side)."""
    if isinstance(exception, APIError):
        # Alpaca uses 429 for rate limiting. 5xx errors are server-side.
        return exception.status_code == 429 or exception.status_code >= 500
    return False


class AlpacaFetcher:
    def __init__(
        self,
        api_key: str | None = None,
        secret_key: str | None = None,
        adjust: bool = True,
        verbose: bool = False,
    ):
        self.api_key = api_key or os.getenv("ALPACA_KEY_ID")
        self.secret_key = secret_key or os.getenv("ALPACA_SECRET_KEY")
        self.adjust = adjust
        self.verbose = verbose

        if not self.api_key or not self.secret_key:
            raise ValueError(
                "Alpaca API key and secret key must be provided or set as environment variables ALPACA_KEY_ID and ALPACA_SECRET_KEY."
            )

        # Using data_v2 for historical bars, paper=True is irrelevant for data.
        self.client = REST(
            self.api_key, self.secret_key, data_feed="sip"
        )  # SIP for free data
        if self.verbose:
            print("AlpacaFetcher initialized. Data feed: SIP (free)")

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=60),  # Exponential backoff
        retry=retry_if_exception_type(
            (APIError, ConnectionError)
        ),  # Retry on APIError and connection issues
        # Consider adding retry_if_exception_type(is_retryable_alpaca_error) for more specific retries
    )
    def fetch(
        self, symbol: str, timeframe_str: str, start_date_str: str, end_date_str: str
    ) -> pd.DataFrame | None:
        """Fetches historical bars for a symbol between start and end dates."""

        start_dt = pendulum.parse(start_date_str).replace(
            tzinfo=None
        )  # Alpaca API expects naive datetime for start/end
        end_dt = pendulum.parse(end_date_str).replace(tzinfo=None)

        # Parse timeframe_str (e.g., "1Min", "1Day", "1H") into Alpaca TimeFrame object
        try:
            # Simple parsing, assumes format like "1Min", "5Day", "1Hour"
            value = (
                int(timeframe_str[:-3])
                if timeframe_str[-3:].lower() in ["min", "day"]
                else int(timeframe_str[:-1])
            )
            unit_str = (
                timeframe_str[-3:].lower()
                if timeframe_str[-3:].lower() in ["min", "day"]
                else timeframe_str[-1:].lower()
            )

            if unit_str == "min":
                tf_unit = TimeFrameUnit.Minute
            elif unit_str == "day":
                tf_unit = TimeFrameUnit.Day
            elif unit_str == "h":
                tf_unit = TimeFrameUnit.Hour
            else:
                raise ValueError(
                    f"Unsupported timeframe unit in '{timeframe_str}'. Supported: Min, Day, H"
                )

            timeframe = TimeFrame(value, tf_unit)
        except ValueError as e:
            print(f"Error parsing timeframe '{timeframe_str}': {e}", file=sys.stderr)
            return None

        if self.verbose:
            print(
                f"Alpaca: Fetching {symbol} ({timeframe_str} -> {timeframe}) from {start_dt.to_iso8601_string()} to {end_dt.to_iso8601_string()}, Adjust: {self.adjust}"
            )

        try:
            # Alpaca API /v2/stocks/{symbol}/bars
            # The end date is inclusive in the request by default with how get_bars works.
            # AlpacaPy automatically handles pagination for longer date ranges.
            request_params = {
                "symbol_or_symbols": symbol,
                "timeframe": timeframe,
                "start": start_dt.isoformat(),  # Ensure ISO format, though AlpacaPy might handle others
                "end": end_dt.isoformat(),
                "adjustment": (
                    "split" if self.adjust else "raw"
                ),  # 'all' for dividends and splits, 'split' for splits only
            }
            if self.verbose:
                print(f"Alpaca API request params: {request_params}")

            bars_df = self.client.get_bars(**request_params).df

            if bars_df.empty:
                if self.verbose:
                    print(
                        f"Alpaca: No data returned for {symbol} in the given range and timeframe."
                    )
                return (
                    pd.DataFrame()
                )  # Return empty DataFrame, not None, for consistency

            # Ensure timezone is UTC as Alpaca returns UTC timestamps
            if bars_df.index.tz is None:
                bars_df.index = bars_df.index.tz_localize("UTC")
            else:
                bars_df.index = bars_df.index.tz_convert("UTC")

            # Filter by exact start and end, as API might return wider range due to market hours/granularity
            # Convert start_dt and end_dt to UTC for comparison with DataFrame index
            start_dt_utc = pendulum.parse(start_date_str).in_timezone("UTC")
            end_dt_utc = (
                pendulum.parse(end_date_str).in_timezone("UTC").end_of("day")
            )  # Ensure end of day for range

            bars_df = bars_df[
                (bars_df.index >= start_dt_utc) & (bars_df.index <= end_dt_utc)
            ]

            if self.verbose:
                print(f"Alpaca: Fetched {len(bars_df)} bars for {symbol}.")
            return bars_df

        except APIError as e:
            # Log specific API errors that tenacity might not catch or if it exhausts retries
            print(f"Alpaca API Error for {symbol}: {e}", file=sys.stderr)
            print(
                f"Response details: {e._response.text if e._response else 'No response details'}",
                file=sys.stderr,
            )
            return None
        except ConnectionError as e:
            print(f"Alpaca Connection Error for {symbol}: {e}", file=sys.stderr)
            return None
        except Exception as e:
            # Catch any other unexpected errors during fetch
            print(
                f"An unexpected error occurred while fetching Alpaca data for {symbol}: {e}",
                file=sys.stderr,
            )
            return None


if __name__ == "__main__":
    # Example Usage (requires ALPACA_KEY_ID and ALPACA_SECRET_KEY environment variables)
    print("Testing AlpacaFetcher...")
    # Load environment variables if .env file is present and python-dotenv is installed
    try:
        from dotenv import load_dotenv

        if load_dotenv():
            print("Loaded .env file")
    except ImportError:
        print(".env file not loaded (python-dotenv not installed or .env not found)")
        pass  # Optional dependency

    api_key_id = os.getenv("ALPACA_KEY_ID")
    api_secret_key = os.getenv("ALPACA_SECRET_KEY")

    if not api_key_id or not api_secret_key:
        print(
            "ALPACA_KEY_ID and ALPACA_SECRET_KEY must be set in environment for testing."
        )
        sys.exit(1)

    fetcher = AlpacaFetcher(verbose=True)

    # Test 1: Daily data for AAPL
    print("\n--- Test Case 1: Daily AAPL data ---")
    aapl_daily = fetcher.fetch("AAPL", "1Day", "2023-01-01", "2023-01-10")
    if aapl_daily is not None:
        print(f"Fetched AAPL daily data (first 5 rows):\n{aapl_daily.head()}")
        print(f"Shape: {aapl_daily.shape}")
    else:
        print("Failed to fetch AAPL daily data.")

    # Test 2: Minute data for SPY
    print("\n--- Test Case 2: Minute SPY data ---")
    # Using a very short period for minute data to limit output and API usage
    spy_minute = fetcher.fetch(
        "SPY", "1Min", "2024-01-01T09:30:00", "2024-01-01T09:35:00"
    )
    if spy_minute is not None:
        print(f"Fetched SPY minute data (first 5 rows):\n{spy_minute.head()}")
        print(f"Shape: {spy_minute.shape}")
    else:
        print("Failed to fetch SPY minute data.")

    # Test 3: Non-existent symbol or bad parameters (should be handled gracefully or by APIError)
    print("\n--- Test Case 3: Non-existent symbol ---")
    non_existent = fetcher.fetch(
        "NONEXISTENTSYMBOL123", "1Day", "2023-01-01", "2023-01-05"
    )
    if non_existent is not None and non_existent.empty:
        print("Correctly returned empty DataFrame for non-existent symbol or no data.")
    elif non_existent is None:
        print("API call failed as expected for non-existent symbol (returned None).")
    else:
        print(f"Unexpectedly got data for non-existent symbol: {non_existent.head()}")

    # Test 4: Invalid timeframe
    print("\n--- Test Case 4: Invalid timeframe ---")
    invalid_tf = fetcher.fetch("AAPL", "1Year", "2023-01-01", "2023-01-05")
    if invalid_tf is None:
        print("Correctly handled invalid timeframe (returned None).")
    else:
        print(f"Unexpectedly processed invalid timeframe: {invalid_tf.head()}")

    # Example of what happens with rate limiting (difficult to test reliably without hammering)
    # print("\n--- Intentionally trying to hit rate limit (may take a while or not trigger) ---")
    # for i in range(250): # Alpaca free plan has 200 requests/minute limit
    #     print(f"Request {i+1}")
    #     fetcher.fetch("MSFT", "1Min", "2024-01-01T09:30:00", "2024-01-01T09:31:00")
    #     if i % 10 == 0:
    #         time.sleep(0.1) # Small pause, but not enough to avoid limit usually
    # print("Finished rate limit test attempts.")
