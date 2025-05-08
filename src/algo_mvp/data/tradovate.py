# Tradovate data fetching implementation
import os
import sys
from datetime import timezone

import pandas as pd
import pendulum
import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

TRADOVATE_DEMO_API_URL = "https://demo.tradovateapi.com/v1"


class TradovateFetcher:
    def __init__(
        self,
        client_id: str | None = None,
        access_token: str | None = None,
        verbose: bool = False,
    ):
        # For demo, client_id might be a user-specific ID or app ID.
        # Access token is key for API calls.
        self.client_id = client_id or os.getenv("TRADOVATE_CLIENT_ID")
        self.access_token = access_token or os.getenv("TRADOVATE_ACCESS_TOKEN")
        self.verbose = verbose

        if not self.client_id:
            # Depending on Tradovate's demo auth, client_id might not always be strictly necessary if a valid access_token is supplied directly.
            # However, it's good practice to require it if it's part of their expected auth flow or for logging.
            print(
                "Warning: TRADOVATE_CLIENT_ID is not set. Proceeding with access token only."
            )

        if not self.access_token:
            raise ValueError(
                "Tradovate access token must be provided or set as TRADOVATE_ACCESS_TOKEN environment variable."
            )

        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.access_token}",
                "Accept": "application/json",
            }
        )

        if self.verbose:
            print(
                f"TradovateFetcher initialized for client ID (if provided): {self.client_id}"
            )

    def _is_retryable_tradovate_error(self, exception):
        """Determines if a Tradovate requests.HTTPError is retryable."""
        if isinstance(exception, requests.exceptions.HTTPError):
            # Retry on 429 (Too Many Requests) or 5xx server errors
            return (
                exception.response.status_code == 429
                or exception.response.status_code >= 500
            )
        return False

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        retry=retry_if_exception_type(
            (requests.exceptions.HTTPError, requests.exceptions.ConnectionError)
        ),
        # Could also use: retry=lambda retry_state: self._is_retryable_tradovate_error(retry_state.outcome.exception())
    )
    def _make_request(self, endpoint: str, params: dict | None = None):
        url = f"{TRADOVATE_DEMO_API_URL}{endpoint}"
        if self.verbose:
            print(f"Tradovate API request: GET {url} with params: {params}")
        response = self.session.get(url, params=params)
        response.raise_for_status()  # Raises HTTPError for bad responses (4XX or 5XX)
        return response.json()

    def _resample_ticks_to_ohlcv(
        self, ticks_df_indexed: pd.DataFrame, timeframe_str: str
    ) -> pd.DataFrame:
        """Resamples tick data (with timestamp index) to OHLCV DataFrame."""
        # Input DataFrame is expected to have a datetime index named 'timestamp'
        if ticks_df_indexed.empty:
            return pd.DataFrame()

        if not isinstance(ticks_df_indexed.index, pd.DatetimeIndex):
            raise ValueError("Input DataFrame must have a DatetimeIndex.")
        if ticks_df_indexed.index.name != "timestamp":
            print(
                f"Warning: Input DataFrame index name is '{ticks_df_indexed.index.name}', expected 'timestamp'."
            )
            # Optionally rename: ticks_df_indexed.index.name = "timestamp"

        # Ensure required columns exist
        if (
            "price" not in ticks_df_indexed.columns
            or "size" not in ticks_df_indexed.columns
        ):
            raise ValueError("Input DataFrame must contain 'price' and 'size' columns.")

        # Determine resampling rule
        rule = timeframe_str
        if "Min" in timeframe_str:
            rule = timeframe_str.replace("Min", "T")
        elif "H" in timeframe_str and "Min" not in timeframe_str:
            rule = timeframe_str

        if self.verbose:
            print(f"Resampling ticks with rule: {rule}")

        # Resample using the DataFrame with the datetime index
        # Use closed='left' and label='left' for standard OHLC behavior
        ohlc = (
            ticks_df_indexed["price"].resample(rule, closed="left", label="left").ohlc()
        )
        volume = (
            ticks_df_indexed["size"].resample(rule, closed="left", label="left").sum()
        )

        # Combine OHLC with Volume
        df_ohlcv = pd.concat([ohlc, volume], axis=1)
        df_ohlcv.rename(columns={"size": "volume"}, inplace=True)
        df_ohlcv = df_ohlcv[["open", "high", "low", "close", "volume"]].copy()
        # Drop rows where resampling produced no data (all OHLC are NaN)
        df_ohlcv.dropna(subset=["open"], inplace=True)

        # Ensure timezone is UTC
        if df_ohlcv.index.tz is None:
            # If the original index was naive, assume UTC based on Tradovate standard
            df_ohlcv.index = df_ohlcv.index.tz_localize("UTC")
        elif df_ohlcv.index.tz != timezone.utc:
            df_ohlcv.index = df_ohlcv.index.tz_convert("UTC")

        df_ohlcv.index.name = "timestamp"  # Ensure index name is set

        return df_ohlcv

    def fetch(
        self, symbol: str, timeframe_str: str, start_date_str: str, end_date_str: str
    ) -> pd.DataFrame | None:
        """Fetches historical tick data for a symbol and resamples to OHLCV."""
        # Tradovate's /history endpoint typically takes `from` and `to` timestamps.
        # It might also have limits on how far back or how much data can be fetched at once.
        # The task mentions "Tradovate ticks->1m resample", so we primarily fetch ticks.

        # Convert start/end to ISO 8601 UTC, which Tradovate expects for /history
        # The API expects timestamps like "2024-01-01T00:00:00.000Z"
        try:
            start_iso = pendulum.parse(start_date_str).in_timezone("UTC").isoformat()
            # For end_date, ensure it covers the whole day if only date is given.
            # The `/history` endpoint seems to expect specific time ranges.
            # If end_date_str is just a date, make it end of that day.
            end_dt_pendulum = pendulum.parse(end_date_str)
            if end_dt_pendulum.time() == pendulum.Time(0, 0, 0):
                end_iso = end_dt_pendulum.end_of("day").in_timezone("UTC").isoformat()
            else:
                end_iso = end_dt_pendulum.in_timezone("UTC").isoformat()

        except Exception as e:
            print(f"Error parsing dates for Tradovate: {e}", file=sys.stderr)
            return None

        if self.verbose:
            print(
                f"Tradovate: Fetching ticks for {symbol} from {start_iso} to {end_iso} for {timeframe_str} resampling."
            )

        # Placeholder for fetching actual tick data
        # The /history/{symbol} endpoint might need specific parameters like `from`, `to`, `type` (e.g. 'tick')
        # This is a simplified representation. Actual API might require pagination or chunking.
        ticks_data_list = []
        try:
            # Example: /history?symbol=MESM5&from=2024-01-01T00:00:00.000Z&to=2024-01-01T01:00:00.000Z&type=Tick
            # This is a conceptual representation. The actual API may have different parameters and pagination.
            # We might need to loop or use a specific Tradovate library if one exists for full history.
            # For now, assume a single call can get the required ticks, or this part needs expansion.
            params = {
                "symbol": symbol,
                "from": start_iso,
                "to": end_iso,
                "type": "Tick",  # Assuming 'Tick' is the type for tick history
            }
            response_json = self._make_request(endpoint="/history", params=params)

            # The structure of response_json needs to be known. Assume it's a list of tick objects.
            # Each tick object might have { timestamp, price, size }
            # Example: { "ticks": [ { "t": "2024-03-01T10:00:00.123Z", "p": 4000.25, "s": 2 }, ... ] }
            # This structure is hypothetical.

            raw_ticks = response_json.get(
                "ticks", []
            )  # Adjust based on actual API response structure
            if not raw_ticks:
                if self.verbose:
                    print(
                        f"Tradovate: No tick data returned for {symbol} in the given range."
                    )
                return pd.DataFrame()  # Return empty DataFrame

            for tick in raw_ticks:
                # Adapt keys based on actual Tradovate tick structure
                # Assuming 't' for timestamp, 'p' for price, 's' for size.
                ticks_data_list.append(
                    {
                        "timestamp": tick.get("t"),
                        "price": tick.get("p"),
                        "size": tick.get("s"),
                    }
                )

            if not ticks_data_list:
                if self.verbose:
                    print(f"Tradovate: No processable ticks found for {symbol}.")
                return pd.DataFrame()

            ticks_df = pd.DataFrame(ticks_data_list)
            ticks_df.dropna(subset=["timestamp", "price", "size"], inplace=True)
            if ticks_df.empty:
                if self.verbose:
                    print(
                        f"Tradovate: Tick data was empty after NaNs dropped for {symbol}."
                    )
                return pd.DataFrame()

            # Convert timestamp and set as index BEFORE resampling
            ticks_df["timestamp"] = pd.to_datetime(ticks_df["timestamp"])
            ticks_df.set_index("timestamp", inplace=True)

            # Resample to OHLCV (passing the indexed DataFrame)
            ohlcv_df = self._resample_ticks_to_ohlcv(ticks_df, timeframe_str)

            if self.verbose:
                print(
                    f"Tradovate: Fetched and resampled {len(ohlcv_df)} bars for {symbol}."
                )
            return ohlcv_df

        except requests.exceptions.HTTPError as e:
            print(
                f"Tradovate API HTTP Error for {symbol}: {e}. Response: {e.response.text if e.response else 'N/A'}",
                file=sys.stderr,
            )
            return None
        except requests.exceptions.ConnectionError as e:
            print(f"Tradovate Connection Error for {symbol}: {e}", file=sys.stderr)
            return None
        except Exception as e:
            print(
                f"An unexpected error occurred while fetching Tradovate data for {symbol}: {e}",
                file=sys.stderr,
            )
            # import traceback; traceback.print_exc()  # For debugging unexpected errors
            return None


# Example Usage (requires TRADOVATE_CLIENT_ID and TRADOVATE_ACCESS_TOKEN environment variables)
if __name__ == "__main__":
    print("Testing TradovateFetcher...")
    try:
        from dotenv import load_dotenv

        if load_dotenv():
            print("Loaded .env file")
    except ImportError:
        print(".env file not loaded (python-dotenv not installed or .env not found)")
        pass  # Optional

    client_id = os.getenv("TRADOVATE_CLIENT_ID")
    access_token = os.getenv("TRADOVATE_ACCESS_TOKEN")

    if (
        not access_token
    ):  # Client ID might be optional for some demo setups if token is direct
        print("TRADOVATE_ACCESS_TOKEN must be set in environment for testing.")
        # print("If using a full auth flow, TRADOVATE_USER, TRADOVATE_PASS, TRADOVATE_APP_ID, TRADOVATE_CID might be needed.")
        sys.exit(1)

    # Replace with a known accessible demo symbol for Tradovate
    # Micro E-mini S&P 500, MESM4 (June 2024 contract as of early 2024)
    # For May 2025, it would be MESM5 or MESK5 (check contract codes)
    # The prompt mentions MESM25 - this seems like a year-based symbol rather than month code.
    # Tradovate symbols usually have month codes like H, M, U, Z.
    # Let's use a common continuous contract symbol like /MES for testing if direct contract is tricky.
    # Or a specific, currently active contract symbol.
    # For now, we'll use the example from the prompt, assuming it's valid in the demo environment.
    test_symbol = "MESM25"  # From prompt
    # test_symbol = "MESU24" # Example: Sep 2024 contract

    fetcher = TradovateFetcher(
        client_id=client_id, access_token=access_token, verbose=True
    )

    print(f"\n--- Test Case 1: 1-minute {test_symbol} data ---")
    # Fetch a very small window for testing to avoid large tick data downloads
    # Tradovate demo might have limited history for ticks (e.g., few days)
    # And specific symbols might only be active for certain periods.
    # Using a recent date for more likely data availability.
    start_fetch_time = (
        pendulum.now("UTC").subtract(days=2).to_datetime_string()
    )  # e.g. "2024-05-06 10:00:00"
    end_fetch_time = (
        pendulum.parse(start_fetch_time).add(minutes=5).to_datetime_string()
    )

    # If using specific dates from prompt, ensure they are valid for the demo environment
    # start_fetch_time = "2024-03-01T10:00:00Z"
    # end_fetch_time = "2024-03-01T10:05:00Z"

    trad_data = fetcher.fetch(test_symbol, "1Min", start_fetch_time, end_fetch_time)

    if trad_data is not None and not trad_data.empty:
        print(
            f"Fetched {test_symbol} 1-minute data (first 5 rows):\n{trad_data.head()}"
        )
        print(f"Shape: {trad_data.shape}")
    elif trad_data is not None and trad_data.empty:
        print(
            f"Fetched {test_symbol} data, but it was empty (no trades in period or after resampling)."
        )
    else:
        print(f"Failed to fetch {test_symbol} data.")

    print("\n--- Test Case 2: Invalid symbol ---")
    invalid_data = fetcher.fetch(
        "INVALIDSYMBOLXYZ123", "1Min", start_fetch_time, end_fetch_time
    )
    if invalid_data is None:
        print(
            "Correctly failed to fetch data for invalid symbol (returned None due to API error)."
        )
    elif invalid_data.empty:
        print(
            "Correctly returned empty dataframe for invalid symbol (no data or caught by API)."
        )
    else:
        print("Unexpectedly got data for invalid symbol.")

    # Note: Full testing of Tradovate would require knowing the exact demo API behavior,
    # available symbols, historical data depth, and rate limits.
    # The /history endpoint documentation is crucial here.
