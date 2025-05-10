# Provider-agnostic façade for fetching data
import json  # For Parquet metadata serialization
import sys  # Keep sys for stderr outpu
from pathlib import Path

import pandas as pd
import pendulum
import pyarrow as pa
import pyarrow.parquet as pq

from algo_mvp.data.alpaca import AlpacaFetcher
from algo_mvp.data.tradovate import TradovateFetcher
from algo_mvp.models import AlpacaConfig, BaseConfig, TradovateConfig

# Attempt to get alpaca-trade-api version, fallback if not found
ALPACA_API_PACKAGE_VERSION = "N/A"
TRADOVATE_API_VERSION_NOTE = "Tradovate API v1 (via requests)"
try:
    import alpaca_trade_api

    ALPACA_API_PACKAGE_VERSION = alpaca_trade_api.__version__
except ImportError:
    pass

# Define a threshold for partitioning (e.g., 50MB)
PARTITION_SIZE_THRESHOLD_BYTES = 50 * 1024 * 1024


class DataFetcher:
    def __init__(self, config: BaseConfig, verbose: bool = False):
        self.config = config
        self.verbose = verbose
        self.provider_fetcher = None

        if self.verbose:
            print(f"DataFetcher initializing for provider: {self.config.provider}")

        if isinstance(self.config, AlpacaConfig):
            # In tests, this init will be mocked
            self.provider_fetcher = AlpacaFetcher(
                adjust=self.config.adjust, verbose=self.verbose
            )
            self.api_version = f"alpaca-trade-api {ALPACA_API_PACKAGE_VERSION}"
        elif isinstance(self.config, TradovateConfig):
            # In tests, this init will be mocked
            self.provider_fetcher = TradovateFetcher(verbose=self.verbose)
            self.api_version = TRADOVATE_API_VERSION_NOTE
        else:
            raise ValueError(
                f"Unsupported configuration provider: {self.config.provider}"
            )

    def _get_base_path(self) -> Path:
        # Sanitize timeframe for directory/file naming if needed (e.g., replace special chars)
        # For now, assume timeframe string is simple enough (e.g., "1Min", "1Day")
        # A more robust approach would be to slugify the timeframe string.
        timeframe_dirname = self.config.timeframe.replace(" ", "_").replace(":", "-")
        return (
            Path("data") / self.config.provider / self.config.symbol / timeframe_dirname
        )

    def _get_parquet_metadata(self, df: pd.DataFrame | None = None) -> dict:
        """Prepares metadata to be stored in the Parquet file."""
        meta = {
            "provider": self.config.provider,
            "symbol": self.config.symbol,
            "timeframe": self.config.timeframe,
            "downloaded_at_utc": pendulum.now("UTC").to_iso8601_string(),
            "api_version": self.api_version,
            "config_start_date": self.config.start,
            "config_end_date": self.config.end,
        }
        if isinstance(self.config, AlpacaConfig):
            meta["alpaca_adjust"] = str(
                self.config.adjust
            )  # Store bools as strings for pq metadata

        if df is not None and not df.empty:
            # Use .isoformat() for pandas Timestamp
            meta["actual_start_date"] = df.index.min().isoformat()
            meta["actual_end_date"] = df.index.max().isoformat()
            meta["num_rows"] = str(len(df))

        # Parquet metadata values must be strings
        return {k: str(v) for k, v in meta.items()}

    def _read_parquet_custom_metadata(self, file_path: Path) -> dict | None:
        try:
            if not file_path.exists():
                return None

            # Use read_metadata instead of ParquetFile.schema.metadata
            metadata = pq.read_metadata(file_path)
            if metadata.metadata:
                # Metadata is stored as bytes, need to decode and parse if it was json stringified
                meta_bytes = metadata.metadata.get(b"algo_mvp_meta")
                if meta_bytes:
                    return json.loads(meta_bytes.decode("utf-8"))
            return None  # No custom metadata found
        except Exception as e:
            if self.verbose:
                print(
                    f"Could not read Parquet metadata from {file_path}: {e}",
                    file=sys.stderr,
                )
            return None

    def _check_existing_data(
        self,
        target_file: Path,
        requested_start_dt: pendulum.DateTime,
        requested_end_dt: pendulum.DateTime,
    ) -> bool:
        """Checks if data for the full requested range already exists in the target file."""
        if not target_file.exists():
            if self.verbose:
                print(f"No existing data file found at {target_file}")
            return False

        metadata = self._read_parquet_custom_metadata(target_file)
        if not metadata:
            if self.verbose:
                print(
                    f"Existing file {target_file} found, but no valid metadata to check range. Will re-fetch."
                )
            return False  # Cannot verify range, so re-fetch

        try:
            # Use 'actual_start_date' or fall back to 'config_start_date' from metadata
            meta_start_str = metadata.get(
                "actual_start_date", metadata.get("config_start_date")
            )
            meta_end_str = metadata.get(
                "actual_end_date", metadata.get("config_end_date")
            )

            if not meta_start_str or not meta_end_str:
                if self.verbose:
                    print(
                        f"Metadata in {target_file} missing date range. Will re-fetch."
                    )
                return False

            # Parse dates from metadata. Pendulum handles ISO strings with/without TZ.
            # Explicitly convert to UTC for reliable comparison.
            # Use try-except for robust parsing
            try:
                existing_start_dt = pendulum.parse(meta_start_str).in_timezone("UTC")
                existing_end_dt = pendulum.parse(meta_end_str).in_timezone("UTC")
            except Exception as parse_err:
                if self.verbose:
                    print(
                        f"Error parsing date from metadata in {target_file}: {parse_err}. Will re-fetch."
                    )
                return False

            # Ensure requested dates are UTC. requested_end_dt should already be end_of('day').
            req_start_utc = requested_start_dt.in_timezone("UTC").start_of(
                "day"
            )  # Compare start of day
            req_end_utc = requested_end_dt.in_timezone("UTC")

            # Check if existing data range fully encapsulates the requested range.
            # Existing start date must be on or before requested start date.
            # Existing end date must be on or after requested end date.
            # For exact date comparison, we need to compare the date parts
            covers_start = existing_start_dt.date() <= req_start_utc.date()

            # For the end date, we need to handle microsecond differences
            # By comparing only date parts, we avoid issues with microsecond precision
            covers_end = existing_end_dt.date() >= req_end_utc.date()

            if covers_start and covers_end:
                if self.verbose:
                    print(
                        f"Existing data at {target_file} fully covers requested range ({req_start_utc.to_date_string()} to {req_end_utc.to_date_string()}). Skipping download."
                    )
                return True
            else:
                if self.verbose:
                    print(
                        f"Existing data at {target_file} does not fully cover requested range."
                    )
                    print(
                        f"  Requested: {req_start_utc.isoformat()} - {req_end_utc.isoformat()}"
                    )
                    print(
                        f"  Existing:  {existing_start_dt.isoformat()} - {existing_end_dt.isoformat()}"
                    )
                    print(f"  Covers Start: {covers_start}, Covers End: {covers_end}")
                return False
        except Exception as e:
            # Catch other unexpected errors during comparison
            if self.verbose:
                print(
                    f"Error comparing date ranges from metadata in {target_file}: {e}. Will re-fetch.",
                    file=sys.stderr,
                )
            return False

    def fetch_data(self, force: bool = False) -> bool:
        base_path_for_timeframe = self._get_base_path()
        single_target_file_path = base_path_for_timeframe.with_suffix(".parquet")

        requested_start_dt = pendulum.parse(self.config.start)
        # For end date, consider it to be end of day for range checks if only date is provided
        requested_end_dt = pendulum.parse(self.config.end)
        if requested_end_dt.time() == pendulum.Time(0, 0, 0):
            requested_end_dt = requested_end_dt.end_of("day")

        if not force:
            # Pass the specific file path to check for non-partitioned case
            if self._check_existing_data(
                single_target_file_path, requested_start_dt, requested_end_dt
            ):
                return True  # Data exists and covers range, success.

        if self.verbose:
            print(
                f"Proceeding to fetch data for {self.config.symbol} ({self.config.timeframe}) from {self.config.start} to {self.config.end}"
            )

        df = self.provider_fetcher.fetch(
            symbol=self.config.symbol,
            timeframe_str=self.config.timeframe,
            start_date_str=self.config.start,
            end_date_str=self.config.end,
        )

        if df is None:
            print(
                f"Failed to fetch data for {self.config.symbol} from provider.",
                file=sys.stderr,
            )
            return False

        if df.empty:
            if self.verbose:
                print(
                    f"No data returned by provider for {self.config.symbol} for the given range/timeframe. Output file will not be created."
                )
            # Consider if an empty parquet file should be written or not. For now, no.
            return True  # Successfully determined no data, not a failure of fetching process itself.

        # Prepare metadata for Parquet file
        pq_file_custom_meta = self._get_parquet_metadata(df)
        # pyarrow schema metadata must be Dict[str, str] but values are stored as bytes.
        # We need to serialize our dict (e.g. to JSON string) to store it under one key.
        algo_mvp_meta_json = json.dumps(pq_file_custom_meta)
        current_schema_meta = {b"algo_mvp_meta": algo_mvp_meta_json.encode("utf-8")}

        # Create a table with metadata
        table = pa.Table.from_pandas(df)
        # Add metadata to schema
        new_schema = table.schema.with_metadata(current_schema_meta)
        # Create a new table with the updated schema
        table = pa.Table.from_pandas(df, schema=new_schema)

        try:
            # Ensure output directory exists
            output_dir_single_file = single_target_file_path.parent
            output_dir_single_file.mkdir(parents=True, exist_ok=True)

            # Partitioning Logic (Simplified: by year if multiple years present and size > threshold)
            # This is a very rough estimate of potential file size from in-memory DF size
            estimated_size_bytes = df.memory_usage(deep=True).sum()
            unique_years = df.index.year.unique()

            if (
                len(unique_years) > 1
                and estimated_size_bytes > PARTITION_SIZE_THRESHOLD_BYTES
            ):
                if self.verbose:
                    print(
                        f"Data for {self.config.symbol} spans multiple years and is large ({estimated_size_bytes / (1024 * 1024):.2f}MB est.). Partitioning by year."
                    )
                # Change base_path_for_timeframe to be a directory for yearly files
                # Files will be like /data/alpaca/AAPL/1Day/2023.parque
                # Ensure the base directory for these yearly files exists
                base_path_for_timeframe.mkdir(parents=True, exist_ok=True)

                for year in unique_years:
                    year_df = df[df.index.year == year]
                    if year_df.empty:
                        continue
                    year_table = pa.Table.from_pandas(year_df)
                    year_meta_for_file = self._get_parquet_metadata(year_df)
                    year_meta_json = json.dumps(year_meta_for_file)
                    year_schema_meta = {
                        b"algo_mvp_meta": year_meta_json.encode("utf-8")
                    }
                    year_schema = year_table.schema.with_metadata(year_schema_meta)
                    year_table = pa.Table.from_pandas(year_df, schema=year_schema)

                    year_file_path = base_path_for_timeframe / f"{year}.parquet"
                    pq.write_table(year_table, year_file_path)
                    if self.verbose:
                        print(
                            f"Successfully wrote year-partitioned data to {year_file_path}"
                        )
            else:
                # Write as a single file
                pq.write_table(table, single_target_file_path)
                if self.verbose:
                    print(f"Successfully wrote data to {single_target_file_path}")

            return True

        except Exception as e:
            print(
                f"Error writing Parquet file for {self.config.symbol}: {e}",
                file=sys.stderr,
            )
            # import traceback; traceback.print_exc() # For debugging
            return False
