# Placeholder for data fetcher tests
import json
from pathlib import Path

# from unittest.mock import MagicMock, call, patch # Remove patch if only mocker is used for patching
from unittest.mock import MagicMock, call  # Keep MagicMock, call if used

import pandas as pd
import pendulum
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import algo_mvp.data.alpaca  # noqa: F401

# Import the actual modules to ensure coverage while avoiding redefinition errors
import algo_mvp.data.fetcher  # noqa: F401
import algo_mvp.data.tradovate  # noqa: F401

# Import the actual classes to mock their __init__
from algo_mvp.data import alpaca as alpaca_module  # noqa: F401
from algo_mvp.data import tradovate as tradovate_module  # noqa: F401
from algo_mvp.data.fetcher import PARTITION_SIZE_THRESHOLD_BYTES, DataFetcher
from algo_mvp.models import AlpacaConfig, TradovateConfig

# pytestmark = pytest.mark.skip(
#     reason="Temporarily skipping all tests in test_fetchers.py to focus on live trading module and avoid alpaca_trade_api dependency issue."
# )


@pytest.fixture
def alpaca_config_fixture():
    return AlpacaConfig(
        provider="alpaca",
        symbol="AAPL",
        timeframe="1Day",
        start="2023-01-01",
        end="2023-01-10",
        adjust=True,
    )


@pytest.fixture
def tradovate_config_fixture():
    return TradovateConfig(
        provider="tradovate",
        symbol="MESM25",
        timeframe="1Min",
        start="2024-03-01T10:00:00Z",
        end="2024-03-01T10:05:00Z",
    )


@pytest.fixture
def sample_dataframe():
    dates = pd.to_datetime(
        [
            "2023-01-01 00:00:00",
            "2023-01-01 00:01:00",
            "2024-05-10 10:00:00",
            "2024-05-10 10:01:00",
        ]
    ).tz_localize("UTC")
    data = {
        "open": [100, 101, 150, 151],
        "high": [102, 101.5, 152, 151.5],
        "low": [99, 100.5, 149, 150.5],
        "close": [101, 101.2, 151, 151.2],
        "volume": [1000, 1200, 2000, 2200],
    }
    return pd.DataFrame(data, index=pd.Index(dates, name="timestamp"))


@pytest.fixture
def large_multiyear_dataframe():
    # Create a dataframe that spans multiple years and would be > threshold
    # For simplicity, let's assume 10000 rows of 5 columns (float64) + index (datetime64)
    # float64 is 8 bytes. datetime64 is 8 bytes.
    # 10000 rows * (5*8 + 8) bytes = 10000 * 48 bytes = 480,000 bytes ~ 0.48MB
    # To exceed 50MB (PARTITION_SIZE_THRESHOLD_BYTES), we'd need ~100x more rows.
    # For test speed, we'll rely on mocking estimated_size_bytes instead of huge DF.
    # But we do need data spanning multiple years for the partitioning logic to trigger.
    dates_2022 = pd.date_range(start="2022-01-01", end="2022-01-05", freq="D", tz="UTC")
    dates_2023 = pd.date_range(start="2023-01-01", end="2023-01-05", freq="D", tz="UTC")
    all_dates = dates_2022.union(dates_2023)
    data_val = range(len(all_dates))
    data = {
        "open": data_val,
        "high": data_val,
        "low": data_val,
        "close": data_val,
        "volume": data_val,
    }
    return pd.DataFrame(data, index=pd.Index(all_dates, name="timestamp"))


# --- Mocks for Provider Fetcher Initialization ---
@pytest.fixture
def mock_alpaca_provider_instance(mocker):
    """Provides a mock AlpacaFetcher instance and patches the class where DataFetcher uses it."""
    mock_instance = MagicMock(
        spec=alpaca_module.AlpacaFetcher
    )  # Still spec with original for type hint
    mock_instance.fetch = MagicMock()

    # Patch AlpacaFetcher in the module where DataFetcher looks it up
    patched_class = mocker.patch("algo_mvp.data.fetcher.AlpacaFetcher", autospec=True)
    patched_class.return_value = mock_instance
    return mock_instance


@pytest.fixture
def mock_tradovate_provider_instance(mocker):
    """Provides a mock TradovateFetcher instance and patches the class where DataFetcher uses it."""
    mock_instance = MagicMock(spec=tradovate_module.TradovateFetcher)
    mock_instance.fetch = MagicMock()

    patched_class = mocker.patch(
        "algo_mvp.data.fetcher.TradovateFetcher", autospec=True
    )
    patched_class.return_value = mock_instance
    return mock_instance


# --- Tests for DataFetcher --- #


def test_data_fetcher_initialization(
    mocker, alpaca_config_fixture, tradovate_config_fixture
):
    """Test DataFetcher initialization selects correct (mocked) provider."""
    # We need to mock the provider fetcher inits *before* DataFetcher is instantiated
    mock_alpaca_init = mocker.patch.object(
        alpaca_module.AlpacaFetcher, "__init__", return_value=None
    )
    mock_tradovate_init = mocker.patch.object(
        tradovate_module.TradovateFetcher, "__init__", return_value=None
    )

    df_alpaca = DataFetcher(config=alpaca_config_fixture, verbose=True)
    mock_alpaca_init.assert_called_once()
    assert df_alpaca.api_version.startswith("alpaca-trade-api")
    assert isinstance(df_alpaca.provider_fetcher, alpaca_module.AlpacaFetcher)

    # Reset mock for the next instantiation if needed, or use separate tests
    mock_alpaca_init.reset_mock()

    df_tradovate = DataFetcher(config=tradovate_config_fixture, verbose=True)
    mock_tradovate_init.assert_called_once()
    assert df_tradovate.api_version == "Tradovate API v1 (via requests)"
    assert isinstance(df_tradovate.provider_fetcher, tradovate_module.TradovateFetcher)


# Use the init mocks for tests that instantiate DataFetcher
def test_get_base_path(
    mock_alpaca_provider_instance,
    mock_tradovate_provider_instance,
    alpaca_config_fixture,
    tradovate_config_fixture,
):
    """Test _get_base_path method."""
    fetcher_alpaca = DataFetcher(alpaca_config_fixture)
    expected_path_alpaca = Path("data/alpaca/AAPL/1Day")
    assert fetcher_alpaca._get_base_path() == expected_path_alpaca

    alpaca_config_fixture.timeframe = "1 Hour"
    fetcher_alpaca_hourly = DataFetcher(alpaca_config_fixture)
    expected_path_alpaca_hourly = Path("data/alpaca/AAPL/1_Hour")
    assert fetcher_alpaca_hourly._get_base_path() == expected_path_alpaca_hourly

    fetcher_tradovate = DataFetcher(tradovate_config_fixture)
    expected_path_tradovate = Path("data/tradovate/MESM25/1Min")
    assert fetcher_tradovate._get_base_path() == expected_path_tradovate

    # Test with more complex timeframe needing sanitization
    alpaca_config_fixture.timeframe = "5 Min:Test"
    fetcher_alpaca_complex_tf = DataFetcher(alpaca_config_fixture)
    expected_path_alpaca_complex = Path("data/alpaca/AAPL/5_Min-Test")
    assert fetcher_alpaca_complex_tf._get_base_path() == expected_path_alpaca_complex

    # Test with only colon needing sanitization
    alpaca_config_fixture.timeframe = "1Day:Official"
    fetcher_alpaca_colon_tf = DataFetcher(alpaca_config_fixture)
    expected_path_alpaca_colon = Path("data/alpaca/AAPL/1Day-Official")
    assert fetcher_alpaca_colon_tf._get_base_path() == expected_path_alpaca_colon


def test_get_parquet_metadata(
    mock_alpaca_provider_instance, alpaca_config_fixture, sample_dataframe
):
    fetcher = DataFetcher(alpaca_config_fixture)
    metadata = fetcher._get_parquet_metadata(sample_dataframe)
    assert metadata["provider"] == "alpaca"
    assert (
        metadata["actual_start_date"] == sample_dataframe.index.min().isoformat()
    )  # Check using .isoformat()
    assert metadata["actual_end_date"] == sample_dataframe.index.max().isoformat()
    assert pendulum.parse(metadata["downloaded_at_utc"])  # Checks if parseable


def test_get_parquet_metadata_no_df(
    mock_alpaca_provider_instance, alpaca_config_fixture
):
    fetcher = DataFetcher(alpaca_config_fixture)
    metadata = fetcher._get_parquet_metadata(None)
    assert "actual_start_date" not in metadata


def test_read_parquet_custom_metadata(
    mock_alpaca_provider_instance, tmp_path, alpaca_config_fixture, sample_dataframe
):
    fetcher = DataFetcher(alpaca_config_fixture)
    test_file = tmp_path / "test.parquet"
    assert (
        fetcher._read_parquet_custom_metadata(test_file) is None
    )  # File doesn't exist

    # Write table without specific metadata key
    table_no_meta = pa.Table.from_pandas(sample_dataframe)
    pq.write_table(table_no_meta, test_file)
    assert fetcher._read_parquet_custom_metadata(test_file) is None  # No algo_mvp_meta

    # Write table with correct metadata key
    expected_meta_dict = {"key": "value", "number_str": "123"}
    meta_json = json.dumps(expected_meta_dict)
    schema_meta = {b"algo_mvp_meta": meta_json.encode("utf-8")}
    table_with_meta = pa.Table.from_pandas(sample_dataframe)
    # Correct way: Add metadata to schema, then write table with that schema
    updated_schema = table_with_meta.schema.with_metadata(schema_meta)
    pq.write_table(
        table_with_meta.cast(updated_schema), test_file
    )  # Write using updated schema
    read_meta = fetcher._read_parquet_custom_metadata(test_file)
    assert read_meta == expected_meta_dict

    # Write table with malformed metadata
    bad_schema_meta = {b"algo_mvp_meta": b"not a json string"}
    table_bad_meta = pa.Table.from_pandas(sample_dataframe)  # Start fresh
    updated_schema_bad = table_bad_meta.schema.with_metadata(bad_schema_meta)
    pq.write_table(table_bad_meta.cast(updated_schema_bad), test_file)
    assert fetcher._read_parquet_custom_metadata(test_file) is None  # Malformed JSON


@pytest.mark.parametrize(
    "file_exists_val, metadata_content, expected_result, verbose_val",
    [
        (False, None, False, False),
        (True, None, False, True),
        (
            True,
            {
                "actual_start_date": "2023-01-01T00:00:00+00:00",
                "actual_end_date": "2023-01-10T23:59:59+00:00",
            },
            True,
            False,
        ),
        (
            True,
            {"config_start_date": "2023-01-01", "config_end_date": "2023-01-10"},
            True,
            False,
        ),
        (
            True,
            {
                "actual_start_date": "2023-01-02T00:00:00+00:00",
                "actual_end_date": "2023-01-09T23:59:59+00:00",
            },
            False,
            False,
        ),
        (
            True,
            {
                "actual_start_date": "2023-01-01T00:00:00+00:00",
                "actual_end_date": "2023-01-08T23:59:59+00:00",
            },
            False,
            False,
        ),
        (
            True,
            {
                "actual_start_date": "2022-12-31T00:00:00+00:00",
                "actual_end_date": "2023-01-11T23:59:59+00:00",
            },
            True,
            False,
        ),
        (True, {"some_other_meta": "value"}, False, False),
        (
            True,
            {
                "actual_start_date": "invalid-date",
                "actual_end_date": "2023-01-10T23:59:59+00:00",
            },
            False,
            True,
        ),
    ],
)
def test_check_existing_data(
    mocker,
    tmp_path,
    mock_alpaca_provider_instance,
    alpaca_config_fixture,
    file_exists_val,
    metadata_content,
    expected_result,
    verbose_val,
):
    fetcher = DataFetcher(alpaca_config_fixture, verbose=verbose_val)
    req_start_dt = pendulum.parse(alpaca_config_fixture.start)  # Already date start
    req_end_dt = pendulum.parse(alpaca_config_fixture.end).end_of(
        "day"
    )  # Ensure end of day

    mock_target_file_arg = MagicMock(spec=Path)
    mock_target_file_arg.exists.return_value = file_exists_val
    # Mock the internal call _read_parquet_custom_metadata
    mocker.patch.object(
        fetcher, "_read_parquet_custom_metadata", return_value=metadata_content
    )

    assert (
        fetcher._check_existing_data(mock_target_file_arg, req_start_dt, req_end_dt)
        == expected_result
    )


def test_fetch_data_skip_if_exists_no_force(
    mocker, mock_alpaca_provider_instance, alpaca_config_fixture
):
    fetcher = DataFetcher(config=alpaca_config_fixture, verbose=True)
    mocker.patch.object(
        fetcher, "_check_existing_data", return_value=True
    )  # Mock to simulate data exists
    # mock_alpaca_provider_instance fixture already mocked provider_fetcher.fetch

    assert fetcher.fetch_data(force=False) is True
    mock_alpaca_provider_instance.fetch.assert_not_called()  # Crucial: provider fetch should not be called


def test_fetch_data_fetch_if_not_exists(
    mocker,
    mock_alpaca_provider_instance,
    alpaca_config_fixture,
    sample_dataframe,
    tmp_path,
):
    fetcher = DataFetcher(config=alpaca_config_fixture, verbose=True)
    # Mock the fetch method directly on provider_fetcher instead of using mock_alpaca_provider_instance.fetch
    mocker.patch.object(
        fetcher.provider_fetcher, "fetch", return_value=sample_dataframe
    )

    mocker.patch.object(fetcher, "_check_existing_data", return_value=False)
    mock_write_table = mocker.patch("pyarrow.parquet.write_table")
    mocker.patch.object(Path, "mkdir")

    base_test_path = tmp_path / "test_data_output"
    mocker.patch.object(fetcher, "_get_base_path", return_value=base_test_path)

    assert fetcher.fetch_data(force=False) is True
    fetcher.provider_fetcher.fetch.assert_called_once()
    mock_write_table.assert_called_once()


def test_fetch_data_force_fetch(
    mocker,
    mock_alpaca_provider_instance,
    alpaca_config_fixture,
    sample_dataframe,
    tmp_path,
):
    fetcher = DataFetcher(config=alpaca_config_fixture, verbose=True)
    # Mock the fetch method directly on provider_fetcher
    mocker.patch.object(
        fetcher.provider_fetcher, "fetch", return_value=sample_dataframe
    )

    mocker.patch.object(fetcher, "_check_existing_data", return_value=True)
    mock_write_table = mocker.patch("pyarrow.parquet.write_table")
    mocker.patch.object(Path, "mkdir")
    base_test_path = tmp_path / "test_data_output_force"
    mocker.patch.object(fetcher, "_get_base_path", return_value=base_test_path)

    assert fetcher.fetch_data(force=True) is True
    fetcher.provider_fetcher.fetch.assert_called_once()
    mock_write_table.assert_called_once()


def test_fetch_data_provider_fails(
    mocker, mock_alpaca_provider_instance, alpaca_config_fixture, capsys
):
    fetcher = DataFetcher(config=alpaca_config_fixture, verbose=True)
    # Mock the fetch method directly on provider_fetcher
    mocker.patch.object(fetcher.provider_fetcher, "fetch", return_value=None)

    mocker.patch.object(fetcher, "_check_existing_data", return_value=False)
    mock_write_table = mocker.patch("pyarrow.parquet.write_table")

    assert fetcher.fetch_data(force=False) is False
    fetcher.provider_fetcher.fetch.assert_called_once()
    mock_write_table.assert_not_called()

    captured = capsys.readouterr()
    assert (
        f"Failed to fetch data for {alpaca_config_fixture.symbol} from provider."
        in captured.err
    )


def test_fetch_data_provider_returns_empty_df(
    mocker, mock_alpaca_provider_instance, alpaca_config_fixture, tmp_path
):
    fetcher = DataFetcher(config=alpaca_config_fixture, verbose=True)
    empty_df = pd.DataFrame()
    # Mock the fetch method directly on provider_fetcher
    mocker.patch.object(fetcher.provider_fetcher, "fetch", return_value=empty_df)

    mocker.patch.object(fetcher, "_check_existing_data", return_value=False)
    mock_write_table = mocker.patch("pyarrow.parquet.write_table")
    base_test_path = tmp_path / "test_data_output_empty"
    mocker.patch.object(fetcher, "_get_base_path", return_value=base_test_path)

    assert (
        fetcher.fetch_data(force=False) is True
    )  # Success because empty is a valid outcome
    fetcher.provider_fetcher.fetch.assert_called_once()
    mock_write_table.assert_not_called()  # Should not write empty df based on current logic


def test_fetch_data_partitioning(
    mocker,
    mock_alpaca_provider_instance,
    alpaca_config_fixture,
    large_multiyear_dataframe,
    tmp_path,
):
    # Use strftime for date string format if needed, .isoformat() is generally preferred
    alpaca_config_fixture.start = large_multiyear_dataframe.index.min().strftime(
        "%Y-%m-%d"
    )
    alpaca_config_fixture.end = large_multiyear_dataframe.index.max().strftime(
        "%Y-%m-%d"
    )
    fetcher = DataFetcher(config=alpaca_config_fixture, verbose=True)
    # Mock the fetch method directly on provider_fetcher
    mocker.patch.object(
        fetcher.provider_fetcher, "fetch", return_value=large_multiyear_dataframe
    )

    mocker.patch.object(fetcher, "_check_existing_data", return_value=False)
    # mock_write_table = mocker.patch("pyarrow.parquet.write_table") # Remove this mock, we want actual writes

    # Create a mock Series for memory_usage return value
    memory_usage_return = pd.Series(
        [PARTITION_SIZE_THRESHOLD_BYTES / 5] * 6
    )  # 6 columns including index
    # Mock memory_usage to return the Series with a sum method
    mocker.patch.object(pd.DataFrame, "memory_usage", return_value=memory_usage_return)

    #    mock_mkdir = mocker.patch.object(Path, "mkdir") # Remove this, we need actual mkdir for pq.write_table

    base_output_dir = tmp_path / fetcher._get_base_path()
    mocker.patch.object(fetcher, "_get_base_path", return_value=base_output_dir)

    assert fetcher.fetch_data(force=False) is True
    fetcher.provider_fetcher.fetch.assert_called_once()

    years = large_multiyear_dataframe.index.year.unique()
    # assert mock_write_table.call_count == len(years) # Cannot assert call_count if not mocked

    expected_calls = []
    for year in years:
        expected_file_path = base_output_dir / f"{year}.parquet"
        expected_calls.append(call(mocker.ANY, expected_file_path))
    # mock_write_table.assert_has_calls(expected_calls, any_order=True) # This is more precise but complex

    #    mock_mkdir.assert_any_call(parents=True, exist_ok=True)  # Cannot assert if not mocked

    # Assert the provider fetcher's fetch method is called
    mock_alpaca_provider_instance.fetch.assert_called_once()

    # For 2022 and 2023, two partitions.
    # Check if files like "data/alpaca/AAPL/1Day_2022.parquet" exist
    base_path = (
        fetcher._get_base_path()
    )  # This is a directory like data/alpaca/AAPL/1Day
    year_files = list(base_path.glob("*.parquet"))  # Check for year-partitioned files
    assert len(year_files) >= 2  # Exactly 2 for 2022 and 2023

    # Example check for one file
    # Example: Check for 2022 data if large_multiyear_dataframe includes it
    # This part of the assertion needs to be robust to the actual years in large_multiyear_dataframe
    # For the provided large_multiyear_dataframe, it has 2022 and 2023.
    assert (base_path / "2022.parquet").exists()
    assert (base_path / "2023.parquet").exists()


@pytest.mark.parametrize(
    "fail_on_first_write", [True, False]
)  # Test failing first and second write
def test_fetch_data_partitioning_write_error(
    mocker,
    mock_alpaca_provider_instance,
    alpaca_config_fixture,
    large_multiyear_dataframe,
    tmp_path,
    capsys,
    fail_on_first_write,
):
    """Test graceful handling of OSError during pq.write_table in partitioning."""
    test_specific_data_dir = tmp_path / "test_data_output_write_error"
    sanitized_timeframe = alpaca_config_fixture.timeframe.replace(" ", "_").replace(
        ":", "-"
    )
    expected_base_under_tmp = (
        test_specific_data_dir
        / alpaca_config_fixture.provider
        / alpaca_config_fixture.symbol
        / sanitized_timeframe
    )
    mocker.patch.object(
        DataFetcher, "_get_base_path", return_value=expected_base_under_tmp
    )

    mock_alpaca_provider_instance.fetch.return_value = large_multiyear_dataframe

    # Mock pq.write_table to raise OSError on the specified attempt
    # It will be called twice (for 2022 and 2023)
    write_attempt = 0
    original_write_table = pq.write_table

    def faulty_write_table(*args, **kwargs):
        nonlocal write_attempt
        write_attempt += 1
        if (fail_on_first_write and write_attempt == 1) or (
            not fail_on_first_write and write_attempt == 2
        ):
            raise OSError("Mocked pq.write_table error")
        return original_write_table(
            *args, **kwargs
        )  # Call original for non-failing attempts

    mocked_pq_write = mocker.patch(
        "pyarrow.parquet.write_table", side_effect=faulty_write_table
    )

    fetcher = DataFetcher(config=alpaca_config_fixture, verbose=True)

    mock_memory_usage_series = MagicMock(spec=pd.Series)
    mock_memory_usage_series.sum.return_value = PARTITION_SIZE_THRESHOLD_BYTES + 1
    mocker.patch.object(
        pd.DataFrame, "memory_usage", return_value=mock_memory_usage_series
    )

    fetch_result = fetcher.fetch_data(force=True)

    assert (
        fetch_result is False
    )  # Should fail because write error is not caught per-partition

    captured = capsys.readouterr()
    # The error is caught by the main try-except in fetch_data
    assert (
        f"Error writing Parquet file for {alpaca_config_fixture.symbol}: Mocked pq.write_table error"
        in captured.err
    )

    # Ensure pq.write_table was called up to the failure point
    if fail_on_first_write:
        assert mocked_pq_write.call_count == 1
        # The first file (2022) should not exist as write failed
        path_2022 = expected_base_under_tmp / "2022.parquet"
        assert not path_2022.exists()
    else:  # Fails on second write (2023)
        assert mocked_pq_write.call_count == 2
        # The first file (2022) should exist
        path_2022 = expected_base_under_tmp / "2022.parquet"
        assert path_2022.exists()
        # The second file (2023) should not exist
        path_2023 = expected_base_under_tmp / "2023.parquet"
        assert not path_2023.exists()


# @pytest.mark.skip(reason="Placeholder test, to be implemented.") # Unskip the test
def test_full_data_fetching_flow(
    mocker,
    alpaca_config_fixture,
    sample_dataframe,
    mock_alpaca_provider_instance,  # Use the fixture that provides a mocked AlpacaFetcher instance
    tmp_path,
    capsys,
):
    """
    Test the full data fetching flow:
    - Mocks the provider's fetch method.
    - Ensures DataFetcher writes the data to a Parquet file.
    - Verifies the content and metadata of the created file.
    """
    config = alpaca_config_fixture
    fetcher = DataFetcher(config=config, verbose=True)  # Initialize DataFetcher

    # mock_alpaca_provider_instance fixture ensures fetcher.provider_fetcher is a mock
    # and its __init__ was effectively bypassed.
    # Configure the mock provider_fetcher.fetch to return our sample_dataframe
    fetcher.provider_fetcher.fetch.return_value = sample_dataframe

    # Mock _get_base_path to use tmp_path for output
    # The actual filename will be <base_path_val_from_config>.parquet
    # So, _get_base_path should return the path *without* the .parquet extension
    sanitized_timeframe = config.timeframe.replace(" ", "_").replace(":", "-")
    # base_path_for_output will be like tmp_path/data/alpaca/AAPL/1Day
    base_path_for_output = (
        tmp_path / config.provider / config.symbol / sanitized_timeframe
    )
    expected_output_file = base_path_for_output.with_suffix(".parquet")

    mocker.patch.object(fetcher, "_get_base_path", return_value=base_path_for_output)

    # Ensure the directory structure exists for writing the file
    # DataFetcher's _write_partitioned_data and _write_single_file handle mkdir
    # but for direct _get_base_path mock, we might need to ensure parent exists
    # if _write_single_file doesn't create all parents. Let's assume it does.
    # If not, an explicit base_path_for_output.parent.mkdir(parents=True, exist_ok=True)
    # might be needed before calling fetch_data depending on fetcher's internal mkdir logic.
    # fetcher._write_single_file itself calls base_path.mkdir(parents=True, exist_ok=True)
    # where base_path is what _get_base_path returns. So, it should be fine.

    # Call fetch_data (force=True to ensure it writes)
    result = fetcher.fetch_data(force=True)
    assert result is True

    # Assertions
    fetcher.provider_fetcher.fetch.assert_called_once_with(
        symbol=config.symbol,
        timeframe_str=config.timeframe,
        start_date_str=config.start,
        end_date_str=config.end,
    )

    # 1. Check if the file was created
    assert expected_output_file.exists(), (
        f"Output file {expected_output_file} was not created."
    )

    # 2. Check the content of the file
    # Read the Parquet file back into a DataFrame
    # The schema might differ slightly (e.g. index name might be lost if not handled by pyarrow)
    # so best to compare specific columns or use pandas testing utilities.
    df_from_file = pd.read_parquet(expected_output_file)

    # Ensure index name is preserved or set for comparison
    # sample_dataframe has index name 'timestamp'
    if df_from_file.index.name != sample_dataframe.index.name:
        df_from_file.index.name = sample_dataframe.index.name

    pd.testing.assert_frame_equal(df_from_file, sample_dataframe, check_dtype=True)

    # 3. Check the custom metadata
    metadata_from_file = fetcher._read_parquet_custom_metadata(expected_output_file)
    assert metadata_from_file is not None
    assert metadata_from_file["provider"] == config.provider
    assert metadata_from_file["symbol"] == config.symbol
    assert metadata_from_file["timeframe"] == config.timeframe
    assert metadata_from_file["config_start_date"] == config.start
    assert metadata_from_file["config_end_date"] == config.end
    assert (
        metadata_from_file["actual_start_date"]
        == sample_dataframe.index.min().isoformat()
    )
    assert (
        metadata_from_file["actual_end_date"]
        == sample_dataframe.index.max().isoformat()
    )
    assert int(metadata_from_file["num_rows"]) == len(sample_dataframe)
    assert "downloaded_at_utc" in metadata_from_file
    assert metadata_from_file["api_version"].startswith("alpaca-trade-api")

    captured = capsys.readouterr()
    assert f"Proceeding to fetch data for {config.symbol}" in captured.out
    assert f"Successfully wrote data to {expected_output_file}" in captured.out


def test_data_fetcher_placeholder_true():
    assert True


# TODO: Add tests for AlpacaFetcher with mocking
# TODO: Add tests for TradovateFetcher with mocking (using responses)
# TODO: Add tests for CLI (config validation, argument parsing)

# New tests for verbose outputs using capsys


def test_read_parquet_custom_metadata_verbose_error(
    mocker,
    mock_alpaca_provider_instance,
    alpaca_config_fixture,
    tmp_path,
    capsys,
    sample_dataframe,
):
    _ = mock_alpaca_provider_instance  # Ensure fixture is activated
    fetcher_verbose = DataFetcher(alpaca_config_fixture, verbose=True)
    test_file = tmp_path / "test_meta_verbose.parquet"
    # Ensure file exists for pq.read_metadata to be called
    sample_dataframe.to_parquet(test_file)  # Use fixture directly

    mocker.patch.object(
        pq, "read_metadata", side_effect=Exception("Mock pq.read_metadata error")
    )
    meta = fetcher_verbose._read_parquet_custom_metadata(test_file)
    assert meta is None
    captured = capsys.readouterr()
    assert "Could not read Parquet metadata" in captured.err
    assert "Mock pq.read_metadata error" in captured.err


def test_check_existing_data_verbose_no_file(
    mock_alpaca_provider_instance, alpaca_config_fixture, capsys
):
    _ = mock_alpaca_provider_instance  # Ensure fixture is activated
    fetcher_verbose = DataFetcher(alpaca_config_fixture, verbose=True)
    mock_target_file = MagicMock(spec=Path)
    mock_target_file.exists.return_value = False
    mock_target_file.__str__.return_value = (
        "mock/path/nonexistent.parquet"  # For print output
    )

    requested_start = pendulum.parse(alpaca_config_fixture.start)
    requested_end = pendulum.parse(alpaca_config_fixture.end)

    assert not fetcher_verbose._check_existing_data(
        mock_target_file, requested_start, requested_end
    )
    captured = capsys.readouterr()
    assert f"No existing data file found at {str(mock_target_file)}" in captured.out


def test_check_existing_data_verbose_parse_error(
    mocker, mock_alpaca_provider_instance, alpaca_config_fixture, tmp_path, capsys
):
    _ = mock_alpaca_provider_instance  # Ensure fixture is activated
    fetcher_verbose = DataFetcher(alpaca_config_fixture, verbose=True)
    mock_target_file = tmp_path / "test_parse_error.parquet"
    # Create a dummy file for exists() check
    mock_target_file.touch()

    mocker.patch.object(
        fetcher_verbose,
        "_read_parquet_custom_metadata",
        return_value={
            "actual_start_date": "invalid-date",
            "actual_end_date": "2023-01-01",
        },
    )

    requested_start = pendulum.parse(alpaca_config_fixture.start)
    requested_end = pendulum.parse(alpaca_config_fixture.end)

    assert not fetcher_verbose._check_existing_data(
        mock_target_file, requested_start, requested_end
    )
    captured = capsys.readouterr()
    assert "Error parsing date from metadata" in captured.out


# Test for line 186 verbose print
# Reverted to use mock_alpaca_provider_instance as @patch was problematic
def test_fetch_data_verbose_fetch_proceeds(
    mocker,
    mock_alpaca_provider_instance,
    alpaca_config_fixture,
    sample_dataframe,
    tmp_path,
    capsys,
):
    config = alpaca_config_fixture
    # mock_alpaca_provider_instance ensures DataFetcher gets a mock provider_fetcher whose __init__ is also bypassed
    fetcher = DataFetcher(config, verbose=True)  # Verbose True

    # At this point, fetcher.provider_fetcher should be the mock_instance from mock_alpaca_provider_instance
    assert fetcher.provider_fetcher is mock_alpaca_provider_instance

    # Mock _check_existing_data to return False (so it proceeds to fetch)
    mocker.patch.object(fetcher, "_check_existing_data", return_value=False)

    # Configure the fetch method on the mock provider instance
    fetcher.provider_fetcher.fetch.return_value = sample_dataframe

    base_path_for_timeframe = (
        tmp_path
        / "data"
        / config.provider
        / config.symbol
        / config.timeframe.replace(" ", "_").replace(":", "-")
    )
    base_path_for_timeframe.mkdir(parents=True, exist_ok=True)

    mocker.patch.object(fetcher, "_get_base_path", return_value=base_path_for_timeframe)

    mock_write_table = mocker.patch.object(pq, "write_table")

    fetcher.fetch_data(force=False)

    captured = capsys.readouterr()
    expected_log = f"Proceeding to fetch data for {config.symbol} ({config.timeframe}) from {config.start} to {config.end}"
    assert expected_log in captured.out
    fetcher.provider_fetcher.fetch.assert_called_once()
    mock_write_table.assert_called_once()


# Test for end_date normalization (lines 183-184)
def test_fetch_data_end_date_normalization(
    mocker, mock_alpaca_provider_instance, alpaca_config_fixture
):
    config = alpaca_config_fixture.model_copy()
    config.end = "2023-01-05"  # Date string, implies midnight

    fetcher = DataFetcher(config, verbose=False)
    mock_check_existing = mocker.patch.object(fetcher, "_check_existing_data")
    mock_provider_fetch = mock_alpaca_provider_instance.fetch
    mock_provider_fetch.return_value = pd.DataFrame()  # Return empty to stop early

    fetcher.fetch_data()

    # Check the arguments passed to _check_existing_data
    call_args = mock_check_existing.call_args
    assert call_args is not None
    _, requested_start_dt, requested_end_dt = call_args[0]

    expected_end_dt = pendulum.parse("2023-01-05").end_of("day")
    assert requested_end_dt == expected_end_dt
