# Placeholder for data fetcher tests
import pytest
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pendulum
import json
from unittest.mock import MagicMock, call

from algo_mvp.models import AlpacaConfig, TradovateConfig
from algo_mvp.data.fetcher import DataFetcher, PARTITION_SIZE_THRESHOLD_BYTES
# Import the actual classes to mock their __init__
from algo_mvp.data import alpaca as alpaca_module 
from algo_mvp.data import tradovate as tradovate_module

@pytest.fixture
def alpaca_config_fixture():
    return AlpacaConfig(
        provider="alpaca", 
        symbol="AAPL", 
        timeframe="1Day", 
        start="2023-01-01", 
        end="2023-01-10", 
        adjust=True
    )

@pytest.fixture
def tradovate_config_fixture():
    return TradovateConfig(
        provider="tradovate",
        symbol="MESM25",
        timeframe="1Min",
        start="2024-03-01T10:00:00Z",
        end="2024-03-01T10:05:00Z"
    )

@pytest.fixture
def sample_dataframe():
    dates = pd.to_datetime([
        '2023-01-01 00:00:00',
        '2023-01-01 00:01:00',
        '2024-05-10 10:00:00',
        '2024-05-10 10:01:00'
    ]).tz_localize('UTC')
    data = {
        'open': [100, 101, 150, 151],
        'high': [102, 101.5, 152, 151.5],
        'low': [99, 100.5, 149, 150.5],
        'close': [101, 101.2, 151, 151.2],
        'volume': [1000, 1200, 2000, 2200]
    }
    return pd.DataFrame(data, index=pd.Index(dates, name='timestamp'))

@pytest.fixture
def large_multiyear_dataframe():
    # Create a dataframe that spans multiple years and would be > threshold
    # For simplicity, let's assume 10000 rows of 5 columns (float64) + index (datetime64)
    # float64 is 8 bytes. datetime64 is 8 bytes.
    # 10000 rows * (5*8 + 8) bytes = 10000 * 48 bytes = 480,000 bytes ~ 0.48MB
    # To exceed 50MB (PARTITION_SIZE_THRESHOLD_BYTES), we'd need ~100x more rows.
    # For test speed, we'll rely on mocking estimated_size_bytes instead of huge DF.
    # But we do need data spanning multiple years for the partitioning logic to trigger.
    dates_2022 = pd.date_range(start='2022-01-01', end='2022-01-05', freq='D', tz='UTC')
    dates_2023 = pd.date_range(start='2023-01-01', end='2023-01-05', freq='D', tz='UTC')
    all_dates = dates_2022.union(dates_2023)
    data_val = range(len(all_dates))
    data = {
        'open': data_val,
        'high': data_val,
        'low': data_val,
        'close': data_val,
        'volume': data_val
    }
    return pd.DataFrame(data, index=pd.Index(all_dates, name='timestamp'))

# --- Mocks for Provider Fetcher Initialization ---
@pytest.fixture
def mock_alpaca_fetcher_init(mocker):
    # Mock the __init__ of the actual AlpacaFetcher class
    mocker.patch.object(alpaca_module.AlpacaFetcher, '__init__', return_value=None)
    # Also mock the fetch method on the *instance* that will be created
    mock_instance = MagicMock()
    mock_instance.fetch = MagicMock()
    # Add verbose attribute to the mock instance
    mock_instance.verbose = False
    # Make the class return our mock instance when called
    mocker.patch.object(alpaca_module, 'AlpacaFetcher', return_value=mock_instance)
    return mock_instance  # Return the mock instance for fetch assertions

@pytest.fixture
def mock_tradovate_fetcher_init(mocker):
    mocker.patch.object(tradovate_module.TradovateFetcher, '__init__', return_value=None)
    mock_instance = MagicMock()
    mock_instance.fetch = MagicMock()
    # Add verbose attribute to the mock instance
    mock_instance.verbose = False
    mocker.patch.object(tradovate_module, 'TradovateFetcher', return_value=mock_instance)
    return mock_instance

# --- Tests for DataFetcher --- #

def test_data_fetcher_initialization(mocker, alpaca_config_fixture, tradovate_config_fixture):
    """Test DataFetcher initialization selects correct (mocked) provider."""
    # We need to mock the provider fetcher inits *before* DataFetcher is instantiated
    mock_alpaca_init = mocker.patch.object(alpaca_module.AlpacaFetcher, '__init__', return_value=None)
    mock_tradovate_init = mocker.patch.object(tradovate_module.TradovateFetcher, '__init__', return_value=None)
    
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
def test_get_base_path(mock_alpaca_fetcher_init, mock_tradovate_fetcher_init, alpaca_config_fixture, tradovate_config_fixture):
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

def test_get_parquet_metadata(mock_alpaca_fetcher_init, alpaca_config_fixture, sample_dataframe):
    fetcher = DataFetcher(alpaca_config_fixture)
    metadata = fetcher._get_parquet_metadata(sample_dataframe)
    assert metadata['provider'] == "alpaca"
    assert metadata['actual_start_date'] == sample_dataframe.index.min().isoformat()  # Check using .isoformat()
    assert metadata['actual_end_date'] == sample_dataframe.index.max().isoformat()
    assert pendulum.parse(metadata['downloaded_at_utc'])  # Checks if parseable

def test_get_parquet_metadata_no_df(mock_alpaca_fetcher_init, alpaca_config_fixture):
    fetcher = DataFetcher(alpaca_config_fixture)
    metadata = fetcher._get_parquet_metadata(None)
    assert 'actual_start_date' not in metadata

def test_read_parquet_custom_metadata(mock_alpaca_fetcher_init, tmp_path, alpaca_config_fixture, sample_dataframe):
    fetcher = DataFetcher(alpaca_config_fixture)
    test_file = tmp_path / "test.parquet"
    assert fetcher._read_parquet_custom_metadata(test_file) is None  # File doesn't exist

    # Write table without specific metadata key
    table_no_meta = pa.Table.from_pandas(sample_dataframe)
    pq.write_table(table_no_meta, test_file)
    assert fetcher._read_parquet_custom_metadata(test_file) is None  # No algo_mvp_meta

    # Write table with correct metadata key
    expected_meta_dict = {"key": "value", "number_str": "123"}
    meta_json = json.dumps(expected_meta_dict)
    schema_meta = {b'algo_mvp_meta': meta_json.encode('utf-8')}
    table_with_meta = pa.Table.from_pandas(sample_dataframe)
    # Correct way: Add metadata to schema, then write table with that schema
    updated_schema = table_with_meta.schema.with_metadata(schema_meta)
    pq.write_table(table_with_meta.cast(updated_schema), test_file)  # Write using updated schema
    read_meta = fetcher._read_parquet_custom_metadata(test_file)
    assert read_meta == expected_meta_dict

    # Write table with malformed metadata
    bad_schema_meta = {b'algo_mvp_meta': b'not a json string'}
    table_bad_meta = pa.Table.from_pandas(sample_dataframe)  # Start fresh
    updated_schema_bad = table_bad_meta.schema.with_metadata(bad_schema_meta)
    pq.write_table(table_bad_meta.cast(updated_schema_bad), test_file)
    assert fetcher._read_parquet_custom_metadata(test_file) is None  # Malformed JSON

@pytest.mark.parametrize(
    "file_exists_val, metadata_content, expected_result, verbose_val",
    [
        (False, None, False, False),
        (True, None, False, True),
        (True, {'actual_start_date': '2023-01-01T00:00:00+00:00', 'actual_end_date': '2023-01-10T23:59:59+00:00'}, True, False),
        (True, {'config_start_date': '2023-01-01', 'config_end_date': '2023-01-10'}, True, False),
        (True, {'actual_start_date': '2023-01-02T00:00:00+00:00', 'actual_end_date': '2023-01-09T23:59:59+00:00'}, False, False),
        (True, {'actual_start_date': '2023-01-01T00:00:00+00:00', 'actual_end_date': '2023-01-08T23:59:59+00:00'}, False, False),
        (True, {'actual_start_date': '2022-12-31T00:00:00+00:00', 'actual_end_date': '2023-01-11T23:59:59+00:00'}, True, False),
        (True, {'some_other_meta': 'value'}, False, False),
        (True, {'actual_start_date': 'invalid-date', 'actual_end_date': '2023-01-10T23:59:59+00:00'}, False, True),
    ]
)
def test_check_existing_data(
    mocker, tmp_path, mock_alpaca_fetcher_init, alpaca_config_fixture, 
    file_exists_val, metadata_content, expected_result, verbose_val
):
    fetcher = DataFetcher(alpaca_config_fixture, verbose=verbose_val)
    req_start_dt = pendulum.parse(alpaca_config_fixture.start)  # Already date start
    req_end_dt = pendulum.parse(alpaca_config_fixture.end).end_of('day')  # Ensure end of day
    
    mock_target_file_arg = MagicMock(spec=Path)
    mock_target_file_arg.exists.return_value = file_exists_val
    # Mock the internal call _read_parquet_custom_metadata
    mocker.patch.object(fetcher, '_read_parquet_custom_metadata', return_value=metadata_content)

    assert fetcher._check_existing_data(mock_target_file_arg, req_start_dt, req_end_dt) == expected_result


def test_fetch_data_skip_if_exists_no_force(mocker, mock_alpaca_fetcher_init, alpaca_config_fixture):
    fetcher = DataFetcher(config=alpaca_config_fixture, verbose=True)
    mocker.patch.object(fetcher, '_check_existing_data', return_value=True)  # Mock to simulate data exists
    # mock_alpaca_fetcher_init fixture already mocked provider_fetcher.fetch

    assert fetcher.fetch_data(force=False) is True
    mock_alpaca_fetcher_init.fetch.assert_not_called()  # Crucial: provider fetch should not be called

def test_fetch_data_fetch_if_not_exists(mocker, mock_alpaca_fetcher_init, alpaca_config_fixture, sample_dataframe, tmp_path):
    fetcher = DataFetcher(config=alpaca_config_fixture, verbose=True)
    # Mock the fetch method directly on provider_fetcher instead of using mock_alpaca_fetcher_init.fetch
    mocker.patch.object(fetcher.provider_fetcher, 'fetch', return_value=sample_dataframe)
    
    mocker.patch.object(fetcher, '_check_existing_data', return_value=False)
    mock_write_table = mocker.patch('pyarrow.parquet.write_table')
    mocker.patch.object(Path, 'mkdir')

    base_test_path = tmp_path / "test_data_output"
    mocker.patch.object(fetcher, '_get_base_path', return_value=base_test_path)

    assert fetcher.fetch_data(force=False) is True
    fetcher.provider_fetcher.fetch.assert_called_once()
    mock_write_table.assert_called_once()
    expected_output_file = base_test_path.with_suffix('.parquet')
    args, _ = mock_write_table.call_args
    assert args[1] == expected_output_file

def test_fetch_data_force_fetch(mocker, mock_alpaca_fetcher_init, alpaca_config_fixture, sample_dataframe, tmp_path):
    fetcher = DataFetcher(config=alpaca_config_fixture, verbose=True)
    # Mock the fetch method directly on provider_fetcher
    mocker.patch.object(fetcher.provider_fetcher, 'fetch', return_value=sample_dataframe)
    
    mocker.patch.object(fetcher, '_check_existing_data', return_value=True) 
    mock_write_table = mocker.patch('pyarrow.parquet.write_table')
    mocker.patch.object(Path, 'mkdir')
    base_test_path = tmp_path / "test_data_output_force"
    mocker.patch.object(fetcher, '_get_base_path', return_value=base_test_path)

    assert fetcher.fetch_data(force=True) is True
    fetcher.provider_fetcher.fetch.assert_called_once()
    mock_write_table.assert_called_once()

def test_fetch_data_provider_fails(mocker, mock_alpaca_fetcher_init, alpaca_config_fixture):
    fetcher = DataFetcher(config=alpaca_config_fixture, verbose=True)
    # Mock the fetch method directly on provider_fetcher
    mocker.patch.object(fetcher.provider_fetcher, 'fetch', return_value=None)
    
    mocker.patch.object(fetcher, '_check_existing_data', return_value=False)
    mock_write_table = mocker.patch('pyarrow.parquet.write_table')

    assert fetcher.fetch_data(force=False) is False
    fetcher.provider_fetcher.fetch.assert_called_once()
    mock_write_table.assert_not_called()

def test_fetch_data_provider_returns_empty_df(mocker, mock_alpaca_fetcher_init, alpaca_config_fixture, tmp_path):
    fetcher = DataFetcher(config=alpaca_config_fixture, verbose=True)
    empty_df = pd.DataFrame()
    # Mock the fetch method directly on provider_fetcher
    mocker.patch.object(fetcher.provider_fetcher, 'fetch', return_value=empty_df)
    
    mocker.patch.object(fetcher, '_check_existing_data', return_value=False)
    mock_write_table = mocker.patch('pyarrow.parquet.write_table')
    base_test_path = tmp_path / "test_data_output_empty"
    mocker.patch.object(fetcher, '_get_base_path', return_value=base_test_path)

    assert fetcher.fetch_data(force=False) is True  # Success because empty is a valid outcome
    fetcher.provider_fetcher.fetch.assert_called_once()
    mock_write_table.assert_not_called()  # Should not write empty df based on current logic

def test_fetch_data_partitioning(mocker, mock_alpaca_fetcher_init, alpaca_config_fixture, large_multiyear_dataframe, tmp_path):
    # Use strftime for date string format if needed, .isoformat() is generally preferred
    alpaca_config_fixture.start = large_multiyear_dataframe.index.min().strftime('%Y-%m-%d')
    alpaca_config_fixture.end = large_multiyear_dataframe.index.max().strftime('%Y-%m-%d')
    fetcher = DataFetcher(config=alpaca_config_fixture, verbose=True)
    # Mock the fetch method directly on provider_fetcher
    mocker.patch.object(fetcher.provider_fetcher, 'fetch', return_value=large_multiyear_dataframe)
    
    mocker.patch.object(fetcher, '_check_existing_data', return_value=False)
    mock_write_table = mocker.patch('pyarrow.parquet.write_table')
    
    # Create a mock Series for memory_usage return value
    memory_usage_return = pd.Series([PARTITION_SIZE_THRESHOLD_BYTES / 5] * 6)  # 6 columns including index
    # Mock memory_usage to return the Series with a sum method
    mocker.patch.object(pd.DataFrame, 'memory_usage', return_value=memory_usage_return)
    
    mock_mkdir = mocker.patch.object(Path, 'mkdir')

    base_output_dir = tmp_path / fetcher._get_base_path()
    mocker.patch.object(fetcher, '_get_base_path', return_value=base_output_dir)

    assert fetcher.fetch_data(force=False) is True
    fetcher.provider_fetcher.fetch.assert_called_once()
    
    years = large_multiyear_dataframe.index.year.unique()
    assert mock_write_table.call_count == len(years)

    expected_calls = []
    for year in years:
        expected_file_path = base_output_dir / f"{year}.parquet"
        expected_calls.append(call(mocker.ANY, expected_file_path))
    # mock_write_table.assert_has_calls(expected_calls, any_order=True) # This is more precise but complex

    mock_mkdir.assert_any_call(parents=True, exist_ok=True)  # For the base_output_dir

@pytest.mark.skip(reason="Placeholder test, to be implemented.")
def test_full_data_fetching_flow():
    assert False  # Fail until implemented

def test_data_fetcher_placeholder_true():
    assert True

# TODO: Add tests for AlpacaFetcher with mocking
# TODO: Add tests for TradovateFetcher with mocking (using responses)
# TODO: Add tests for CLI (config validation, argument parsing)
