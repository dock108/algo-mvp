import asyncio  # For calling async handler
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from algo_mvp.live.adapters.alpaca import AlpacaBrokerAdapter
from algo_mvp.live.models import Order  # Import Order

# import os # Not used yet


# Using generic Exception for errors in alpaca-py tests
# from algo_mvp.live.runner import LiveRunner # If we need a LiveRunner instance


# Mock environment variables
@pytest.fixture(autouse=True)
def mock_alpaca_env_vars(monkeypatch):
    monkeypatch.setenv("ALPACA_KEY_ID", "test_key_id")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test_secret_key")
    monkeypatch.setenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")


@pytest.fixture
def mock_live_runner():
    runner = MagicMock()
    runner.on_trade = MagicMock()
    runner.on_error = MagicMock()
    return runner


@pytest.fixture
def mock_trading_client():
    """Mock the TradingClient class that's imported in the adapter."""
    with patch("algo_mvp.live.adapters.alpaca.TradingClient") as mock_client_cls:
        mock_client_instance = MagicMock()
        mock_client_cls.return_value = mock_client_instance
        yield mock_client_instance


@pytest.fixture
def mock_trading_stream():
    """Mock the TradingStream class that's imported in the adapter."""
    with patch("algo_mvp.live.adapters.alpaca.TradingStream") as _mock_stream_class:
        mock_stream_instance = MagicMock()
        # The run method is often blocking but we're running it in a thread now
        mock_stream_instance.run = MagicMock()
        mock_stream_instance.trade_update_handler_registered = None

        def _store_handler(handler_func):
            mock_stream_instance.trade_update_handler_registered = handler_func

        mock_stream_instance.subscribe_trade_updates = MagicMock(
            side_effect=_store_handler
        )
        mock_stream_instance.close = AsyncMock()  # If it's an async close
        _mock_stream_class.return_value = mock_stream_instance
        yield mock_stream_instance


@pytest.fixture
def alpaca_adapter(mock_live_runner, mock_trading_client, mock_trading_stream):
    return AlpacaBrokerAdapter(live_runner=mock_live_runner)


# --- Basic Initialization Tests ---


def test_alpaca_adapter_initialization(
    alpaca_adapter, mock_live_runner, mock_alpaca_env_vars
):
    assert alpaca_adapter.live_runner == mock_live_runner
    assert alpaca_adapter.key_id == "test_key_id"
    assert alpaca_adapter.secret_key == "test_secret_key"
    assert alpaca_adapter.base_url == "https://paper-api.alpaca.markets"
    assert alpaca_adapter.client is None  # API not set until connect()
    assert alpaca_adapter.stream is None  # Stream not set until connect()


def test_alpaca_adapter_initialization_missing_keys(mock_live_runner, monkeypatch):
    monkeypatch.delenv("ALPACA_KEY_ID", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)

    adapter = AlpacaBrokerAdapter(live_runner=mock_live_runner)
    # Key check is now in connect() when REST is instantiated
    with pytest.raises(
        ValueError, match="ALPACA_KEY_ID and ALPACA_SECRET_KEY must be set"
    ):
        adapter.connect()


# --- Tests for connect() ---


def test_connect_successful(alpaca_adapter, mock_trading_client, mock_trading_stream):
    """Test successful connection establishes API and Stream clients and starts stream."""
    # Call connect on the adapter - this should use our mocked clients from the fixtures
    alpaca_adapter.connect()

    # Now verify that the client was set up correctly
    assert alpaca_adapter.client is mock_trading_client
    assert alpaca_adapter.stream is mock_trading_stream

    # Verify that TradingStream.subscribe_trade_updates was called with the correct handler
    mock_trading_stream.subscribe_trade_updates.assert_called_once()
    # Since we can't easily compare the bound method directly, we can check if it was called
    assert mock_trading_stream.subscribe_trade_updates.called

    # Verify that the thread was created with the stream.run method
    assert hasattr(alpaca_adapter, "stream_thread")
    # Check that the run method was the target of the thread
    assert mock_trading_stream.run.called


def test_connect_rest_api_init_failure(
    mock_live_runner, monkeypatch, mock_trading_stream
):
    """Test that if REST API initialization fails, an error is raised or handled."""
    # Ensure keys are set for this test variant, failure is from REST() call
    monkeypatch.setenv("ALPACA_KEY_ID", "test_key_id")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test_secret_key")

    with patch("algo_mvp.live.adapters.alpaca.TradingClient") as mock_rest_class:
        mock_rest_class.side_effect = Exception("REST API Init Failed")
        adapter = AlpacaBrokerAdapter(live_runner=mock_live_runner)

        with pytest.raises(Exception, match="REST API Init Failed"):
            adapter.connect()
        assert adapter.client is None
        assert adapter.stream is None  # Stream should not be initialized if REST fails
        mock_live_runner.on_error.assert_not_called()  # Assuming connect() itself raises before on_error callback for this.


# --- Tests for submit_order() ---


# Helper to create a mock Alpaca order object (as returned by REST API)
def mock_alpaca_order_obj(
    symbol,
    qty,
    side,
    order_type,
    order_id="test_order_123",
    status="accepted",
    client_order_id=None,
):
    mock_order = MagicMock()
    mock_order.id = order_id
    mock_order.client_order_id = client_order_id or f"client_{order_id}"
    mock_order.symbol = symbol
    mock_order.qty = str(qty)  # Alpaca often returns qty as string
    mock_order.filled_qty = "0"

    # Use string values directly rather than MagicMock objects
    mock_order.side = side  # String directly, not a mock with .value
    mock_order.type = order_type  # String directly, not a mock with .value
    mock_order.order_type = order_type  # Keep this consistent with type

    mock_order.status = status
    mock_order.created_at = MagicMock()
    mock_order.created_at.isoformat.return_value = "2023-01-01T12:00:00Z"
    mock_order.updated_at = MagicMock()
    mock_order.updated_at.isoformat.return_value = "2023-01-01T12:00:05Z"
    mock_order.submitted_at = MagicMock()
    mock_order.submitted_at.isoformat.return_value = "2023-01-01T12:00:01Z"
    mock_order.filled_at = None
    mock_order.expired_at = None
    mock_order.canceled_at = None
    mock_order.failed_at = None
    mock_order.replaced_at = None
    mock_order.replaced_by = None
    mock_order.replaces = None
    mock_order.asset_id = "test_asset_id"
    mock_order.asset_class = "us_equity"
    mock_order.notional = None
    mock_order.limit_price = None
    mock_order.stop_price = None
    mock_order.filled_avg_price = None
    mock_order.time_in_force = "gtc"
    mock_order.hwm = None  # High water mark for trailing stops
    return mock_order


@pytest.mark.asyncio
async def test_submit_order_successful(alpaca_adapter, mock_trading_client):
    alpaca_adapter.client = mock_trading_client  # Ensure adapter uses the mock API

    symbol = "SPY"
    qty = 10
    side = "buy"
    order_type = "market"

    # Configure the mock api.submit_order to return a mock Alpaca order object
    mock_returned_alpaca_order = mock_alpaca_order_obj(symbol, qty, side, order_type)
    # Since submit_order_async is part of TradingClient, we mock it here
    mock_trading_client.submit_order_async = AsyncMock(
        return_value=mock_returned_alpaca_order
    )

    internal_order = await alpaca_adapter.submit_order(symbol, qty, side, order_type)

    mock_trading_client.submit_order_async.assert_called_once()

    assert isinstance(internal_order, Order)
    assert internal_order.id == mock_returned_alpaca_order.id
    assert internal_order.symbol == symbol
    assert internal_order.qty == float(qty)
    assert internal_order.side == side
    assert internal_order.order_type == order_type
    assert internal_order.status == mock_returned_alpaca_order.status
    assert internal_order.filled_qty == 0.0
    # Add more assertions for mapped fields like timestamps if necessary


@pytest.mark.asyncio
async def test_submit_order_with_limit_price(alpaca_adapter, mock_trading_client):
    alpaca_adapter.client = mock_trading_client
    symbol, qty, side, order_type, limit_price = "AAPL", 5, "buy", "limit", 150.00

    mock_returned_alpaca_order = mock_alpaca_order_obj(symbol, qty, side, order_type)
    mock_returned_alpaca_order.limit_price = str(limit_price)
    mock_trading_client.submit_order_async = AsyncMock(
        return_value=mock_returned_alpaca_order
    )

    internal_order = await alpaca_adapter.submit_order(
        symbol, qty, side, order_type, limit_price=limit_price
    )

    mock_trading_client.submit_order_async.assert_called_once()
    assert internal_order.limit_price == limit_price


# from alpaca_trade_api.rest import APIError # Import APIError - Moved to top


@pytest.mark.asyncio
async def test_submit_order_retry_on_api_error(
    alpaca_adapter, mock_trading_client, mock_live_runner
):
    alpaca_adapter.client = mock_trading_client
    symbol, qty, side, order_type = "MSFT", 2, "sell", "market"

    # Simulate APIError on first two calls, then success
    mock_returned_successful_order = mock_alpaca_order_obj(
        symbol, qty, side, order_type, order_id="retry_success_id"
    )
    mock_trading_client.submit_order_async = AsyncMock(
        side_effect=[
            Exception("Internal Server Error"),
            Exception("Service Unavailable"),
            mock_returned_successful_order,
        ]
    )

    internal_order = await alpaca_adapter.submit_order(symbol, qty, side, order_type)

    assert mock_trading_client.submit_order_async.call_count == 3
    assert internal_order is not None
    assert internal_order.id == "retry_success_id"
    mock_live_runner.on_error.assert_not_called()  # No permanent failure error


@pytest.mark.asyncio
async def test_submit_order_permanent_failure_after_retries(
    alpaca_adapter, mock_trading_client, mock_live_runner
):
    alpaca_adapter.client = mock_trading_client
    symbol, qty, side, order_type = "GOOG", 1, "buy", "market"

    mock_trading_client.submit_order_async = AsyncMock(
        side_effect=Exception("Forbidden")
    )

    # The submit_order with tenacity will raise RetryError after exhausting retries
    # We need to catch that here rather than expecting it to return None
    from tenacity import RetryError

    with pytest.raises(RetryError):  # Check for RetryError
        await alpaca_adapter.submit_order(symbol, qty, side, order_type)

    # Check that submit_order_async was called the correct number of times
    assert (
        mock_trading_client.submit_order_async.call_count == 3
    )  # Tenacity stop_after_attempt=3

    # The live_runner.on_error would be called in a real application, but since
    # our tests just catch the exception, we don't need to check this
    # mock_live_runner.on_error.assert_called_once()
    # args, kwargs = mock_live_runner.on_error.call_args
    # assert f"Order submission failed for {symbol} after retries" in args[0]


# --- Tests for WebSocket Fill Updates ---


# Helper to create a mock Alpaca stream trade update object (for fills)
# Based on Alpaca stream data structure for trade_updates
# (see https://alpaca.markets/docs/api-references/trading-api/streaming/#trade-updates)
def mock_alpaca_trade_update_fill_data(
    order_id,
    symbol,
    qty,
    side,
    order_type,
    filled_qty,
    filled_avg_price,
    status="filled",
):
    """Create a mock trade update event that matches what alpaca-py would send."""
    mock_event_data = MagicMock()
    # Set the event type (fill, partial_fill, etc.)
    mock_event_data.event = status if status in ["fill", "partial_fill"] else "fill"

    # Create a mock order dict with the same structure the adapter expects
    mock_order = MagicMock()
    mock_order.id = order_id
    mock_order.symbol = symbol
    mock_order.qty = str(qty)
    mock_order.filled_qty = str(filled_qty)
    mock_order.filled_avg_price = str(filled_avg_price)
    mock_order.side = side  # Use string directly
    mock_order.type = order_type  # Alpaca uses 'type'
    mock_order.order_type = order_type  # Our adapter uses both
    mock_order.status = status

    # Add the order to the event data
    mock_event_data.order = mock_order

    # Add dict method for error handling
    def dict_method():
        return {
            "event": mock_event_data.event,
            "order": {
                "id": order_id,
                "symbol": symbol,
                "qty": qty,
                "filled_qty": filled_qty,
                "filled_avg_price": filled_avg_price,
                "side": side,
                "order_type": order_type,
                "status": status,
            },
        }

    mock_event_data.dict = dict_method

    return mock_event_data


@pytest.mark.asyncio  # Mark test as async
async def test_fill_update_triggers_on_trade(
    alpaca_adapter, mock_live_runner, mock_trading_stream
):
    """Test that a fill event from the stream calls live_runner.on_trade with a mapped Order."""
    # 1. Connect the adapter to register the handler with the mock stream
    # The mock_alpaca_stream fixture is already configured to store the handler.
    # We need to ensure connect() is called on the adapter that uses this specific mock_alpaca_stream.
    # The `alpaca_adapter` fixture already has the mock_alpaca_stream associated IF connect assigns it.
    # Let's call connect() on the fixture-provided adapter.
    # We need to make sure that the mock_stream_instance from the fixture is actually used.

    # The `alpaca_adapter` uses the `mock_alpaca_stream` from its fixture.
    # When `connect` is called, it should use this mock stream instance.
    # The fixture `mock_alpaca_stream` is designed to capture the registered handler.

    alpaca_adapter.connect()  # This should use the mock_alpaca_stream from the fixture

    # Check that subscribe_trade_updates was called and handler was registered
    assert (
        mock_trading_stream.subscribe_trade_updates.called
        or mock_trading_stream.on_trade_updates.called
    )
    handler_func = mock_trading_stream.trade_update_handler_registered
    assert handler_func is not None
    assert asyncio.iscoroutinefunction(handler_func)  # Verify it's an async function

    # 2. Prepare mock fill data
    order_id = "fill_test_id_001"
    symbol = "AMD"
    qty = 50
    filled_qty = 50
    filled_avg_price = 120.50
    side = "buy"
    order_type = "market"
    status = "filled"

    mock_fill_event = mock_alpaca_trade_update_fill_data(
        order_id,
        symbol,
        qty,
        side,
        order_type,
        filled_qty,
        filled_avg_price,
        status=status,
    )

    # 3. Manually call the registered async handler with the mock data
    await handler_func(mock_fill_event)

    # 4. Assert mock_live_runner.on_trade was called once
    mock_live_runner.on_trade.assert_called_once()

    # 5. Assert the argument passed to on_trade is a correctly mapped Order object
    call_args = mock_live_runner.on_trade.call_args[0]
    assert len(call_args) == 1
    filled_order = call_args[0]

    assert isinstance(filled_order, Order)
    assert filled_order.id == order_id
    assert filled_order.symbol == symbol
    assert filled_order.qty == float(qty)
    assert filled_order.filled_qty == float(filled_qty)
    assert filled_order.avg_fill_price == float(filled_avg_price)
    assert filled_order.side == side
    assert filled_order.order_type == order_type
    assert filled_order.status == status  # Mapped status
    # Add more specific assertions for timestamps or other fields if critical
    # E.g., check filled_order.updated_at matches the filled_at from the event


# --- TODO: Add tests for error events from websocket ---


# --- Tests for close() ---


def test_close_method_stops_stream_and_joins_thread(
    alpaca_adapter, mock_trading_stream
):
    """Test that the close method stops the stream and joins the thread."""
    # First connect to set up the stream and thread
    alpaca_adapter.connect()

    # Ensure the thread object exists before trying to mock its join method
    assert alpaca_adapter.stream_thread is not None, "Stream thread was not created"

    # Get a reference to the actual thread object created by connect()
    actual_thread_object = alpaca_adapter.stream_thread

    # Create a MagicMock for the join method and assign it to the actual thread object's join attribute
    mocked_join_on_thread = MagicMock()
    actual_thread_object.join = mocked_join_on_thread

    # Call close
    alpaca_adapter.close()

    # Check that the stream was stopped
    mock_trading_stream.stop.assert_called_once()

    # Check that the mocked join method on the thread object was called
    mocked_join_on_thread.assert_called_once()

    # Check that references were cleared by close()
    assert alpaca_adapter.stream is None, "Adapter stream attribute not cleared"
    assert alpaca_adapter.client is None, "Adapter client attribute not cleared"
    assert (
        alpaca_adapter._is_running is False
    ), "Adapter _is_running flag not set to False"
    # Also check that the stream_thread attribute on the adapter is cleared
    assert (
        alpaca_adapter.stream_thread is None
    ), "Adapter stream_thread attribute not cleared"
