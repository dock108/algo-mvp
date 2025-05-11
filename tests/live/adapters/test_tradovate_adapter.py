import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import httpx
import pytest
import pytest_asyncio
import websockets

from algo_mvp.live.adapters.tradovate import (
    TRADOVATE_BASE_URL,
    TRADOVATE_WS_URL,
    TradovateBrokerAdapter,
)
from algo_mvp.live.models import Fill, Order, Position

# Configure a logger for the test file itself if needed for test-specific logging
test_file_logger = logging.getLogger(__name__)

# Default Env Vars for tests
TEST_CLIENT_ID = "test_client_id"
TEST_USERNAME = "test_username"
TEST_PASSWORD = "test_password"
TEST_ACCESS_TOKEN = "test_access_token_123"
TEST_USER_ID = 12345


@pytest.fixture
def mock_env_vars(monkeypatch):
    monkeypatch.setenv("TRADOVATE_CLIENT_ID", TEST_CLIENT_ID)
    monkeypatch.setenv("TRADOVATE_USERNAME", TEST_USERNAME)
    monkeypatch.setenv("TRADOVATE_PASSWORD", TEST_PASSWORD)
    monkeypatch.setenv("TRADOVATE_BASE_URL", TRADOVATE_BASE_URL)
    monkeypatch.setenv("TRADOVATE_WS_URL", TRADOVATE_WS_URL)


@pytest.fixture
def mock_live_runner():
    runner = MagicMock()
    runner.on_trade = AsyncMock()
    runner.on_order_update = AsyncMock()
    runner.on_broker_event = AsyncMock()
    return runner


@pytest.fixture
def mock_http_client():
    with patch("httpx.AsyncClient") as mock_client_class_constructor:
        mock_client_instance = AsyncMock(name="AsyncClient_instance")
        mock_client_instance.aclose = AsyncMock(name="aclose_method")

        # Default mock for response from client.post()
        default_post_response = AsyncMock(name="default_post_response_object")
        default_post_response.status_code = 200
        default_post_response.json = MagicMock(
            name="post_response_json_method", return_value={}
        )
        default_post_response.raise_for_status = MagicMock(
            name="post_response_raise_method"
        )
        mock_client_instance.post = AsyncMock(
            name="post_method", return_value=default_post_response
        )

        # Default mock for response from client.get()
        default_get_response = AsyncMock(name="default_get_response_object")
        default_get_response.status_code = 200
        default_get_response.json = MagicMock(
            name="get_response_json_method", return_value={}
        )
        default_get_response.raise_for_status = MagicMock(
            name="get_response_raise_method"
        )
        mock_client_instance.get = AsyncMock(
            name="get_method", return_value=default_get_response
        )

        # Default mock for response from client.request() (general)
        default_request_response = AsyncMock(name="default_request_response_object")
        default_request_response.status_code = 200
        default_request_response.json = MagicMock(
            name="request_response_json_method", return_value={}
        )
        default_request_response.raise_for_status = MagicMock(
            name="request_response_raise_method"
        )
        mock_client_instance.request = AsyncMock(
            name="request_method", return_value=default_request_response
        )

        # Default mock for response from client.delete()
        default_delete_response = AsyncMock(name="default_delete_response_object")
        default_delete_response.status_code = 200
        default_delete_response.raise_for_status = MagicMock(
            name="delete_response_raise_method"
        )
        default_delete_response.json = MagicMock(
            name="delete_response_json_method", return_value=None
        )
        mock_client_instance.delete = AsyncMock(
            name="delete_method", return_value=default_delete_response
        )

        mock_client_class_constructor.return_value = mock_client_instance
        yield mock_client_class_constructor


@pytest_asyncio.fixture
async def adapter(mock_env_vars, mock_live_runner, mock_http_client):
    adapter_instance = TradovateBrokerAdapter(
        runner=mock_live_runner,
        client_id=TEST_CLIENT_ID,
        username=TEST_USERNAME,
        password=TEST_PASSWORD,
    )
    yield adapter_instance
    if (
        adapter_instance._stop_event.is_set()
        or adapter_instance.ws_connection
        or adapter_instance.http_client
    ):
        await adapter_instance.close()


@pytest_asyncio.fixture
async def connected_adapter(adapter, mock_http_client, mock_websocket_connect):
    token_response_dict = {
        "accessToken": TEST_ACCESS_TOKEN,
        "userId": TEST_USER_ID,
        "userStatus": "Authorized",
        "expirationTime": "2099-01-01T00:00:00Z",
        "name": TEST_USERNAME,
        "hasLive": False,
    }
    account_list_response_dict = [
        {
            "id": 54321,
            "name": "TestAccount",
            "balance": 100000,
            "userId": TEST_USER_ID,
            "buyingPower": 100000,
        }
    ]

    mock_http_client.return_value.post.return_value.json.return_value = (
        token_response_dict
    )
    mock_http_client.return_value.request.return_value.json.return_value = (
        account_list_response_dict
    )
    mock_http_client.return_value.request.return_value.status_code = 200

    await adapter.connect()
    yield adapter
    await adapter.close()  # Ensure cleanup happens


@pytest.fixture
def mock_websocket_connect():
    with patch("websockets.connect", new_callable=AsyncMock) as mock_connect:
        mock_ws_instance = AsyncMock()
        mock_ws_instance.closed = False
        # Only set initial connection messages, let tests set their own messages
        mock_ws_instance.recv.side_effect = ["o", 'a[{"userStatus":"Authorized"}]']
        mock_ws_instance.send = AsyncMock()
        mock_ws_instance.close = AsyncMock()
        mock_connect.return_value = mock_ws_instance
        yield mock_connect


# --- Basic Initialization Tests ---


@pytest.mark.asyncio
async def test_adapter_initialization(adapter):
    assert adapter.client_id == TEST_CLIENT_ID
    assert adapter.username == TEST_USERNAME
    assert adapter.password == TEST_PASSWORD
    assert adapter.base_url == TRADOVATE_BASE_URL
    assert adapter.ws_url == TRADOVATE_WS_URL
    assert adapter.access_token_details is None
    assert adapter.ws_connection is None
    assert not adapter._stop_event.is_set()


@pytest.mark.asyncio
async def test_adapter_init_missing_creds(mock_live_runner, monkeypatch):
    monkeypatch.delenv("TRADOVATE_CLIENT_ID", raising=False)
    monkeypatch.delenv("TRADOVATE_USERNAME", raising=False)
    monkeypatch.delenv("TRADOVATE_PASSWORD", raising=False)
    with pytest.raises(ValueError, match="Tradovate credentials .* must be provided"):
        TradovateBrokerAdapter(
            runner=mock_live_runner, client_id=None, username=None, password=None
        )


# --- Connection Tests (OAuth, WS) ---


@pytest.mark.asyncio
async def test_connect_successful(adapter, mock_http_client, mock_websocket_connect):
    token_response = {
        "accessToken": TEST_ACCESS_TOKEN,
        "userId": TEST_USER_ID,
        "userStatus": "Authorized",
        "expirationTime": "2099-01-01T00:00:00Z",
        "name": TEST_USERNAME,
        "hasLive": False,
    }
    mock_http_client.return_value.post.return_value.json.return_value = token_response

    account_response = [
        {"id": 54321, "name": "TestAccount", "balance": 100000, "userId": TEST_USER_ID}
    ]
    mock_http_client.return_value.request.return_value.json.return_value = (
        account_response
    )
    mock_http_client.return_value.request.return_value.status_code = 200

    await adapter.connect()

    assert adapter.access_token_details is not None
    assert adapter.access_token_details.accessToken == TEST_ACCESS_TOKEN
    assert adapter.ws_connection is not None
    mock_websocket_connect.assert_called_once_with(
        TRADOVATE_WS_URL, ssl=True, ping_interval=None
    )

    expected_auth_msg = f"authorize\n0\n\n{TEST_ACCESS_TOKEN}"

    assert adapter.ws_connection.send.call_count > 0, (
        "ws_connection.send was not called"
    )

    # Get the first argument of the first call to send
    actual_sent_arg = adapter.ws_connection.send.call_args_list[0][0][0]

    assert actual_sent_arg == expected_auth_msg, (
        f"Auth message mismatch. Expected: {repr(expected_auth_msg)}, Actual: {repr(actual_sent_arg)}"
    )

    assert adapter.heartbeat_task is not None
    assert adapter.ws_listener_task is not None
    assert adapter._data_poller_task is not None
    adapter.runner.on_broker_event.assert_called_with(
        {"type": "connect", "status": "success"}
    )

    await adapter.close()


@pytest.mark.asyncio
async def test_connect_token_request_fails(adapter, mock_http_client):
    http_status_error = httpx.HTTPStatusError(
        "Unauthorized",
        request=MagicMock(),
        response=AsyncMock(
            status_code=401, text="Unauthorized Text"
        ),  # text needs to be a string
    )

    # Configure the mock for the POST response
    post_response_mock = mock_http_client.return_value.post.return_value
    post_response_mock.status_code = 401  # Ensure status code is set for the response
    post_response_mock.json.return_value = {"error": "Unauthorized JSON"}
    post_response_mock.raise_for_status.side_effect = http_status_error

    await adapter.connect()

    assert adapter.access_token_details is None
    assert adapter.ws_connection is None
    # The error message in on_broker_event comes from str(e) where e is HTTPStatusError
    adapter.runner.on_broker_event.assert_called_with(
        {"type": "connect", "status": "failure", "error": str(http_status_error)}
    )
    assert adapter.is_connecting is False


@pytest.mark.asyncio
async def test_rest_call_with_expired_token_refreshes_token(adapter, mock_http_client):
    assert adapter.access_token_details is None

    expired_time = datetime.now(timezone.utc) - timedelta(minutes=1)
    expired_token_dict = {
        "accessToken": "expired_token",
        "userId": TEST_USER_ID,
        "userStatus": "Authorized",
        "expirationTime": expired_time.isoformat().replace("+00:00", "Z"),
        "name": TEST_USERNAME,
        "hasLive": False,
    }
    fresh_token_dict = {
        "accessToken": "new_fresh_token",
        "userId": TEST_USER_ID,
        "userStatus": "Authorized",
        "expirationTime": (datetime.now(timezone.utc) + timedelta(hours=1))
        .isoformat()
        .replace("+00:00", "Z"),
        "name": TEST_USERNAME,
        "hasLive": False,
    }

    # Mock responses for POST calls to token endpoint
    post_mock = mock_http_client.return_value.post

    # Response for 1st POST (expired token)
    resp1_post = AsyncMock(name="response_post_expired")
    resp1_post.status_code = 200
    resp1_post.json = MagicMock(return_value=expired_token_dict)
    resp1_post.raise_for_status = MagicMock()

    # Response for 2nd POST (fresh token)
    resp2_post = AsyncMock(name="response_post_fresh")
    resp2_post.status_code = 200
    resp2_post.json = MagicMock(return_value=fresh_token_dict)
    resp2_post.raise_for_status = MagicMock()

    # Sequence of responses for POST calls
    post_mock.side_effect = [resp1_post, resp2_post, resp1_post, resp2_post]

    # Mock response for GET /account/list
    account_list_dict = [
        {"id": 54321, "balance": 1000, "name": "TestAccount", "buyingPower": 5000}
    ]
    # The general .request() method is used by _make_request if method is GET
    get_account_list_response = AsyncMock(name="get_account_list_response")
    get_account_list_response.status_code = 200
    get_account_list_response.json = MagicMock(return_value=account_list_dict)
    get_account_list_response.raise_for_status = MagicMock()

    # Set default for GET, but specific for /account/list if necessary
    mock_http_client.return_value.request.return_value = (
        get_account_list_response  # If get_cash uses .request
    )

    # Capture headers
    captured_headers = {}
    original_request_method = mock_http_client.return_value.request

    async def capture_request_side_effect(method, endpoint, headers=None, **kwargs):
        nonlocal captured_headers
        if (
            endpoint == "/account/list"
        ):  # Assuming get_cash makes a GET to /account/list
            if headers:  # headers might be None if not passed
                captured_headers = headers.copy()
        # Return the default mock response for .request or a specific one if needed
        return await original_request_method(
            method, endpoint, headers=headers, **kwargs
        )

    mock_http_client.return_value.request.side_effect = capture_request_side_effect

    # --- First call to get_cash ---
    # Expect: POST (expired), POST (fresh)
    await adapter.get_cash()
    assert post_mock.call_count == 2
    assert adapter.access_token_details.accessToken == "new_fresh_token"
    assert captured_headers.get("Authorization") == "Bearer new_fresh_token"

    # --- Reset token and call get_cash again ---
    adapter.access_token_details = None
    captured_headers = {}  # Reset for next capture

    # Expect: POST (expired), POST (fresh)
    await adapter.get_cash()
    assert post_mock.call_count == 4  # Total POSTs
    assert adapter.access_token_details.accessToken == "new_fresh_token"
    assert captured_headers.get("Authorization") == "Bearer new_fresh_token"

    await adapter.close()


@pytest.mark.asyncio
async def test_websocket_reconnects_on_token_expiry_during_heartbeat(
    adapter, mock_http_client, mock_websocket_connect
):
    initial_expiration = datetime.now(timezone.utc) + timedelta(seconds=5)
    initial_token_dict = {
        "accessToken": "short_lived_token",
        "userId": TEST_USER_ID,
        "userStatus": "Authorized",
        "expirationTime": initial_expiration.isoformat().replace("+00:00", "Z"),
        "name": TEST_USERNAME,
        "hasLive": False,
    }
    mock_http_client.return_value.post.return_value.json.return_value = (
        initial_token_dict
    )

    # _get_initial_account_data in _ws_connect uses _make_request -> client.request()
    account_list_for_connect = [
        {"id": 54321, "name": "TestAccount", "balance": 100000, "userId": TEST_USER_ID}
    ]
    mock_http_client.return_value.request.return_value.json.return_value = (
        account_list_for_connect
    )
    mock_http_client.return_value.request.return_value.status_code = 200

    await adapter.connect()
    assert adapter.access_token_details is not None
    assert adapter.access_token_details.accessToken == "short_lived_token"
    original_ws_connection = adapter.ws_connection

    mock_websocket_connect.reset_mock()

    # Configure mock for the .post() method of the client instance for re-authentication
    fresh_token_dict = {
        "accessToken": "new_heartbeat_token",
        "userId": TEST_USER_ID,
        "userStatus": "Authorized",
        "expirationTime": (datetime.now(timezone.utc) + timedelta(hours=1))
        .isoformat()
        .replace("+00:00", "Z"),
        "name": TEST_USERNAME,
        "hasLive": False,
    }
    # The client's post method (AsyncMock) should return a response mock (AsyncMock)
    # whose json method (MagicMock) returns fresh_token_dict.
    mock_http_client.return_value.post.return_value.json.return_value = fresh_token_dict
    mock_http_client.return_value.post.return_value.status_code = (
        200  # Ensure re-auth post is OK
    )
    mock_http_client.return_value.post.reset_mock()  # Reset for asserting the specific re-auth call

    # For the account list fetch during the new _ws_connect call
    account_list_for_reconnect = [
        {
            "id": 54321,
            "name": "TestAccount",
            "balance": 100000,
            "userId": TEST_USER_ID,
            "buyingPower": 200000,
        }
    ]
    mock_http_client.return_value.request.return_value.json.return_value = (
        account_list_for_reconnect
    )
    mock_http_client.return_value.request.return_value.status_code = 200

    adapter.access_token_details.expirationTime = datetime.now(
        timezone.utc
    ) - timedelta(minutes=1)

    if original_ws_connection:  # Ensure it's not None
        original_ws_connection.send.side_effect = (
            websockets.exceptions.ConnectionClosed(None, None)
        )  # Simpler close

    # --- Store original asyncio.sleep before patching ---
    original_asyncio_sleep = asyncio.sleep

    async def mock_actual_short_sleep(delay):
        # Call the original asyncio.sleep
        await original_asyncio_sleep(0.001)

    with patch("asyncio.sleep", new=mock_actual_short_sleep):
        connect_called_event = asyncio.Event()
        original_connect_method = adapter.connect

        async def connect_wrapper(*args, **kwargs):
            nonlocal original_connect_method  # Ensure it's from the correct scope
            # Call the original connect method that's part of the adapter instance
            await original_connect_method(*args, **kwargs)
            connect_called_event.set()
            # We might not want to stop the event here, let heartbeat task manage its lifecycle if it can.
            # Or, if this is specifically to stop after one reconnect attempt in heartbeat:
            # adapter._stop_event.set()

        adapter.connect = AsyncMock(side_effect=connect_wrapper)

        # To reliably test the heartbeat's reconnect:
        # 1. Ensure heartbeat task is running
        # 2. Simulate condition that makes it call connect()
        # We've made original_ws_connection.send fail.
        # The _send_ws_heartbeat loop will catch ConnectionClosed, set ws_connection to None,
        # then call self.connect() (which is now connect_wrapper).

        # If heartbeat_task was cancelled or not started, start a controlled one.
        if adapter.heartbeat_task and not adapter.heartbeat_task.done():
            adapter.heartbeat_task.cancel()  # Cancel previous
            try:
                await adapter.heartbeat_task
            except asyncio.CancelledError:
                pass

        # Clear stop event before starting a new heartbeat cycle for the test
        adapter._stop_event.clear()
        heartbeat_test_task = asyncio.create_task(adapter._send_ws_heartbeat())

        try:
            await asyncio.wait_for(
                connect_called_event.wait(), timeout=10
            )  # Increased timeout
        except asyncio.TimeoutError:
            pytest.fail(
                "_send_ws_heartbeat did not trigger connect (via connect_wrapper) within timeout"
            )
        finally:
            adapter._stop_event.set()  # Stop all tasks including the new heartbeat
            heartbeat_test_task.cancel()
            try:
                await heartbeat_test_task
            except asyncio.CancelledError:
                pass
            adapter.connect = original_connect_method  # Restore original
            adapter._stop_event.clear()

    mock_http_client.return_value.post.assert_called_once()  # For the new token during reconnect
    assert adapter.access_token_details.accessToken == "new_heartbeat_token"
    mock_websocket_connect.assert_called_once()
    assert adapter.ws_connection is not original_ws_connection
    assert adapter.ws_connection is not None

    await adapter.close()


# --- WebSocket Message Handling Tests ---


@pytest.mark.asyncio
async def test_websocket_handles_fill_event(connected_adapter, mock_live_runner):
    adapter = connected_adapter

    # Stop any existing tasks
    adapter._stop_event.set()
    if adapter.ws_listener_task and not adapter.ws_listener_task.done():
        adapter.ws_listener_task.cancel()
        try:
            await adapter.ws_listener_task
        except asyncio.CancelledError:
            pass
    if adapter.heartbeat_task and not adapter.heartbeat_task.done():
        adapter.heartbeat_task.cancel()
        try:
            await adapter.heartbeat_task
        except asyncio.CancelledError:
            pass
    adapter.ws_listener_task = None
    adapter.heartbeat_task = None

    # Prepare the WebSocket message
    tradovate_fill_data = {
        "tradeId": "fill-12345",
        "orderId": "order-67890",
        "contractId": 1001,
        "contract": {
            "id": 1001,
            "name": "MESM25",
            "expirationDate": "2025-06-20",
            "cvs": 0.01,
            "tickSize": 0.25,
        },
        "timestamp": "2023-10-26T10:30:00.123Z",
        "action": "Buy",
        "qty": 2,
        "price": 4500.75,
        "active": True,
        "userId": TEST_USER_ID,
        "accountId": adapter._account_id,
        "relatedTradeId": None,
        "commission": 0.85,
    }
    ws_message = f'a[{{"e":"fill","d":{json.dumps(tradovate_fill_data)}}}]'

    if not adapter.ws_connection:
        pytest.fail("connected_adapter did not establish ws_connection")

    # Set up the fill event tracking
    fill_processed = asyncio.Event()
    original_on_trade = mock_live_runner.on_trade

    async def wrapped_on_trade(*args, **kwargs):
        await original_on_trade(*args, **kwargs)
        fill_processed.set()

    mock_live_runner.on_trade = wrapped_on_trade

    # Configure WebSocket mock with our test message
    adapter.ws_connection.recv.side_effect = [ws_message, asyncio.CancelledError()]

    # Start a new listener task
    adapter._stop_event.clear()
    test_listener_task = asyncio.create_task(adapter._ws_listen())

    try:
        # Wait for the fill to be processed
        await asyncio.wait_for(fill_processed.wait(), timeout=2)
    except asyncio.TimeoutError:
        pytest.fail("Fill event was not processed within timeout")
    finally:
        # Clean up
        adapter._stop_event.set()
        if test_listener_task and not test_listener_task.done():
            test_listener_task.cancel()
            try:
                await test_listener_task
            except asyncio.CancelledError:
                pass
        mock_live_runner.on_trade = original_on_trade

    # Verify the fill was processed correctly
    mock_live_runner.on_trade.assert_called_once()
    called_fill_arg = mock_live_runner.on_trade.call_args[0][0]

    assert isinstance(called_fill_arg, Fill)
    assert called_fill_arg.id == "fill-12345"
    assert called_fill_arg.order_id == "order-67890"
    assert called_fill_arg.symbol == "MESM25"
    assert called_fill_arg.qty == 2
    assert called_fill_arg.price == 4500.75
    assert called_fill_arg.side == "buy"
    assert called_fill_arg.timestamp == datetime(
        2023, 10, 26, 10, 30, 0, 123000, tzinfo=timezone.utc
    )
    assert called_fill_arg.commission == 0.85


@pytest.mark.asyncio
async def test_websocket_handles_order_update_event(
    connected_adapter, mock_live_runner
):
    adapter = connected_adapter

    # Stop any existing tasks
    adapter._stop_event.set()
    if adapter.ws_listener_task and not adapter.ws_listener_task.done():
        adapter.ws_listener_task.cancel()
        try:
            await adapter.ws_listener_task
        except asyncio.CancelledError:
            pass
    if adapter.heartbeat_task and not adapter.heartbeat_task.done():
        adapter.heartbeat_task.cancel()
        try:
            await adapter.heartbeat_task
        except asyncio.CancelledError:
            pass
    adapter.ws_listener_task = None
    adapter.heartbeat_task = None

    tradovate_order_data = {
        "orderId": "order-78901",
        "orderStatus": "Filled",
        "action": "Sell",
        "ordQty": 1,
        "orderType": "Limit",
        "price": 4510.00,
        "stopPrice": None,
        "contractId": 1002,
        "contract": {
            "id": 1002,
            "name": "ESU25",
        },
        "timestamp": "2023-10-26T11:00:00.456Z",
        "updatedTimestamp": "2023-10-26T11:00:05.789Z",
        "userId": TEST_USER_ID,
        "accountId": adapter._account_id,
        "avgPx": 4510.00,
        "cumQty": 1,
        "leavesQty": 0,
    }
    ws_message = f'a[{{"e":"order","d":{json.dumps(tradovate_order_data)}}}]'

    if not adapter.ws_connection:
        pytest.fail("connected_adapter did not establish ws_connection")

    # Set up the order update event tracking
    order_processed = asyncio.Event()
    original_on_order_update = mock_live_runner.on_order_update

    async def wrapped_on_order_update(*args, **kwargs):
        await original_on_order_update(*args, **kwargs)
        order_processed.set()

    mock_live_runner.on_order_update = wrapped_on_order_update

    # Configure WebSocket mock with our test message
    adapter.ws_connection.recv.side_effect = [ws_message, asyncio.CancelledError()]

    # Start a new listener task
    adapter._stop_event.clear()
    test_listener_task = asyncio.create_task(adapter._ws_listen())

    try:
        # Wait for the order to be processed
        await asyncio.wait_for(order_processed.wait(), timeout=2)
    except asyncio.TimeoutError:
        pytest.fail("Order event was not processed within timeout")
    finally:
        # Clean up
        adapter._stop_event.set()
        if test_listener_task and not test_listener_task.done():
            test_listener_task.cancel()
            try:
                await test_listener_task
            except asyncio.CancelledError:
                pass
        mock_live_runner.on_order_update = original_on_order_update

    # Verify the order was processed correctly
    mock_live_runner.on_order_update.assert_called_once()
    called_order_arg = mock_live_runner.on_order_update.call_args[0][0]

    assert isinstance(called_order_arg, Order)
    assert called_order_arg.id == "order-78901"
    assert called_order_arg.symbol == "ESU25"
    assert called_order_arg.qty == 1
    assert called_order_arg.side == "sell"
    assert called_order_arg.order_type == "limit"
    assert called_order_arg.status == "filled"
    assert called_order_arg.filled_qty == 1
    assert called_order_arg.avg_fill_price == 4510.00
    assert called_order_arg.limit_price == 4510.00
    assert called_order_arg.created_at == datetime(
        2023, 10, 26, 11, 0, 0, 456000, tzinfo=timezone.utc
    )
    assert called_order_arg.updated_at == datetime(
        2023, 10, 26, 11, 0, 5, 789000, tzinfo=timezone.utc
    )


# --- Order Management Tests ---


@pytest.mark.asyncio
async def test_submit_market_order(connected_adapter, mock_http_client):
    adapter = connected_adapter
    symbol = "MESM25"
    qty = 2.0
    side = "buy"

    # Mock the response from POST /order/placeorder
    # This response is an acknowledgement; full status comes via WebSocket.
    order_submission_response = {
        "orderId": "market-order-id-123",
    }

    # Store original side_effect and reset mock for clean assertion
    original_request_side_effect = mock_http_client.return_value.request.side_effect
    mock_http_client.return_value.request.reset_mock()  # Reset call count and history

    async def custom_request_mock(method, endpoint, headers=None, json=None, **kwargs):
        test_file_logger.debug(f"custom_request_mock called with: {method} {endpoint}")
        if endpoint == "/order/placeorder" and method == "POST":
            mock_response = AsyncMock(status_code=200)
            mock_response.json = MagicMock(return_value=order_submission_response)
            mock_response.raise_for_status = MagicMock()
            return mock_response
        elif endpoint == "/account/list" and method == "GET":
            mock_response = AsyncMock(status_code=200)
            mock_response.json = MagicMock(
                return_value=[
                    {
                        "id": adapter._account_id or 54321,
                        "name": "TestAccount",
                        "balance": 100000,
                        "userId": TEST_USER_ID,
                        "buyingPower": 100000,
                    }
                ]
            )
            mock_response.raise_for_status = MagicMock()
            return mock_response
        test_file_logger.error(
            f"Unexpected call to {method} {endpoint} in custom_request_mock"
        )
        error_response = AsyncMock(status_code=404)
        error_response.json = MagicMock(return_value={"error": "Not Found"})
        error_response.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError(
                "Not Found", request=MagicMock(), response=error_response
            )
        )
        return error_response

    mock_http_client.return_value.request.side_effect = custom_request_mock

    try:
        assert adapter._account_id is not None, (
            "_account_id not set by connected_adapter"
        )

        submitted_order = await adapter.submit_order(
            symbol=symbol, qty=qty, side=side, order_type="market"
        )

        assert submitted_order is not None
        assert submitted_order.id == "market-order-id-123"
        assert submitted_order.symbol == symbol
        assert submitted_order.qty == qty
        assert submitted_order.side == side
        assert submitted_order.order_type == "market"
        assert submitted_order.status == "submitted"  # Initial status from submit_order

        # Verify the HTTP request payload
        expected_payload = {
            "accountId": adapter._account_id,
            "action": "Buy",
            "symbol": symbol,
            "orderQty": int(qty),
            "orderType": "Market",
            "isAutomated": True,
        }

        # _make_request calls self.http_client.request.
        # The mock_http_client fixture provides mock_client_class.return_value (which is mock_client_instance)
        # So we check mock_client_instance.request
        # _make_request adds an Authorization header if a token exists.
        expected_headers = {}
        if adapter.access_token_details and not adapter.access_token_details.is_expired:
            expected_headers["Authorization"] = (
                f"Bearer {adapter.access_token_details.accessToken}"
            )

        mock_http_client.return_value.request.assert_called_once_with(
            "POST", "/order/placeorder", json=expected_payload, headers=expected_headers
        )
    finally:
        # Restore original side_effect to prevent interference with other tests/teardown
        mock_http_client.return_value.request.side_effect = original_request_side_effect


@pytest.mark.asyncio
async def test_submit_limit_order(connected_adapter, mock_http_client):
    adapter = connected_adapter
    symbol = "ESU25"
    qty = 1.0
    side = "sell"
    limit_price = 4550.25

    order_submission_response = {
        "orderId": "limit-order-id-456",
    }

    original_request_side_effect = mock_http_client.return_value.request.side_effect
    mock_http_client.return_value.request.reset_mock()

    async def custom_request_mock(method, endpoint, headers=None, json=None, **kwargs):
        test_file_logger.debug(
            f"custom_request_mock (limit) called with: {method} {endpoint}"
        )
        if endpoint == "/order/placeorder" and method == "POST":
            mock_response = AsyncMock(status_code=200)
            mock_response.json = MagicMock(return_value=order_submission_response)
            mock_response.raise_for_status = MagicMock()
            return mock_response
        elif endpoint == "/account/list" and method == "GET":
            mock_response = AsyncMock(status_code=200)
            mock_response.json = MagicMock(
                return_value=[
                    {
                        "id": adapter._account_id or 54321,
                        "name": "TestAccount",
                        "balance": 100000,
                        "userId": TEST_USER_ID,
                        "buyingPower": 100000,
                    }
                ]
            )
            mock_response.raise_for_status = MagicMock()
            return mock_response
        test_file_logger.error(
            f"Unexpected call to {method} {endpoint} in custom_request_mock (limit)"
        )
        error_response = AsyncMock(status_code=404)
        error_response.json = MagicMock(return_value={"error": "Not Found"})
        error_response.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError(
                "Not Found", request=MagicMock(), response=error_response
            )
        )
        return error_response

    mock_http_client.return_value.request.side_effect = custom_request_mock

    try:
        assert adapter._account_id is not None, (
            "_account_id not set by connected_adapter"
        )

        submitted_order = await adapter.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            order_type="limit",
            limit_price=limit_price,
        )

        assert submitted_order is not None
        assert submitted_order.id == "limit-order-id-456"
        assert submitted_order.symbol == symbol
        assert submitted_order.qty == qty
        assert submitted_order.side == side
        assert submitted_order.order_type == "limit"
        assert submitted_order.status == "submitted"
        assert submitted_order.limit_price == limit_price

        expected_payload = {
            "accountId": adapter._account_id,
            "action": "Sell",
            "symbol": symbol,
            "orderQty": int(qty),
            "orderType": "Limit",
            "isAutomated": True,
            "price": limit_price,
        }
        expected_headers = {}
        if adapter.access_token_details and not adapter.access_token_details.is_expired:
            expected_headers["Authorization"] = (
                f"Bearer {adapter.access_token_details.accessToken}"
            )

        mock_http_client.return_value.request.assert_called_once_with(
            "POST", "/order/placeorder", json=expected_payload, headers=expected_headers
        )
    finally:
        mock_http_client.return_value.request.side_effect = original_request_side_effect


@pytest.mark.asyncio
async def test_submit_stop_order(connected_adapter, mock_http_client):
    adapter = connected_adapter
    symbol = "MNQH24"
    qty = 3.0
    side = "buy"
    stop_price = 18000.00

    order_submission_response = {
        "orderId": "stop-order-id-789",
    }

    original_request_side_effect = mock_http_client.return_value.request.side_effect
    mock_http_client.return_value.request.reset_mock()

    async def custom_request_mock(method, endpoint, headers=None, json=None, **kwargs):
        test_file_logger.debug(
            f"custom_request_mock (stop) called with: {method} {endpoint}"
        )
        if endpoint == "/order/placeorder" and method == "POST":
            mock_response = AsyncMock(status_code=200)
            mock_response.json = MagicMock(return_value=order_submission_response)
            mock_response.raise_for_status = MagicMock()
            return mock_response
        elif endpoint == "/account/list" and method == "GET":
            mock_response = AsyncMock(status_code=200)
            mock_response.json = MagicMock(
                return_value=[
                    {
                        "id": adapter._account_id or 54321,
                        "name": "TestAccount",
                        "balance": 100000,
                        "userId": TEST_USER_ID,
                        "buyingPower": 100000,
                    }
                ]
            )
            mock_response.raise_for_status = MagicMock()
            return mock_response
        test_file_logger.error(
            f"Unexpected call to {method} {endpoint} in custom_request_mock (stop)"
        )
        error_response = AsyncMock(status_code=404)
        error_response.json = MagicMock(return_value={"error": "Not Found"})
        error_response.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError(
                "Not Found", request=MagicMock(), response=error_response
            )
        )
        return error_response

    mock_http_client.return_value.request.side_effect = custom_request_mock

    try:
        assert adapter._account_id is not None, (
            "_account_id not set by connected_adapter"
        )

        submitted_order = await adapter.submit_order(
            symbol=symbol, qty=qty, side=side, order_type="stop", stop_price=stop_price
        )

        assert submitted_order is not None
        assert submitted_order.id == "stop-order-id-789"
        assert submitted_order.symbol == symbol
        assert submitted_order.qty == qty
        assert submitted_order.side == side
        assert submitted_order.order_type == "stop"
        assert submitted_order.status == "submitted"
        assert submitted_order.stop_price == stop_price

        expected_payload = {
            "accountId": adapter._account_id,
            "action": "Buy",
            "symbol": symbol,
            "orderQty": int(qty),
            "orderType": "Stop",  # Note: submit_order capitalizes this, but Tradovate might expect "StopMarket"
            "isAutomated": True,
            "stopPrice": stop_price,
        }
        expected_headers = {}
        if adapter.access_token_details and not adapter.access_token_details.is_expired:
            expected_headers["Authorization"] = (
                f"Bearer {adapter.access_token_details.accessToken}"
            )

        mock_http_client.return_value.request.assert_called_once_with(
            "POST", "/order/placeorder", json=expected_payload, headers=expected_headers
        )
    finally:
        mock_http_client.return_value.request.side_effect = original_request_side_effect


@pytest.mark.asyncio
async def test_cancel_order_successful(connected_adapter, mock_http_client):
    adapter = connected_adapter
    order_id_to_cancel = "order-to-cancel-123"

    # No specific response body is usually expected for a successful DELETE, status code is key.
    # The .json() method might not even be called by the adapter for DELETE if it checks status.
    # However, our mock setup for httpx.AsyncClient provides a .json by default.
    # Let's ensure the mock response for DELETE is simple.

    original_request_side_effect = mock_http_client.return_value.request.side_effect
    mock_http_client.return_value.request.reset_mock()

    async def custom_request_mock(method, endpoint, headers=None, json=None, **kwargs):
        test_file_logger.debug(
            f"custom_request_mock (cancel) called with: {method} {endpoint}"
        )
        if (
            endpoint == f"/order/cancelorder/{order_id_to_cancel}"
            and method == "DELETE"
        ):
            mock_response = AsyncMock(status_code=200)  # Or 204 No Content
            mock_response.json = MagicMock(
                return_value=None
            )  # Tradovate might return empty or minimal JSON
            mock_response.raise_for_status = MagicMock()
            return mock_response
        elif (
            endpoint == "/account/list" and method == "GET"
        ):  # Fallback from connected_adapter
            mock_response = AsyncMock(status_code=200)
            mock_response.json = MagicMock(
                return_value=[
                    {
                        "id": adapter._account_id or 54321,
                        "name": "TestAccount",
                        "balance": 100000,
                        "userId": TEST_USER_ID,
                        "buyingPower": 100000,
                    }
                ]
            )
            mock_response.raise_for_status = MagicMock()
            return mock_response
        test_file_logger.error(
            f"Unexpected call to {method} {endpoint} in custom_request_mock (cancel)"
        )
        error_response = AsyncMock(status_code=404)
        error_response.json = MagicMock(
            return_value={"error": "Not Found for cancel test"}
        )
        error_response.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError(
                "Not Found", request=MagicMock(), response=error_response
            )
        )
        return error_response

    mock_http_client.return_value.request.side_effect = custom_request_mock

    try:
        assert adapter.access_token_details is not None, (
            "Access token not set by connected_adapter"
        )

        result = await adapter.cancel_order(order_id_to_cancel)

        assert result is True

        expected_headers = {}
        if adapter.access_token_details and not adapter.access_token_details.is_expired:
            expected_headers["Authorization"] = (
                f"Bearer {adapter.access_token_details.accessToken}"
            )

        # For DELETE, there's no json payload in the call itself.
        mock_http_client.return_value.request.assert_called_once_with(
            "DELETE",
            f"/order/cancelorder/{order_id_to_cancel}",
            headers=expected_headers,
        )
    finally:
        mock_http_client.return_value.request.side_effect = original_request_side_effect


@pytest.mark.asyncio
async def test_cancel_order_fails(connected_adapter, mock_http_client):
    adapter = connected_adapter
    order_id_to_cancel = "order-that-fails-cancel-456"

    original_request_side_effect = mock_http_client.return_value.request.side_effect
    mock_http_client.return_value.request.reset_mock()

    # This will be the mock HTTP response object for the failed call
    failed_http_response_mock = AsyncMock(
        status_code=404,  # Example: Not Found
    )
    failed_http_response_mock.json = MagicMock(
        return_value={"error": "Order not found"}
    )
    # Configure raise_for_status to actually raise an error, as _make_request would expect
    http_error = httpx.HTTPStatusError(
        message="404 Not Found", request=MagicMock(), response=failed_http_response_mock
    )
    failed_http_response_mock.raise_for_status = MagicMock(side_effect=http_error)

    async def custom_request_mock(method, endpoint, headers=None, json=None, **kwargs):
        test_file_logger.debug(
            f"custom_request_mock (cancel_fails) called with: {method} {endpoint}"
        )
        if (
            endpoint == f"/order/cancelorder/{order_id_to_cancel}"
            and method == "DELETE"
        ):
            return failed_http_response_mock  # Return the pre-configured failed response mock
        elif (
            endpoint == "/account/list" and method == "GET"
        ):  # Fallback from connected_adapter
            mock_response = AsyncMock(status_code=200)
            mock_response.json = MagicMock(
                return_value=[
                    {
                        "id": adapter._account_id or 54321,
                        "name": "TestAccount",
                        "balance": 100000,
                        "userId": TEST_USER_ID,
                        "buyingPower": 100000,
                    }
                ]
            )
            mock_response.raise_for_status = MagicMock()
            return mock_response
        test_file_logger.error(
            f"Unexpected call to {method} {endpoint} in custom_request_mock (cancel_fails)"
        )
        # Fallback for genuinely unexpected calls
        unhandled_error_response = AsyncMock(status_code=500)
        unhandled_error_response.json = MagicMock(
            return_value={"error": "Server error in test mock"}
        )
        unhandled_error_response.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError(
                "Server error", request=MagicMock(), response=unhandled_error_response
            )
        )
        return unhandled_error_response

    mock_http_client.return_value.request.side_effect = custom_request_mock

    # --- Store original asyncio.sleep before patching ---
    original_asyncio_sleep = asyncio.sleep

    async def mock_actual_short_sleep(delay):
        # Call the original asyncio.sleep for a very short duration
        await original_asyncio_sleep(0.001)

    with patch("asyncio.sleep", new=mock_actual_short_sleep):
        try:
            result = await adapter.cancel_order(order_id_to_cancel)
            assert result is False  # Expecting False on failure

            expected_headers = {}
            if (
                adapter.access_token_details
                and not adapter.access_token_details.is_expired
            ):
                expected_headers["Authorization"] = (
                    f"Bearer {adapter.access_token_details.accessToken}"
                )

            # Check call count due to tenacity retries
            assert (
                mock_http_client.return_value.request.call_count == 4
            )  # stop_after_attempt(4)

            # Check that any of the calls were made with the correct parameters
            mock_http_client.return_value.request.assert_any_call(
                "DELETE",
                f"/order/cancelorder/{order_id_to_cancel}",
                headers=expected_headers,
            )
        finally:
            mock_http_client.return_value.request.side_effect = (
                original_request_side_effect
            )
            # asyncio.sleep is restored automatically by with patch exiting


# --- Data Retrieval Tests ---


@pytest.mark.asyncio
async def test_get_cash_successful(connected_adapter, mock_http_client):
    adapter = connected_adapter
    expected_cash_balance = 12345.67
    expected_buying_power = 50000.00  # Well above margin guard threshold

    account_list_response = [
        {
            "id": adapter._account_id,
            "name": "TestAccount",
            "balance": expected_cash_balance,
            "userId": TEST_USER_ID,
            "buyingPower": expected_buying_power,
            "currencyId": 1,
            "active": True,
            "accountType": "Futures",
        },
        {
            "id": 99999,
            "name": "OtherAccount",
            "balance": 500.00,
            "userId": TEST_USER_ID,
            "buyingPower": 1000.00,
            "currencyId": 1,
            "active": True,
            "accountType": "Futures",
        },  # Another account to ensure correct filtering
    ]

    original_request_side_effect = mock_http_client.return_value.request.side_effect
    mock_http_client.return_value.request.reset_mock()

    async def custom_request_mock(method, endpoint, headers=None, json=None, **kwargs):
        test_file_logger.debug(
            f"custom_request_mock (get_cash) called with: {method} {endpoint}"
        )
        if endpoint == "/account/list" and method == "GET":
            mock_response = AsyncMock(status_code=200)
            mock_response.json = MagicMock(return_value=account_list_response)
            mock_response.raise_for_status = MagicMock()
            return mock_response
        test_file_logger.error(
            f"Unexpected call to {method} {endpoint} in custom_request_mock (get_cash)"
        )
        error_response = AsyncMock(status_code=404)
        error_response.json = MagicMock(
            return_value={"error": "Not Found for get_cash test"}
        )
        error_response.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError(
                "Not Found", request=MagicMock(), response=error_response
            )
        )
        return error_response

    mock_http_client.return_value.request.side_effect = custom_request_mock

    try:
        assert adapter._account_id is not None, (
            "_account_id not set by connected_adapter"
        )

        # Clear previous cash to ensure it's updated by the call
        adapter._cash = {}  # Initialize as empty dict

        retrieved_cash_dict = await adapter.get_cash()

        expected_cash_dict = {"USD": expected_cash_balance}
        assert retrieved_cash_dict == expected_cash_dict
        assert adapter._cash == expected_cash_dict

        expected_headers = {}
        if adapter.access_token_details and not adapter.access_token_details.is_expired:
            expected_headers["Authorization"] = (
                f"Bearer {adapter.access_token_details.accessToken}"
            )

        mock_http_client.return_value.request.assert_called_once_with(
            "GET",
            "/account/list",
            headers=expected_headers,  # _make_request adds headers
        )
    finally:
        mock_http_client.return_value.request.side_effect = original_request_side_effect


@pytest.mark.asyncio
async def test_get_cash_margin_guard_warning(
    connected_adapter, mock_http_client, caplog
):
    adapter = connected_adapter
    cash_balance = 1000.00
    low_buying_power = 40.00  # Below $50 threshold

    account_list_response = [
        {
            "id": adapter._account_id,
            "name": "TestAccount",
            "balance": cash_balance,
            "userId": TEST_USER_ID,
            "buyingPower": low_buying_power,
            "currencyId": 1,
            "active": True,
            "accountType": "Futures",
        }
    ]

    original_request_side_effect = mock_http_client.return_value.request.side_effect
    mock_http_client.return_value.request.reset_mock()

    async def custom_request_mock(method, endpoint, headers=None, json=None, **kwargs):
        if endpoint == "/account/list" and method == "GET":
            mock_response = AsyncMock(status_code=200)
            mock_response.json = MagicMock(return_value=account_list_response)
            mock_response.raise_for_status = MagicMock()
            return mock_response
        error_response = AsyncMock(status_code=404)
        error_response.json = MagicMock(
            return_value={"error": "Not Found for get_cash margin test"}
        )
        error_response.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError(
                "Not Found", request=MagicMock(), response=error_response
            )
        )
        return error_response

    mock_http_client.return_value.request.side_effect = custom_request_mock

    try:
        with caplog.at_level(
            logging.WARNING, logger="algo_mvp.live.adapters.tradovate"
        ):
            await adapter.get_cash()

        assert adapter._cash == {"USD": cash_balance}  # Cash should still be updated

        expected_log_message = f"Margin Alert: Account {adapter._account_id} buying power ({low_buying_power} USD) is below $50."
        assert any(expected_log_message in message for message in caplog.messages)
        # Verify the HTTP call was made
        mock_http_client.return_value.request.assert_called_once_with(
            "GET", "/account/list", headers=ANY
        )  # Use ANY without mock.

    finally:
        mock_http_client.return_value.request.side_effect = original_request_side_effect


@pytest.mark.asyncio
async def test_get_cash_fetches_account_id_if_not_set(adapter, mock_http_client):
    # Use the basic 'adapter' fixture which is not auto-connected
    # Ensure _account_id is None initially
    adapter._account_id = None
    adapter._cash = {}  # Start with empty cash

    expected_account_id = 12345  # A different ID for this test
    expected_cash_balance = 777.77

    # Mock the token response for the _get_access_token call
    # This will be triggered by _make_request when it sees no token.
    token_response_dict = {
        "accessToken": "token_for_get_cash_test",
        "userId": TEST_USER_ID,  # Ensure TEST_USER_ID matches account
        "userStatus": "Authorized",
        "expirationTime": (datetime.now(timezone.utc) + timedelta(hours=1))
        .isoformat()
        .replace("+00:00", "Z"),
        "name": TEST_USERNAME,
        "hasLive": False,
    }
    # Ensure the POST mock (used by _get_access_token) is ready BEFORE any call that might trigger it.
    mock_http_client.return_value.post.return_value.status_code = 200
    mock_http_client.return_value.post.return_value.json.return_value = (
        token_response_dict
    )
    mock_http_client.return_value.post.return_value.raise_for_status = (
        MagicMock()
    )  # Ensure no error on post

    # This will be the response for the _get_initial_account_data call within get_cash
    initial_account_list_response = [
        {
            "id": expected_account_id,
            "name": "FetchedAccount",
            "balance": expected_cash_balance,
            "userId": TEST_USER_ID,
            "buyingPower": 10000,
            "currencyId": 1,
            "active": True,
            "accountType": "Futures",
        }
    ]
    # This could be the response for a subsequent call if get_cash called /account/list again, but it shouldn't if _get_initial_account_data sets _cash.
    # For this test, _get_initial_account_data sets _cash, so get_cash might not need to make its own /account/list call.
    # The current get_cash implementation calls /account/list regardless of _get_initial_account_data.

    original_request_side_effect = mock_http_client.return_value.request.side_effect
    mock_http_client.return_value.request.reset_mock()

    # We expect two calls to /account/list if _account_id is None:
    # 1. Inside _get_initial_account_data (triggered by get_cash)
    # 2. Inside get_cash itself after _account_id is set.
    # Both will use the same mock response for this test for simplicity.
    async def custom_request_mock(method, endpoint, headers=None, json=None, **kwargs):
        if endpoint == "/account/list" and method == "GET":
            mock_response = AsyncMock(status_code=200)
            mock_response.json = MagicMock(return_value=initial_account_list_response)
            mock_response.raise_for_status = MagicMock()
            return mock_response
        # Fallback for other unexpected calls
        error_response = AsyncMock(status_code=404)
        error_response.json = MagicMock(return_value={"error": "Not Found"})
        error_response.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError(
                "Not Found", request=MagicMock(), response=error_response
            )
        )
        return error_response

    mock_http_client.return_value.request.side_effect = custom_request_mock

    try:
        # Call get_cash when _account_id is None
        retrieved_cash = await adapter.get_cash()

        assert adapter._account_id == expected_account_id
        assert retrieved_cash == {"USD": expected_cash_balance}
        assert adapter._cash == {"USD": expected_cash_balance}

        # Assert that /account/list was called (at least once, could be twice)
        # _get_initial_account_data calls it, then get_cash calls it again.
        assert mock_http_client.return_value.request.call_count >= 1
        mock_http_client.return_value.request.assert_any_call(
            "GET", "/account/list", headers=ANY
        )

    finally:
        mock_http_client.return_value.request.side_effect = original_request_side_effect
        # Clean up tasks if adapter.connect was implicitly called via _get_initial_account_data needing a token
        # However, this test uses the 'adapter' fixture which doesn't auto-connect or get token initially.
        # _get_initial_account_data will call _make_request which tries to get token if needed.
        # For this test, assume token exists or _get_access_token is mocked if it was triggered.
        # The adapter.close() in the fixture should handle task cleanup if any were started.


@pytest.mark.asyncio
async def test_get_cash_fails_to_fetch_account_list(
    connected_adapter, mock_http_client, caplog
):
    adapter = connected_adapter
    # adapter._account_id is set by connected_adapter
    # Explicitly set adapter._cash to ensure it's a dict for this test, overriding any prior incorrect state.
    # This mirrors the expected state after connected_adapter's connect() call if successful.
    adapter._cash = {"USD": 100000.0}
    initial_cash_state = adapter._cash.copy()  # Capture initial state

    original_request_side_effect = mock_http_client.return_value.request.side_effect
    mock_http_client.return_value.request.reset_mock()

    http_error = httpx.HTTPStatusError(
        "Server Error", request=MagicMock(), response=AsyncMock(status_code=500)
    )

    async def custom_request_mock(method, endpoint, headers=None, json=None, **kwargs):
        if endpoint == "/account/list" and method == "GET":
            # Simulate failure for the /account/list call
            mock_response = AsyncMock(status_code=500)
            mock_response.json = MagicMock(
                return_value={"error": "Internal Server Error"}
            )
            mock_response.raise_for_status = MagicMock(side_effect=http_error)
            return mock_response
        # Fallback for other unexpected calls (should not happen in this specific test flow for get_cash)
        error_response = AsyncMock(status_code=404)
        error_response.json = MagicMock(return_value={"error": "Not Found"})
        error_response.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError(
                "Not Found", request=MagicMock(), response=error_response
            )
        )
        return error_response

    mock_http_client.return_value.request.side_effect = custom_request_mock

    # --- Store original asyncio.sleep before patching ---
    original_asyncio_sleep = asyncio.sleep

    async def mock_actual_short_sleep(delay):
        # Call the original asyncio.sleep for a very short duration
        await original_asyncio_sleep(0.001)

    with patch("asyncio.sleep", new=mock_actual_short_sleep):
        try:
            with caplog.at_level(
                logging.ERROR, logger="algo_mvp.live.adapters.tradovate"
            ):
                retrieved_cash = await adapter.get_cash()

            # Should return the initial cash state because the new fetch failed
            assert retrieved_cash == initial_cash_state
            assert (
                adapter._cash == initial_cash_state
            )  # Internal state should also remain unchanged

            assert any(
                "Failed to get cash balance" in message for message in caplog.messages
            )

            # Check that /account/list was called (and retried by tenacity)
            assert (
                mock_http_client.return_value.request.call_count == 4
            )  # stop_after_attempt(4)
            mock_http_client.return_value.request.assert_any_call(
                "GET", "/account/list", headers=ANY
            )

        finally:
            mock_http_client.return_value.request.side_effect = (
                original_request_side_effect
            )
            # asyncio.sleep is restored automatically by with patch exiting


@pytest.mark.asyncio
async def test_get_positions_successful(connected_adapter, mock_http_client):
    adapter = connected_adapter
    assert adapter._account_id is not None, (
        "_account_id not set by connected_adapter fixture"
    )

    tradovate_position_data = [
        {
            "accountId": adapter._account_id,
            "contractId": 12345,
            "timestamp": "2023-10-27T10:00:00Z",
            "netPos": 5,
            "avgEntryPrice": 4500.25,
            "contract": {"id": 12345, "name": "ESZ23"},
        },
        {
            "accountId": adapter._account_id,
            "contractId": 67890,
            "timestamp": "2023-10-27T10:05:00Z",
            "netPos": -2,
            "avgEntryPrice": 150.75,
            "contract": {"id": 67890, "name": "CLF24"},
        },
        {
            "accountId": 99999,  # Different account
            "contractId": 11111,
            "timestamp": "2023-10-27T10:05:00Z",
            "netPos": 10,
            "avgEntryPrice": 1.00,
            "contract": {"id": 11111, "name": "OTHERACC"},
        },
    ]

    original_request_side_effect = mock_http_client.return_value.request.side_effect
    mock_http_client.return_value.request.reset_mock()

    async def custom_request_mock(method, endpoint, headers=None, json=None, **kwargs):
        if endpoint == "/position/list" and method == "GET":
            mock_response = AsyncMock(status_code=200)
            mock_response.json = MagicMock(return_value=tradovate_position_data)
            mock_response.raise_for_status = MagicMock()
            return mock_response
        # Fallback for other calls like /account/list if connect() is somehow re-triggered
        elif endpoint == "/account/list" and method == "GET":
            mock_response = AsyncMock(status_code=200)
            mock_response.json = MagicMock(
                return_value=[
                    {
                        "id": adapter._account_id,
                        "name": "TestAccount",
                        "balance": 100000,
                        "userId": TEST_USER_ID,
                        "buyingPower": 100000,
                    }
                ]
            )
            mock_response.raise_for_status = MagicMock()
            return mock_response
        error_response = AsyncMock(status_code=404)
        error_response.json = MagicMock(
            return_value={"error": "Not Found in get_positions test"}
        )
        error_response.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError(
                "Not Found", request=MagicMock(), response=error_response
            )
        )
        return error_response

    mock_http_client.return_value.request.side_effect = custom_request_mock

    try:
        retrieved_positions = await adapter.get_positions()

        assert len(retrieved_positions) == 2
        assert adapter._positions == retrieved_positions

        pos1 = next(p for p in retrieved_positions if p.symbol == "ESZ23")
        assert pos1.qty == 5
        assert pos1.avg_entry_price == 4500.25

        pos2 = next(p for p in retrieved_positions if p.symbol == "CLF24")
        assert pos2.qty == -2
        assert pos2.avg_entry_price == 150.75

        expected_headers = {}
        if adapter.access_token_details and not adapter.access_token_details.is_expired:
            expected_headers["Authorization"] = (
                f"Bearer {adapter.access_token_details.accessToken}"
            )

        mock_http_client.return_value.request.assert_called_once_with(
            "GET", "/position/list", headers=expected_headers
        )
    finally:
        mock_http_client.return_value.request.side_effect = original_request_side_effect


@pytest.mark.asyncio
async def test_get_positions_empty(connected_adapter, mock_http_client):
    adapter = connected_adapter
    assert adapter._account_id is not None

    empty_position_data = [
        {
            "accountId": 99999,  # Position for another account
            "contractId": 11111,
            "timestamp": "2023-10-27T10:05:00Z",
            "netPos": 10,
            "avgEntryPrice": 1.00,
            "contract": {"id": 11111, "name": "OTHERACC"},
        }
    ]
    # Or simply an empty list if the API returns nothing for the user,
    # and the adapter filters by account_id correctly
    # empty_position_data = []

    original_request_side_effect = mock_http_client.return_value.request.side_effect
    mock_http_client.return_value.request.reset_mock()

    async def custom_request_mock(method, endpoint, headers=None, json=None, **kwargs):
        if endpoint == "/position/list" and method == "GET":
            mock_response = AsyncMock(status_code=200)
            mock_response.json = MagicMock(
                return_value=empty_position_data
            )  # Could also be just []
            mock_response.raise_for_status = MagicMock()
            return mock_response
        elif endpoint == "/account/list" and method == "GET":
            mock_response = AsyncMock(status_code=200)
            mock_response.json = MagicMock(
                return_value=[
                    {
                        "id": adapter._account_id,
                        "name": "TestAccount",
                        "balance": 100000,
                        "userId": TEST_USER_ID,
                        "buyingPower": 100000,
                    }
                ]
            )
            mock_response.raise_for_status = MagicMock()
            return mock_response
        error_response = AsyncMock(status_code=404)
        error_response.json = MagicMock(return_value={"error": "Not Found"})
        error_response.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError(
                "Not Found", request=MagicMock(), response=error_response
            )
        )
        return error_response

    mock_http_client.return_value.request.side_effect = custom_request_mock

    try:
        adapter._positions = [
            Position(symbol="dummy", qty=1, avg_entry_price=100)
        ]  # Pre-set some positions
        retrieved_positions = await adapter.get_positions()

        assert len(retrieved_positions) == 0
        assert adapter._positions == retrieved_positions
        assert adapter._positions == []

        mock_http_client.return_value.request.assert_called_once_with(
            "GET", "/position/list", headers=ANY
        )
    finally:
        mock_http_client.return_value.request.side_effect = original_request_side_effect


@pytest.mark.asyncio
async def test_get_positions_api_error(connected_adapter, mock_http_client, caplog):
    adapter = connected_adapter
    assert adapter._account_id is not None

    # Pre-set some positions to test if they are preserved on API error
    initial_positions = [
        Position(symbol="ESZ23", qty=2, avg_entry_price=4500.00),
        Position(symbol="NQU23", qty=-1, avg_entry_price=15000.00),
    ]
    adapter._positions = initial_positions.copy()

    original_request_side_effect = mock_http_client.return_value.request.side_effect
    mock_http_client.return_value.request.reset_mock()

    http_error = httpx.HTTPStatusError(
        "Server Error", request=MagicMock(), response=AsyncMock(status_code=500)
    )

    async def custom_request_mock(method, endpoint, headers=None, json=None, **kwargs):
        if endpoint == "/position/list" and method == "GET":
            mock_response = AsyncMock(status_code=500)
            mock_response.json = MagicMock(
                return_value={"error": "Internal Server Error"}
            )
            mock_response.raise_for_status = MagicMock(side_effect=http_error)
            return mock_response
        elif endpoint == "/account/list" and method == "GET":
            mock_response = AsyncMock(status_code=200)
            mock_response.json = MagicMock(
                return_value=[
                    {
                        "id": adapter._account_id,
                        "name": "TestAccount",
                        "balance": 100000,
                        "userId": TEST_USER_ID,
                        "buyingPower": 100000,
                    }
                ]
            )
            mock_response.raise_for_status = MagicMock()
            return mock_response
        error_response = AsyncMock(status_code=404)
        error_response.json = MagicMock(return_value={"error": "Not Found"})
        error_response.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError(
                "Not Found", request=MagicMock(), response=error_response
            )
        )
        return error_response

    mock_http_client.return_value.request.side_effect = custom_request_mock

    try:
        with caplog.at_level(logging.ERROR, logger="algo_mvp.live.adapters.tradovate"):
            retrieved_positions = await adapter.get_positions()

        assert (
            retrieved_positions == initial_positions
        )  # Should return cached positions
        assert (
            adapter._positions == initial_positions
        )  # Internal state should remain unchanged

        assert any("Failed to get positions" in message for message in caplog.messages)

        # Check that /position/list was called (and retried by tenacity)
        assert (
            mock_http_client.return_value.request.call_count == 4
        )  # stop_after_attempt(4) in _make_request
        mock_http_client.return_value.request.assert_any_call(
            "GET", "/position/list", headers=ANY
        )
    finally:
        mock_http_client.return_value.request.side_effect = original_request_side_effect


@pytest.mark.asyncio
async def test_get_positions_fetches_account_id_if_not_set(
    adapter, mock_http_client, caplog
):
    # Use the basic 'adapter' fixture which is not auto-connected
    adapter._account_id = None  # Ensure it starts as None
    adapter._positions = []  # Start with empty positions

    expected_account_id = 67890
    tradovate_position_data = [
        {
            "accountId": expected_account_id,
            "contractId": 123,
            "netPos": 10,
            "avgEntryPrice": 100.0,
            "contract": {"id": 123, "name": "SYMTEST"},
        }
    ]

    # Mock token response (needed if _get_initial_account_data triggers _make_request -> _get_access_token)
    token_response_dict = {
        "accessToken": "token_for_pos_test",
        "userId": TEST_USER_ID,
        "expirationTime": (datetime.now(timezone.utc) + timedelta(hours=1))
        .isoformat()
        .replace("+00:00", "Z"),
        "userStatus": "Authorized",
        "name": TEST_USERNAME,
        "hasLive": False,
    }
    mock_http_client.return_value.post.return_value.status_code = 200
    mock_http_client.return_value.post.return_value.json.return_value = (
        token_response_dict
    )
    mock_http_client.return_value.post.return_value.raise_for_status = (
        MagicMock()
    )  # Ensure no error on post

    # Mock account list response (for _get_initial_account_data)
    account_list_response = [
        {
            "id": expected_account_id,
            "name": "FetchedAcc",
            "balance": 50000,
            "userId": TEST_USER_ID,
            "buyingPower": 50000,
        }
    ]

    original_request_side_effect = mock_http_client.return_value.request.side_effect
    mock_http_client.return_value.request.reset_mock()

    # We expect calls to /account/list (to get account_id) and then /position/list
    async def custom_request_mock(method, endpoint, headers=None, json=None, **kwargs):
        if endpoint == "/account/list" and method == "GET":
            mock_response = AsyncMock(status_code=200)
            mock_response.json = MagicMock(return_value=account_list_response)
            mock_response.raise_for_status = MagicMock()
            return mock_response
        elif endpoint == "/position/list" and method == "GET":
            mock_response = AsyncMock(status_code=200)
            # Ensure that this mock is only called *after* _account_id would have been set
            assert adapter._account_id == expected_account_id, (
                "_account_id not set before /position/list call"
            )
            mock_response.json = MagicMock(return_value=tradovate_position_data)
            mock_response.raise_for_status = MagicMock()
            return mock_response
        error_response = AsyncMock(status_code=404)
        error_response.json = MagicMock(return_value={"error": "Not Found"})
        error_response.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError(
                "Not Found", request=MagicMock(), response=error_response
            )
        )
        return error_response

    mock_http_client.return_value.request.side_effect = custom_request_mock

    try:
        retrieved_positions = await adapter.get_positions()

        assert adapter._account_id == expected_account_id
        assert len(retrieved_positions) == 1
        assert retrieved_positions[0].symbol == "SYMTEST"
        assert retrieved_positions[0].qty == 10
        assert adapter._positions == retrieved_positions

        # Check calls: at least one to /account/list and one to /position/list
        # Due to tenacity, counts could be higher if there were (mocked) failures initially
        # For this happy path, expect one of each after token is set.
        account_list_call = any(
            call.args == ("GET", "/account/list")
            and isinstance(call.kwargs.get("headers"), dict)
            for call in mock_http_client.return_value.request.call_args_list
        )
        position_list_call = any(
            call.args == ("GET", "/position/list")
            and isinstance(call.kwargs.get("headers"), dict)
            for call in mock_http_client.return_value.request.call_args_list
        )
        assert account_list_call, "GET /account/list was not called correctly"
        assert position_list_call, "GET /position/list was not called correctly"

    finally:
        mock_http_client.return_value.request.side_effect = original_request_side_effect
        # Clean up tasks if adapter.connect was implicitly called via _get_initial_account_data needing a token
        # However, this test uses the 'adapter' fixture which doesn't auto-connect or get token initially.
        # _get_initial_account_data will call _make_request which tries to get token if needed.
        # For this test, assume token exists or _get_access_token is mocked if it was triggered.
        # The adapter.close() in the fixture should handle task cleanup if any were started.


# More tests to be added...
