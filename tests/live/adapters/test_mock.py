import pytest

from algo_mvp.live.adapters.mock import MockBrokerAdapter


@pytest.fixture
def mock_adapter():
    return MockBrokerAdapter()


@pytest.mark.asyncio
async def test_mock_adapter_initial_state(mock_adapter: MockBrokerAdapter):
    # Check individual counters are zero initially
    assert mock_adapter.submit_order_calls == 0
    assert mock_adapter.cancel_order_calls == 0
    assert mock_adapter.get_cash_calls == 0
    assert mock_adapter.get_positions_calls == 0

    assert await mock_adapter.get_cash() == {"USD": 100000.0}
    assert await mock_adapter.get_positions() == []
    # After the calls above, counts for get_cash and get_positions will be 1
    assert mock_adapter.get_cash_calls == 1
    assert mock_adapter.get_positions_calls == 1


@pytest.mark.asyncio
async def test_mock_adapter_submit_buy_order(mock_adapter: MockBrokerAdapter):
    initial_cash_calls = mock_adapter.get_cash_calls  # Track before action
    initial_submit_calls = mock_adapter.submit_order_calls

    initial_cash_dict = await mock_adapter.get_cash()
    initial_cash_value = initial_cash_dict["USD"]
    symbol = "TEST/USD"
    qty = 10
    limit_price = 150.0

    order_response = await mock_adapter.submit_order(
        symbol, qty, "buy", "limit", limit_price
    )

    assert order_response["id"].startswith("mock_order_id_")
    assert order_response["status"] == "filled"
    assert mock_adapter.submit_order_calls == initial_submit_calls + 1

    current_cash_dict = await mock_adapter.get_cash()
    assert current_cash_dict["USD"] == initial_cash_value - (qty * limit_price)

    positions_list = await mock_adapter.get_positions()
    found_position = next((p for p in positions_list if p.symbol == symbol), None)
    assert found_position is not None, f"Position for {symbol} not found"
    assert found_position.qty == qty

    assert (
        mock_adapter.get_cash_calls == initial_cash_calls + 2
    )  # one for initial_cash, one for assert


@pytest.mark.asyncio
async def test_mock_adapter_submit_sell_order(mock_adapter: MockBrokerAdapter):
    symbol = "TEST/USD"
    qty_to_buy = 20
    buy_price = 100.0
    # First, establish a position
    await mock_adapter.submit_order(symbol, qty_to_buy, "buy", "limit", buy_price)

    # Reset/track counts after setup for this specific test section
    mock_adapter.submit_order_calls = 0  # Reset for this part of test
    mock_adapter.get_cash_calls = 0  # Reset
    mock_adapter.get_positions_calls = 0  # Reset

    cash_after_buy_dict = await mock_adapter.get_cash()
    cash_after_buy_value = cash_after_buy_dict["USD"]

    qty_to_sell = 5
    sell_price = 110.0
    order_response = await mock_adapter.submit_order(
        symbol, qty_to_sell, "sell", "limit", sell_price
    )

    assert order_response["status"] == "filled"
    assert mock_adapter.submit_order_calls == 1

    current_cash_dict = await mock_adapter.get_cash()
    assert current_cash_dict["USD"] == cash_after_buy_value + (qty_to_sell * sell_price)

    positions_list = await mock_adapter.get_positions()
    found_position = next((p for p in positions_list if p.symbol == symbol), None)
    assert found_position is not None, f"Position for {symbol} not found"
    assert found_position.qty == qty_to_buy - qty_to_sell

    assert mock_adapter.get_cash_calls == 2  # one for cash_after_buy, one for assert
    assert mock_adapter.get_positions_calls == 1


@pytest.mark.asyncio
async def test_mock_adapter_submit_market_order(mock_adapter: MockBrokerAdapter):
    initial_cash_dict = await mock_adapter.get_cash()
    initial_cash_value = initial_cash_dict["USD"]
    symbol = "MARKET/USD"
    qty = 7

    # Reset relevant counters before the tested action
    mock_adapter.submit_order_calls = 0
    # mock_adapter.get_cash_calls = 0 # get_cash_calls will be 1 due to initial_cash_dict
    get_cash_calls_before_action = mock_adapter.get_cash_calls
    mock_adapter.get_positions_calls = 0

    await mock_adapter.submit_order(
        symbol, qty, "buy", "market"
    )  # limit_price is None for market
    assert mock_adapter.submit_order_calls == 1

    current_cash_dict = await mock_adapter.get_cash()
    assert current_cash_dict["USD"] == initial_cash_value - (
        qty * 100.0
    )  # Default price

    positions_list = await mock_adapter.get_positions()
    found_position = next((p for p in positions_list if p.symbol == symbol), None)
    assert found_position is not None, f"Position for {symbol} not found"
    assert found_position.qty == qty

    # get_cash was called for initial_cash_dict and for current_cash_dict
    assert mock_adapter.get_cash_calls == get_cash_calls_before_action + 1
    assert mock_adapter.get_positions_calls == 1


@pytest.mark.asyncio
async def test_mock_adapter_cancel_order(mock_adapter: MockBrokerAdapter):
    order_id_to_cancel = "some_order_id_123"
    initial_cancel_calls = mock_adapter.cancel_order_calls

    cancel_successful = await mock_adapter.cancel_order(order_id_to_cancel)

    assert cancel_successful is True
    assert mock_adapter.cancel_order_calls == initial_cancel_calls + 1


@pytest.mark.asyncio
async def test_mock_adapter_get_cash_updates(mock_adapter: MockBrokerAdapter):
    # Reset relevant counters
    mock_adapter.submit_order_calls = 0
    mock_adapter.get_cash_calls = 0  # Reset before action
    initial_cash_dict_before_order = (
        await mock_adapter.get_cash()
    )  # Call to get initial state
    initial_cash_value = initial_cash_dict_before_order["USD"]
    get_cash_calls_at_start = mock_adapter.get_cash_calls  # Should be 1 now

    await mock_adapter.submit_order(
        "CASH/TEST", 1, "buy", "market", limit_price=None
    )  # Market order, qty 1, uses mock default price 100
    assert mock_adapter.submit_order_calls == 1

    final_cash_dict = await mock_adapter.get_cash()
    assert final_cash_dict["USD"] == initial_cash_value - 100.0  # Cost is 1 * 100 = 100
    # get_cash called for initial state and by the final assert
    assert mock_adapter.get_cash_calls == get_cash_calls_at_start + 1


@pytest.mark.asyncio
async def test_mock_adapter_get_positions_updates(mock_adapter: MockBrokerAdapter):
    # Reset relevant counters
    mock_adapter.submit_order_calls = 0
    mock_adapter.get_positions_calls = 0  # Reset before action

    # Call get_positions once before main assertions to establish baseline call count
    _ = await mock_adapter.get_positions()
    get_positions_calls_at_start = mock_adapter.get_positions_calls  # Should be 1

    await mock_adapter.submit_order(
        "POS/TEST1", 3, "buy", "market"
    )  # price will be default 100
    await mock_adapter.submit_order("POS/TEST2", 7, "buy", "limit", limit_price=20.0)
    assert mock_adapter.submit_order_calls == 2

    positions_list = await mock_adapter.get_positions()

    pos1 = next((p for p in positions_list if p.symbol == "POS/TEST1"), None)
    pos2 = next((p for p in positions_list if p.symbol == "POS/TEST2"), None)

    assert pos1 is not None and pos1.qty == 3
    assert pos2 is not None and pos2.qty == 7
    assert len(positions_list) == 2
    # get_positions called for baseline and by the main assertion part
    assert mock_adapter.get_positions_calls == get_positions_calls_at_start + 1


@pytest.mark.asyncio
async def test_mock_adapter_multiple_orders_same_symbol(
    mock_adapter: MockBrokerAdapter,
):
    symbol = "MULTI/USD"
    # Reset counters for this test scenario
    mock_adapter.submit_order_calls = 0
    mock_adapter.get_cash_calls = 0  # Reset before action
    mock_adapter.get_positions_calls = 0  # Reset before action

    # Call to establish baseline call counts after reset
    _ = await mock_adapter.get_cash()
    _ = await mock_adapter.get_positions()
    get_cash_calls_at_start = mock_adapter.get_cash_calls  # Should be 1
    get_positions_calls_at_start = mock_adapter.get_positions_calls  # Should be 1

    await mock_adapter.submit_order(
        symbol, 10, "buy", "limit", limit_price=100.0
    )  # Cash: 100k - 1k = 99k, Pos: 10
    await mock_adapter.submit_order(
        symbol, 5, "sell", "limit", limit_price=110.0
    )  # Cash: 99k + 550 = 99550, Pos: 5
    await mock_adapter.submit_order(
        symbol, 3, "buy", "limit", limit_price=105.0
    )  # Cash: 99550 - 315 = 99235, Pos: 8

    current_cash_dict = await mock_adapter.get_cash()
    assert current_cash_dict["USD"] == 99235.0

    positions_list = await mock_adapter.get_positions()
    found_position = next((p for p in positions_list if p.symbol == symbol), None)
    assert found_position is not None, f"Position for {symbol} not found"
    assert found_position.qty == 8

    assert mock_adapter.submit_order_calls == 3
    # get_cash called for baseline + final assert
    assert mock_adapter.get_cash_calls == get_cash_calls_at_start + 1
    # get_positions called for baseline + final assert
    assert mock_adapter.get_positions_calls == get_positions_calls_at_start + 1
