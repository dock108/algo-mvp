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

    assert await mock_adapter.get_cash() == 100000.0
    assert await mock_adapter.get_positions() == {}
    # After the calls above, counts for get_cash and get_positions will be 1
    assert mock_adapter.get_cash_calls == 1
    assert mock_adapter.get_positions_calls == 1


@pytest.mark.asyncio
async def test_mock_adapter_submit_buy_order(mock_adapter: MockBrokerAdapter):
    initial_cash_calls = mock_adapter.get_cash_calls  # Track before action
    initial_submit_calls = mock_adapter.submit_order_calls

    initial_cash = await mock_adapter.get_cash()
    symbol = "TEST/USD"
    qty = 10
    limit_price = 150.0

    order_response = await mock_adapter.submit_order(
        symbol, qty, "buy", "limit", limit_price
    )

    assert order_response["id"].startswith("mock_order_id_")
    assert order_response["status"] == "filled"
    assert mock_adapter.submit_order_calls == initial_submit_calls + 1

    assert await mock_adapter.get_cash() == initial_cash - (qty * limit_price)
    positions = await mock_adapter.get_positions()
    assert positions[symbol] == qty
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

    cash_after_buy = await mock_adapter.get_cash()

    qty_to_sell = 5
    sell_price = 110.0
    order_response = await mock_adapter.submit_order(
        symbol, qty_to_sell, "sell", "limit", sell_price
    )

    assert order_response["status"] == "filled"
    assert mock_adapter.submit_order_calls == 1
    assert await mock_adapter.get_cash() == cash_after_buy + (qty_to_sell * sell_price)
    positions = await mock_adapter.get_positions()
    assert positions[symbol] == qty_to_buy - qty_to_sell
    assert mock_adapter.get_cash_calls == 2  # one for cash_after_buy, one for assert
    assert mock_adapter.get_positions_calls == 1


@pytest.mark.asyncio
async def test_mock_adapter_submit_market_order(mock_adapter: MockBrokerAdapter):
    initial_cash = await mock_adapter.get_cash()
    symbol = "MARKET/USD"
    qty = 7

    # Reset relevant counters before the tested action
    mock_adapter.submit_order_calls = 0
    mock_adapter.get_cash_calls = 0
    mock_adapter.get_positions_calls = 0

    await mock_adapter.submit_order(
        symbol, qty, "buy", "market"
    )  # limit_price is None for market
    assert mock_adapter.submit_order_calls == 1
    assert await mock_adapter.get_cash() == initial_cash - (
        qty * 100.0
    )  # Default price
    positions = await mock_adapter.get_positions()
    assert positions[symbol] == qty
    assert mock_adapter.get_cash_calls == 1  # Was 2, should be 1 after counter reset
    assert mock_adapter.get_positions_calls == 1


@pytest.mark.asyncio
async def test_mock_adapter_cancel_order(mock_adapter: MockBrokerAdapter):
    order_id_to_cancel = "some_order_id_123"
    initial_cancel_calls = mock_adapter.cancel_order_calls

    cancel_response = await mock_adapter.cancel_order(order_id_to_cancel)

    assert cancel_response["id"] == order_id_to_cancel
    assert cancel_response["status"] == "cancelled"
    assert mock_adapter.cancel_order_calls == initial_cancel_calls + 1


@pytest.mark.asyncio
async def test_mock_adapter_get_cash_updates(mock_adapter: MockBrokerAdapter):
    # Reset relevant counters
    mock_adapter.submit_order_calls = 0
    mock_adapter.get_cash_calls = 0

    await mock_adapter.submit_order(
        "CASH/TEST", 1, "buy", "market", limit_price=None
    )  # Market order, qty 1, uses mock default price 100
    assert mock_adapter.submit_order_calls == 1
    assert await mock_adapter.get_cash() == 100000.0 - 100.0  # Cost is 1 * 100 = 100
    assert mock_adapter.get_cash_calls == 1  # get_cash called once by the assert


@pytest.mark.asyncio
async def test_mock_adapter_get_positions_updates(mock_adapter: MockBrokerAdapter):
    # Reset relevant counters
    mock_adapter.submit_order_calls = 0
    mock_adapter.get_positions_calls = 0

    await mock_adapter.submit_order("POS/TEST1", 3, "buy", "market", 10.0)
    await mock_adapter.submit_order("POS/TEST2", 7, "buy", "limit", 20.0)
    assert mock_adapter.submit_order_calls == 2

    positions = await mock_adapter.get_positions()
    assert positions["POS/TEST1"] == 3
    assert positions["POS/TEST2"] == 7
    assert len(positions) == 2
    assert mock_adapter.get_positions_calls == 1


@pytest.mark.asyncio
async def test_mock_adapter_multiple_orders_same_symbol(
    mock_adapter: MockBrokerAdapter,
):
    symbol = "MULTI/USD"
    # Reset counters for this test scenario
    mock_adapter.submit_order_calls = 0
    mock_adapter.get_cash_calls = 0
    mock_adapter.get_positions_calls = 0

    await mock_adapter.submit_order(
        symbol, 10, "buy", "limit", 100.0
    )  # Cash: 100k - 1k = 99k, Pos: 10
    await mock_adapter.submit_order(
        symbol, 5, "sell", "limit", 110.0
    )  # Cash: 99k + 550 = 99550, Pos: 5
    await mock_adapter.submit_order(
        symbol, 3, "buy", "limit", 105.0
    )  # Cash: 99550 - 315 = 99235, Pos: 8

    assert await mock_adapter.get_cash() == 99235.0
    positions = await mock_adapter.get_positions()
    assert positions[symbol] == 8
    assert mock_adapter.submit_order_calls == 3
    assert mock_adapter.get_cash_calls == 1  # Called by assert
    assert mock_adapter.get_positions_calls == 1  # Called by assert
