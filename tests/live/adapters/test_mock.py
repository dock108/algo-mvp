from collections import defaultdict

import pytest

from algo_mvp.live.adapters.mock import MockBrokerAdapter


@pytest.fixture
def mock_adapter():
    return MockBrokerAdapter()


@pytest.mark.asyncio
async def test_mock_adapter_initial_state(mock_adapter: MockBrokerAdapter):
    assert mock_adapter.call_counts == defaultdict(int)  # Check this before other calls
    assert await mock_adapter.get_cash() == 100000.0
    assert await mock_adapter.get_positions() == {}
    # After the calls above, counts for get_cash and get_positions will be 1
    assert mock_adapter.call_counts["get_cash"] == 1
    assert mock_adapter.call_counts["get_positions"] == 1


@pytest.mark.asyncio
async def test_mock_adapter_submit_buy_order(mock_adapter: MockBrokerAdapter):
    initial_cash = await mock_adapter.get_cash()
    symbol = "TEST/USD"
    qty = 10
    limit_price = 150.0

    order_response = await mock_adapter.submit_order(
        symbol, qty, "buy", "limit", limit_price
    )

    assert order_response["id"] == "mock_order_id"
    assert order_response["status"] == "filled"
    assert mock_adapter.call_counts["submit_order"] == 1

    assert await mock_adapter.get_cash() == initial_cash - (qty * limit_price)
    positions = await mock_adapter.get_positions()
    assert positions[symbol] == qty


@pytest.mark.asyncio
async def test_mock_adapter_submit_sell_order(mock_adapter: MockBrokerAdapter):
    symbol = "TEST/USD"
    qty_to_buy = 20
    buy_price = 100.0
    # First, establish a position
    await mock_adapter.submit_order(symbol, qty_to_buy, "buy", "limit", buy_price)
    mock_adapter.call_counts.clear()  # Reset count after setup
    cash_after_buy = await mock_adapter.get_cash()

    qty_to_sell = 5
    sell_price = 110.0
    order_response = await mock_adapter.submit_order(
        symbol, qty_to_sell, "sell", "limit", sell_price
    )

    assert order_response["status"] == "filled"
    assert mock_adapter.call_counts["submit_order"] == 1
    assert await mock_adapter.get_cash() == cash_after_buy + (qty_to_sell * sell_price)
    positions = await mock_adapter.get_positions()
    assert positions[symbol] == qty_to_buy - qty_to_sell


@pytest.mark.asyncio
async def test_mock_adapter_submit_market_order(mock_adapter: MockBrokerAdapter):
    # Market orders use a default price of 100.0 in the mock
    initial_cash = await mock_adapter.get_cash()
    symbol = "MARKET/USD"
    qty = 7

    await mock_adapter.submit_order(symbol, qty, "buy", "market")
    assert mock_adapter.call_counts["submit_order"] == 1
    assert await mock_adapter.get_cash() == initial_cash - (
        qty * 100.0
    )  # Default price
    positions = await mock_adapter.get_positions()
    assert positions[symbol] == qty


@pytest.mark.asyncio
async def test_mock_adapter_cancel_order(mock_adapter: MockBrokerAdapter):
    order_id_to_cancel = "some_order_id_123"
    cancel_response = await mock_adapter.cancel_order(order_id_to_cancel)

    assert cancel_response["id"] == order_id_to_cancel
    assert cancel_response["status"] == "cancelled"
    assert mock_adapter.call_counts["cancel_order"] == 1


@pytest.mark.asyncio
async def test_mock_adapter_get_cash_updates(mock_adapter: MockBrokerAdapter):
    await mock_adapter.submit_order("CASH/TEST", 1, "buy", "market", 50)  # Buys at 50
    assert await mock_adapter.get_cash() == 100000.0 - 50.0
    assert (
        mock_adapter.call_counts["get_cash"] == 1
    )  # get_cash called once by the assert
    # submit_order also internally modifies cash, but get_cash counts explicit calls to it.


@pytest.mark.asyncio
async def test_mock_adapter_get_positions_updates(mock_adapter: MockBrokerAdapter):
    await mock_adapter.submit_order("POS/TEST1", 3, "buy", "market", 10)
    await mock_adapter.submit_order("POS/TEST2", 7, "buy", "limit", 20)
    positions = await mock_adapter.get_positions()
    assert positions["POS/TEST1"] == 3
    assert positions["POS/TEST2"] == 7
    assert len(positions) == 2
    assert mock_adapter.call_counts["get_positions"] == 1


@pytest.mark.asyncio
async def test_mock_adapter_multiple_orders_same_symbol(
    mock_adapter: MockBrokerAdapter,
):
    symbol = "MULTI/USD"
    await mock_adapter.submit_order(
        symbol, 10, "buy", "limit", 100
    )  # Cash: 100k - 1k = 99k, Pos: 10
    await mock_adapter.submit_order(
        symbol, 5, "sell", "limit", 110
    )  # Cash: 99k + 550 = 99550, Pos: 5
    await mock_adapter.submit_order(
        symbol, 3, "buy", "limit", 105
    )  # Cash: 99550 - 315 = 99235, Pos: 8

    assert await mock_adapter.get_cash() == 99235.0
    positions = await mock_adapter.get_positions()
    assert positions[symbol] == 8
    assert mock_adapter.call_counts["submit_order"] == 3
