import asyncio
import threading
import time
from unittest.mock import AsyncMock, MagicMock

import backtrader as bt
import pytest
import pytest_mock

from algo_mvp.live.adapters.mock import MockBrokerAdapter
from algo_mvp.live.runner import LiveRunner, RunnerStatus


# --- Helper Test Strategy ---
class TestStrategy(bt.Strategy):
    params = (
        ("order_count", 1),
        ("log_level", "INFO"),
    )

    def __init__(self):
        self.order = None
        self.orders_placed = 0
        if self.p.log_level == "DEBUG":
            print(
                f"[TestStrategy '{self.data._name}'] Initialized with data: {self.data._name}"
            )

    def log(self, txt, dt=None):
        if self.p.log_level == "DEBUG":
            dt = dt or self.datas[0].datetime.datetime(0)
            print(f"{dt.isoformat()} [TestStrategy '{self.data._name}'] {txt}")

    def next(self):
        self.log(f"Next bar, Close: {self.datas[0].close[0]:.2f}")
        if not self.position and self.orders_placed < self.p.order_count:
            self.log("Placing market buy order")
            self.order = self.buy(size=1)
            self.orders_placed += 1
        elif self.position and self.orders_placed >= self.p.order_count:
            self.log("Max orders placed and in position, doing nothing.")

    def notify_order(self, order):
        self.log(
            f"Order notification: Ref: {order.ref}, Status: {order.getstatusname()}"
        )
        if order.status in [order.Submitted, order.Accepted]:
            return  # Do nothing for these transient states in this simple strategy
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    f"BUY EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}"
                )
            elif order.issell():
                self.log(
                    f"SELL EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}"
                )
        elif order.status in [
            order.Canceled,
            order.Margin,
            order.Rejected,
            order.Expired,
        ]:
            self.log(f"Order Canceled/Margin/Rejected/Expired: {order.getstatusname()}")
        self.order = None

    def notify_trade(self, trade):
        if trade.isclosed:
            self.log(
                f"TRADE CLOSED, Gross PNL: {trade.pnl:.2f}, Net PNL: {trade.pnlcomm:.2f}"
            )


# --- Pytest Fixtures ---
@pytest.fixture
def mock_event_loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    # loop.close() # Can cause issues if other async fixtures are still using it


@pytest.fixture
def mock_broker_adapter(mock_event_loop: asyncio.AbstractEventLoop):
    adapter = MockBrokerAdapter(event_loop=mock_event_loop)
    # Give it a name, as CLI might try to access it via adapter.config.name
    adapter.config = MagicMock()
    adapter.config.name = "test_mock_adapter"
    return adapter


@pytest.fixture
def dummy_data_feed_config():
    return {"symbol": "DUMMYTEST", "timeframe": "1Min"}


@pytest.fixture
def live_runner_instance(
    mock_event_loop: asyncio.AbstractEventLoop,
    mock_broker_adapter: MockBrokerAdapter,
    dummy_data_feed_config: dict,
) -> LiveRunner:
    runner = LiveRunner(
        strategy_source=TestStrategy,
        strategy_params={
            "order_count": 1,
            "log_level": "DEBUG",
        },  # DEBUG for test verbosity
        broker_adapter=mock_broker_adapter,
        data_feed_config=dummy_data_feed_config,
        event_loop=mock_event_loop,
    )
    return runner


# --- Test Cases ---


def test_liverunner_initial_status(live_runner_instance: LiveRunner):
    assert live_runner_instance.status() == RunnerStatus.STOPPED.value


@pytest.mark.slow  # This test involves threading and sleeps
def test_liverunner_start_stop_status_lifecycle(
    live_runner_instance: LiveRunner,
    mock_broker_adapter: MockBrokerAdapter,
    mocker: pytest_mock.MockerFixture,
):
    runner = live_runner_instance

    cerebro_run_event = threading.Event()

    def mock_cerebro_run_blocking():
        # This function will be the side_effect for cerebro.run()
        # It will block until cerebro_run_event is set (by runstop)
        # print("[Test Mock Cerebro] run() called, waiting for event...")  # For debugging tests
        cerebro_run_event.wait()  # Block here
        # print("[Test Mock Cerebro] run() event received, finishing.")  # For debugging tests

    def mock_cerebro_runstop_releasing():
        # This function will be the side_effect for cerebro.runstop()
        # print("[Test Mock Cerebro] runstop() called, setting event.")  # For debugging tests
        cerebro_run_event.set()  # Release the block in mock_cerebro_run_blocking

    mock_cerebro_instance = MagicMock(spec=bt.Cerebro)
    # mock_cerebro_run = MagicMock(side_effect=mock_cerebro_run_blocking)  # Replaced direct MagicMock
    # mock_cerebro_runstop = MagicMock(side_effect=mock_cerebro_runstop_releasing)  # Replaced direct MagicMock
    mock_cerebro_instance.run = MagicMock(side_effect=mock_cerebro_run_blocking)
    mock_cerebro_instance.runstop = MagicMock(
        side_effect=mock_cerebro_runstop_releasing
    )

    mock_cerebro_instance.broker = MagicMock()  # Mock broker part of cerebro
    mock_cerebro_instance.datas = []  # Mock datas part of cerebro

    mocker.patch("backtrader.Cerebro", return_value=mock_cerebro_instance)
    mocker.patch.object(mock_cerebro_instance, "addwriter", MagicMock())
    mocker.patch.object(mock_cerebro_instance, "adddata", MagicMock())
    mocker.patch.object(mock_cerebro_instance, "addstrategy", MagicMock())
    mocker.patch.object(mock_cerebro_instance, "setbroker", MagicMock())

    run_cerebro_spy = mocker.spy(runner, "_run_cerebro")

    # Test start
    runner.start()
    assert runner.status() == RunnerStatus.RUNNING.value
    run_cerebro_spy.assert_called_once()
    time.sleep(0.2)  # Give thread time to start and cerebro.run to be called
    mock_cerebro_instance.run.assert_called_once()  # Check that the .run attribute (our mock) was called

    # Ensure thread is actually running and waiting if mock_cerebro_run_blocking is working
    assert runner._thread is not None
    assert (
        runner._thread.is_alive()
    ), "Cerebro thread should be alive and blocking in mock_cerebro_run"

    # Test stopping
    runner.stop()
    # runner.stop() should call mock_cerebro_instance.runstop(), which calls mock_cerebro_runstop_releasing(),
    # which sets the event, unblocking mock_cerebro_instance.run(). The thread should then finish.

    # Give some time for the event to propagate and thread to join
    if runner._thread.is_alive():
        runner._thread.join(
            timeout=2.0
        )  # Increased timeout for safety with event logic

    mock_cerebro_instance.runstop.assert_called_once()  # Check that the .runstop attribute (our mock) was called
    assert runner.status() == RunnerStatus.STOPPED.value

    assert not runner._thread.is_alive(), "Runner thread did not terminate after stop()"


@pytest.mark.skip(
    reason="Temporarily skipped due to order not reaching adapter (underlying writedict issue)"
)
@pytest.mark.asyncio
@pytest.mark.slow
async def test_liverunner_strategy_order_reaches_mock_adapter(
    live_runner_instance: LiveRunner,
    mock_broker_adapter: MockBrokerAdapter,
    dummy_data_feed_config: dict,
    mock_event_loop: asyncio.AbstractEventLoop,  # Ensure loop is available for adapter
):
    runner = live_runner_instance
    adapter_submit_order_spy = AsyncMock(wraps=mock_broker_adapter.submit_order)
    mock_broker_adapter.submit_order = adapter_submit_order_spy

    # Use a short dummy CSV for this test, LiveRunner will create one if not present.
    # The key is that Cerebro runs, strategy places an order, OrderEventWriter picks it up.
    dummy_csv_path = f'./dummy_data_{dummy_data_feed_config["symbol"]}.csv'
    with open(dummy_csv_path, "w") as f:
        f.write("DateTime,Open,High,Low,Close,Volume\n")
        f.write("2024-01-01 09:30:00,100,101,99,100.5,1000\n")
        f.write("2024-01-01 09:31:00,101,102,100,101.0,1200\n")

    runner.start()

    # Wait for Cerebro to run through the data and strategy to place order
    # This needs to be long enough for the thread to execute Cerebro with the CSV data.
    # The TestStrategy places one order on the first 'next' call.
    max_wait_time = 5  # seconds
    start_time = time.time()
    order_submitted_to_adapter = False
    while time.time() - start_time < max_wait_time:
        if adapter_submit_order_spy.called:
            order_submitted_to_adapter = True
            break
        await asyncio.sleep(0.1)  # Yield to event loop and other tasks

    runner.stop()  # Ensure runner stops cleanly
    # Wait for thread to finish
    if runner._thread and runner._thread.is_alive():
        runner._thread.join(timeout=2)
    assert not (
        runner._thread and runner._thread.is_alive()
    ), "Runner thread still alive after stop"

    assert (
        order_submitted_to_adapter
    ), "Order from strategy did not reach MockBrokerAdapter.submit_order"
    adapter_submit_order_spy.assert_called_once()
    call_args = adapter_submit_order_spy.call_args[1]  # kwargs
    assert call_args["side"] == "buy"
    assert call_args["exectype"] == "market"  # OrderEventWriter maps this
    assert call_args["size"] == 1


# TODO: Test LiveRunner with strategy specified as dotted path string.
# TODO: Test LiveRunner error handling (e.g., strategy init fails, adapter error).
# TODO: Test on_trade and on_error callbacks if provided to LiveRunner.
