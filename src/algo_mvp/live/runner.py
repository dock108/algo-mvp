import asyncio
import datetime
import importlib
import threading
from enum import Enum
from typing import Any, Callable, Dict, Type, Union

import backtrader as bt
from rich.console import Console
from rich.text import Text

# Assuming MockBrokerAdapter is the main one for now
# from algo_mvp.live.adapters.mock import MockBrokerAdapter
# For now, to avoid circular dependency if runner is imported by adapter's __init__ or vice-versa
# We'll assume the adapter instance is passed in and typed as Any for now.

console = Console()


class RunnerStatus(Enum):
    STOPPED = "stopped"
    RUNNING = "running"
    ERROR = "error"


class LiveRunner:
    def __init__(
        self,
        strategy_source: Union[str, Type[bt.Strategy]],
        strategy_params: Dict[str, Any],
        broker_adapter: Any,  # Protocol: BrokerAdapterProtocol (methods: submit_order, cancel_order, etc.)
        data_feed_config: Dict[
            str, Any
        ],  # e.g. {'symbol': 'MESM25', 'timeframe': '1Min', ...}
        event_loop: asyncio.AbstractEventLoop | None = None,
        on_trade: Callable[[bt.Order, bt.Trade], None] | None = None,
        on_error: Callable[[Exception], None] | None = None,
    ):
        self.strategy_source = strategy_source
        self.strategy_params = strategy_params
        self.broker_adapter = broker_adapter
        self.data_feed_config = data_feed_config
        self.loop = (
            event_loop or asyncio.get_event_loop()
        )  # Ensure we have an event loop
        self.on_trade_callback = on_trade
        self.on_error_callback = on_error

        self._status: RunnerStatus = RunnerStatus.STOPPED
        self._cerebro: bt.Cerebro | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = (
            threading.Event()
        )  # For signaling cerebro loop to stop from other threads

        self._strategy_name = self._get_strategy_name()
        self._log_prefix = f"[Runner-{self.data_feed_config.get('symbol', 'Unknown')}-{self._strategy_name}]"

        # Setup callbacks with the broker adapter if it supports them
        if hasattr(self.broker_adapter, "set_on_order_status_change_callback"):
            self.broker_adapter.set_on_order_status_change_callback(
                self._handle_order_status_change_from_adapter
            )

        if hasattr(self.broker_adapter, "set_on_trade_callback"):
            self.broker_adapter.set_on_trade_callback(
                self._handle_trade_event_from_adapter
            )

    def _get_strategy_name(self) -> str:
        if isinstance(self.strategy_source, str):
            # e.g., "algo_mvp.backtest.strategies.vwap_atr:VWAPATRStrategy"
            return self.strategy_source.split(":")[-1].split(".")[-1]
        else:
            return self.strategy_source.__name__

    def _log(self, message: str, level: str = "info"):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        log_message = Text()
        log_message.append(f"{timestamp} ", style="dim blue")
        log_message.append(f"{self._log_prefix} ", style="bold magenta")
        if level == "error":
            log_message.append(message, style="bold red")
        elif level == "warning":
            log_message.append(message, style="yellow")
        else:
            log_message.append(message)
        console.print(log_message)

    # Callback from adapter for order status changes
    def _handle_order_status_change_from_adapter(self, order: bt.Order):
        self._log(
            f"[Adapter] Order Update: Ref {order.ref}, Status {order.getstatusname()}"
        )
        # This is where LiveRunner would react to adapter-driven order status changes.
        # For example, if an order is filled by the broker, the adapter tells us here.
        # We might then update Cerebro's internal state or notify other components.
        pass  # Add pass back, or implement logic

    # Callback from adapter for trade events
    def _handle_trade_event_from_adapter(self, order: bt.Order, trade: bt.Trade):
        self._log(
            f"[Adapter] Trade Event: OrderRef {order.ref}, TradeID {trade.ref}, Size {trade.size}, Price {trade.price}"
        )
        if self.on_trade_callback:
            try:
                self.on_trade_callback(order, trade)
            except (
                Exception
            ) as e_trade_cb:  # Rename exception variable to avoid conflict if on_error_callback also raises
                self._log(
                    f"Error in on_trade_callback (from adapter): {e_trade_cb}",
                    level="error",
                )
                if self.on_error_callback:
                    self.on_error_callback(e_trade_cb)

    def _load_strategy_class(self) -> Type[bt.Strategy]:
        if isinstance(self.strategy_source, str):
            module_path, class_name = self.strategy_source.split(":")
            module = importlib.import_module(module_path)
            strategy_class = getattr(module, class_name)
            return strategy_class
        return self.strategy_source  # type: ignore

    def _initialize_cerebro(self):
        self._log("Initializing Cerebro...")
        self._cerebro = bt.Cerebro(stdstats=False)  # No standard stats for live

        # Broker setup
        # The adapter is expected to provide a Backtrader-compatible broker instance.
        # For MockBrokerAdapter, get_backtrader_broker() returns a basic bt.brokers.BackBroker().
        # This broker will have its own cash management unless synchronized.
        # For a live setup, a custom bt.BrokerBase that uses the async adapter methods is ideal.
        # For this skeleton, we use the basic broker from the adapter.
        if not hasattr(self.broker_adapter, "get_backtrader_broker"):
            err_msg = "Broker adapter must have a 'get_backtrader_broker' method."
            self._log(err_msg, level="error")
            self._status = RunnerStatus.ERROR
            if self.on_error_callback:
                self.on_error_callback(RuntimeError(err_msg))
            raise RuntimeError(err_msg)

        internal_broker = self.broker_adapter.get_backtrader_broker()
        self._cerebro.setbroker(internal_broker)
        self._log(f"Cerebro broker set from adapter: {type(internal_broker).__name__}")

        # Strategy loading
        try:
            strategy_class = self._load_strategy_class()
            self._cerebro.addstrategy(strategy_class, **self.strategy_params)
            self._log(
                f"Strategy '{strategy_class.__name__}' added with params: {self.strategy_params}"
            )
        except Exception as e_strat_load:
            self._log(f"Error loading strategy: {e_strat_load}", level="error")
            self._status = RunnerStatus.ERROR
            if self.on_error_callback:
                self.on_error_callback(e_strat_load)
            raise

        # Data Feed (Simplified placeholder for skeleton)
        # A real live runner would use a live data feed from the broker adapter or a dedicated source.
        # For this skeleton, we use a dummy CSV to allow Cerebro to run.
        symbol = self.data_feed_config.get("symbol", "UNKNOWN_SYMBOL")
        timeframe_str = self.data_feed_config.get(
            "timeframe", "1Min"
        )  # e.g., 1Min, 5Min, 1H, 1D

        # Map timeframe string to Backtrader TimeFrame and compression
        # This is a simplified mapping
        tf_map = {
            "1Min": (bt.TimeFrame.Minutes, 1),
            "5Min": (bt.TimeFrame.Minutes, 5),
            "1H": (bt.TimeFrame.Minutes, 60),
            "1D": (bt.TimeFrame.Days, 1),
        }
        bt_timeframe, bt_compression = tf_map.get(
            timeframe_str, (bt.TimeFrame.Minutes, 1)
        )

        dummy_csv_path = f'./dummy_data_{symbol.replace("/","_")}.csv'
        try:
            with open(dummy_csv_path, "r") as f:
                # Check if has more than header
                if len(f.readlines()) < 2:
                    raise FileNotFoundError  # Trigger recreation if empty or just header
        except FileNotFoundError:
            self._log(
                f"Creating dummy CSV data for {symbol} at {dummy_csv_path}",
                level="warning",
            )
            with open(dummy_csv_path, "w") as f_csv:
                f_csv.write(
                    "DateTime,Open,High,Low,Close,Volume\n"
                )  # Adjusted header to match common CSV formats
                # Add a couple of recent dummy bars for Cerebro to process
                now = datetime.datetime.now(datetime.timezone.utc)
                for i in range(2, 0, -1):  # Two bars: now-2min, now-1min
                    dt = now - datetime.timedelta(minutes=i)
                    # Format as YYYY-MM-DDTHH:MM:SS for bt.feeds.GenericCSVData default dtformat
                    # However, our dtformat is %Y-%m-%d %H:%M:%S, so use that.
                    # dt_str = dt.strftime('%Y-%m-%dT%H:%M:%S')
                    dt_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                    f_csv.write(f"{dt_str},100,101,99,100.5,1000\n")

        data = bt.feeds.GenericCSVData(
            dataname=dummy_csv_path,
            dtformat=("%Y-%m-%d %H:%M:%S"),  # Standard format
            datetime=0,
            open=1,
            high=2,
            low=3,
            close=4,
            volume=5,
            openinterest=-1,
            timeframe=bt_timeframe,
            compression=bt_compression,
            name=symbol,
        )
        self._cerebro.adddata(data, name=symbol)
        self._log(
            f"Dummy CSV data feed added for {symbol} ({timeframe_str}). Path: {dummy_csv_path}"
        )

        # Add a writer to capture order events from cerebro and pass to broker_adapter
        # This is crucial for linking strategy's self.buy()/sell() to the actual adapter.
        self._cerebro.addwriter(
            OrderEventWriter, broker_adapter=self.broker_adapter, live_runner=self
        )
        self._log("OrderEventWriter added to Cerebro.")

    def _run_cerebro(self):
        if not self._cerebro:
            self._log("Cerebro not initialized. Cannot run.", level="error")
            self._status = RunnerStatus.ERROR
            # No on_error_callback here, as start() should have caught this during init.
            return

        try:
            self._log("Starting Cerebro run loop in background thread...")
            # self._status = RunnerStatus.RUNNING  # Status is set in start() before thread launch

            # cerebro.run() is blocking. It will run until:
            # 1. End of data (for historical/CSV feeds)
            # 2. cerebro.stop() is called (not typically used for graceful live stop)
            # 3. cerebro.runstop() is called (preferred for graceful stop signal)
            # 4. An unhandled exception occurs within the strategy or Cerebro.

            # For our dummy CSV, it will finish quickly. For a live feed, it would block indefinitely
            # until data source ends or runstop() is called.
            self._cerebro.run()

            # After cerebro.run() completes (either naturally or by runstop()):
            if self._stop_event.is_set():
                self._log(
                    "Cerebro run loop was stopped by external signal (runstop). Tidy exit."
                )
            else:
                self._log(
                    "Cerebro run loop finished naturally (e.g., end of data for CSV). Runner stopping."
                )

        except Exception as e_cerebro_run:
            self._log(f"Exception in Cerebro run loop: {e_cerebro_run}", level="error")
            self._status = RunnerStatus.ERROR  # Mark runner as errored
            if self.on_error_callback:
                self.on_error_callback(e_cerebro_run)  # Notify external error handler
        finally:
            # This finally block executes regardless of how cerebro.run() exited.
            current_status_before_stop = self._status
            self._status = RunnerStatus.STOPPED  # Ensure status is STOPPED
            self._log(
                f"Cerebro thread finished. Runner status transitioned from {current_status_before_stop.value} to {self._status.value}."
            )
            # self._stop_event.clear()  # Clear event if it was set, for potential restart logic (not fully implemented)

    def start(self):
        if self._status == RunnerStatus.RUNNING:
            self._log("Runner is already running.", level="warning")
            return
        if self._thread and self._thread.is_alive():
            self._log(
                "Runner thread is already alive though status is not RUNNING.Investigate!",
                level="warning",
            )
            return  # Avoid starting another thread if one is lingering

        self._log("Starting LiveRunner...")
        self._stop_event.clear()  # Clear any previous stop signal
        self._status = RunnerStatus.STOPPED  # Reset status before attempting to start

        try:
            self._initialize_cerebro()  # This can raise exceptions and set ERROR status
            if self._status == RunnerStatus.ERROR:
                self._log(
                    "Cerebro initialization failed. Cannot start LiveRunner.",
                    level="error",
                )
                return  # Do not proceed if initialization set an error status
        except Exception:  # Catching generic Exception as e_init was unused.
            # _initialize_cerebro should log its own errors and call on_error_callback
            # self._log(f"Critical error during Cerebro initialization: {e_init}", level="error")
            # self._status = RunnerStatus.ERROR  # Should be set by _initialize_cerebro
            # if self.on_error_callback: self.on_error_callback(e_init)
            return  # Stop if initialization raised an unhandled exception that wasn't caught by itself to set ERROR

        self._thread = threading.Thread(target=self._run_cerebro, daemon=True)
        self._status = (
            RunnerStatus.RUNNING
        )  # Set status to RUNNING before starting thread
        self._thread.start()
        self._log(
            f"LiveRunner started. Cerebro thread '{self._thread.name}' initiated."
        )

    def stop(self):
        if self._status != RunnerStatus.RUNNING and not (
            self._thread and self._thread.is_alive()
        ):
            self._log(
                f"Runner is not running (status: {self._status.value}) or thread not alive. Stop command ignored.",
                level="warning",
            )
            return

        self._log("Attempting to stop LiveRunner gracefully...")
        self._stop_event.set()  # Signal the _run_cerebro loop (if it checks this event)

        if self._cerebro:
            try:
                # cerebro.runstop() is the Backtrader way to ask the running strategy/loop to stop.
                # It sets an internal flag that cerebro.run() checks.
                self._cerebro.runstop()
                self._log(
                    "cerebro.runstop() called to signal Cerebro loop termination."
                )
            except Exception as e_runstop:
                self._log(
                    f"Exception calling cerebro.runstop(): {e_runstop}", level="error"
                )
                # Potentially trigger on_error_callback if this is critical
                if self.on_error_callback:
                    self.on_error_callback(e_runstop)

        if self._thread and self._thread.is_alive():
            self._log(f"Waiting for Cerebro thread '{self._thread.name}' to join...")
            self._thread.join(
                timeout=10.0
            )  # Wait for up to 10 seconds for the thread to finish
            if self._thread.is_alive():
                self._log(
                    f"Cerebro thread '{self._thread.name}' did not join in time. It might be stuck.",
                    level="warning",
                )
            else:
                self._log(f"Cerebro thread '{self._thread.name}' joined successfully.")
        else:
            self._log("No active Cerebro thread to join, or thread already finished.")

        # Status should be set to STOPPED by the _run_cerebro finally block.
        # If thread didn't join or crashed, force status if it's not already STOPPED.
        if self._status != RunnerStatus.STOPPED:
            self._log(
                f"Forcing status to STOPPED post-join attempt (was {self._status.value}).",
                level="warning",
            )
            self._status = RunnerStatus.STOPPED

        self._log("LiveRunner stop sequence complete.")

    def status(self) -> str:
        # Could add more checks, e.g., self._thread.is_alive() if status is RUNNING
        if (
            self._status == RunnerStatus.RUNNING
            and self._thread
            and not self._thread.is_alive()
        ):
            self._log(
                "Status is RUNNING but thread is not alive. Correcting to ERROR.",
                level="error",
            )
            self._status = (
                RunnerStatus.ERROR
            )  # Mark as error if thread died unexpectedly
            # Consider calling on_error_callback with a custom error
            if self.on_error_callback:
                self.on_error_callback(
                    RuntimeError("LiveRunner thread died unexpectedly.")
                )
        return self._status.value

    # Generic error handler that can be called internally or by adapter
    def _handle_error_event(self, exc: Exception, source: str = "LiveRunner"):
        self._log(f"Error Event from {source}: {exc}", level="error")
        self._status = RunnerStatus.ERROR
        if self.on_error_callback:
            try:
                self.on_error_callback(exc)
            except Exception as e_cb_err:  # Error within the error callback itself
                self._log(
                    f"Exception in on_error_callback itself: {e_cb_err}", level="error"
                )


# Example usage (for testing within this file, normally CLI drives this)
async def _test_live_runner():  # pragma: no cover
    # This will be fleshed out for local testing of the runner if needed.
    # For now, it's a placeholder.
    console.print("[bold yellow]LiveRunner test execution placeholder.[/bold yellow]")
    # Add basic setup and start/stop calls here later.
    pass


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(_test_live_runner())


# Custom Backtrader Writer to intercept order creation/notification from strategy
# and pass it to the broker adapter.
class OrderEventWriter(bt.WriterBase):
    params = (
        ("broker_adapter", None),  # Will be injected by LiveRunner
        ("live_runner", None),  # Will be injected by LiveRunner
        ("csv", False),  # Explicitly disable CSV for this writer
    )

    def __init__(self):
        super().__init__()
        self.p.csv = (
            False  # Force self.p.csv to False to try and prevent writedict error
        )
        # Ensure parameters are passed
        if self.p.broker_adapter is None:
            raise ValueError(
                "OrderEventWriter requires 'broker_adapter' from LiveRunner."
            )
        if self.p.live_runner is None:
            raise ValueError("OrderEventWriter requires 'live_runner' from LiveRunner.")

        self.broker_adapter = self.p.broker_adapter
        self.live_runner: LiveRunner = (
            self.p.live_runner
        )  # For typing and access to _log, loop
        self.loop = self.live_runner.loop
        self._processed_cerebro_order_refs = (
            set()
        )  # Track refs of orders sent to adapter

    def start(self):
        # Called once at the beginning of a Backtrader run
        pass

    def stop(self):
        # Called once at the end of a Backtrader run
        pass

    # Add open/close methods as a speculative fix for 'writedict' issue
    def open(self, **kwargs):
        pass

    def close(self, **kwargs):
        pass

    # Add a dummy writedict to see if it gets called and to prevent AttributeError
    def writedict(self, adict: dict):
        if self.live_runner:
            self.live_runner._log(
                f"[OrderEventWriter] WARNING: writedict unexpectedly called with {adict}",
                level="warning",
            )
        pass  # Do nothing else

    def next(self):
        # Called on each bar. Not typically for order events directly.
        pass

    def notify_order(self, order: bt.Order):
        # This method is called by Cerebro when an order changes status.
        self.live_runner._log(
            f"[Cerebro Writer] Notified order: Ref {order.ref}, Status {order.getstatusname()}, Size {order.size}, Price {order.price}"
        )

        # If order is created by strategy (via cerebro.broker) and not yet processed by us:
        if (
            order.status == bt.Order.Submitted
            and order.ref not in self._processed_cerebro_order_refs
        ):
            self.live_runner._log(
                f"[Cerebro Writer] Intercepted new order from strategy: Ref {order.ref}"
            )
            self._processed_cerebro_order_refs.add(order.ref)

            # The order object (order) was created by Backtrader's internal broker.
            # We need to submit this intent to our actual async broker adapter.
            async def do_submit_to_adapter():
                try:
                    # Map Cerebro order details to adapter's expected format
                    # Order types might need mapping if not using bt.Order constants directly in adapter
                    exectype_map = {
                        bt.Order.Market: "market",
                        bt.Order.Limit: "limit",
                        bt.Order.Stop: "stop",
                        bt.Order.StopLimit: "stoplimit",
                    }
                    adapter_exectype = exectype_map.get(
                        order.exectype, "market"
                    )  # Default to market

                    submitted_adapter_order = await self.broker_adapter.submit_order(
                        owner=order.owner,  # Strategy instance
                        data=order.data,  # Data feed
                        side="buy" if order.isbuy() else "sell",
                        exectype=adapter_exectype,
                        size=order.created.size,  # Original requested size
                        price=(
                            order.created.price
                            if order.exectype in [bt.Order.Limit, bt.Order.StopLimit]
                            else None
                        ),
                        # TODO: Add other relevant params like 'valid' if adapter supports
                    )
                    if submitted_adapter_order:
                        self.live_runner._log(
                            f"[Cerebro Writer] Order {order.ref} submitted to adapter. Adapter Order Ref: {submitted_adapter_order.ref}"
                        )
                        # Here, you might want to link cerebro order.ref with adapter_order.ref if they differ
                        # and if the adapter provides its own reference.
                        # For mock, they might be the same if adapter reuses Cerebro's order objects.
                    else:
                        self.live_runner._log(
                            f"[Cerebro Writer] Adapter failed to submit order {order.ref}. No adapter order returned.",
                            level="warning",
                        )
                        # How to reflect this back to Cerebro? Cerebro's order is already 'Submitted'.
                        # We could try to cancel Cerebro's internal order, or mark it as rejected.
                        # For now, just log. This is a key integration point for robustness.
                        # order.reject()  # This would change cerebro state
                except Exception as e_submit:
                    self.live_runner._log(
                        f"[Cerebro Writer] Exception submitting order {order.ref} to adapter: {e_submit}",
                        level="error",
                    )
                    if (
                        self.live_runner.on_error_callback
                    ):  # Use the runner's main error callback
                        self.live_runner.on_error_callback(e_submit)

            if self.loop and self.loop.is_running():
                asyncio.run_coroutine_threadsafe(do_submit_to_adapter(), self.loop)
            else:
                # This is a critical failure: cannot communicate with async adapter.
                self.live_runner._log(
                    "[Cerebro Writer] CRITICAL: No running event loop to submit order to adapter!",
                    level="error",
                )
                # Potentially raise an error or notify runner to stop.

        elif order.status in [
            bt.Order.Completed,
            bt.Order.Canceled,
            bt.Order.Expired,
            bt.Order.Margin,
            bt.Order.Rejected,
        ]:
            self.live_runner._log(
                f"[Cerebro Writer] Order {order.ref} reached terminal state in Cerebro: {order.getstatusname()}"
            )
            if order.ref in self._processed_cerebro_order_refs:
                self._processed_cerebro_order_refs.remove(order.ref)  # Clean up

        # Note: We are NOT calling self.live_runner._handle_order_status_change here anymore
        # because that handler is for status changes coming FROM the adapter.
        # Order status changes within Cerebro are logged here directly.

    def notify_trade(self, trade: bt.Trade):
        # This is called by Cerebro when its internal broker processes a fill and creates/updates a trade.
        # With an external adapter, the adapter should be the source of truth for trades.
        # The adapter should call `_handle_trade_event_from_adapter` on the LiveRunner.
        self.live_runner._log(
            f"[Cerebro Writer] Notified of Cerebro-internal trade: Ref {trade.ref}, OrderRef {trade.orderref}, Size {trade.size}, Price {trade.price}, Status {trade.status_names[trade.status]}"
        )
        # We do not propagate this as a primary trade event if an external adapter is used,
        # to avoid duplicate trade processing. The adapter's trade notification is primary.
        pass

    def notify_cashvalue(self, cash, value):
        # self.live_runner._log(f"[Cerebro Writer] Cerebro Cash: {cash:.2f}, Portfolio Value: {value:.2f}")
        pass

    def notify_fund(self, cash, value, fundvalue, shares):
        pass  # Not used typically
