import importlib
import logging  # Import logging
import threading

import backtrader as bt

from algo_mvp.live.adapters import AlpacaBrokerAdapter  # Import AlpacaBrokerAdapter

# import pendulum # No longer used

# Configure a logger for LiveRunner
logger = logging.getLogger(__name__)  # Use module's name for the logger
# You might want to set a default level and handler if not configured elsewhere
# For example:
# if not logger.hasHandlers():
#     handler = logging.StreamHandler()
#     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     handler.setFormatter(formatter)
#     logger.addHandler(handler)
#     logger.setLevel(logging.INFO)  # Or logging.DEBUG for more verbosity


class LiveRunner:
    def __init__(
        self,
        strategy_path: str,
        params: dict,
        # broker_adapter, # Replaced by broker_config
        broker_config: dict,  # New: to specify provider e.g. {'provider': 'alpaca'}
        datafeed_config: dict,
        on_trade=None,
        on_error=None,
    ):
        self.strategy_path = strategy_path
        self.params = params
        # self.broker_adapter = broker_adapter # Replaced
        self.broker_adapter = None
        self.datafeed_config = (
            datafeed_config  # Example: {'symbol': 'MESM25', 'timeframe': '1Min'}
        )
        self.on_trade = on_trade
        self.on_error = on_error
        self._status = "stopped"
        self._thread = None
        self.cerebro = None
        # self.console = Console()  # Replaced by logger

        # Instantiate broker adapter based on config
        provider = broker_config.get("provider")
        if provider == "alpaca":
            # AlpacaBrokerAdapter expects the LiveRunner instance for callbacks
            self.broker_adapter = AlpacaBrokerAdapter(live_runner=self)
            self._log("Initialized AlpacaBrokerAdapter.", level=logging.INFO)
            # Attempt to connect the broker adapter immediately
            try:
                self.broker_adapter.connect()
                self._log(
                    "AlpacaBrokerAdapter connected successfully.", level=logging.INFO
                )
            except Exception as e:
                self._log(
                    f"Failed to connect AlpacaBrokerAdapter: {e}", level=logging.ERROR
                )
                # Optionally re-raise or handle this critical failure
                # For now, we log and continue, but the adapter might not be usable.
                if self.on_error:
                    self.on_error(f"Failed to connect AlpacaBrokerAdapter: {e}")
        # elif provider == 'tradovate': # Example for another provider
        # self.broker_adapter = TradovateBrokerAdapter(live_runner=self, **broker_config.get('settings', {}))
        # self._log(f"Initialized TradovateBrokerAdapter.", level=logging.INFO)
        elif provider == "mock":
            # For testing: use the mock adapter provided in the config
            self.broker_adapter = broker_config.get("adapter")
            if self.broker_adapter:
                self._log("Using provided mock broker adapter.", level=logging.INFO)
            else:
                error_msg = "No mock adapter provided in broker_config."
                self._log(error_msg, level=logging.ERROR)
                raise ValueError(error_msg)
        else:
            error_msg = f"Unsupported broker provider: {provider}. Please choose 'alpaca' or 'mock'."
            self._log(error_msg, level=logging.ERROR)
            raise ValueError(error_msg)

    def _log(
        self, message: str, level: int = logging.INFO, style: str = ""
    ):  # Added level, style might be deprecated
        # Rich styling won't directly apply to standard logger without custom handlers/formatters
        # For simplicity, we'll just log the message.
        # The 'style' parameter might need to be re-thought or mapped to log levels.

        # Map rich styles to log levels if desired, or simplify
        if style == "bold red":
            level = logging.ERROR
        elif style == "yellow":
            level = logging.WARNING
        elif style == "green":
            level = logging.INFO  # Or logging.DEBUG if "green" implies success details
        elif style == "blue":
            level = logging.DEBUG  # Or logging.INFO

        # Standard logging doesn't use pendulum directly in the log message like rich's console.log timestamp
        # Timestamps are typically handled by the logger's formatter.
        logger.log(level, f"LiveRunner: {message}")

    def _import_strategy(self):
        module_path, class_name = self.strategy_path.rsplit(":", 1)
        module = importlib.import_module(module_path)
        strategy_class = getattr(module, class_name)
        return strategy_class

    def start(self):
        if self._status == "running":
            self._log("Already running.", level=logging.WARNING)  # Use logging level
            return

        self._log("Starting...", level=logging.INFO)  # Use logging level
        try:
            self._status = "running"  # Set to running optimistically

            self.cerebro = bt.Cerebro(
                stdstats=False
            )  # Disable standard Backtrader stats for live trading

            # Here you would integrate the broker adapter with Backtrader's broker interface
            # For now, we'll just pass it along, but a real integration is more complex.
            # self.cerebro.setbroker(self.broker_adapter) # This would need a Backtrader compatible broker

            StrategyClass = self._import_strategy()
            self.cerebro.addstrategy(StrategyClass, **self.params)

            # Data feed setup would go here. This is highly dependent on the data source.
            # For a true live feed, you'd use something like cerebro.adddata_live(...)
            # For now, this is a placeholder.
            self._log(
                f"Datafeed config: {self.datafeed_config}", level=logging.DEBUG
            )  # Use logging level

            # The on_trade and on_error callbacks would be connected to the strategy or broker events.
            # This is a simplified representation.
            if self.on_trade:
                # This is not a direct Backtrader mechanism; actual implementation depends on strategy/broker design
                self._log(
                    "on_trade callback registered.", level=logging.DEBUG
                )  # Use logging level
            if self.on_error:
                self._log(
                    "on_error callback registered.", level=logging.DEBUG
                )  # Use logging level

            def run_cerebro():
                try:
                    self.cerebro.run()
                    self._status = "stopped"
                    self._log(
                        "Run loop finished.", level=logging.INFO
                    )  # Use logging level
                except Exception as e:
                    self._status = "error"
                    self._log(
                        f"Error during run: {e}", level=logging.ERROR
                    )  # Use logging level
                    if self.on_error:
                        self.on_error(e)

            self._thread = threading.Thread(target=run_cerebro, daemon=True)
            self._thread.start()
            self._log("Started successfully.", level=logging.INFO)  # Use logging level

        except Exception as e:
            self._status = "error"
            self._log(
                f"Error during start: {e}", level=logging.ERROR
            )  # Use logging level
            if self.on_error:
                self.on_error(e)
            # Do not proceed to start thread if setup failed. self._thread remains None.

    def stop(self):
        if self._status != "running":
            self._log("Not running.", level=logging.WARNING)  # Use logging level
            return

        self._log("Stopping...", level=logging.INFO)  # Use logging level
        if self.cerebro:
            self.cerebro.runstop()  # Request graceful stop

        if self._thread and self._thread.is_alive():
            self._thread.join(
                timeout=5
            )  # Wait for thread to finish, adjusted timeout to 5s
            if self._thread.is_alive():
                self._log(
                    "Thread did not stop in time.", level=logging.ERROR
                )  # Use logging level
                self._status = "error"
            else:
                self._log(
                    "Stopped successfully.", level=logging.INFO
                )  # Use logging level
        else:
            self._log(
                "No active thread to stop.", level=logging.WARNING
            )  # Use logging level

        self._status = "stopped"

    def status(self) -> str:
        return self._status
