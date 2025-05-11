import backtrader as bt


class DummyStrategy(bt.Strategy):
    """A simple strategy that does nothing, used for testing only."""

    params = (("param1", None),)

    def __init__(self):
        """Initialize the strategy."""
        pass

    def next(self):
        """Called for each bar update."""
        pass
