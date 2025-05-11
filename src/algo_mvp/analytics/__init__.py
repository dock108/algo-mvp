"""Analytics API for P&L curves, trade logs, and performance statistics.

This package provides a unified interface for querying trade data, P&L,
and performance metrics from the algo_mvp database.
"""

from algo_mvp.analytics.api import AnalyticsAPI

__all__ = ["AnalyticsAPI"]
