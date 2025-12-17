"""backtest - A portfolio backtesting library.

This module exposes the stable public API for the backtest library.
"""

from backtest.api import (
    BacktestRequest,
    BacktestResponse,
    MetricsResult,
    build_weights,
    compute_metrics,
    download_prices,
    list_available_symbols,
    load_prices,
    run_backtest,
)
from backtest.data_loader import DownloadSummary, DownloadedSymbol
from backtest.engine import BacktestEngine, BacktestParams, BacktestResult
from backtest.portfolio import PortfolioConfig

__all__ = [
    # Main API functions
    "run_backtest",
    "load_prices",
    "download_prices",
    "list_available_symbols",
    "build_weights",
    "compute_metrics",
    # Request/Response types
    "BacktestRequest",
    "BacktestResponse",
    "MetricsResult",
    # Engine types (for advanced usage)
    "BacktestEngine",
    "BacktestParams",
    "BacktestResult",
    "PortfolioConfig",
    # Data types
    "DownloadSummary",
    "DownloadedSymbol",
]

