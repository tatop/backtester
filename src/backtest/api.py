"""Stable public API facade for the backtest library.

This module provides a clean, stable interface for running backtests
that both the CLI and WebUI can use.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from backtest.data_loader import (
    DownloadSummary,
    align_price_data,
    download_yahoo,
    load_multiple_symbols,
    load_price_series,
)
from backtest.engine import BacktestEngine, BacktestParams, BacktestResult
from backtest.metrics import (
    compute_all_metrics,
    compute_cagr,
    compute_max_drawdown,
    compute_sharpe_ratio,
    compute_total_return,
    compute_annualized_volatility,
)
from backtest.portfolio import PortfolioConfig, normalize_weights


@dataclass
class BacktestRequest:
    """Request parameters for running a backtest."""

    symbols: list[str]
    weights: Optional[list[float]] = None
    initial_capital: float = 10_000.0
    rebalance_frequency: str = "yearly"
    transaction_cost: float = 0.001
    align_method: str = "inner"
    data_dir: str = "data"
    benchmark: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None


@dataclass
class MetricsResult:
    """Numeric metrics from a backtest (machine-readable)."""

    total_return: float
    cagr: float
    volatility: float
    max_drawdown: float
    sharpe_ratio: float

    def as_formatted_dict(self) -> dict[str, str]:
        """Return metrics as formatted strings (CLI-friendly)."""
        return {
            "total_return": f"{self.total_return:.2%}",
            "cagr": f"{self.cagr:.2%}",
            "volatility": f"{self.volatility:.2f}%",
            "max_drawdown": f"{self.max_drawdown:.2%}",
            "sharpe_ratio": f"{self.sharpe_ratio:.2f}",
        }


@dataclass
class BacktestResponse:
    """Response from running a backtest."""

    nav_series: pd.Series
    weights_over_time: Optional[pd.DataFrame]
    metrics: MetricsResult
    final_nav: float
    benchmark_nav_series: Optional[pd.Series] = None
    benchmark_metrics: Optional[MetricsResult] = None


def build_weights(symbols: list[str], weights: Optional[list[float]]) -> dict[str, float]:
    """Build a weights dictionary from symbols and optional weight values.

    Args:
        symbols: List of symbol names.
        weights: Optional list of weights. If None, equal weights are used.

    Returns:
        Dictionary mapping symbol names to normalized weights.

    Raises:
        ValueError: If weights length doesn't match symbols length.
    """
    if weights is None:
        equal_weight = 1.0 / len(symbols)
        weights = [equal_weight] * len(symbols)
    if len(weights) != len(symbols):
        raise ValueError("Il numero di pesi deve corrispondere al numero di simboli.")
    weight_dict = {sym: float(w) for sym, w in zip(symbols, weights)}
    return normalize_weights(weight_dict)


def compute_metrics(nav_series: pd.Series) -> MetricsResult:
    """Compute all performance metrics for a NAV series.

    Args:
        nav_series: Series of portfolio NAV values indexed by date.

    Returns:
        MetricsResult with numeric values for all metrics.
    """
    return MetricsResult(
        total_return=compute_total_return(nav_series),
        cagr=compute_cagr(nav_series),
        volatility=compute_annualized_volatility(nav_series),
        max_drawdown=compute_max_drawdown(nav_series),
        sharpe_ratio=compute_sharpe_ratio(nav_series),
    )


def run_backtest(request: BacktestRequest) -> BacktestResponse:
    """Run a portfolio backtest.

    Args:
        request: BacktestRequest with all parameters.

    Returns:
        BacktestResponse with NAV series, weights, and metrics.

    Raises:
        FileNotFoundError: If CSV files for symbols are missing.
        ValueError: If data is invalid or parameters are incorrect.
    """
    data_path = Path(request.data_dir)
    missing_files = [
        symbol
        for symbol in request.symbols
        if not (data_path / f"{symbol}.csv").exists()
    ]
    if missing_files:
        raise FileNotFoundError(f"Mancano i file CSV per: {', '.join(missing_files)}")

    weights = build_weights(request.symbols, request.weights)
    prices = load_multiple_symbols(request.symbols, data_dir=request.data_dir)
    prices = align_price_data(prices, method=request.align_method)

    benchmark_series = None
    benchmark_label = None
    if request.benchmark:
        try:
            benchmark_series = load_price_series(request.benchmark, data_dir=request.data_dir)
            benchmark_label = benchmark_series.name
        except Exception:
            benchmark_series = None

    engine = BacktestEngine(
        prices_df=prices,
        portfolio_config=PortfolioConfig(
            weights=weights, initial_capital=request.initial_capital
        ),
        params=BacktestParams(
            rebalance_frequency=request.rebalance_frequency,
            transaction_cost=request.transaction_cost,
            start_date=request.start_date,
            end_date=request.end_date,
        ),
        benchmark_prices=benchmark_series,
        benchmark_label=benchmark_label,
    )
    result = engine.run()

    metrics = compute_metrics(result.nav_series)

    benchmark_metrics = None
    if result.benchmark_nav_series is not None:
        benchmark_metrics = compute_metrics(result.benchmark_nav_series)

    return BacktestResponse(
        nav_series=result.nav_series,
        weights_over_time=result.weights_over_time,
        metrics=metrics,
        final_nav=result.metrics["final_nav"] if result.metrics else result.nav_series.iloc[-1],
        benchmark_nav_series=result.benchmark_nav_series,
        benchmark_metrics=benchmark_metrics,
    )


def load_prices(
    symbols: list[str],
    data_dir: str = "data",
    align_method: str = "inner",
) -> pd.DataFrame:
    """Load and align price data for multiple symbols.

    Args:
        symbols: List of symbol names to load.
        data_dir: Directory containing CSV files.
        align_method: Method for aligning time series ('inner', 'outer', 'ffill', 'bfill').

    Returns:
        DataFrame with aligned price data, one column per symbol.
    """
    prices = load_multiple_symbols(symbols, data_dir=data_dir)
    return align_price_data(prices, method=align_method)


def download_prices(
    symbols: list[str],
    start: Optional[str] = None,
    end: Optional[str] = None,
    data_dir: str = "data",
) -> DownloadSummary:
    """Download historical price data from Yahoo Finance.

    Args:
        symbols: List of ticker symbols to download.
        start: Start date (YYYY-MM-DD), defaults to 5 years ago.
        end: End date (YYYY-MM-DD), defaults to today.
        data_dir: Directory to save CSV files.

    Returns:
        DownloadSummary with success/error information.
    """
    return download_yahoo(symbols, start=start, end=end, data_dir=data_dir)


def list_available_symbols(data_dir: str = "data") -> list[str]:
    """List all available symbols (CSV files) in the data directory.

    Args:
        data_dir: Directory containing CSV files.

    Returns:
        List of symbol names (without .csv extension).
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        return []
    return sorted(p.stem for p in data_path.glob("*.csv"))

