"""Pydantic schemas for the backtest web API."""

from typing import Optional

from pydantic import BaseModel, Field


class BacktestRequestSchema(BaseModel):
    """Request schema for running a backtest."""

    symbols: list[str] = Field(..., description="List of ticker symbols", min_length=1)
    weights: Optional[list[float]] = Field(
        None, description="Portfolio weights (must match symbols length, sum to 1)"
    )
    initial_capital: float = Field(10_000.0, gt=0, description="Initial capital")
    rebalance_frequency: str = Field(
        "yearly",
        description="Rebalance frequency",
        pattern="^(none|monthly|quarterly|yearly)$",
    )
    transaction_cost: float = Field(
        0.001, ge=0, le=0.1, description="Transaction cost as decimal (e.g., 0.001 = 0.1%)"
    )
    align_method: str = Field(
        "inner",
        description="Price alignment method",
        pattern="^(inner|outer|ffill|bfill)$",
    )
    benchmark: Optional[str] = Field("SPY", description="Benchmark symbol for comparison")
    start_date: Optional[str] = Field(None, description="Backtest start date (YYYY-MM-DD)")
    end_date: Optional[str] = Field(None, description="Backtest end date (YYYY-MM-DD)")


class MetricsSchema(BaseModel):
    """Metrics response schema."""

    total_return: float
    cagr: float
    volatility: float
    max_drawdown: float
    sharpe_ratio: float


class NavPointSchema(BaseModel):
    """Single NAV data point."""

    date: str
    value: float


class WeightPointSchema(BaseModel):
    """Weights at a single point in time."""

    date: str
    weights: dict[str, float]


class BacktestResponseSchema(BaseModel):
    """Response schema for a completed backtest."""

    metrics: MetricsSchema
    final_nav: float
    nav_series: list[NavPointSchema]
    weights_over_time: Optional[list[WeightPointSchema]] = None
    benchmark_metrics: Optional[MetricsSchema] = None
    benchmark_nav_series: Optional[list[NavPointSchema]] = None


class SymbolsResponseSchema(BaseModel):
    """Response schema for available symbols."""

    symbols: list[str]


class DownloadRequestSchema(BaseModel):
    """Request schema for downloading price data."""

    symbols: list[str] = Field(..., description="Symbols to download", min_length=1)
    start: Optional[str] = Field(None, description="Start date (YYYY-MM-DD)")
    end: Optional[str] = Field(None, description="End date (YYYY-MM-DD)")


class DownloadedSymbolSchema(BaseModel):
    """Info about a successfully downloaded symbol."""

    symbol: str
    rows: int
    start: str
    end: str


class DownloadResponseSchema(BaseModel):
    """Response schema for download operation."""

    successes: list[DownloadedSymbolSchema]
    errors: dict[str, str]


class ErrorResponseSchema(BaseModel):
    """Error response schema."""

    detail: str

