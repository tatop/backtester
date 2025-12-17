"""FastAPI application for the backtest web UI."""

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backtest import (
    BacktestRequest,
    download_prices,
    list_available_symbols,
    run_backtest,
)

from backtest_webui.schemas import (
    BacktestRequestSchema,
    BacktestResponseSchema,
    DownloadedSymbolSchema,
    DownloadRequestSchema,
    DownloadResponseSchema,
    ErrorResponseSchema,
    MetricsSchema,
    NavPointSchema,
    SymbolsResponseSchema,
    WeightPointSchema,
)

# Determine data directory - use environment variable or default
DATA_DIR = os.environ.get("BACKTEST_DATA_DIR", "data")

app = FastAPI(
    title="Backtest Web UI",
    description="Web interface for portfolio backtesting",
    version="0.1.0",
)

# CORS middleware for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
STATIC_DIR = Path(__file__).parent.parent.parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=FileResponse)
async def root():
    """Serve the main HTML page."""
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return FileResponse(index_path)


@app.get("/api/symbols", response_model=SymbolsResponseSchema)
async def get_symbols():
    """List all available symbols (CSV files in data directory)."""
    symbols = list_available_symbols(data_dir=DATA_DIR)
    return SymbolsResponseSchema(symbols=symbols)


@app.post(
    "/api/backtest",
    response_model=BacktestResponseSchema,
    responses={400: {"model": ErrorResponseSchema}, 404: {"model": ErrorResponseSchema}},
)
async def execute_backtest(request: BacktestRequestSchema):
    """Run a portfolio backtest with the given parameters."""
    try:
        # Convert Pydantic schema to library request
        backtest_request = BacktestRequest(
            symbols=request.symbols,
            weights=request.weights,
            initial_capital=request.initial_capital,
            rebalance_frequency=request.rebalance_frequency,
            transaction_cost=request.transaction_cost,
            align_method=request.align_method,
            data_dir=DATA_DIR,
            benchmark=request.benchmark,
            start_date=request.start_date,
            end_date=request.end_date,
        )

        response = run_backtest(backtest_request)

        # Convert NAV series to list of points
        nav_series = [
            NavPointSchema(date=str(date.date()), value=float(value))
            for date, value in response.nav_series.items()
        ]

        # Convert weights over time
        weights_over_time = None
        if response.weights_over_time is not None:
            weights_over_time = [
                WeightPointSchema(
                    date=str(date.date()),
                    weights={col: float(row[col]) for col in response.weights_over_time.columns},
                )
                for date, row in response.weights_over_time.iterrows()
            ]

        # Convert metrics
        metrics = MetricsSchema(
            total_return=response.metrics.total_return,
            cagr=response.metrics.cagr,
            volatility=response.metrics.volatility,
            max_drawdown=response.metrics.max_drawdown,
            sharpe_ratio=response.metrics.sharpe_ratio,
        )

        # Convert benchmark if present
        benchmark_nav_series = None
        benchmark_metrics = None
        if response.benchmark_nav_series is not None:
            benchmark_nav_series = [
                NavPointSchema(date=str(date.date()), value=float(value))
                for date, value in response.benchmark_nav_series.items()
            ]
        if response.benchmark_metrics is not None:
            benchmark_metrics = MetricsSchema(
                total_return=response.benchmark_metrics.total_return,
                cagr=response.benchmark_metrics.cagr,
                volatility=response.benchmark_metrics.volatility,
                max_drawdown=response.benchmark_metrics.max_drawdown,
                sharpe_ratio=response.benchmark_metrics.sharpe_ratio,
            )

        return BacktestResponseSchema(
            metrics=metrics,
            final_nav=response.final_nav,
            nav_series=nav_series,
            weights_over_time=weights_over_time,
            benchmark_metrics=benchmark_metrics,
            benchmark_nav_series=benchmark_nav_series,
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post(
    "/api/download",
    response_model=DownloadResponseSchema,
    responses={400: {"model": ErrorResponseSchema}},
)
async def download_data(request: DownloadRequestSchema):
    """Download price data from Yahoo Finance."""
    try:
        summary = download_prices(
            symbols=request.symbols,
            start=request.start,
            end=request.end,
            data_dir=DATA_DIR,
        )

        successes = [
            DownloadedSymbolSchema(
                symbol=s.symbol,
                rows=s.rows,
                start=str(s.start.date()),
                end=str(s.end.date()),
            )
            for s in summary.successes
        ]

        return DownloadResponseSchema(successes=successes, errors=summary.errors)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


def main():
    """Entry point for running the web server."""
    import uvicorn

    # Resolve data directory relative to workspace root when running as script
    global DATA_DIR
    if not Path(DATA_DIR).is_absolute():
        # Look for data directory relative to current working directory
        cwd_data = Path.cwd() / DATA_DIR
        if cwd_data.exists():
            DATA_DIR = str(cwd_data)

    uvicorn.run(
        "backtest_webui.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )


if __name__ == "__main__":
    main()

