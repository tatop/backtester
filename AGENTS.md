# AGENTS.md

## Commands
- **Install deps**: `uv sync`
- **Run CLI**: `uv run python main.py --symbols SPY STOXX50 --weights 0.5 0.5 --plot`
- **Type check**: `uv run mypy src/`
- **Format**: `uv run ruff format .`
- **Lint**: `uv run ruff check .`
- **Test**: No tests configured yet; use `uv run pytest` if added

## Architecture
- `main.py` - Entry point, calls CLI module
- `src/backtest/` - Core package:
  - `CLI.py` - Argument parsing, orchestrates backtest
  - `engine.py` - BacktestEngine, BacktestParams, BacktestResult (main simulation logic)
  - `portfolio.py` - PortfolioConfig, weight validation/normalization
  - `data_loader.py` - CSV loading, price alignment
  - `metrics.py` - Performance metrics (CAGR, Sharpe, drawdown)
  - `plotting.py` - Bokeh visualizations
- `data/` - CSV files with Date,Close columns

## Code Style
- Python 3.13+, use type hints (Dict, Optional, list[...])
- Dataclasses for config/result objects
- Use numpy for vectorized math, pandas for time series
- Italian comments acceptable; keep code in English
- No external backtest engines; all logic is manual
