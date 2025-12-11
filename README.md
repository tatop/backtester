# Passive Portfolio Backtester

A small Python playground for testing passive ETF portfolios built entirely with in-house logic. The repository contains a CLI that loads local CSV data, simulates a share-based portfolio with optional rebalancing and transaction costs, computes standard metrics, and (optionally) renders a Bokeh dashboard.

## Features
- Load one CSV per symbol from `data/`, auto-detect date/price columns, deduplicate dates, and align the price series by intersection, union, forward fill, or backward fill.
- Create a static portfolio with validated weights (defaults to equal weight) and configurable initial capital.
- Simulate daily NAV using explicit share counts, rebalance monthly/quarterly/yearly (or never), and deduct proportional transaction costs at each trade.
- Export NAV, weights-through-time, and trades; expose summary metrics (final NAV, total return, CAGR, volatility, max drawdown, Sharpe).
- Plot an interactive dashboard (equity, drawdown, stacked weights, KPI cards) using Bokeh.

## Project Layout
```
main.py              # Entry point -> backtest.CLI.main
src/backtest/
  CLI.py             # argparse + run_cli_backtest orchestrator
  data_loader.py     # load_price_series(), align_price_data()
  portfolio.py       # PortfolioConfig, weight helpers
  engine.py          # BacktestEngine, BacktestParams/Result
  metrics.py         # NAV->metrics utilities
  plotting.py        # Bokeh dashboard
```
`data/` must contain one CSV per symbol with at least `Date` and price (e.g. `Close`) columns.

## Requirements & Setup
- Python 3.13+
- Managed by [uv](https://github.com/astral-sh/uv); dependencies are listed in `pyproject.toml`.

```sh
uv sync
```

## Preparing Data
- Place CSVs like `data/SPY.csv`, `data/STOXX50.csv`.
- Required columns: a date column (`Date`, `Datetime`, etc.) and at least one numeric price column (priority: Adj Close → Close → first numeric).
- Dates are parsed with pandas; duplicated dates are dropped (keep first).

## CLI Usage
Typical run:
```sh
uv run python main.py \
  --symbols SPY STOXX50 \
  --weights 0.5 0.5 \
  --initial 10000 \
  --rebalance yearly \
  --transaction-cost 0.001 \
  --align inner \
  --plot
```

### Arguments
| Flag | Description |
| --- | --- |
| `--symbols` | Space-separated tickers (default `SPY STOXX50`). |
| `--weights` | Portfolio weights (sum = 1, defaults to equal weight). |
| `--initial` | Initial capital in base currency (default 10,000). |
| `--rebalance` | `none`, `monthly`, `quarterly`, `yearly` (default yearly). |
| `--transaction-cost` | Proportional cost per trade (e.g. `0.001` = 0.1%). |
| `--data-dir` | CSV directory (default `data`). |
| `--align` | Series alignment: `inner`, `outer`, `ffill`, `bfill`. |
| `--plot` | When provided, opens the Bokeh dashboard. |

Console output lists formatted metrics (`Total Return`, `CAGR`, `Volatility`, `Max Drawdown`, `Sharpe`) plus the final NAV. If plotting is enabled, the browser opens with NAV, drawdown, and stacked weights charts.

## Development
- **Format**: `uv run ruff format .`
- **Lint**: `uv run ruff check .`
- **Type check**: `uv run mypy src/`
- **Run CLI**: `uv run python main.py --symbols SPY STOXX50 --weights 0.5 0.5 --plot`

Tests are not set up yet; add pytest cases under `tests/` and run `uv run pytest` once available.

## Next Steps
Ideas for future iterations:
1. Add periodic contributions / withdrawals.
2. Support FX conversion for multi-currency data.
3. Extend metrics (Sortino, rolling stats) and export to CSV/JSON.
4. Automate data fetching (e.g., yfinance) alongside the CSV loader.
