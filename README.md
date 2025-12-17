# Passive Portfolio Backtester

A Python playground for testing passive ETF portfolios built entirely with in-house logic. The project provides both a CLI and a web interface for simulating share-based portfolios with rebalancing, transaction costs, and benchmark comparisons.

## Features
- **Multiple interfaces**: Command-line tool and REST API with web UI
- **Flexible data sources**: Load local CSV files or download historical data from Yahoo Finance
- **Intelligent data handling**: Auto-detect date/price columns, deduplicate dates, and align price series using intersection, union, forward fill, or backward fill
- **Realistic simulation**: Share-based portfolio tracking with configurable rebalancing (monthly/quarterly/yearly) and proportional transaction costs
- **Benchmark comparison**: Compare your portfolio against any single-asset benchmark
- **Comprehensive metrics**: Total return, CAGR, volatility, max drawdown, and Sharpe ratio
- **Interactive visualization**: Bokeh dashboard with equity curves, drawdown charts, weight evolution, and KPI cards

## Project Layout
```
main.py                       # CLI entry point
src/backtest/
  CLI.py                      # Command-line interface
  api.py                      # Public API facade (used by CLI & WebUI)
  data_loader.py              # CSV loading, Yahoo Finance downloads, alignment
  portfolio.py                # PortfolioConfig, weight validation
  engine.py                   # BacktestEngine simulation logic
  metrics.py                  # Performance calculations
  plotting.py                 # Bokeh visualizations
apps/webui/
  src/backtest_webui/
    api.py                    # FastAPI REST endpoints
    schemas.py                # Pydantic request/response models
  static/                     # HTML/JS/CSS frontend
data/                         # CSV files (Date, Close columns)
```

## Requirements & Setup
- Python 3.13+
- Managed by [uv](https://github.com/astral-sh/uv)
- Core dependencies: pandas, numpy, bokeh, yfinance

Install the CLI and library:
```sh
uv sync
```

Install the web UI (from `apps/webui/`):
```sh
cd apps/webui
uv sync
```

## Data Management

### Option 1: Download from Yahoo Finance
Use the CLI to fetch historical data automatically:
```sh
uv run python main.py download --symbols SPY QQQ --start 2020-01-01
```

Or use the web UI's download endpoint.

### Option 2: Provide Local CSV Files
Place CSV files in the `data/` directory (e.g., `data/SPY.csv`, `data/STOXX50.csv`).

**Required columns:**
- A date column: `Date`, `Datetime`, etc.
- At least one price column (priority: `Adj Close` > `Close` > first numeric column)

Dates are auto-parsed by pandas; duplicates are dropped (keeping the first occurrence).

## CLI Usage

### Run a Backtest
```sh
uv run python main.py \
  --symbols SPY STOXX50 \
  --weights 0.5 0.5 \
  --initial 10000 \
  --rebalance yearly \
  --transaction-cost 0.001 \
  --align inner \
  --benchmark SPY \
  --plot
```

### CLI Arguments
| Flag | Description |
| --- | --- |
| `--symbols` | Space-separated tickers (default: `SPY STOXX50`) |
| `--weights` | Portfolio weights (must sum to 1; defaults to equal weight) |
| `--initial` | Initial capital in base currency (default: 10,000) |
| `--rebalance` | Rebalancing frequency: `none`, `monthly`, `quarterly`, `yearly` (default: `yearly`) |
| `--transaction-cost` | Proportional cost per trade, e.g., `0.001` = 0.1% (default: 0.001) |
| `--data-dir` | Directory containing CSV files (default: `data`) |
| `--align` | Price series alignment method: `inner`, `outer`, `ffill`, `bfill` (default: `inner`) |
| `--benchmark` | Optional benchmark symbol for comparison (e.g., `SPY`) |
| `--plot` | Open interactive Bokeh dashboard in browser |

The CLI outputs formatted metrics (Total Return, CAGR, Volatility, Max Drawdown, Sharpe Ratio) to the console. When `--benchmark` is provided, both portfolio and benchmark metrics are displayed side-by-side. If `--plot` is enabled, an interactive dashboard opens with equity curves, drawdown charts, and weight evolution.

## Web UI Usage

Start the web server:
```sh
cd apps/webui
uv run uvicorn backtest_webui.api:app --reload
```

Access the interface at `http://localhost:8000`

### API Endpoints

**`GET /api/symbols`** - List available symbols in the data directory

**`POST /api/backtest`** - Run a portfolio backtest
```json
{
  "symbols": ["SPY", "STOXX50"],
  "weights": [0.5, 0.5],
  "initial_capital": 10000,
  "rebalance_frequency": "yearly",
  "transaction_cost": 0.001,
  "align_method": "inner",
  "benchmark": "SPY"
}
```

**`POST /api/download`** - Download historical data from Yahoo Finance
```json
{
  "symbols": ["SPY", "QQQ"],
  "start": "2020-01-01",
  "end": "2024-01-01"
}
```

### Environment Variables
- `BACKTEST_DATA_DIR`: Path to CSV data directory (default: `data` relative to current directory)

## Development

### Code Quality Tools
```sh
# Format code
uv run ruff format .

# Lint
uv run ruff check .

# Type check
uv run mypy src/

# Run a sample backtest
uv run python main.py --symbols SPY STOXX50 --weights 0.5 0.5 --plot
```

### Testing
Tests are not yet implemented. Add pytest cases under `tests/` and run:
```sh
uv run pytest
```

### Architecture Notes
- The project uses a clean separation between the core library (`src/backtest/`) and interfaces (CLI in `main.py`, Web UI in `apps/webui/`)
- The `api.py` module provides a stable facade for both CLI and Web UI to consume
- All simulation logic is manual (no external backtest engines); share-based accounting with explicit rebalancing
- Type hints are used throughout (Python 3.13+ style with `list[...]`, `dict[...]`)
- Italian comments are acceptable; code remains in English

## Future Enhancements
Ideas for upcoming iterations:
1. Periodic contributions and withdrawals (dollar-cost averaging scenarios)
2. Multi-currency support with FX conversion
3. Additional metrics: Sortino ratio, Calmar ratio, rolling statistics
4. Export results to CSV/JSON/Excel
5. Tax-aware simulations (capital gains, dividends)
6. Monte Carlo simulation and scenario analysis
