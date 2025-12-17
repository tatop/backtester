# Backtest WebUI

Web interface for the backtest portfolio library.

## Installation

From the `apps/webui` directory:

```bash
uv sync
```

## Running the Server

```bash
# From apps/webui directory
uv run uvicorn backtest_webui.api:app --reload

# Or use the entry point
uv run backtest-webui
```

The server will start at `http://localhost:8000`.

## Environment Variables

- `BACKTEST_DATA_DIR`: Path to the directory containing CSV price data (default: `data` relative to current working directory)

## API Endpoints

### `GET /api/symbols`
List all available symbols (CSV files in data directory).

### `POST /api/backtest`
Run a portfolio backtest.

Request body:
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

### `POST /api/download`
Download price data from Yahoo Finance.

Request body:
```json
{
    "symbols": ["SPY", "QQQ"],
    "start": "2020-01-01",
    "end": "2024-01-01"
}
```

## Development

The WebUI is a separate distribution that depends on the `backtest` library. During development, it uses a path dependency to the library.

When publishing, change the dependency in `pyproject.toml` to a version requirement:

```toml
dependencies = [
    "backtest>=0.1.0",
    ...
]
```

