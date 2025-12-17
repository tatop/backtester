---
name: Engine time frame filtering
overview: Add optional start/end date parameters to the backtest engine so runs can be limited to a specific historical window, with clipping to available data, and thread the new params through CLI and WebUI without breaking existing callers.
todos:
  - id: engine-timeframe
    content: Add start_date/end_date to BacktestParams and filter prices_df/benchmark inside BacktestEngine.__init__ with clipping + empty-range error.
    status: pending
  - id: api-threading
    content: Extend BacktestRequest with start_date/end_date and pass them into BacktestParams in run_backtest.
    status: pending
    dependencies:
      - engine-timeframe
  - id: cli-bt-flags
    content: Add --bt-start/--bt-end to CLI and include them in BacktestRequest creation.
    status: pending
    dependencies:
      - api-threading
  - id: webui-timeframe
    content: Add optional start_date/end_date to WebUI BacktestRequestSchema and pass them into BacktestRequest.
    status: pending
    dependencies:
      - api-threading
---

# Add backtest time frame (start/end dates)

## Goal

Allow running the same price data through `BacktestEngine` but **only over a specified date window**, using **explicit `start_date` / `end_date`** and **clipping to available data**.

## Proposed changes

- **Add engine parameters**: Extend `BacktestParams` in [`src/backtest/engine.py`](/Users/pietromagni/Development/Python/Finanza/backtest/src/backtest/engine.py) with optional `start_date` and `end_date`.
- **Filter inside the engine**: In `BacktestEngine.__init__`, apply the time filter to `prices_df` (and align/clip `benchmark_prices` to the filtered index) before any simulation state is initialized.
- **Date parsing**: accept `None`, `str` (`YYYY-MM-DD`), or `pd.Timestamp`.
- **Clipping behavior**: if requested dates exceed the data range, intersect with the available range; if the intersection is empty, raise `ValueError` describing the available date span.
- **Inclusivity**: treat both `start_date` and `end_date` as inclusive.
- **Thread through stable API**: Update `BacktestRequest` and `run_backtest` in [`src/backtest/api.py`](/Users/pietromagni/Development/Python/Finanza/backtest/src/backtest/api.py) to accept and pass `start_date`/`end_date` into `BacktestParams`.
- **CLI flags**: Add separate backtest flags to [`src/backtest/CLI.py`](/Users/pietromagni/Development/Python/Finanza/backtest/src/backtest/CLI.py) as requested:
- `--bt-start` and `--bt-end` (strings `YYYY-MM-DD`)
- map them to `BacktestRequest.start_date` / `BacktestRequest.end_date`
- **WebUI API**: Extend [`apps/webui/src/backtest_webui/schemas.py`](/Users/pietromagni/Development/Python/Finanza/backtest/apps/webui/src/backtest_webui/schemas.py) and request construction in [`apps/webui/src/backtest_webui/api.py`](/Users/pietromagni/Development/Python/Finanza/backtest/apps/webui/src/backtest_webui/api.py) to optionally pass `start_date`/`end_date`.

## Notes / compatibility

- Existing callers that don’t pass `start_date`/`end_date` will behave exactly as today (full available history).
- The filtered date window will be reflected naturally in the returned `nav_series` index (and benchmark series, if enabled).

## Implementation todos

- `engine-timeframe`: Add `start_date`/`end_date` to `BacktestParams` and implement clipping/filtering in `BacktestEngine.__init__`.
- `api-threading`: Add `start_date`/`end_date` to `BacktestRequest` and pass through `run_backtest`.
- `cli-bt-flags`: Add `--bt-start/--bt-end` and map into `BacktestRequest`.
- `webui-timeframe`: Extend WebUI request schema and mapping to support optional `start_date`/`end_date`.