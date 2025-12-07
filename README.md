# Passive Portfolio Backtester

A modular Python framework for backtesting passive ETF portfolios.
The system is designed for clarity, extensibility, and full control over the backtesting logic.
No external backtesting engines are used: all portfolio mechanics (weights, rebalancing, NAV computation) are implemented manually.

The project supports:

* Local CSV data loading
* Custom portfolio configurations
* Flexible rebalancing rules
* Standard performance metrics
* CLI usage
* Interactive visualizations (via Bokeh)

---

## Features

### Core

* Load price data for any set of ETFs from local CSV files.
* Build portfolios with custom weights and initial capital.
* Compute NAV using precise share-based portfolio logic.
* Optional rebalancing:

  * None
  * Monthly
  * Quarterly
  * Yearly
* Optional transaction costs.

### Metrics

* Total Return
* CAGR
* Annualized Volatility
* Max Drawdown
* Sharpe Ratio
* Daily returns

### Plotting

* Equity curve (NAV over time)
* Drawdown curve
* Weight allocation over time (for rebalanced portfolios)

### CLI

* Run complete backtests directly from the terminal.
* Output results to stdout.
* Optional flag to display interactive charts in the browser.

---

## Project Structure

```
src/
    main.py

data/
    <your_etf_files>.csv

pyproject.toml
README.md
```

---

## Installation

### Requirements

* Python >= 3.10
* pandas
* numpy
* bokeh

Install dependencies:

```sh
uv sync
```

(If no `pyproject.toml` exists, install manually.)

---

## Preparing the Data

Each ETF must have a CSV file stored in the `data/` directory.

Expected CSV format:

```
Date,Close
2020-01-01,100.25
2020-01-02,100.40
...
```

* `Date` must be parseable as YYYY-MM-DD.
* `Close` should ideally be adjusted prices.

---

## Usage (CLI) not yet implemented

Run a full backtest from the command line:

```sh
python main.py \
    --symbols VWCE AGGH \
    --weights 0.6 0.4 \
    --initial 10000 \
    --rebalance yearly \
    --plot
```

### Arguments

| Flag          | Description                              |
| ------------- | ---------------------------------------- |
| `--symbols`   | List of ETF symbols                      |
| `--weights`   | Corresponding weights (must sum to 1)    |
| `--initial`   | Initial capital                          |
| `--rebalance` | `none`, `monthly`, `quarterly`, `yearly` |
| `--plot`      | Show Bokeh charts                        |

Example without plotting:

```sh
python main.py --symbols VWCE AGGH --weights 0.5 0.5 --initial 5000
```

---

## Output Examples

### Console Output

```
Backtest Results
----------------
Total Return: 42.8%
CAGR: 8.3%
Volatility (ann.): 12.5%
Max Drawdown: -18.7%
Sharpe Ratio: 0.66
```

### Graphs

If `--plot` is used:

* The NAV curve will open in your browser
* Additional plots available (e.g., weights over time)

---

## Extending the Framework

This project is designed to be extended.
Some possible extensions:

* Periodic contributions (PAC / DCA)
* Multi-currency portfolios with FX conversion
* Factor attribution analysis
* Monte Carlo simulations
* Custom rebalancing rules (e.g., tolerance bands)
* Transaction cost models

Each module is intentionally simple, with clear interfaces that allow progressive sophistication.

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

All code should follow a clean modular structure consistent with the existing layout.

---

## License

Specify a license here (MIT recommended).

---

If you'd like, I can also prepare:

* A `requirements.txt`
* A sample `main.py`
* Example CSV files
* A version of the README tailored to a specific GitHub repository name.
