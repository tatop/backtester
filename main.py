import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Optional
from bokeh.plotting import figure, show
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, HoverTool, NumeralTickFormatter, Div
from bokeh.palettes import Category10

def load_price_series(symbol: str, data_dir: str = "data") -> pd.Series:
    """Carica la serie dei prezzi per un singolo ETF da file CSV."""
    csv_path = Path(data_dir) / f"{symbol}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"File CSV non trovato per {symbol}: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Il file {csv_path} è vuoto.")

    # Identifica la colonna delle date (default: la prima colonna).
    date_cols = [c for c in df.columns if c.lower() in {"date", "datetime", "time"}]
    date_col = date_cols[0] if date_cols else df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)

    # Identifica la colonna del prezzo (default: prima colonna numerica disponibile).
    price_candidates = ["adj close", "adj_close", "close", "price", "value"]
    price_col = None
    for candidate in price_candidates:
        matches = [c for c in df.columns if c.lower() == candidate]
        if matches:
            price_col = matches[0]
            break
    if price_col is None:
        numeric_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
        if not numeric_cols:
            raise ValueError(f"Impossibile identificare la colonna prezzo in {csv_path}.")
        price_col = numeric_cols[0]

    series = df[price_col].astype(float).sort_index()
    series = series[~series.index.duplicated(keep="first")]
    series.name = symbol
    return series


def load_multiple_symbols(symbols: list[str], data_dir: str = "data") -> pd.DataFrame:
    """Carica e unisce le serie dei prezzi di più ETF in un unico DataFrame."""
    series_list = [load_price_series(symbol, data_dir=data_dir) for symbol in symbols]
    if not series_list:
        return pd.DataFrame()

    combined = pd.concat(series_list, axis=1)
    return combined.sort_index()


def align_price_data(prices_df: pd.DataFrame, method: str = "inner") -> pd.DataFrame:
    """Allinea le serie temporali delle diverse colonne (ETF)."""
    if prices_df.empty:
        return prices_df

    method = method.lower()
    df = prices_df.sort_index()

    if method == "inner":
        return df.dropna(how="any")
    if method == "outer":
        return df
    if method == "ffill":
        return df.ffill().dropna(how="all")
    if method == "bfill":
        return df.bfill().dropna(how="all")

    raise ValueError("Metodo di allineamento non riconosciuto. Usa 'inner', 'outer', 'ffill' o 'bfill'.")

@dataclass
class PortfolioConfig:
    weights: Dict[str, float]
    initial_capital: float = 10000.0


def validate_weights(weights: Dict[str, float]) -> None:
    """Verifica che i pesi siano validi (>=0 e somma ≈ 1)."""
    if not all(0 <= w <= 1 for w in weights.values()):
        raise ValueError("I pesi devono essere compresi tra 0 e 1.")
    if not np.isclose(sum(weights.values()), 1):
        raise ValueError("La somma dei pesi deve essere uguale a 1.")
    pass


def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    """Normalizza i pesi in modo che la somma sia 1."""
    total = sum(weights.values())
    return {symbol: weight / total for symbol, weight in weights.items()}

@dataclass
class BacktestParams:
    rebalance_frequency: str = "none"  # "none", "monthly", "quarterly", "yearly"
    transaction_cost: float = 0.0      # es. 0.001 = 0.1% per trade

@dataclass
class BacktestResult:
    nav_series: pd.Series
    weights_over_time: Optional[pd.DataFrame] = None
    trades: Optional[pd.DataFrame] = None
    metrics: Optional[Dict[str, float]] = None


class BacktestEngine:
    def __init__(
        self,
        prices_df: pd.DataFrame,
        portfolio_config,
        params: BacktestParams,
    ) -> None:
        if prices_df.empty:
            raise ValueError("Il DataFrame dei prezzi è vuoto.")

        validate_weights(portfolio_config.weights)
        self.weights = normalize_weights(portfolio_config.weights)

        missing = set(self.weights) - set(prices_df.columns)
        if missing:
            raise ValueError(f"Mancano i prezzi per: {', '.join(sorted(missing))}")

        self.symbols = list(self.weights.keys())
        self.prices_df = prices_df.sort_index()[self.symbols]
        self.params = params
        self.initial_capital = float(portfolio_config.initial_capital)

        self.positions = np.zeros(len(self.symbols), dtype=float)
        self.current_nav = 0.0
        self.nav_history: list[float] = []
        self.weight_history: list[Dict[str, float]] = []
        self.trades_history: list[Dict[str, float]] = []

        self.rebalance_dates = set(self._generate_rebalance_dates())

    def run(self) -> BacktestResult:
        """Esegue il backtest completo e restituisce risultati."""
        self._initialize_positions()

        dates = self.prices_df.index
        # Salta la prima data perché già gestita in _initialize_positions.
        for date in dates[1:]:
            if date in self.rebalance_dates:
                self._rebalance_portfolio(date)
            else:
                self._step_without_rebalance(date)

        nav_series = pd.Series(self.nav_history, index=dates, name="NAV")
        weights_df = pd.DataFrame(self.weight_history, index=dates) if self.weight_history else None
        trades_df = pd.DataFrame(self.trades_history).set_index("date") if self.trades_history else None
        metrics = {"final_nav": self.nav_history[-1]} if self.nav_history else None
        return BacktestResult(nav_series=nav_series, weights_over_time=weights_df, trades=trades_df, metrics=metrics)

    def _initialize_positions(self) -> None:
        """Calcola le quote iniziali per ciascun ETF."""
        first_prices = self.prices_df.iloc[0].to_numpy()
        weights_vec = np.array([self.weights[s] for s in self.symbols])

        trade_value = self.initial_capital
        cost = trade_value * self.params.transaction_cost
        investable = trade_value - cost

        self.positions = investable * weights_vec / first_prices
        self.current_nav = float(np.dot(self.positions, first_prices))

        self.nav_history = [self.current_nav]
        self.weight_history = [dict(zip(self.symbols, (self.positions * first_prices) / self.current_nav))]
        self.trades_history = [{"date": self.prices_df.index[0], **dict(zip(self.symbols, self.positions))}]

    def _generate_rebalance_dates(self) -> list[pd.Timestamp]:
        """Determina le date di ribilanciamento in base alla frequenza."""
        if self.prices_df.empty:
            return []

        freq = self.params.rebalance_frequency.lower()
        if freq == "none":
            return []

        period_map = {"monthly": "M", "quarterly": "Q", "yearly": "A"}
        if freq not in period_map:
            raise ValueError("Frequenza di rebalance non riconosciuta.")

        idx = self.prices_df.index
        # Converto in Series per usare shift posizionale (PeriodIndex.shift() sposta di 1 periodo nel tempo)
        periods = pd.Series(idx.to_period(period_map[freq]), index=idx)
        change_mask = periods != periods.shift(1)
        # Escludi la prima data (già inizializzata).
        return list(periods.index[change_mask][1:])

    def _step_without_rebalance(self, date) -> None:
        """Aggiorna il valore del portafoglio in una data senza ribilanciamento."""
        prices = self.prices_df.loc[date].to_numpy()
        self.current_nav = float(np.dot(self.positions, prices))
        weights = (self.positions * prices) / self.current_nav

        self.nav_history.append(self.current_nav)
        self.weight_history.append(dict(zip(self.symbols, weights)))

    def _rebalance_portfolio(self, date) -> None:
        """Esegue il ribilanciamento del portafoglio in una data di rebalance."""
        prices = self.prices_df.loc[date].to_numpy()
        weights_vec = np.array([self.weights[s] for s in self.symbols])

        current_values = self.positions * prices
        # Aggiorna il NAV con i prezzi correnti prima di calcolare i nuovi target.
        self.current_nav = float(current_values.sum())
        target_values = self.current_nav * weights_vec

        turnover = np.abs(target_values - current_values).sum()
        cost = turnover * self.params.transaction_cost
        available_nav = self.current_nav - cost

        new_target_values = available_nav * weights_vec
        new_positions = new_target_values / prices

        trades = new_positions - self.positions
        self.positions = new_positions
        self.current_nav = float(np.dot(self.positions, prices))

        self.nav_history.append(self.current_nav)
        self.weight_history.append(dict(zip(self.symbols, (self.positions * prices) / self.current_nav)))
        self.trades_history.append({"date": date, **dict(zip(self.symbols, trades))})

def compute_returns(nav_series: pd.Series) -> pd.Series:
    """Calcola i rendimenti periodali dal NAV."""
    if nav_series.size < 2:
        return pd.Series(dtype=float)

    nav_values = nav_series.to_numpy(dtype=float)
    returns = nav_values[1:] / nav_values[:-1] - 1.0
    return pd.Series(returns, index=nav_series.index[1:], name="returns")


def compute_total_return(nav_series: pd.Series) -> float:
    """Calcola il rendimento totale sul periodo di backtest."""
    if nav_series.empty:
        return float("nan")
    start, end = float(nav_series.iloc[0]), float(nav_series.iloc[-1])
    return end / start - 1.0


def compute_cagr(nav_series: pd.Series) -> float:
    """Calcola il CAGR."""
    if nav_series.size < 2:
        return float("nan")

    start, end = float(nav_series.iloc[0]), float(nav_series.iloc[-1])
    if start <= 0:
        return float("nan")

    idx = nav_series.index
    years = nav_series.size / 252
    if isinstance(idx, (pd.DatetimeIndex, pd.PeriodIndex, pd.TimedeltaIndex)):
        start_date = idx[0].to_timestamp() if isinstance(idx, pd.PeriodIndex) else pd.Timestamp(idx[0])
        end_date = idx[-1].to_timestamp() if isinstance(idx, pd.PeriodIndex) else pd.Timestamp(idx[-1])
        elapsed_days = (end_date - start_date).days
        if elapsed_days > 0:
            years = elapsed_days / 365.25

    return (end / start) ** (1.0 / years) - 1.0 if years > 0 else float("nan")


def compute_annualized_volatility(nav_series: pd.Series, periods_per_year: int = 12) -> float:
    """Calcola la volatilità annualizzata con deviazione standard campionaria."""
    returns = compute_returns(nav_series)
    if returns.size < 2:
        return float("nan")

    ret_values = returns.to_numpy(dtype=float)
    valid_obs = np.count_nonzero(~np.isnan(ret_values))
    if valid_obs < 2:
        return float("nan")

    vol = np.nanstd(ret_values, ddof=1)
    return float(vol * np.sqrt(periods_per_year) * 100.0)


def compute_max_drawdown(nav_series: pd.Series) -> float:
    """Calcola il massimo drawdown."""
    if nav_series.empty:
        return float("nan")

    nav_values = nav_series.to_numpy(dtype=float)
    cum_max = np.maximum.accumulate(nav_values)
    drawdowns = nav_values / cum_max - 1.0
    return float(drawdowns.min())

# Sharpe = (CAGR - Tasso Risk-Free) / Volatilità
def compute_sharpe_ratio(
    nav_series: pd.Series,
    risk_free_rate: float = 0.002,
    periods_per_year: int = 252,
) -> float:
    """Calcola lo Sharpe ratio."""
    cagr = compute_cagr(nav_series)
    if cagr == 0:
        return float("nan")

    vol = compute_annualized_volatility(nav_series) / 100
    if vol == 0:
        return float("nan")
    return float((cagr - risk_free_rate) / vol)


def compute_all_metrics(nav_series: pd.Series) -> dict:
    """Restituisce tutte le metriche principali in un dizionario."""
    return {
        "total_return": f"{compute_total_return(nav_series):.2%}",
        "cagr": f"{compute_cagr(nav_series):.2%}",
        "volatility": f"{compute_annualized_volatility(nav_series):.2f}%",
        "max_drawdown": f"{compute_max_drawdown(nav_series):.2%}",
        "sharpe_ratio": f"{compute_sharpe_ratio(nav_series):.2f}",
    }

def plot_backtest_dashboard(nav_series: pd.Series, weights_df: pd.DataFrame = None):
    """Crea un dashboard con equity curve, drawdown e pesi del portafoglio."""
    dates = nav_series.index

    metrics_cards = []
    metrics_info = {
        "CAGR": (compute_cagr(nav_series), "{:.2%}", "#0ea5e9"),
        "Volatility": (compute_annualized_volatility(nav_series), "{:.2f}%", "#22c55e"),
        "Max Drawdown": (compute_max_drawdown(nav_series), "{:.2%}", "#f97316"),
        "Sharpe Ratio": (compute_sharpe_ratio(nav_series), "{:.2f}", "#eab308"),
    }

    for title, (value, fmt, accent) in metrics_info.items():
        display_value = "—" if pd.isna(value) else fmt.format(value)
        metrics_cards.append(
            Div(
                text=f"""
<div style="
    background:#0f172a;
    color:#e2e8f0;
    border-radius:10px;
    padding:12px 16px;
    box-shadow:0 2px 8px rgba(0,0,0,0.15);
    border-top:4px solid {accent};
">
  <div style="font-size:12px; letter-spacing:0.5px; text-transform:uppercase; color:#94a3b8; font-weight:700;">
    {title}
  </div>
  <div style="font-size:28px; font-weight:800; margin-top:8px; color:#e2e8f0;">
    {display_value}
  </div>
</div>
""",
                width=220,
            )
        )
    metrics_row = row(*metrics_cards, sizing_mode="stretch_width")

    p1 = figure(
        title="Equity Curve",
        x_axis_type="datetime",
        height=250,
        sizing_mode="stretch_width",
    )
    p1.line(dates, nav_series.values, line_width=2, color="navy", legend_label="NAV")
    p1.add_tools(HoverTool(tooltips=[("Date", "@x{%F}"), ("NAV", "@y{0,0.00}")], formatters={"@x": "datetime"}))
    p1.legend.location = "top_left"

    nav_values = nav_series.to_numpy(dtype=float)
    cum_max = np.maximum.accumulate(nav_values)
    drawdowns = (nav_values / cum_max - 1.0) * 100

    p2 = figure(
        title="Drawdown (%)",
        x_axis_type="datetime",
        height=200,
        sizing_mode="stretch_width",
        x_range=p1.x_range,
    )
    p2.varea(x=dates, y1=0, y2=drawdowns, fill_color="crimson", fill_alpha=0.6)
    p2.line(dates, drawdowns, line_width=1, color="darkred")
    p2.add_tools(HoverTool(tooltips=[("Date", "@x{%F}"), ("DD", "@y{0.00}%")], formatters={"@x": "datetime"}))

    if weights_df is not None and not weights_df.empty:
        p3 = figure(
            title="Portfolio Weights",
            x_axis_type="datetime",
            height=200,
            sizing_mode="stretch_width",
            x_range=p1.x_range,
            y_range=(0, 1),
        )
        symbols = weights_df.columns.tolist()
        colors = Category10[max(3, len(symbols))][:len(symbols)]
        weights_source = ColumnDataSource(weights_df.assign(date=weights_df.index))
        p3.varea_stack(
            stackers=symbols,
            x="date",
            color=colors,
            alpha=0.8,
            legend_label=symbols,
            source=weights_source,
        )
        p3.yaxis.formatter = NumeralTickFormatter(format="0%")
        p3.legend.location = "top_left"
        p3.legend.click_policy = "hide"
        p3.add_tools(
            HoverTool(
                tooltips=[("Date", "@date{%F}"), ("Asset", "$name"), ("Weight", "@$name{0.00%}")],
                formatters={"@date": "datetime"},
                mode="vline",
            )
        )
        layout = column(metrics_row, p1, p2, p3, sizing_mode="stretch_width")
    else:
        layout = column(metrics_row, p1, p2, sizing_mode="stretch_width")

    show(layout)
    return layout


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Esegui un backtest di portafoglio statico.")
    parser.add_argument("--symbols", nargs="+", help="Ticker/nomi degli ETF (es. SPY STOXX50).")
    parser.add_argument("--weights", nargs="+", type=float, help="Pesi corrispondenti (somma=1).")
    parser.add_argument("--initial", type=float, default=10_000.0, help="Capitale iniziale (default: 10000).")
    parser.add_argument(
        "--rebalance",
        choices=["none", "monthly", "quarterly", "yearly"],
        default="yearly",
        help="Frequenza di ribilanciamento (default: yearly).",
    )
    parser.add_argument(
        "--transaction-cost",
        type=float,
        default=0.001,
        help="Costo di transazione (default: 0.001 = 0.1%%).",
    )
    parser.add_argument("--data-dir", default="data", help="Directory dei CSV (default: data).")
    parser.add_argument(
        "--align",
        choices=["inner", "outer", "ffill", "bfill"],
        default="inner",
        help="Metodo di allineamento serie (default: inner).",
    )
    parser.add_argument("--plot", action="store_true", help="Mostra i grafici Bokeh.")
    return parser.parse_args()


def _build_weights(symbols: list[str], weights: Optional[list[float]]) -> dict:
    if weights is None:
        # Default: pesi uguali se non specificati.
        equal_weight = 1.0 / len(symbols)
        weights = [equal_weight] * len(symbols)
    if len(weights) != len(symbols):
        raise ValueError("Il numero di pesi deve corrispondere al numero di simboli.")
    return {sym: float(w) for sym, w in zip(symbols, weights)}


def run_cli_backtest() -> None:
    args = _parse_args()

    symbols = args.symbols or ["SPY", "STOXX50"]
    weights = _build_weights(symbols, args.weights)

    prices = load_multiple_symbols(symbols, data_dir=args.data_dir)
    prices = align_price_data(prices, method=args.align)

    engine = BacktestEngine(
        prices_df=prices,
        portfolio_config=PortfolioConfig(weights=weights, initial_capital=args.initial),
        params=BacktestParams(rebalance_frequency=args.rebalance, transaction_cost=args.transaction_cost),
    )
    result = engine.run()

    metrics = compute_all_metrics(result.nav_series)
    print("Backtest Results")
    print("----------------")
    for k, v in metrics.items():
        print(f"{k.replace('_', ' ').title()}: {v}")
    print(f"Final NAV: {result.metrics['final_nav']:.2f}")

    if args.plot:
        plot_backtest_dashboard(result.nav_series, result.weights_over_time)


if __name__ == "__main__":
    run_cli_backtest()
