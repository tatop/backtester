from dataclasses import dataclass
from typing import Dict, Optional
import pandas as pd
from backtest.portfolio import validate_weights, normalize_weights
import numpy as np

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

