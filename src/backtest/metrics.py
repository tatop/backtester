import numpy as np
import pandas as pd

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
