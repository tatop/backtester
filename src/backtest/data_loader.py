from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf


DEFAULT_LOOKBACK_DAYS = 365 * 10


@dataclass
class DownloadedSymbol:
    symbol: str
    path: Path
    rows: int
    start: pd.Timestamp
    end: pd.Timestamp


@dataclass
class DownloadSummary:
    successes: list[DownloadedSymbol]
    errors: dict[str, str]


def _coerce_date(
    value: str | datetime | pd.Timestamp | None, fallback: pd.Timestamp
) -> pd.Timestamp:
    """Return a timezone-naive timestamp, using the fallback when value is None."""
    if value is None:
        return fallback

    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert(None)
    return timestamp.normalize()


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
            raise ValueError(
                f"Impossibile identificare la colonna prezzo in {csv_path}."
            )
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

    raise ValueError(
        "Metodo di allineamento non riconosciuto. Usa 'inner', 'outer', 'ffill' o 'bfill'."
    )


def download_yahoo(
    symbols: list[str],
    start: str | datetime | pd.Timestamp | None = None,
    end: str | datetime | pd.Timestamp | None = None,
    data_dir: str = "data",
) -> DownloadSummary:
    """Scarica dati storici da Yahoo Finance e salva in CSV.

    Args:
        symbols: Lista di ticker (es: ['SPY', 'QQQ', 'IWM']).
        start: Data inizio (YYYY-MM-DD), default: 10 anni fa.
        end: Data fine (YYYY-MM-DD), default: oggi.
        data_dir: Directory dove salvare i file CSV.
    """

    end_ts = _coerce_date(end, pd.Timestamp.now().normalize())
    start_ts = _coerce_date(start, end_ts - pd.Timedelta(days=DEFAULT_LOOKBACK_DAYS))
    if start_ts >= end_ts:
        raise ValueError("La data di inizio deve essere precedente alla data di fine.")

    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    successes: list[DownloadedSymbol] = []
    errors: dict[str, str] = {}

    for symbol in symbols:
        print(f"Scaricamento {symbol} da {start_ts.date()} a {end_ts.date()}...")
        try:
            data = yf.download(
                symbol,
                start=start_ts.strftime("%Y-%m-%d"),
                end=end_ts.strftime("%Y-%m-%d"),
                progress=False,
                group_by="ticker",
                threads=False,
            )
            if data.empty:
                raise ValueError("Nessun dato trovato.")

            price_col = (
                "Adj Close"
                if "Adj Close" in data.columns
                else "Close"
                if "Close" in data.columns
                else None
            )
            if price_col is None:
                raise ValueError("Colonna prezzo non presente (Close/Adj Close).")

            prices = data[price_col].dropna()
            if prices.empty:
                raise ValueError("Colonna prezzo vuota dopo la pulizia.")

            index = pd.to_datetime(prices.index)
            if getattr(index, "tz", None) is not None:
                index = index.tz_convert(None)
            prices.index = index

            prices = prices[~prices.index.duplicated(keep="first")].sort_index()
            cleaned = pd.DataFrame({"Date": prices.index, "Close": prices.values})

            csv_path = data_path / f"{symbol}.csv"
            cleaned.to_csv(csv_path, index=False)
            successes.append(
                DownloadedSymbol(
                    symbol=symbol,
                    path=csv_path,
                    rows=len(cleaned),
                    start=cleaned["Date"].iloc[0],
                    end=cleaned["Date"].iloc[-1],
                )
            )
            print(f"✓ Salvato: {csv_path} ({len(cleaned)} righe)")
        except Exception as e:
            errors[symbol] = str(e)
            print(f"✗ Errore scaricamento {symbol}: {errors[symbol]}")

    return DownloadSummary(successes=successes, errors=errors)
