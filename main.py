from pathlib import Path

import numpy as np
import pandas as pd


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

def main():
    data_dir = "data"
    prices_df = load_price_series(symbol="SPY", data_dir=data_dir)
    aligned_prices_df = align_price_data(prices_df)
    print(aligned_prices_df)

if __name__ == "__main__":
    main()
