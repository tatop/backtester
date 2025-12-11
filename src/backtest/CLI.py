import argparse

from backtest.metrics import compute_all_metrics
from backtest.plotting import plot_backtest_dashboard
from backtest.data_loader import load_multiple_symbols, align_price_data, download_yahoo
from backtest.portfolio import _build_weights, PortfolioConfig
from backtest.engine import BacktestEngine, BacktestParams


def parse_args() -> argparse.Namespace:
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
    parser.add_argument("--download", action="store_true", help="Scarica dati da Yahoo Finance prima del backtest.")
    parser.add_argument("--start", type=str, default=None, help="Data inizio download (YYYY-MM-DD, default: 10 anni fa).")
    parser.add_argument("--end", type=str, default=None, help="Data fine download (YYYY-MM-DD, default: oggi).")
    return parser.parse_args()


def run_cli_backtest(args: argparse.Namespace) -> None:
    symbols = args.symbols or ["SPY", "STOXX50"]
    
    # Download data if requested
    if args.download:
        print(f"⬇️  Scaricamento dati per {symbols}...")
        download_yahoo(symbols, start=args.start, end=args.end, data_dir=args.data_dir)
        print()
    
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


def main() -> None:
    args = parse_args()
    run_cli_backtest(args)


if __name__ == "__main__":
    main()
