import argparse
from pathlib import Path

from backtest.api import (
    BacktestRequest,
    download_prices,
    run_backtest,
)
from backtest.plotting import plot_backtest_dashboard


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Esegui un backtest di portafoglio statico."
    )
    parser.add_argument(
        "--symbols", nargs="+", help="Ticker/nomi degli ETF (es. SPY STOXX50)."
    )
    parser.add_argument(
        "--weights", nargs="+", type=float, help="Pesi corrispondenti (somma=1)."
    )
    parser.add_argument(
        "--initial",
        type=float,
        default=10_000.0,
        help="Capitale iniziale (default: 10000).",
    )
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
    parser.add_argument(
        "--data-dir", default="data", help="Directory dei CSV (default: data)."
    )
    parser.add_argument(
        "--align",
        choices=["inner", "outer", "ffill", "bfill"],
        default="inner",
        help="Metodo di allineamento serie (default: inner).",
    )
    parser.add_argument("--plot", action="store_true", help="Mostra i grafici Bokeh.")
    parser.add_argument(
        "--benchmark",
        default="SPY",
        help="Ticker benchmark per confronto (default: SPY).",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Scarica dati da Yahoo Finance prima del backtest.",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Data inizio download (YYYY-MM-DD, default: 5 anni fa).",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="Data fine download (YYYY-MM-DD, default: oggi).",
    )
    parser.add_argument(
        "--bt-start",
        type=str,
        default=None,
        help="Data inizio backtest (YYYY-MM-DD, default: inizio dati).",
    )
    parser.add_argument(
        "--bt-end",
        type=str,
        default=None,
        help="Data fine backtest (YYYY-MM-DD, default: fine dati).",
    )
    return parser.parse_args()


def run_cli_backtest(args: argparse.Namespace) -> None:
    symbols = args.symbols or ["SPY", "STOXX50"]

    # Download data if requested
    if args.download:
        print(f"⬇️  Scaricamento dati per {symbols}...")
        try:
            summary = download_prices(
                symbols, start=args.start, end=args.end, data_dir=args.data_dir
            )
        except ValueError as exc:
            raise SystemExit(f"Parametri download non validi: {exc}") from exc

        if summary.errors:
            print("⚠️  Problemi durante il download:")
            for symbol, message in summary.errors.items():
                print(f"   {symbol}: {message}")
        print()

    data_path = Path(args.data_dir)
    missing_files = [
        symbol for symbol in symbols if not (data_path / f"{symbol}.csv").exists()
    ]
    if missing_files:
        raise SystemExit(f"Mancano i file CSV per: {', '.join(missing_files)}")

    # Build request using the stable API
    request = BacktestRequest(
        symbols=symbols,
        weights=args.weights,
        initial_capital=args.initial,
        rebalance_frequency=args.rebalance,
        transaction_cost=args.transaction_cost,
        align_method=args.align,
        data_dir=args.data_dir,
        benchmark=args.benchmark,
        start_date=args.bt_start,
        end_date=args.bt_end,
    )

    response = run_backtest(request)

    # Print metrics using formatted output
    formatted = response.metrics.as_formatted_dict()
    print("Backtest Results")
    print("----------------")
    for k, v in formatted.items():
        print(f"{k.replace('_', ' ').title()}: {v}")
    print(f"Final NAV: {response.final_nav:.2f}")

    if response.benchmark_nav_series is not None and response.benchmark_metrics is not None:
        bench_formatted = response.benchmark_metrics.as_formatted_dict()
        print()
        print(f"Benchmark ({response.benchmark_nav_series.name})")
        print("------------------------")
        for k, v in bench_formatted.items():
            print(f"{k.replace('_', ' ').title()}: {v}")
        print(f"Final NAV: {response.benchmark_nav_series.iloc[-1]:.2f}")

    if args.plot:
        plot_backtest_dashboard(
            response.nav_series,
            weights_df=response.weights_over_time,
            benchmark_series=response.benchmark_nav_series,
        )


def main() -> None:
    args = parse_args()
    run_cli_backtest(args)


if __name__ == "__main__":
    main()
