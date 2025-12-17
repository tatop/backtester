from backtest.metrics import (
    compute_total_return,
    compute_cagr,
    compute_annualized_volatility,
    compute_max_drawdown,
    compute_sharpe_ratio,
)
from bokeh.plotting import figure, show
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, HoverTool, NumeralTickFormatter, Div
from bokeh.palettes import Category10
import numpy as np
import pandas as pd


def plot_backtest_dashboard(
    nav_series: pd.Series,
    weights_df: pd.DataFrame | None = None,
    benchmark_series: pd.Series | None = None,
):
    """Crea un dashboard con equity curve, drawdown e pesi del portafoglio."""
    dates = nav_series.index

    sections: list[tuple[str, pd.Series]] = [("Portfolio", nav_series)]
    if benchmark_series is not None:
        benchmark_series = (
            benchmark_series.reindex(nav_series.index).ffill().bfill().dropna()
        )
        if not benchmark_series.empty:
            label = benchmark_series.name or "Benchmark"
            sections.append((f"Benchmark ({label})", benchmark_series))

    def _render_metrics(title: str, series: pd.Series) -> str:
        metrics_info = {
            "Total Return": (compute_total_return(series), "{:.2%}"),
            "CAGR": (compute_cagr(series), "{:.2%}"),
            "Volatility": (compute_annualized_volatility(series), "{:.2f}%"),
            "Max Drawdown": (compute_max_drawdown(series), "{:.2%}"),
            "Sharpe": (compute_sharpe_ratio(series), "{:.2f}"),
        }
        items = " ".join(
            f'<div style="display:inline-block;margin-right:48px;text-align:center;">'
            f'<div style="font-size:24px;font-weight:600;">{"—" if pd.isna(val) else fmt.format(val)}</div>'
            f'<div style="color:#888;font-size:11px;text-transform:uppercase;margin-top:4px;">{metric}</div></div>'
            for metric, (val, fmt) in metrics_info.items()
        )
        return (
            f'<div style="padding:12px 0;font-family:system-ui,sans-serif;">'
            f'<div style="font-size:13px;font-weight:600;margin-bottom:4px;">{title}</div>'
            f"{items}</div>"
        )

    metrics_div = Div(
        text="".join(_render_metrics(title, series) for title, series in sections),
        sizing_mode="stretch_width",
    )

    p1 = figure(
        title="Equity Curve",
        x_axis_type="datetime",
        height=280,
        sizing_mode="stretch_width",
    )
    nav_source = ColumnDataSource({"date": dates, "value": nav_series.values})
    nav_renderer = p1.line(
        "date",
        "value",
        source=nav_source,
        line_width=2,
        color="navy",
        legend_label="Portfolio",
        name="Portfolio",
    )
    hover_renderers = [nav_renderer]
    if len(sections) > 1:
        # last section corresponds to the benchmark series
        benchmark_label, aligned_series = sections[-1]
        bench_source = ColumnDataSource(
            {"date": aligned_series.index, "value": aligned_series.values}
        )
        bench_renderer = p1.line(
            "date",
            "value",
            source=bench_source,
            line_width=2,
            color="darkorange",
            line_dash="dashed",
            legend_label=benchmark_label,
            name=benchmark_label,
        )
        hover_renderers.append(bench_renderer)
    p1.add_tools(
        HoverTool(
            renderers=hover_renderers,
            tooltips=[
                ("Series", "$name"),
                ("Date", "@date{%F}"),
                ("NAV", "@value{0,0.00}"),
            ],
            formatters={"@date": "datetime"},
        )
    )
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
    p2.add_tools(
        HoverTool(
            tooltips=[("Date", "@x{%F}"), ("DD", "@y{0.00}%")],
            formatters={"@x": "datetime"},
        )
    )

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
        colors = Category10[max(3, len(symbols))][: len(symbols)]
        # weights_source = ColumnDataSource(weights_df.assign(date=weights_df.index))
        cumulative = np.zeros(len(weights_df))
        for sym, color in zip(symbols, colors):
            y1 = cumulative.copy()
            y2 = cumulative + weights_df[sym].values
            cumulative = y2
            source = ColumnDataSource(
                {
                    "date": weights_df.index,
                    "y1": y1,
                    "y2": y2,
                    "weight": weights_df[sym].values,
                }
            )
            r = p3.varea(
                x="date",
                y1="y1",
                y2="y2",
                fill_color=color,
                fill_alpha=0.8,
                legend_label=sym,
                source=source,
            )
            p3.add_tools(
                HoverTool(
                    renderers=[r],
                    tooltips=[
                        ("Date", "@date{%F}"),
                        ("Asset", sym),
                        ("Weight", "@weight{0.00%}"),
                    ],
                    formatters={"@date": "datetime"},
                )
            )
        p3.yaxis.formatter = NumeralTickFormatter(format="0%")
        p3.legend.location = "top_left"
        p3.legend.click_policy = "hide"
        layout = column(metrics_div, p1, p2, p3, sizing_mode="stretch_width")
    else:
        layout = column(metrics_div, p1, p2, sizing_mode="stretch_width")
    show(layout)
    return layout
