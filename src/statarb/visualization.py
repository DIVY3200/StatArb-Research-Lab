"""Plotly figures for the research dashboard.

Design rules: categorical colours are assigned in a fixed order and follow the entity
(asset A is always slot 1, asset B slot 2); a single y-axis per plot (measures with
different units go into stacked subplots, never a secondary axis); thin 2px lines,
recessive hairline grids, and a hover layer on every chart. The palette is a validated
colour-vision-deficiency-safe set with separate light and dark steps.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_LIGHT = {
    "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"],
    "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7",
    "neutral": "#f0efec", "neg": "#e34948", "pos": "#2a78d6",
}
_DARK = {
    "series": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
    "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781", "grid": "#2c2c2a", "axis": "#383835",
    "neutral": "#383835", "neg": "#e66767", "pos": "#3987e5",
}
_PERIOD_FILL = {"train": 0.0, "validation": 0.06, "test": 0.12}


def palette(dark: bool = False) -> dict:
    return _DARK if dark else _LIGHT


def _layout(fig: go.Figure, title: str, dark: bool, height: int = 380, yaxis_title: str | None = None,
            hovermode: str = "x unified") -> go.Figure:
    p = palette(dark)
    fig.update_layout(
        title={"text": title, "x": 0.0, "xanchor": "left", "font": {"size": 15, "color": p["ink"]}},
        height=height,
        margin={"l": 60, "r": 20, "t": 56, "b": 40},
        hovermode=hovermode,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "xanchor": "right", "x": 1.0,
                "font": {"color": p["ink2"]}},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "system-ui, -apple-system, Segoe UI, sans-serif", "color": p["ink2"], "size": 12},
    )
    fig.update_xaxes(showgrid=False, linecolor=p["axis"], tickfont={"color": p["muted"]}, zeroline=False)
    fig.update_yaxes(gridcolor=p["grid"], gridwidth=1, linecolor=p["axis"], tickfont={"color": p["muted"]},
                     zerolinecolor=p["axis"], zerolinewidth=1)
    if yaxis_title:
        fig.update_yaxes(title_text=yaxis_title, title_font={"color": p["muted"], "size": 12})
    return fig


def _line(x, y, name: str, color: str, width: float = 2.0, dash: str | None = None, hover_fmt: str = ".4f",
          showlegend: bool = True) -> go.Scatter:
    return go.Scatter(
        x=x, y=y, name=name, mode="lines", line={"color": color, "width": width, "dash": dash},
        hovertemplate=f"%{{y:{hover_fmt}}}<extra>{name}</extra>", showlegend=showlegend,
    )


def add_period_shading(fig: go.Figure, splits: dict, dates: pd.DatetimeIndex, dark: bool = False,
                       visible_end: int | None = None, row: int | str | None = None) -> go.Figure:
    """Shade validation and test periods and label every period at the top of the plot."""
    p = palette(dark)
    for name, split in splits.items():
        end_idx = split.end_idx if visible_end is None else min(split.end_idx, visible_end)
        if split.start_idx >= end_idx:
            continue
        x0, x1 = dates[split.start_idx], dates[end_idx - 1]
        opacity = _PERIOD_FILL.get(name, 0.0)
        kwargs = {"row": row, "col": 1} if row is not None else {}
        if opacity > 0:
            fig.add_vrect(x0=x0, x1=x1, fillcolor=p["muted"], opacity=opacity, line_width=0, layer="below", **kwargs)
        fig.add_vline(x=x0, line={"color": p["axis"], "width": 1}, **kwargs)
        fig.add_annotation(x=x0, y=1.0, yref="paper", xref="x", text=f" {name}", showarrow=False,
                           xanchor="left", yanchor="top", font={"size": 11, "color": p["muted"]})
    return fig


# --------------------------------------------------------------------------- exploratory
def price_history_figure(prices: pd.DataFrame, names: dict[str, str], dark: bool = False) -> go.Figure:
    """Each asset's price on its own panel (different price scales are never forced onto one axis)."""
    p = palette(dark)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06)
    for i, col in enumerate(("A", "B")):
        fig.add_trace(_line(prices.index, prices[col], names[col], p["series"][i], hover_fmt=".2f"), row=i + 1, col=1)
    _layout(fig, "Price history (prices used for research)", dark, height=480)
    for i, col in enumerate(("A", "B")):
        fig.update_yaxes(title_text=names[col], title_font={"color": p["muted"], "size": 12}, row=i + 1, col=1)
    return fig


def normalized_price_figure(prices: pd.DataFrame, names: dict[str, str], dark: bool = False) -> go.Figure:
    p = palette(dark)
    norm = 100 * prices / prices.iloc[0]
    fig = go.Figure([_line(norm.index, norm[c], names[c], p["series"][i], hover_fmt=".1f") for i, c in enumerate(("A", "B"))])
    return _layout(fig, "Normalised prices (first observation = 100)", dark, yaxis_title="Index")


def returns_figure(returns: pd.DataFrame, names: dict[str, str], dark: bool = False) -> go.Figure:
    p = palette(dark)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, shared_yaxes=True, vertical_spacing=0.06)
    for i, col in enumerate(("A", "B")):
        fig.add_trace(go.Scatter(x=returns.index, y=returns[col], name=names[col], mode="lines",
                                 line={"color": p["series"][i], "width": 1},
                                 hovertemplate=f"%{{y:.2%}}<extra>{names[col]}</extra>"), row=i + 1, col=1)
    _layout(fig, "Daily simple returns", dark, height=440)
    fig.update_yaxes(tickformat=".0%")
    for i, col in enumerate(("A", "B")):
        fig.update_yaxes(title_text=names[col], title_font={"color": p["muted"], "size": 12}, row=i + 1, col=1)
    return fig


def rolling_volatility_figure(vol: pd.DataFrame, names: dict[str, str], window: int, dark: bool = False) -> go.Figure:
    p = palette(dark)
    fig = go.Figure([_line(vol.index, vol[c], names[c], p["series"][i], hover_fmt=".1%") for i, c in enumerate(("A", "B"))])
    _layout(fig, f"Rolling {window}-day annualised volatility", dark)
    fig.update_yaxes(tickformat=".0%")
    return fig


def rolling_correlation_figure(corr: pd.Series, window: int, dark: bool = False) -> go.Figure:
    p = palette(dark)
    fig = go.Figure([_line(corr.index, corr, "Return correlation", p["series"][0], hover_fmt=".2f", showlegend=False)])
    _layout(fig, f"Rolling {window}-day correlation of daily returns", dark, yaxis_title="Correlation")
    fig.update_yaxes(range=[-1, 1])
    return fig


def log_price_scatter_figure(log_prices: pd.DataFrame, names: dict[str, str], intercept: float, hedge_ratio: float,
                             train_end_idx: int, dark: bool = False) -> go.Figure:
    """log P_A against log P_B; training points vs later points, with the training-fitted line."""
    p = palette(dark)
    train, later = log_prices.iloc[:train_end_idx], log_prices.iloc[train_end_idx:]
    fig = go.Figure()
    for frame, label, color in ((train, "Training period", p["series"][0]), (later, "After training", p["series"][1])):
        if frame.empty:
            continue
        fig.add_trace(go.Scatter(
            x=frame["B"], y=frame["A"], mode="markers", name=label,
            marker={"color": color, "size": 5, "opacity": 0.55, "line": {"width": 0}},
            customdata=np.array([str(d.date()) for d in frame.index]),
            hovertemplate=f"%{{customdata}}<br>log {names['B']}: %{{x:.3f}}<br>log {names['A']}: %{{y:.3f}}<extra>{label}</extra>",
        ))
    xs = np.linspace(log_prices["B"].min(), log_prices["B"].max(), 50)
    fig.add_trace(go.Scatter(x=xs, y=intercept + hedge_ratio * xs, mode="lines", name="Training OLS fit",
                             line={"color": p["ink"], "width": 2},
                             hovertemplate="fit: %{y:.3f}<extra></extra>"))
    _layout(fig, f"log {names['A']} vs log {names['B']} (fit estimated on training data only)", dark, height=440,
            hovermode="closest")
    fig.update_xaxes(title_text=f"log price {names['B']}", title_font={"color": p["muted"]})
    fig.update_yaxes(title_text=f"log price {names['A']}")
    return fig


# --------------------------------------------------------------------------- spread and signals
def spread_figure(spread: pd.Series, splits: dict, mean: float, std: float, dark: bool = False) -> go.Figure:
    p = palette(dark)
    fig = go.Figure([_line(spread.index, spread, "Spread", p["series"][0], showlegend=False)])
    for k, label in ((0, "training mean"), (2, "+2 training std"), (-2, "-2 training std")):
        fig.add_hline(y=mean + k * std, line={"color": p["muted"], "width": 1, "dash": "dot" if k else "solid"},
                      annotation_text=label, annotation_font={"size": 10, "color": p["muted"]},
                      annotation_position="bottom right")
    add_period_shading(fig, splits, spread.index, dark, visible_end=len(spread))
    return _layout(fig, "Spread: log A - intercept - hedge ratio x log B", dark, yaxis_title="Log spread")


def zscore_figure(zscore: pd.Series, splits: dict, entry: float, exit_: float, stop: float | None,
                  signal_frames: list[pd.DataFrame], dark: bool = False) -> go.Figure:
    """Z-score with thresholds and the *signal-time* entry and exit events of each period's backtest."""
    p = palette(dark)
    fig = go.Figure([_line(zscore.index, zscore, "z-score", p["ink2"], width=1.5, hover_fmt=".2f", showlegend=False)])
    for level, label in ((entry, "entry"), (-entry, "entry"), (exit_, "exit"), (-exit_, "exit")):
        fig.add_hline(y=level, line={"color": p["muted"], "width": 1, "dash": "dash" if label == "entry" else "dot"},
                      annotation_text=f"{label} {level:+.2f}", annotation_font={"size": 10, "color": p["muted"]},
                      annotation_position="top right")
    if stop is not None:
        for level in (stop, -stop):
            fig.add_hline(y=level, line={"color": p["neg"], "width": 1}, annotation_text=f"stop {level:+.2f}",
                          annotation_font={"size": 10, "color": p["muted"]}, annotation_position="top right")
    events = pd.concat(signal_frames) if signal_frames else pd.DataFrame(columns=["zscore", "event"])
    marker_specs = (
        ("entry_long", "Enter long spread (long A, short B)", p["series"][0], "triangle-up"),
        ("entry_short", "Enter short spread (short A, long B)", p["series"][1], "triangle-down"),
        ("exit", "Exit (reverted)", p["ink"], "circle-open"),
        ("stop_loss", "Stop-loss exit", p["neg"], "x"),
        ("max_holding", "Max-holding exit", p["muted"], "square-open"),
    )
    for event, label, color, symbol in marker_specs:
        sel = events.loc[events["event"] == event]
        if sel.empty:
            continue
        fig.add_trace(go.Scatter(x=sel.index, y=sel["zscore"], mode="markers", name=label,
                                 marker={"color": color, "size": 10, "symbol": symbol, "line": {"width": 2, "color": color}},
                                 hovertemplate=f"%{{x|%Y-%m-%d}}<br>z = %{{y:.2f}}<extra>{label}</extra>"))
    add_period_shading(fig, splits, zscore.index, dark, visible_end=len(zscore))
    return _layout(fig, "Z-score (training mean and std) with signal events", dark, height=440,
                   yaxis_title="z", hovermode="closest")


# --------------------------------------------------------------------------- portfolio
def equity_vs_benchmarks_figure(strategy_returns: pd.Series, benchmarks: pd.DataFrame, title: str,
                                dark: bool = False) -> go.Figure:
    """Growth of 1 for the strategy and its benchmarks on one common base (no secondary axis)."""
    p = palette(dark)
    fig = go.Figure()
    fig.add_trace(_line(strategy_returns.index, (1 + strategy_returns).cumprod(), "Pairs strategy (net)",
                        p["series"][0], width=2.5, hover_fmt=".3f"))
    slot = 1
    for col in benchmarks.columns:
        color = p["muted"] if col.startswith("Cash") else p["series"][slot]
        if not col.startswith("Cash"):
            slot += 1
        fig.add_trace(_line(benchmarks.index, (1 + benchmarks[col]).cumprod(), col, color, width=1.5, hover_fmt=".3f"))
    return _layout(fig, title, dark, yaxis_title="Growth of 1")


def gross_net_figure(daily: pd.DataFrame, dark: bool = False) -> go.Figure:
    """Gross vs net growth (top) and cumulative costs in currency (bottom): two units, two panels."""
    p = palette(dark)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.65, 0.35])
    fig.add_trace(_line(daily.index, (1 + daily["gross_return"]).cumprod(), "Gross (before costs)", p["series"][1],
                        hover_fmt=".4f"), row=1, col=1)
    fig.add_trace(_line(daily.index, (1 + daily["net_return"]).cumprod(), "Net (after costs)", p["series"][0],
                        hover_fmt=".4f"), row=1, col=1)
    fig.add_trace(_line(daily.index, daily["cumulative_costs"], "Cumulative costs", p["ink2"], hover_fmt=",.0f",
                        showlegend=False), row=2, col=1)
    _layout(fig, "Gross vs net performance and cumulative costs", dark, height=480)
    fig.update_yaxes(title_text="Growth of 1", title_font={"color": p["muted"], "size": 12}, row=1, col=1)
    fig.update_yaxes(title_text="Costs (currency)", title_font={"color": p["muted"], "size": 12}, row=2, col=1)
    return fig


def drawdown_figure(series: dict[str, pd.Series], dark: bool = False) -> go.Figure:
    p = palette(dark)
    fig = go.Figure()
    for i, (name, equity) in enumerate(series.items()):
        dd = equity / equity.cummax() - 1.0
        fig.add_trace(_line(dd.index, dd, name, p["series"][i], hover_fmt=".2%"))
    _layout(fig, "Drawdown from running peak", dark)
    fig.update_yaxes(tickformat=".0%")
    return fig


def exposure_figure(daily: pd.DataFrame, names: dict[str, str], dark: bool = False) -> go.Figure:
    """Leg weights (top) and gross/net leverage (bottom), both as fractions of equity."""
    p = palette(dark)
    equity = daily["equity"].where(daily["equity"] > 0)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, shared_yaxes=False, vertical_spacing=0.08)
    fig.add_trace(_line(daily.index, daily["notional_a"] / equity, f"{names['A']} weight", p["series"][0], hover_fmt=".1%"), row=1, col=1)
    fig.add_trace(_line(daily.index, daily["notional_b"] / equity, f"{names['B']} weight", p["series"][1], hover_fmt=".1%"), row=1, col=1)
    fig.add_trace(_line(daily.index, daily["gross_leverage"], "Gross exposure", p["series"][2], hover_fmt=".1%"), row=2, col=1)
    fig.add_trace(_line(daily.index, daily["net_leverage"], "Net exposure", p["series"][3], hover_fmt=".1%"), row=2, col=1)
    _layout(fig, "Leg weights and exposure (fraction of equity)", dark, height=500)
    fig.update_yaxes(tickformat=".0%")
    fig.update_yaxes(title_text="Leg weight", title_font={"color": p["muted"], "size": 12}, row=1, col=1)
    fig.update_yaxes(title_text="Exposure", title_font={"color": p["muted"], "size": 12}, row=2, col=1)
    return fig


def trade_return_figure(trades: pd.DataFrame, dark: bool = False) -> go.Figure:
    p = palette(dark)
    closed = trades.loc[trades["status"] == "closed"] if not trades.empty else trades
    fig = go.Figure()
    if not closed.empty:
        colors = [p["pos"] if r > 0 else p["neg"] for r in closed["net_return"]]
        fig.add_trace(go.Bar(
            x=closed["trade_id"].astype(str), y=closed["net_return"], marker={"color": colors, "line": {"width": 0}},
            customdata=np.stack([closed["direction"], closed["exit_reason"], closed["holding_days"]], axis=1),
            hovertemplate="Trade %{x}<br>net return %{y:.2%}<br>%{customdata[0]}, exit: %{customdata[1]}<br>"
                          "held %{customdata[2]} days<extra></extra>",
            name="Net return per trade", showlegend=False,
        ))
    _layout(fig, "Net return per completed trade (on equity at entry)", dark, hovermode="closest")
    fig.update_xaxes(title_text="Trade #", title_font={"color": p["muted"]}, type="category")
    fig.update_yaxes(tickformat=".1%")
    fig.update_layout(bargap=0.25 if len(closed) >= 8 else 0.6)
    return fig


# --------------------------------------------------------------------------- robustness
def parameter_heatmap(grid: pd.DataFrame, metric: str = "sharpe_ratio", title: str | None = None,
                      dark: bool = False) -> go.Figure:
    """Diverging heatmap (red negative, neutral zero, blue positive) of one metric over the grid."""
    p = palette(dark)
    pivot = grid.pivot(index="exit_z", columns="entry_z", values=metric).sort_index(ascending=False)
    trades = grid.pivot(index="exit_z", columns="entry_z", values="completed_trades").reindex_like(pivot)
    values = pivot.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    bound = float(np.nanmax(np.abs(finite))) if finite.size else 1.0
    bound = bound or 1.0
    text = np.where(np.isfinite(values), np.vectorize(lambda v: f"{v:.2f}")(np.nan_to_num(values)), "n/a")
    fig = go.Figure(go.Heatmap(
        z=values, x=[f"{c:g}" for c in pivot.columns], y=[f"{r:g}" for r in pivot.index],
        colorscale=[[0.0, p["neg"]], [0.5, p["neutral"]], [1.0, p["pos"]]], zmin=-bound, zmax=bound,
        text=text, texttemplate="%{text}", textfont={"color": p["ink"], "size": 12},
        customdata=trades.to_numpy(), xgap=2, ygap=2,
        hovertemplate="entry %{x}, exit %{y}<br>" + metric + ": %{z:.3f}<br>trades: %{customdata}<extra></extra>",
        colorbar={"title": {"text": metric.replace("_", " "), "font": {"color": p["muted"]}}},
    ))
    _layout(fig, title or f"{metric.replace('_', ' ')} by entry/exit threshold", dark, height=380, hovermode="closest")
    fig.update_xaxes(title_text="entry |z|", title_font={"color": p["muted"]}, type="category")
    fig.update_yaxes(title_text="exit |z|", type="category", showgrid=False)
    return fig


def cost_sensitivity_figure(tables: dict[str, pd.DataFrame], dark: bool = False) -> go.Figure:
    p = palette(dark)
    fig = go.Figure()
    for i, (name, frame) in enumerate(tables.items()):
        fig.add_trace(go.Scatter(x=frame["cost_bps_per_leg_per_side"], y=frame["cumulative_return"], name=name,
                                 mode="lines+markers", line={"color": p["series"][i], "width": 2},
                                 marker={"size": 8, "color": p["series"][i]},
                                 hovertemplate="%{x:g} bps: %{y:.2%}<extra>" + name + "</extra>"))
    _layout(fig, "Net cumulative return vs trading cost (per leg, per side)", dark)
    fig.update_xaxes(title_text="Cost (bps)", title_font={"color": p["muted"]})
    fig.update_yaxes(tickformat=".1%")
    return fig


def walk_forward_figure(windows: pd.DataFrame, dark: bool = False) -> go.Figure:
    p = palette(dark)
    traded = windows["traded"].astype(bool)
    colors = [(p["pos"] if r > 0 else p["neg"]) if t else p["muted"] for r, t in zip(windows["window_return"], traded)]
    labels = [f"W{w}" for w in windows["window"]]
    fig = go.Figure(go.Bar(
        x=labels, y=windows["window_return"], marker={"color": colors, "line": {"width": 0}},
        customdata=np.stack([windows["eval_start"], windows["eval_end"], windows["eg_p_value"].round(4),
                             np.where(traded, "traded", "not traded (failed EG test)")], axis=1),
        hovertemplate="%{x}: %{customdata[0]} to %{customdata[1]}<br>return %{y:.2%}<br>fit-window EG p = "
                      "%{customdata[2]}<br>%{customdata[3]}<extra></extra>",
        text=["" if t else "no trade" for t in traded], textposition="outside",
        textfont={"size": 10, "color": p["muted"]}, cliponaxis=False,
        showlegend=False,
    ))
    _layout(fig, "Walk-forward: net return per out-of-sample window (grey = not traded)", dark, hovermode="closest")
    fig.update_yaxes(tickformat=".1%")
    fig.update_layout(bargap=0.25)
    return fig


def screen_pvalue_figure(table: pd.DataFrame, significance: float, dark: bool = False) -> go.Figure:
    p = palette(dark)
    data = table.dropna(subset=["p_value"]).copy()
    data["pair"] = data["dependent"] + " ~ " + data["independent"]
    fig = go.Figure(go.Bar(
        x=data["pair"], y=data["p_value"], marker={"color": p["series"][0], "line": {"width": 0}},
        customdata=data["bh_adjusted_p"],
        hovertemplate="%{x}<br>raw p = %{y:.4f}<br>BH-adjusted p = %{customdata:.4f}<extra></extra>",
        showlegend=False,
    ))
    fig.add_hline(y=significance, line={"color": p["muted"], "width": 1, "dash": "dash"},
                  annotation_text=f"raw {significance:.0%} level", annotation_font={"size": 10, "color": p["muted"]})
    _layout(fig, "Training-period Engle-Granger p-values by pair", dark, hovermode="closest")
    fig.update_yaxes(range=[0, 1])
    fig.update_layout(bargap=0.3)
    return fig


# --------------------------------------------------------------------------- machine learning
def probability_figure(prob: pd.Series, target: pd.Series, rows: pd.Series, threshold: float, splits: dict,
                       title: str, dark: bool = False) -> go.Figure:
    """Predicted reversion probability on labelled rows, coloured by the realised target."""
    p = palette(dark)
    sel = rows & prob.notna()
    fig = go.Figure()
    for value, label, color, symbol in ((1, "Actual: reverted (y = 1)", p["series"][0], "circle"),
                                        (0, "Actual: did not revert (y = 0)", p["series"][1], "x")):
        mask = sel & (target == value)
        fig.add_trace(go.Scatter(
            x=prob.index[mask], y=prob[mask], mode="markers", name=label,
            marker={"color": color, "size": 7, "symbol": symbol, "opacity": 0.75, "line": {"width": 1, "color": color}},
            hovertemplate="%{x|%Y-%m-%d}<br>p = %{y:.3f}<extra>" + label + "</extra>",
        ))
    fig.add_hline(y=threshold, line={"color": p["muted"], "width": 1, "dash": "dash"},
                  annotation_text=f"threshold {threshold:.2f}", annotation_font={"size": 10, "color": p["muted"]},
                  annotation_position="bottom right")
    add_period_shading(fig, splits, prob.index, dark, visible_end=len(prob))
    _layout(fig, title, dark, height=420, yaxis_title="Predicted P(reversion)", hovermode="closest")
    fig.update_yaxes(range=[-0.02, 1.02])
    return fig


def calibration_figure(tables: dict[str, pd.DataFrame], dark: bool = False) -> go.Figure:
    """Reliability diagram: observed frequency vs mean predicted probability per quantile bin."""
    p = palette(dark)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration",
                             line={"color": p["muted"], "width": 1, "dash": "dot"}, hoverinfo="skip"))
    for i, (name, table) in enumerate(tables.items()):
        if table.empty:
            continue
        fig.add_trace(go.Scatter(
            x=table["mean_predicted"], y=table["observed_frequency"], mode="lines+markers", name=name,
            line={"color": p["series"][i], "width": 2}, marker={"size": 9, "color": p["series"][i]},
            customdata=table["count"],
            hovertemplate="predicted %{x:.2f}<br>observed %{y:.2f}<br>rows %{customdata}<extra>" + name + "</extra>",
        ))
    _layout(fig, "Calibration (reliability diagram)", dark, height=400, hovermode="closest")
    fig.update_xaxes(range=[0, 1], title_text="Mean predicted probability", title_font={"color": p["muted"]})
    fig.update_yaxes(range=[0, 1], title_text="Observed reversion frequency")
    return fig


def confusion_figure(matrix: list[list[int]], title: str, dark: bool = False) -> go.Figure:
    """2x2 confusion matrix (rows: actual, columns: predicted), sequential single-hue scale."""
    p = palette(dark)
    z = np.asarray(matrix, dtype=float)
    fig = go.Figure(go.Heatmap(
        z=z, x=["Pred. no reversion (0)", "Pred. reversion (1)"], y=["Actual 0", "Actual 1"],
        colorscale=[[0.0, p["neutral"]], [1.0, p["pos"]]], showscale=False, xgap=2, ygap=2,
        text=z.astype(int).astype(str), texttemplate="%{text}", textfont={"size": 14, "color": p["ink"]},
        hovertemplate="%{y}, %{x}: %{z:.0f}<extra></extra>",
    ))
    _layout(fig, title, dark, height=300, hovermode="closest")
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return fig


def importance_figure(values: pd.Series, title: str, errors: pd.Series | None = None, dark: bool = False,
                      diverging: bool = False) -> go.Figure:
    """Horizontal bars sorted by magnitude; diverging colours for signed coefficients."""
    p = palette(dark)
    order = values.abs().sort_values().index if diverging else values.sort_values().index
    v = values.loc[order]
    colors = [p["pos"] if x >= 0 else p["neg"] for x in v] if diverging else p["series"][0]
    fig = go.Figure(go.Bar(
        x=v, y=list(order), orientation="h", marker={"color": colors, "line": {"width": 0}},
        error_x={"type": "data", "array": errors.loc[order].to_numpy(), "color": p["muted"], "thickness": 1}
        if errors is not None and errors.notna().any() else None,
        hovertemplate="%{y}: %{x:.4f}<extra></extra>", showlegend=False,
    ))
    _layout(fig, title, dark, height=max(320, 26 * len(v) + 90), hovermode="closest")
    fig.update_layout(bargap=0.3)
    return fig


def variant_equity_figure(returns: dict[str, pd.Series], title: str, dark: bool = False) -> go.Figure:
    p = palette(dark)
    fig = go.Figure()
    for i, (name, r) in enumerate(returns.items()):
        fig.add_trace(_line(r.index, (1 + r).cumprod(), name, p["series"][i], width=2.5 if i == 0 else 1.75,
                            hover_fmt=".3f"))
    return _layout(fig, title, dark, height=420, yaxis_title="Growth of 1 (net of costs)")


def variant_drawdown_figure(returns: dict[str, pd.Series], dark: bool = False) -> go.Figure:
    p = palette(dark)
    fig = go.Figure()
    for i, (name, r) in enumerate(returns.items()):
        equity = (1 + r).cumprod()
        fig.add_trace(_line(r.index, equity / equity.cummax() - 1, name, p["series"][i], width=1.5, hover_fmt=".2%"))
    _layout(fig, "Drawdown from running peak", dark, height=360)
    fig.update_yaxes(tickformat=".0%")
    return fig


def variant_costs_figure(costs: pd.Series, dark: bool = False) -> go.Figure:
    p = palette(dark)
    fig = go.Figure(go.Bar(x=list(costs.index), y=costs.to_numpy(), marker={"color": p["series"][0], "line": {"width": 0}},
                           hovertemplate="%{x}: %{y:,.0f}<extra></extra>", showlegend=False))
    _layout(fig, "Total transaction costs (both legs, commission + slippage)", dark, height=340, hovermode="closest")
    fig.update_layout(bargap=0.45)
    return fig


def walk_forward_variant_heatmap(windows: pd.DataFrame, variants: dict[str, str], dark: bool = False) -> go.Figure:
    """Window-by-window net return for each variant (diverging: red loss, neutral zero, blue gain)."""
    p = palette(dark)
    cols = [f"return_{k}" for k in variants]
    z = windows[cols].to_numpy(dtype=float).T
    bound = float(np.nanmax(np.abs(z))) if np.isfinite(z).any() else 1.0
    bound = bound or 1.0
    text = np.vectorize(lambda v: "" if not np.isfinite(v) or v == 0 else f"{v:.1%}")(z)
    fig = go.Figure(go.Heatmap(
        z=z, x=[f"W{w}" for w in windows["window"]], y=list(variants.values()),
        colorscale=[[0.0, p["neg"]], [0.5, p["neutral"]], [1.0, p["pos"]]], zmin=-bound, zmax=bound,
        text=text, texttemplate="%{text}", textfont={"size": 10, "color": p["ink"]}, xgap=2, ygap=2,
        hovertemplate="%{y}, %{x}: %{z:.2%}<extra></extra>", colorbar={"tickformat": ".0%"},
    ))
    _layout(fig, "Walk-forward net return per window and variant (blank = flat)", dark,
            height=110 + 46 * len(variants), hovermode="closest")
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return fig


__all__ = [
    "add_period_shading",
    "calibration_figure",
    "confusion_figure",
    "importance_figure",
    "probability_figure",
    "variant_costs_figure",
    "variant_drawdown_figure",
    "variant_equity_figure",
    "walk_forward_variant_heatmap",
    "cost_sensitivity_figure",
    "drawdown_figure",
    "equity_vs_benchmarks_figure",
    "exposure_figure",
    "gross_net_figure",
    "log_price_scatter_figure",
    "normalized_price_figure",
    "palette",
    "parameter_heatmap",
    "price_history_figure",
    "returns_figure",
    "rolling_correlation_figure",
    "rolling_volatility_figure",
    "screen_pvalue_figure",
    "spread_figure",
    "trade_return_figure",
    "walk_forward_figure",
    "zscore_figure",
]
