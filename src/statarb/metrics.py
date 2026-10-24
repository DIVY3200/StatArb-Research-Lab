"""Performance metrics and benchmarks.

Conventions
-----------
* Daily simple returns; ``periods_per_year`` (default 252) is used for annualisation.
* The annual risk-free rate is converted to a daily geometric rate
  ``(1 + rf) ** (1 / periods_per_year) - 1`` and subtracted before computing the Sharpe ratio.
* Metrics that are undefined for the sample (too few observations, zero volatility, no
  drawdown, no completed trades, or a period shorter than ``min_years`` for CAGR) are
  returned as ``NaN`` rather than misleading infinities.
* Benchmarks are buy-and-hold positions evaluated on exactly the same dates as the
  strategy, with the first day's return set to zero (positions are bought at the first
  close of the period, just as the strategy starts flat at that close). Benchmarks pay no
  transaction costs. Note that the pairs strategy is a long-short position sized to a gross
  exposure limit, so its returns are not directly comparable to a fully-invested long-only
  stock: compare risk-adjusted figures and drawdowns, not only raw returns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_EPS = 1e-12


def daily_risk_free(rf_annual: float, periods_per_year: int = 252) -> float:
    return (1.0 + rf_annual) ** (1.0 / periods_per_year) - 1.0


def _clean(returns: pd.Series) -> pd.Series:
    return pd.Series(returns, dtype=float).dropna()


def cumulative_return(returns: pd.Series) -> float:
    r = _clean(returns)
    if r.empty:
        return float("nan")
    return float(np.prod(1.0 + r.to_numpy()) - 1.0)


def cagr(returns: pd.Series, periods_per_year: int = 252, min_years: float = 1.0) -> float:
    r = _clean(returns)
    years = len(r) / periods_per_year
    if r.empty or years < max(min_years, _EPS):
        return float("nan")
    growth = float(np.prod(1.0 + r.to_numpy()))
    if growth <= 0:
        return float("nan")
    return growth ** (1.0 / years) - 1.0


def annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    r = _clean(returns)
    if len(r) < 2:
        return float("nan")
    return float(r.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(returns: pd.Series, rf_annual: float = 0.0, periods_per_year: int = 252) -> float:
    r = _clean(returns)
    if len(r) < 2:
        return float("nan")
    excess = r - daily_risk_free(rf_annual, periods_per_year)
    sd = float(excess.std(ddof=1))
    if not np.isfinite(sd) or sd < _EPS:
        return float("nan")
    return float(excess.mean() / sd * np.sqrt(periods_per_year))


def sortino_ratio(returns: pd.Series, rf_annual: float = 0.0, periods_per_year: int = 252) -> float:
    """Annualised mean excess return over downside deviation (target = risk-free rate).

    Downside deviation is ``sqrt(mean(min(excess, 0) ** 2))`` over *all* observations. NaN when
    there are fewer than two observations or no downside.
    """
    r = _clean(returns)
    if len(r) < 2:
        return float("nan")
    excess = r - daily_risk_free(rf_annual, periods_per_year)
    downside = float(np.sqrt(np.mean(np.minimum(excess.to_numpy(), 0.0) ** 2)))
    if not np.isfinite(downside) or downside < _EPS:
        return float("nan")
    return float(excess.mean() / downside * np.sqrt(periods_per_year))


def equity_curve(returns: pd.Series, initial: float = 1.0) -> pd.Series:
    return initial * (1.0 + _clean(returns)).cumprod()


def _max_drawdown_from_returns(returns: pd.Series) -> float:
    """Max drawdown of the compounded returns, measured from the starting capital."""
    r = _clean(returns)
    if r.empty:
        return float("nan")
    equity = np.concatenate([[1.0], np.cumprod(1.0 + r.to_numpy())])
    return float((equity / np.maximum.accumulate(equity) - 1.0).min())


def drawdown_series(equity: pd.Series) -> pd.Series:
    equity = pd.Series(equity, dtype=float)
    return equity / equity.cummax() - 1.0


def max_drawdown(equity: pd.Series) -> float:
    equity = pd.Series(equity, dtype=float).dropna()
    if len(equity) < 2:
        return float("nan")
    return float(drawdown_series(equity).min())


def calmar_ratio(returns: pd.Series, periods_per_year: int = 252, min_years: float = 1.0) -> float:
    growth_rate = cagr(returns, periods_per_year, min_years)
    mdd = _max_drawdown_from_returns(returns)
    if not np.isfinite(growth_rate) or not np.isfinite(mdd) or abs(mdd) < _EPS:
        return float("nan")
    return float(growth_rate / abs(mdd))


def return_metrics(
    returns: pd.Series, rf_annual: float = 0.0, periods_per_year: int = 252, min_years: float = 1.0
) -> dict[str, float]:
    r = _clean(returns)
    return {
        "observations": int(len(r)),
        "cumulative_return": cumulative_return(r),
        "cagr": cagr(r, periods_per_year, min_years),
        "annualized_volatility": annualized_volatility(r, periods_per_year),
        "sharpe_ratio": sharpe_ratio(r, rf_annual, periods_per_year),
        "sortino_ratio": sortino_ratio(r, rf_annual, periods_per_year),
        "max_drawdown": _max_drawdown_from_returns(r),
        "calmar_ratio": calmar_ratio(r, periods_per_year, min_years),
    }


def trade_statistics(trades: pd.DataFrame) -> dict[str, float]:
    """Statistics over *completed* (closed) trades, including forced end-of-period exits."""
    closed = trades.loc[trades["status"] == "closed"] if not trades.empty else trades
    n = int(len(closed))
    stats: dict[str, float] = {"completed_trades": n}
    if n == 0:
        stats.update(
            win_rate=float("nan"), avg_trade_net_return=float("nan"), median_trade_net_return=float("nan"),
            avg_trade_gross_return=float("nan"), avg_holding_days=float("nan"), best_trade_net_return=float("nan"),
            worst_trade_net_return=float("nan"), profit_factor=float("nan"),
        )
        return stats
    net = closed["net_pnl"].astype(float)
    wins, losses = net[net > 0].sum(), net[net < 0].sum()
    stats.update(
        win_rate=float((net > 0).mean()),
        avg_trade_net_return=float(closed["net_return"].mean()),
        median_trade_net_return=float(closed["net_return"].median()),
        avg_trade_gross_return=float(closed["gross_return"].mean()),
        avg_holding_days=float(closed["holding_days"].mean()),
        best_trade_net_return=float(closed["net_return"].max()),
        worst_trade_net_return=float(closed["net_return"].min()),
        profit_factor=float(wins / abs(losses)) if losses < 0 else float("nan"),
    )
    return stats


def exposure_statistics(daily: pd.DataFrame, initial_capital: float, periods_per_year: int = 252) -> dict[str, float]:
    years = len(daily) / periods_per_year
    invested = daily["position"] != 0
    total_turnover = float(daily["turnover"].fillna(0.0).sum())
    return {
        "pct_time_invested": float(invested.mean()) if len(daily) else float("nan"),
        "avg_gross_leverage": float(daily["gross_leverage"].mean()),
        "avg_gross_leverage_when_invested": float(daily.loc[invested, "gross_leverage"].mean()) if invested.any() else float("nan"),
        "avg_net_leverage": float(daily["net_leverage"].mean()),
        "max_gross_leverage": float(daily["gross_leverage"].max()),
        "total_turnover": total_turnover,
        "annualized_turnover": total_turnover / years if years > 0 else float("nan"),
        "total_transaction_costs": float(daily["trade_cost"].sum()),
        "total_borrow_costs": float(daily["borrow_cost"].sum()),
        "total_costs_pct_of_initial_capital": float((daily["trade_cost"].sum() + daily["borrow_cost"].sum()) / initial_capital),
    }


def strategy_metrics(result, rf_annual: float = 0.0, periods_per_year: int = 252, min_years: float = 1.0) -> dict[str, float]:
    """All headline metrics for a :class:`~statarb.backtester.BacktestResult`."""
    net = return_metrics(result.returns, rf_annual, periods_per_year, min_years)
    gross = return_metrics(result.gross_returns, rf_annual, periods_per_year, min_years)
    metrics: dict[str, float] = dict(net)
    metrics.update({f"gross_{k}": v for k, v in gross.items() if k != "observations"})
    metrics["final_equity"] = float(result.equity.iloc[-1])
    metrics.update(trade_statistics(result.trades))
    metrics.update(exposure_statistics(result.daily, result.config.initial_capital, periods_per_year))
    return metrics


def benchmark_returns(prices: pd.DataFrame, names: dict[str, str] | None = None,
                      rf_annual: float = 0.0, periods_per_year: int = 252) -> pd.DataFrame:
    """Daily returns of passive benchmarks on the same dates as ``prices``.

    Columns: buy-and-hold A, buy-and-hold B, a 50/50 buy-and-hold portfolio of the two
    assets (equal initial dollars, not rebalanced), and cash earning the risk-free rate.
    The first row is zero because positions are established at the first close.
    """
    names = names or {"A": "A", "B": "B"}
    rets = (prices / prices.shift(1) - 1.0).fillna(0.0)
    growth = prices / prices.iloc[0]
    half_half = 0.5 * growth["A"] + 0.5 * growth["B"]
    rf_d = daily_risk_free(rf_annual, periods_per_year)
    cash = pd.Series(rf_d, index=prices.index)
    cash.iloc[0] = 0.0
    return pd.DataFrame(
        {
            f"Buy & hold {names['A']}": rets["A"],
            f"Buy & hold {names['B']}": rets["B"],
            "50/50 buy & hold": (half_half / half_half.shift(1) - 1.0).fillna(0.0),
            "Cash (risk-free)": cash,
        },
        index=prices.index,
    )


def comparison_table(
    strategy_net: pd.Series,
    strategy_gross: pd.Series,
    benchmarks: pd.DataFrame,
    rf_annual: float = 0.0,
    periods_per_year: int = 252,
    min_years: float = 1.0,
) -> pd.DataFrame:
    """Side-by-side return metrics for the strategy (net and gross) and benchmarks."""
    columns = {"Pairs strategy (net)": strategy_net, "Pairs strategy (gross, before costs)": strategy_gross}
    columns.update({col: benchmarks[col] for col in benchmarks.columns})
    table = {name: return_metrics(series, rf_annual, periods_per_year, min_years) for name, series in columns.items()}
    return pd.DataFrame(table)


__all__ = [
    "annualized_volatility",
    "benchmark_returns",
    "cagr",
    "calmar_ratio",
    "comparison_table",
    "cumulative_return",
    "daily_risk_free",
    "drawdown_series",
    "equity_curve",
    "exposure_statistics",
    "max_drawdown",
    "return_metrics",
    "sharpe_ratio",
    "sortino_ratio",
    "strategy_metrics",
    "trade_statistics",
]
