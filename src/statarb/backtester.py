"""Event-driven two-leg backtester with explicit execution timing and costs.

Timing model (close-to-close, delayed execution)
------------------------------------------------
* ``signals.target_position[t]`` is decided from information available at the close of bar ``t``.
* It is executed at the close of bar ``t + execution_lag`` (``execution_lag >= 1``), at that
  bar's closing price, mirroring the "wait one day" convention of Gatev, Goetzmann and
  Rouwenhorst (2006). A signal can therefore never earn the price move that produced it.
* A position established at the close of bar ``s`` earns returns from bar ``s + 1`` onward.
  On each bar, P&L is first marked to market on units carried from the previous close, then
  trades for that bar are executed and their costs deducted.
* Signals whose execution bar would fall beyond the data are never executed. With
  ``close_at_period_end`` the final bar liquidates any open position (paying exit costs)
  and no new position is opened on it, so the last observation adds no artificial return.

Accounting
----------
* Units are fixed between trades. Equity = previous equity + mark-to-market P&L
  - borrow fees - transaction costs. Capital not used as margin earns zero interest and
  short-sale proceeds earn no rebate (a conservative simplification).
* ``gross_return`` excludes trading and borrow costs but uses the same positions as the
  net run, so ``net_return <= gross_return`` on every bar.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from .portfolio import CostModel, target_leg_weights


@dataclass(frozen=True)
class BacktestConfig:
    initial_capital: float = 1_000_000.0
    max_gross_exposure: float = 1.0
    execution_lag: int = 1
    commission_bps: float = 5.0
    slippage_bps: float = 5.0
    borrow_bps_annual: float = 0.0
    rebalance_tolerance: float | None = 0.10
    close_at_period_end: bool = True
    short_selling_allowed: bool = True
    periods_per_year: int = 252

    def __post_init__(self) -> None:
        if self.execution_lag < 1:
            raise ValueError("execution_lag must be >= 1 to avoid trading on the signal bar's own close")
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        if self.max_gross_exposure <= 0:
            raise ValueError("max_gross_exposure must be positive")

    @classmethod
    def from_research_config(cls, config, **overrides) -> "BacktestConfig":
        values = dict(
            initial_capital=config.portfolio.initial_capital,
            max_gross_exposure=config.portfolio.max_gross_exposure,
            execution_lag=config.strategy.execution_lag,
            commission_bps=config.costs.commission_bps,
            slippage_bps=config.costs.slippage_bps,
            borrow_bps_annual=config.costs.borrow_bps_annual,
            rebalance_tolerance=config.portfolio.rebalance_tolerance,
            close_at_period_end=config.portfolio.close_at_period_end,
            short_selling_allowed=config.costs.short_selling_allowed,
            periods_per_year=config.metrics.periods_per_year,
        )
        values.update(overrides)
        return cls(**values)

    def cost_model(self) -> CostModel:
        return CostModel(self.commission_bps, self.slippage_bps, self.borrow_bps_annual, self.periods_per_year)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class BacktestResult:
    daily: pd.DataFrame
    trades: pd.DataFrame
    signals: pd.DataFrame
    hedge_ratio: float
    config: BacktestConfig
    warnings: list[str] = field(default_factory=list)

    @property
    def equity(self) -> pd.Series:
        return self.daily["equity"]

    @property
    def returns(self) -> pd.Series:
        return self.daily["net_return"]

    @property
    def gross_returns(self) -> pd.Series:
        return self.daily["gross_return"]

    @property
    def closed_trades(self) -> pd.DataFrame:
        if self.trades.empty:
            return self.trades
        return self.trades.loc[self.trades["status"] == "closed"]


TRADE_COLUMNS = [
    "trade_id", "status", "direction", "entry_signal_date", "entry_date", "exit_signal_date", "exit_date",
    "exit_reason", "entry_z", "exit_z", "holding_days", "entry_equity", "entry_price_a", "entry_price_b",
    "exit_price_a", "exit_price_b", "units_a", "units_b", "entry_notional_a", "entry_notional_b",
    "rebalances", "gross_pnl", "transaction_costs", "borrow_costs", "net_pnl", "gross_return", "net_return",
]


def run_backtest(
    prices: pd.DataFrame,
    signals: pd.DataFrame,
    hedge_ratio: float,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """Simulate both legs of the pairs strategy over one evaluation period.

    ``prices`` must have columns ``A`` and ``B`` (the same prices used for returns, e.g.
    adjusted closes) and share its index with ``signals`` (which needs ``target_position``
    and optionally ``event`` and ``zscore``). The period always starts flat.
    """
    config = config or BacktestConfig()
    _validate(prices, signals, hedge_ratio)
    cost = config.cost_model()
    target = signals["target_position"].to_numpy(dtype=int)
    if not config.short_selling_allowed and np.any(target != 0):
        raise ValueError(
            "Short selling is disabled, but every pairs-trade position requires shorting one leg. "
            "Enable short selling (an explicit research assumption) or do not trade this strategy."
        )
    events = signals["event"].astype(str).to_numpy() if "event" in signals else np.array([""] * len(signals))
    zvals = signals["zscore"].to_numpy(float) if "zscore" in signals else np.full(len(signals), np.nan)

    dates = prices.index
    pa = prices["A"].to_numpy(float)
    pb = prices["B"].to_numpy(float)
    n = len(prices)
    lag = config.execution_lag
    limit = config.max_gross_exposure
    tol = config.rebalance_tolerance

    qa = qb = 0.0
    state = 0
    equity = float(config.initial_capital)
    prev_leverage = 0.0
    realized_cum = 0.0
    ruined = False
    warnings: list[str] = []
    open_trade: dict | None = None
    trades: list[dict] = []
    rec: dict[str, list] = {k: [] for k in (
        "signal_target", "position", "event", "signal_date_executed", "units_a", "units_b", "price_a",
        "price_b", "notional_a", "notional_b", "long_exposure", "short_exposure", "gross_exposure",
        "net_exposure", "gross_leverage", "net_leverage", "cash", "pnl", "trade_cost", "borrow_cost",
        "traded_notional", "turnover", "equity", "gross_return", "net_return", "unrealized_pnl",
        "realized_pnl_cum",
    )}

    def open_position(i: int, desired: int, sig_i: int) -> float:
        nonlocal qa, qb, state, open_trade, equity
        wa, wb = target_leg_weights(desired, hedge_ratio, limit)
        # Size so that gross notional equals `limit` x equity *after* paying the entry cost.
        budget = equity / (1.0 + (abs(wa) + abs(wb)) * cost.trade_cost_rate)
        na, nb = wa * budget, wb * budget
        entry_equity = equity
        c = cost.trade_cost(na, nb)
        equity -= c
        qa, qb, state = na / pa[i], nb / pb[i], desired
        open_trade = {
            "trade_id": len(trades) + 1, "status": "open",
            "direction": "long_spread" if desired > 0 else "short_spread",
            "entry_signal_date": dates[sig_i], "entry_date": dates[i], "exit_signal_date": pd.NaT,
            "exit_date": pd.NaT, "exit_reason": "", "entry_z": zvals[sig_i], "exit_z": np.nan,
            "holding_days": 0, "entry_equity": entry_equity, "entry_price_a": pa[i], "entry_price_b": pb[i],
            "exit_price_a": np.nan, "exit_price_b": np.nan, "units_a": qa, "units_b": qb,
            "entry_notional_a": na, "entry_notional_b": nb, "rebalances": 0, "gross_pnl": 0.0,
            "transaction_costs": c, "borrow_costs": 0.0, "_entry_bar": i,
        }
        return abs(na) + abs(nb)

    def close_position(i: int, reason: str, sig_i: int) -> float:
        nonlocal qa, qb, state, open_trade, equity, realized_cum
        na, nb = -qa * pa[i], -qb * pb[i]
        c = cost.trade_cost(na, nb)
        equity -= c
        trade = open_trade
        trade["transaction_costs"] += c
        forced = reason in ("end_of_period", "equity_depleted")
        trade.update(
            status="closed", exit_reason=reason or "signal", exit_date=dates[i],
            exit_signal_date=dates[i] if forced else dates[sig_i],
            exit_z=zvals[i] if forced else zvals[sig_i], holding_days=i - trade["_entry_bar"],
            exit_price_a=pa[i], exit_price_b=pb[i],
        )
        _finalise_trade(trade)
        realized_cum += trade["net_pnl"]
        trades.append(trade)
        open_trade = None
        qa = qb = 0.0
        state = 0
        return abs(na) + abs(nb)

    for i in range(n):
        # 1. Mark to market positions carried from the previous close.
        if i > 0:
            pnl = qa * (pa[i] - pa[i - 1]) + qb * (pb[i] - pb[i - 1])
            short_prev = -(min(qa * pa[i - 1], 0.0) + min(qb * pb[i - 1], 0.0))
            borrow = cost.borrow_cost(short_prev)
        else:
            pnl = borrow = 0.0
        equity_prev_close = equity
        equity += pnl - borrow
        if open_trade is not None:
            open_trade["gross_pnl"] += pnl
            open_trade["borrow_costs"] += borrow
        if equity <= 0 and not ruined:
            ruined = True
            warnings.append(f"Equity depleted on {dates[i].date()}; all positions closed and trading stopped.")
        equity_before_trades = equity

        # 2. Position desired for execution at this close (signal decided `lag` bars earlier).
        sig_i = i - lag
        desired = int(target[sig_i]) if sig_i >= 0 else 0
        reason = events[sig_i] if sig_i >= 0 else ""
        if ruined:
            desired, reason = 0, "equity_depleted"
        elif i == n - 1 and config.close_at_period_end:
            desired, reason = 0, "end_of_period"

        # 3. Execute.
        traded = 0.0
        cost_before = equity
        event = ""
        if desired != state:
            if state != 0:
                traded += close_position(i, reason, sig_i)
                event = reason or "exit"
            if desired != 0:
                traded += open_position(i, desired, sig_i)
                entry_label = events[sig_i] or ("entry_long" if desired > 0 else "entry_short")
                event = entry_label if event == "" else f"{event}+{entry_label}"
        elif state != 0 and tol is not None and prev_leverage > limit * (1.0 + tol):
            # Leverage observed at the previous close exceeded the limit: scale back at this close.
            wa, wb = target_leg_weights(state, hedge_ratio, limit)
            na_new, nb_new = wa * equity, wb * equity
            da, db = na_new - qa * pa[i], nb_new - qb * pb[i]
            c = cost.trade_cost(da, db)
            equity -= c
            qa, qb = na_new / pa[i], nb_new / pb[i]
            open_trade["transaction_costs"] += c
            open_trade["rebalances"] += 1
            traded = abs(da) + abs(db)
            event = "rebalance"
        trade_cost = cost_before - equity

        # 4. Record end-of-bar state.
        na, nb = qa * pa[i], qb * pb[i]
        long_exp = max(na, 0.0) + max(nb, 0.0)
        short_exp = -(min(na, 0.0) + min(nb, 0.0))
        gross = long_exp + short_exp
        leverage = gross / equity if equity > 0 else np.nan
        prev_leverage = leverage if np.isfinite(leverage) else 0.0
        base = equity_prev_close if i > 0 else float(config.initial_capital)
        rec["signal_target"].append(int(target[i]))
        rec["position"].append(state)
        rec["event"].append(event)
        rec["signal_date_executed"].append(dates[sig_i] if event and sig_i >= 0 else pd.NaT)
        rec["units_a"].append(qa)
        rec["units_b"].append(qb)
        rec["price_a"].append(pa[i])
        rec["price_b"].append(pb[i])
        rec["notional_a"].append(na)
        rec["notional_b"].append(nb)
        rec["long_exposure"].append(long_exp)
        rec["short_exposure"].append(short_exp)
        rec["gross_exposure"].append(gross)
        rec["net_exposure"].append(na + nb)
        rec["gross_leverage"].append(leverage)
        rec["net_leverage"].append((na + nb) / equity if equity > 0 else np.nan)
        rec["cash"].append(equity - na - nb)
        rec["pnl"].append(pnl)
        rec["trade_cost"].append(trade_cost)
        rec["borrow_cost"].append(borrow)
        rec["traded_notional"].append(traded)
        rec["turnover"].append(traded / equity_before_trades if equity_before_trades > 0 else np.nan)
        rec["equity"].append(equity)
        rec["gross_return"].append(pnl / base if i > 0 else 0.0)
        rec["net_return"].append(equity / base - 1.0)
        rec["unrealized_pnl"].append(open_trade["gross_pnl"] if open_trade is not None else 0.0)
        rec["realized_pnl_cum"].append(realized_cum)

    if open_trade is not None:
        _finalise_trade(open_trade)
        trades.append(open_trade)
        warnings.append("A position is still open at the end of the period (close_at_period_end is disabled).")

    daily = pd.DataFrame(rec, index=dates)
    daily["cumulative_costs"] = (daily["trade_cost"] + daily["borrow_cost"]).cumsum()
    daily["drawdown"] = daily["equity"] / daily["equity"].cummax() - 1.0
    trade_frame = pd.DataFrame(trades, columns=TRADE_COLUMNS + ["_entry_bar"]).drop(columns="_entry_bar")
    return BacktestResult(daily, trade_frame, signals.copy(), float(hedge_ratio), config, warnings)


def _finalise_trade(trade: dict) -> None:
    trade["net_pnl"] = trade["gross_pnl"] - trade["transaction_costs"] - trade["borrow_costs"]
    trade["gross_return"] = trade["gross_pnl"] / trade["entry_equity"]
    trade["net_return"] = trade["net_pnl"] / trade["entry_equity"]


def _validate(prices: pd.DataFrame, signals: pd.DataFrame, hedge_ratio: float) -> None:
    if not {"A", "B"}.issubset(prices.columns):
        raise ValueError("prices must contain columns 'A' and 'B'")
    if "target_position" not in signals:
        raise ValueError("signals must contain a 'target_position' column")
    if not prices.index.equals(signals.index):
        raise ValueError("prices and signals must share an identical index")
    if len(prices) < 2:
        raise ValueError("at least two observations are required for a backtest")
    values = prices[["A", "B"]].to_numpy(float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("prices must be finite and strictly positive")
    if not set(np.unique(signals["target_position"].to_numpy())).issubset({-1, 0, 1}):
        raise ValueError("target_position values must be -1, 0 or +1")
    if not np.isfinite(hedge_ratio):
        raise ValueError("hedge_ratio must be finite")


__all__ = ["BacktestConfig", "BacktestResult", "TRADE_COLUMNS", "run_backtest"]
