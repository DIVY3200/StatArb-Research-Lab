"""Event-driven two-leg backtester with explicit execution timing and costs.

Timing model (delayed execution)
--------------------------------
* ``signals.target_position[t]`` is decided from information available at the close of bar ``t``.
* It is executed ``execution_lag`` bars later (``execution_lag >= 1``) at that bar's
  - **close** (``execution_price="close"``, baseline): the "wait one day" convention of Gatev,
    Goetzmann and Rouwenhorst (2006); or
  - **open** (``execution_price="open"``): the first executable price after the signal. Opening
    prices must be on the same adjustment basis as the closes (see ``AlignedPair.opens``); bars
    without a usable open fall back to the close and are counted in the warnings.
* On each bar, P&L is first marked on the units carried from the previous close up to the fill
  price ("overnight" part), then trades are executed and costs deducted, then the units held after
  trading are marked from the fill price to the close ("intraday" part, zero for close fills). A
  signal therefore never earns the price move that produced it.
* Signals whose execution bar would fall beyond the data are never executed. With
  ``close_at_period_end`` the final bar liquidates any open position (paying exit costs) and no new
  position is opened on it, so the last observation adds no artificial return.

Accounting
----------
* Units are fixed between trades. Equity = previous equity + P&L + interest - borrow - costs.
* Interest (``cash_interest_rate``, off in the baseline) is credited daily on the prior-close cash
  balance ``equity - long notional + short notional``: idle capital, short-sale proceeds, and the
  cost-of-carry that a short futures position earns as the future converges to spot.
* Costs: ``generic`` profile = commission + slippage per side on both legs (baseline arithmetic);
  ``nse`` profile = per-leg, per-side statutory schedule (:class:`~statarb.portfolio.MarketCostModel`).
* ``short_instrument="futures"``: short legs are single-stock futures. They pay no borrow fee, are
  rolled every ``futures_roll_every_bars`` bars (cost of a futures buy + sell + roll spread), are
  rounded down to whole lots when a lot size is given (both legs scaled together to keep the hedge
  ratio), and their initial margin is reported (``futures_margin``) with a warning if the long cash
  leg plus margin exceeds equity.
* ``gross_return`` includes P&L and interest but excludes trading and borrow costs, so
  ``net_return <= gross_return`` on every bar.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from .metrics import daily_risk_free_series
from .portfolio import CostModel, MarketCostModel, lot_scale, target_leg_weights

_MARKET_FIELDS = (
    "stt_delivery_bps", "stt_futures_sell_bps", "exchange_cash_bps", "exchange_futures_bps", "sebi_fee_bps",
    "stamp_delivery_buy_bps", "stamp_futures_buy_bps", "gst_rate", "brokerage_bps", "brokerage_flat_cash",
    "brokerage_flat_futures", "dp_charge_cash_sell", "futures_roll_spread_bps",
)


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
    execution_price: str = "close"
    cost_profile: str = "generic"
    short_instrument: str = "cash"
    cash_interest_rate: float = 0.0
    cash_interest_by_year: tuple[tuple[int, float], ...] = ()  # calendar-year schedule; overrides the constant
    stt_delivery_bps: float = 10.0
    stt_futures_sell_bps: float = 2.0
    exchange_cash_bps: float = 0.297
    exchange_futures_bps: float = 0.173
    sebi_fee_bps: float = 0.01
    stamp_delivery_buy_bps: float = 1.5
    stamp_futures_buy_bps: float = 0.2
    gst_rate: float = 0.18
    brokerage_bps: float = 0.0
    brokerage_flat_cash: float = 0.0
    brokerage_flat_futures: float = 0.0
    dp_charge_cash_sell: float = 0.0
    futures_roll_every_bars: int = 21
    futures_roll_spread_bps: float = 2.0
    futures_margin_rate: float = 0.2
    futures_lot_size_a: float | None = None
    futures_lot_size_b: float | None = None

    def __post_init__(self) -> None:
        if self.execution_lag < 1:
            raise ValueError("execution_lag must be >= 1 to avoid trading on the signal bar's own close")
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        if self.max_gross_exposure <= 0:
            raise ValueError("max_gross_exposure must be positive")
        if self.execution_price not in {"close", "open"}:
            raise ValueError("execution_price must be close or open")
        if self.cost_profile not in {"generic", "nse"}:
            raise ValueError("cost_profile must be generic or nse")
        if self.short_instrument not in {"cash", "futures"}:
            raise ValueError("short_instrument must be cash or futures")
        if not -0.05 <= self.cash_interest_rate <= 0.5:
            raise ValueError("cash_interest_rate must be an annual decimal rate")
        schedule = self.cash_interest_by_year
        schedule = schedule.items() if isinstance(schedule, dict) else schedule
        schedule = tuple(sorted((int(y), float(r)) for y, r in schedule))
        if any(not -0.05 <= r <= 0.5 for _, r in schedule):
            raise ValueError("cash_interest_by_year rates must be annual decimal rates")
        object.__setattr__(self, "cash_interest_by_year", schedule)
        if self.futures_roll_every_bars < 1:
            raise ValueError("futures_roll_every_bars must be at least 1")
        for lot in (self.futures_lot_size_a, self.futures_lot_size_b):
            if lot is not None and lot <= 0:
                raise ValueError("futures lot sizes must be positive")

    @classmethod
    def from_research_config(cls, config, tickers: tuple[str, str] | None = None, **overrides) -> "BacktestConfig":
        """Build from a :class:`~statarb.config.ResearchConfig`; ``tickers`` resolve futures lot sizes."""
        c = config.costs
        values = dict(
            initial_capital=config.portfolio.initial_capital,
            max_gross_exposure=config.portfolio.max_gross_exposure,
            execution_lag=config.strategy.execution_lag,
            commission_bps=c.commission_bps,
            slippage_bps=c.slippage_bps,
            borrow_bps_annual=c.borrow_bps_annual,
            rebalance_tolerance=config.portfolio.rebalance_tolerance,
            close_at_period_end=config.portfolio.close_at_period_end,
            short_selling_allowed=c.short_selling_allowed,
            periods_per_year=config.metrics.periods_per_year,
            execution_price=config.strategy.execution_price,
            cost_profile=c.profile,
            short_instrument=c.short_instrument,
            cash_interest_rate=config.metrics.risk_free_rate if config.portfolio.cash_interest else 0.0,
            cash_interest_by_year=config.metrics.risk_free_by_year if config.portfolio.cash_interest else (),
            futures_roll_every_bars=c.futures_roll_every_bars,
            futures_margin_rate=c.futures_margin_rate,
            futures_lot_size_a=c.lot_size(tickers[0]) if tickers else None,
            futures_lot_size_b=c.lot_size(tickers[1]) if tickers else None,
            **{name: getattr(c, name) for name in _MARKET_FIELDS},
        )
        values.update(overrides)
        return cls(**values)

    @property
    def simple_costs(self) -> bool:
        """True for the baseline cost model (generic profile, cash shorts)."""
        return self.cost_profile == "generic" and self.short_instrument == "cash"

    def cost_model(self) -> CostModel:
        return CostModel(self.commission_bps, self.slippage_bps, self.borrow_bps_annual, self.periods_per_year)

    def market_cost_model(self) -> MarketCostModel:
        return MarketCostModel(
            profile=self.cost_profile, commission_bps=self.commission_bps, slippage_bps=self.slippage_bps,
            borrow_bps_annual=self.borrow_bps_annual, periods_per_year=self.periods_per_year,
            short_instrument=self.short_instrument, **{name: getattr(self, name) for name in _MARKET_FIELDS},
        )

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
    "rolls",
]


def run_backtest(
    prices: pd.DataFrame,
    signals: pd.DataFrame,
    hedge_ratio: float,
    config: BacktestConfig | None = None,
    opens: pd.DataFrame | None = None,
    accrue_first_bar: bool = False,
) -> BacktestResult:
    """Simulate both legs of the pairs strategy over one evaluation period.

    ``prices`` must have columns ``A`` and ``B`` (the closes used for returns, e.g. adjusted closes)
    and share its index with ``signals`` (which needs ``target_position`` and optionally ``event``
    and ``zscore``). ``opens`` (same index and adjustment basis) is required for
    ``execution_price="open"``. The period always starts flat.

    ``accrue_first_bar``: the period continues an account that was flat in cash at the previous close
    (a chained walk-forward window after the first), so bar 0 earns one bar of cash interest. A
    stand-alone period has no holding period before its first bar, which therefore earns nothing.
    """
    config = config or BacktestConfig()
    _validate(prices, signals, hedge_ratio)
    cost = config.cost_model()
    market = config.market_cost_model()
    simple = config.simple_costs
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
    use_open = config.execution_price == "open"
    if use_open:
        if opens is None:
            raise ValueError("execution_price='open' needs opening prices (pass AlignedPair.opens for the period)")
        if not opens.index.equals(prices.index):
            raise ValueError("opens must share the price index")
        oa, ob = opens["A"].to_numpy(float), opens["B"].to_numpy(float)
        usable = np.isfinite(oa) & (oa > 0) & np.isfinite(ob) & (ob > 0)
        fa, fb = np.where(usable, oa, pa), np.where(usable, ob, pb)
    else:
        usable = np.ones(n, dtype=bool)
        fa, fb = pa, pb
    futures_mode = config.short_instrument == "futures"
    lots = (config.futures_lot_size_a, config.futures_lot_size_b)
    rate = config.cash_interest_rate
    if config.cash_interest_by_year:
        r_cash_bar = daily_risk_free_series(dict(config.cash_interest_by_year), prices.index,
                                            config.periods_per_year).to_numpy()
        r_cash = bool(np.any(r_cash_bar))
    else:
        r_cash = (1.0 + rate) ** (1.0 / config.periods_per_year) - 1.0 if rate else 0.0
        r_cash_bar = None

    qa = qb = 0.0
    state = 0
    equity = float(config.initial_capital)
    prev_leverage = 0.0
    realized_cum = 0.0
    ruined = False
    margin_breach = False
    lot_blocked = 0
    fallback_fills = 0
    bars_since_roll = [0, 0]
    warnings: list[str] = []
    open_trade: dict | None = None
    trades: list[dict] = []
    rec: dict[str, list] = {k: [] for k in (
        "signal_target", "position", "event", "signal_date_executed", "units_a", "units_b", "price_a",
        "price_b", "notional_a", "notional_b", "long_exposure", "short_exposure", "gross_exposure",
        "net_exposure", "gross_leverage", "net_leverage", "cash", "pnl", "trade_cost", "borrow_cost",
        "traded_notional", "turnover", "equity", "gross_return", "net_return", "unrealized_pnl",
        "realized_pnl_cum", "interest", "fill_price_a", "fill_price_b", "futures_margin",
    )}

    def legs_trade_cost(qa_old: float, qa_new: float, qb_old: float, qb_new: float, i: int) -> float:
        """Market-schedule cost of moving both legs; a leg trade touching a short uses futures if configured."""
        total = 0.0
        for q_old, q_new, price in ((qa_old, qa_new, fa[i]), (qb_old, qb_new, fb[i])):
            dq = q_new - q_old
            if dq == 0:
                continue
            instrument = "futures" if futures_mode and (q_old < 0 or q_new < 0) else "cash"
            total += market.leg_cost(dq * price, instrument, "buy" if dq > 0 else "sell")
        return total

    def round_to_lots(qa_new: float, qb_new: float):
        """Round futures (short) legs down to whole lots; a single futures leg scales both legs together."""
        if not futures_mode or (lots[0] is None and lots[1] is None):
            return qa_new, qb_new
        futures_legs = [(k, q, lot) for k, (q, lot) in enumerate(zip((qa_new, qb_new), lots))
                        if q < 0 and lot is not None]
        if not futures_legs:
            return qa_new, qb_new
        if len(futures_legs) == 1:
            scale = lot_scale(futures_legs[0][1], futures_legs[0][2])
            return None if scale is None else (qa_new * scale, qb_new * scale)
        out = [qa_new, qb_new]
        for k, q, lot in futures_legs:
            scale = lot_scale(q, lot)
            if scale is None:
                return None
            out[k] = q * scale
        return out[0], out[1]

    def open_position(i: int, desired: int, sig_i: int) -> tuple[float, bool]:
        nonlocal qa, qb, state, open_trade, equity, lot_blocked
        wa, wb = target_leg_weights(desired, hedge_ratio, limit)
        if simple:
            # Size so that gross notional equals `limit` x equity *after* paying the entry cost.
            budget = equity / (1.0 + (abs(wa) + abs(wb)) * cost.trade_cost_rate)
            na, nb = wa * budget, wb * budget
            c = cost.trade_cost(na, nb)
            qa_new, qb_new = na / fa[i], nb / fb[i]
        else:
            inst_a, inst_b = market.instrument(wa), market.instrument(wb)
            side_a, side_b = ("buy" if wa > 0 else "sell"), ("buy" if wb > 0 else "sell")
            proportional = (abs(wa) * market.rate_bps(inst_a, side_a) + abs(wb) * market.rate_bps(inst_b, side_b)) / 1e4
            flat = (market.flat_fee(inst_a, side_a) if wa else 0.0) + (market.flat_fee(inst_b, side_b) if wb else 0.0)
            budget = max(equity - flat, 0.0) / (1.0 + proportional)
            rounded = round_to_lots(wa * budget / fa[i], wb * budget / fb[i])
            if rounded is None:
                lot_blocked += 1
                return 0.0, False
            qa_new, qb_new = rounded
            na, nb = qa_new * fa[i], qb_new * fb[i]
            c = legs_trade_cost(0.0, qa_new, 0.0, qb_new, i)
        entry_equity = equity
        equity -= c
        qa, qb, state = qa_new, qb_new, desired
        bars_since_roll[0] = bars_since_roll[1] = 0
        open_trade = {
            "trade_id": len(trades) + 1, "status": "open",
            "direction": "long_spread" if desired > 0 else "short_spread",
            "entry_signal_date": dates[sig_i], "entry_date": dates[i], "exit_signal_date": pd.NaT,
            "exit_date": pd.NaT, "exit_reason": "", "entry_z": zvals[sig_i], "exit_z": np.nan,
            "holding_days": 0, "entry_equity": entry_equity, "entry_price_a": fa[i], "entry_price_b": fb[i],
            "exit_price_a": np.nan, "exit_price_b": np.nan, "units_a": qa, "units_b": qb,
            "entry_notional_a": na, "entry_notional_b": nb, "rebalances": 0, "gross_pnl": 0.0,
            "transaction_costs": c, "borrow_costs": 0.0, "rolls": 0, "_entry_bar": i,
        }
        return abs(na) + abs(nb), True

    def close_position(i: int, reason: str, sig_i: int) -> float:
        nonlocal qa, qb, state, open_trade, equity, realized_cum
        na, nb = -qa * fa[i], -qb * fb[i]
        c = cost.trade_cost(na, nb) if simple else legs_trade_cost(qa, 0.0, qb, 0.0, i)
        equity -= c
        trade = open_trade
        trade["transaction_costs"] += c
        forced = reason in ("end_of_period", "equity_depleted")
        trade.update(
            status="closed", exit_reason=reason or "signal", exit_date=dates[i],
            exit_signal_date=dates[i] if forced else dates[sig_i],
            exit_z=zvals[i] if forced else zvals[sig_i], holding_days=i - trade["_entry_bar"],
            exit_price_a=fa[i], exit_price_b=fb[i],
        )
        _finalise_trade(trade)
        realized_cum += trade["net_pnl"]
        trades.append(trade)
        open_trade = None
        qa = qb = 0.0
        state = 0
        return abs(na) + abs(nb)

    for i in range(n):
        # 1. Mark the units carried from the previous close to this bar's fill price ("overnight").
        if i > 0:
            pnl = qa * (fa[i] - pa[i - 1]) + qb * (fb[i] - pb[i - 1])
            if simple:
                short_prev = -(min(qa * pa[i - 1], 0.0) + min(qb * pb[i - 1], 0.0))
                borrow = cost.borrow_cost(short_prev)
            else:
                borrow = sum(market.borrow_cost(q * p) for q, p in ((qa, pa[i - 1]), (qb, pb[i - 1]))
                             if q < 0 and market.instrument(q) == "cash")
            if r_cash:
                long_prev = max(qa * pa[i - 1], 0.0) + max(qb * pb[i - 1], 0.0)
                short_all = -(min(qa * pa[i - 1], 0.0) + min(qb * pb[i - 1], 0.0))
                rate_i = r_cash if r_cash_bar is None else r_cash_bar[i]
                interest = rate_i * (equity - long_prev + short_all)
            else:
                interest = 0.0
        else:
            pnl = borrow = 0.0
            # Flat at the start, so the whole equity is cash; only a chained window has a bar to accrue over.
            interest = (r_cash if r_cash_bar is None else r_cash_bar[0]) * equity \
                if accrue_first_bar and r_cash else 0.0
        if use_open and not usable[i]:
            fallback_fills += 1
        equity_prev_close = equity
        equity += pnl - borrow + interest
        if open_trade is not None:
            open_trade["gross_pnl"] += pnl
            open_trade["borrow_costs"] += borrow
        if equity <= 0 and not ruined:
            ruined = True
            warnings.append(f"Equity depleted on {dates[i].date()}; all positions closed and trading stopped.")
        equity_before_trades = equity

        # 2. Position desired for execution at this bar (signal decided `lag` bars earlier).
        sig_i = i - lag
        desired = int(target[sig_i]) if sig_i >= 0 else 0
        reason = events[sig_i] if sig_i >= 0 else ""
        if ruined:
            desired, reason = 0, "equity_depleted"
        elif i == n - 1 and config.close_at_period_end:
            desired, reason = 0, "end_of_period"

        # 3. Execute at the fill price.
        traded = 0.0
        cost_before = equity
        event = ""
        state_before = state
        if desired != state:
            if state != 0:
                traded += close_position(i, reason, sig_i)
                event = reason or "exit"
            if desired != 0:
                notional, opened = open_position(i, desired, sig_i)
                traded += notional
                if opened:
                    entry_label = events[sig_i] or ("entry_long" if desired > 0 else "entry_short")
                else:
                    entry_label = "lot_size_block"
                event = entry_label if event == "" else f"{event}+{entry_label}"
        elif state != 0 and tol is not None and prev_leverage > limit * (1.0 + tol):
            # Leverage observed at the previous close exceeded the limit: scale back at this bar.
            wa, wb = target_leg_weights(state, hedge_ratio, limit)
            na_new, nb_new = wa * equity, wb * equity
            if simple:
                da, db = na_new - qa * fa[i], nb_new - qb * fb[i]
                c = cost.trade_cost(da, db)
                qa_new, qb_new = na_new / fa[i], nb_new / fb[i]
            else:
                rounded = round_to_lots(na_new / fa[i], nb_new / fb[i])
                qa_new, qb_new = rounded if rounded is not None else (qa, qb)
                da, db = (qa_new - qa) * fa[i], (qb_new - qb) * fb[i]
                c = legs_trade_cost(qa, qa_new, qb, qb_new, i)
            equity -= c
            qa, qb = qa_new, qb_new
            open_trade["transaction_costs"] += c
            open_trade["rebalances"] += 1
            traded = abs(da) + abs(db)
            event = "rebalance"
        # Roll futures legs held through this bar (only when the same position continues).
        if futures_mode and state != 0 and state == state_before and open_trade is not None:
            for k, (q, price) in enumerate(((qa, fa[i]), (qb, fb[i]))):
                if q >= 0:
                    continue
                bars_since_roll[k] += 1
                if bars_since_roll[k] >= config.futures_roll_every_bars:
                    c = market.roll_cost(q * price)
                    equity -= c
                    open_trade["transaction_costs"] += c
                    open_trade["rolls"] += 1
                    traded += 2.0 * abs(q * price)
                    bars_since_roll[k] = 0
                    event = "roll" if event == "" else f"{event}+roll"
        trade_cost = cost_before - equity

        # 4. Mark the units held after trading from the fill price to the close ("intraday").
        if use_open and i > 0:
            pnl_intraday = qa * (pa[i] - fa[i]) + qb * (pb[i] - fb[i])
            equity += pnl_intraday
            pnl += pnl_intraday
            if open_trade is not None:
                open_trade["gross_pnl"] += pnl_intraday

        # 5. Record end-of-bar state.
        na, nb = qa * pa[i], qb * pb[i]
        long_exp = max(na, 0.0) + max(nb, 0.0)
        short_exp = -(min(na, 0.0) + min(nb, 0.0))
        gross = long_exp + short_exp
        leverage = gross / equity if equity > 0 else np.nan
        prev_leverage = leverage if np.isfinite(leverage) else 0.0
        futures_margin = config.futures_margin_rate * short_exp if futures_mode else 0.0
        if futures_mode and long_exp + futures_margin > equity and not margin_breach:
            margin_breach = True
            warnings.append(f"On {dates[i].date()} the long cash leg plus futures margin exceeded equity.")
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
        rec["gross_return"].append((pnl + interest) / base)
        rec["net_return"].append(equity / base - 1.0)
        rec["unrealized_pnl"].append(open_trade["gross_pnl"] if open_trade is not None else 0.0)
        rec["realized_pnl_cum"].append(realized_cum)
        rec["interest"].append(interest)
        rec["fill_price_a"].append(fa[i])
        rec["fill_price_b"].append(fb[i])
        rec["futures_margin"].append(futures_margin)

    if open_trade is not None:
        _finalise_trade(open_trade)
        trades.append(open_trade)
        warnings.append("A position is still open at the end of the period (close_at_period_end is disabled).")
    if fallback_fills:
        warnings.append(f"{fallback_fills} bar(s) had no usable opening price; those fills used the close.")
    if lot_blocked:
        warnings.append(f"{lot_blocked} entry attempt(s) were skipped because the position was smaller than one "
                        "futures lot.")

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
