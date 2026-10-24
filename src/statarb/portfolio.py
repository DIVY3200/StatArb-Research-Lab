"""Two-leg position sizing and cost model.

From hedge ratio to leg weights
-------------------------------
For small moves, ``d spread ≈ r_A - hedge_ratio * r_B`` (log returns). A portfolio
holding dollar weights ``w_A = +k`` and ``w_B = -k * hedge_ratio`` of equity therefore
earns approximately ``k * d spread``: the log-price regression coefficient maps to a
**dollar (notional) ratio**, not to a share ratio. The implied share ratio is
``units_B / units_A = -hedge_ratio * P_A / P_B``.

Sizing convention (gross-exposure normalised)
---------------------------------------------
``k = max_gross_exposure / (1 + |hedge_ratio|)`` so that at entry
``|w_A| + |w_B| = max_gross_exposure`` (e.g. 1.0 = 100% of current equity).
For ``hedge_ratio = 1`` this is dollar-neutral; otherwise the position carries net
exposure ``k * (1 - hedge_ratio)`` by design, because it follows the estimated relation.

Units are fixed between trades (no daily rebalancing), so leg weights drift with prices.
If gross leverage drifts above ``max_gross_exposure * (1 + rebalance_tolerance)`` the
backtester scales the position back to the limit (with costs).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


def target_leg_weights(state: int, hedge_ratio: float, max_gross_exposure: float) -> tuple[float, float]:
    """Dollar weights (fractions of equity) for legs A and B in a given spread state."""
    if state not in (-1, 0, 1):
        raise ValueError("state must be -1, 0 or +1")
    if max_gross_exposure <= 0:
        raise ValueError("max_gross_exposure must be positive")
    if state == 0:
        return 0.0, 0.0
    scale = max_gross_exposure / (1.0 + abs(hedge_ratio))
    return state * scale, -state * hedge_ratio * scale


@dataclass(frozen=True)
class CostModel:
    """Linear transaction-cost and borrow-fee model.

    * Every traded dollar on either leg pays ``commission_bps + slippage_bps``
      (entries, exits, and rebalancing alike).
    * Short positions accrue ``borrow_bps_annual`` per year on their prior-close
      notional, charged daily as ``bps / 1e4 / periods_per_year``.

    These are generic research assumptions, not a model of any particular broker.
    """

    commission_bps: float = 5.0
    slippage_bps: float = 5.0
    borrow_bps_annual: float = 0.0
    periods_per_year: int = 252

    def __post_init__(self) -> None:
        if min(self.commission_bps, self.slippage_bps, self.borrow_bps_annual) < 0:
            raise ValueError("costs must be non-negative")
        if self.periods_per_year <= 0:
            raise ValueError("periods_per_year must be positive")

    @property
    def trade_cost_rate(self) -> float:
        return (self.commission_bps + self.slippage_bps) / 1e4

    @property
    def daily_borrow_rate(self) -> float:
        return self.borrow_bps_annual / 1e4 / self.periods_per_year

    def trade_cost(self, traded_notional_a: float, traded_notional_b: float) -> float:
        return (abs(traded_notional_a) + abs(traded_notional_b)) * self.trade_cost_rate

    def borrow_cost(self, short_notional: float) -> float:
        return abs(short_notional) * self.daily_borrow_rate

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class MarketCostModel:
    """Per-leg, per-side cost schedule for the ``generic`` and ``nse`` profiles (all rates in bps).

    * ``generic``: every trade pays ``commission_bps + slippage_bps``, whatever the instrument or side.
    * ``nse``: cash-delivery and single-stock-futures schedules built from statutory components:
      STT (delivery: buys and sells; futures: sells only), exchange transaction charges, the SEBI fee,
      stamp duty (buys only), GST on brokerage + exchange + SEBI charges, plus brokerage, slippage
      and optional flat per-order fees. The values are the researcher's understanding of the late-2024
      schedule and must be checked against current circulars and broker contract notes.
    * Short legs held as futures (``short_instrument="futures"``) pay no borrow fee; each roll costs a
      futures buy + sell on the rolled notional plus ``futures_roll_spread_bps``.
    """

    profile: str = "generic"
    commission_bps: float = 5.0
    slippage_bps: float = 5.0
    borrow_bps_annual: float = 0.0
    periods_per_year: int = 252
    short_instrument: str = "cash"
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
    futures_roll_spread_bps: float = 2.0

    def __post_init__(self) -> None:
        if self.profile not in {"generic", "nse"}:
            raise ValueError("profile must be generic or nse")
        if self.short_instrument not in {"cash", "futures"}:
            raise ValueError("short_instrument must be cash or futures")

    def instrument(self, units: float) -> str:
        """Instrument used for a position of ``units`` (shorts use futures when configured)."""
        return "futures" if self.short_instrument == "futures" and units < 0 else "cash"

    def rate_bps(self, instrument: str, side: str) -> float:
        if instrument not in {"cash", "futures"} or side not in {"buy", "sell"}:
            raise ValueError("instrument must be cash/futures and side buy/sell")
        if self.profile == "generic":
            return self.commission_bps + self.slippage_bps
        if instrument == "cash":
            fees = self.exchange_cash_bps + self.sebi_fee_bps + self.brokerage_bps
            rate = self.stt_delivery_bps + fees * (1.0 + self.gst_rate)
            rate += self.stamp_delivery_buy_bps if side == "buy" else 0.0
        else:
            fees = self.exchange_futures_bps + self.sebi_fee_bps + self.brokerage_bps
            rate = fees * (1.0 + self.gst_rate)
            rate += self.stamp_futures_buy_bps if side == "buy" else self.stt_futures_sell_bps
        return rate + self.slippage_bps

    def flat_fee(self, instrument: str, side: str) -> float:
        if self.profile == "generic":
            return 0.0
        if instrument == "cash":
            fee = self.brokerage_flat_cash * (1.0 + self.gst_rate)
            return fee + (self.dp_charge_cash_sell if side == "sell" else 0.0)
        return self.brokerage_flat_futures * (1.0 + self.gst_rate)

    def leg_cost(self, traded_notional: float, instrument: str, side: str) -> float:
        """Cost of one leg trade (zero for no trade)."""
        if traded_notional == 0:
            return 0.0
        return abs(traded_notional) * self.rate_bps(instrument, side) / 1e4 + self.flat_fee(instrument, side)

    def roll_cost(self, futures_notional: float) -> float:
        """Close the expiring contract and open the next one on the same notional."""
        if futures_notional == 0:
            return 0.0
        rate = self.rate_bps("futures", "buy") + self.rate_bps("futures", "sell") + self.futures_roll_spread_bps
        return abs(futures_notional) * rate / 1e4 + self.flat_fee("futures", "buy") + self.flat_fee("futures", "sell")

    def borrow_cost(self, short_cash_notional: float) -> float:
        """Borrow fee for one day on a cash short (futures shorts pay none)."""
        return abs(short_cash_notional) * self.borrow_bps_annual / 1e4 / self.periods_per_year


def lot_scale(units: float, lot_size: float | None) -> float | None:
    """Factor (<= 1) that rounds ``units`` toward zero to whole lots; ``None`` if not even one lot fits."""
    if lot_size is None or units == 0:
        return 1.0
    lots = np.floor(abs(units) / lot_size)
    if lots < 1:
        return None
    return float(lots * lot_size / abs(units))


__all__ = ["CostModel", "MarketCostModel", "lot_scale", "target_leg_weights"]
