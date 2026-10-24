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


__all__ = ["CostModel", "target_leg_weights"]
