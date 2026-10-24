"""Spread construction and training-normalised z-scores.

    spread_t = log(P_A,t) - intercept - hedge_ratio * log(P_B,t)
    z_t      = (spread_t - training_spread_mean) / training_spread_std

All four parameters (intercept, hedge ratio, mean, std) are estimated on the fitting
window and then held fixed while the model is applied to the following evaluation
window. Because ``z_t`` depends only on prices at ``t`` and on parameters fixed before
the window starts, computing it for every evaluation date at once introduces no
look-ahead bias.

A positive z-score means A is expensive relative to B under the fitted relation; a
negative z-score means A is cheap relative to B.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .hedge_ratio import HedgeRatioResult, estimate_hedge_ratio


def compute_spread(log_a: pd.Series, log_b: pd.Series, intercept: float, hedge_ratio: float) -> pd.Series:
    if not log_a.index.equals(log_b.index):
        raise ValueError("log_a and log_b must share an identical index.")
    spread = log_a - intercept - hedge_ratio * log_b
    spread.name = "spread"
    return spread


def compute_zscore(spread: pd.Series, mean: float, std: float, min_std: float = 1e-8) -> pd.Series:
    """Standardise a spread with externally supplied (training) parameters."""
    if not np.isfinite(std) or std < min_std:
        raise ValueError(
            f"Spread standard deviation {std!r} is below the minimum {min_std}; the training spread is "
            "degenerate and z-scores would be unstable."
        )
    z = (spread - mean) / std
    z.name = "zscore"
    return z


@dataclass(frozen=True)
class SpreadModel:
    """Frozen spread parameters estimated on one fitting window."""

    intercept: float
    hedge_ratio: float
    mean: float
    std: float
    fit_start: str
    fit_end: str
    n_obs: int

    def spread(self, log_a: pd.Series, log_b: pd.Series) -> pd.Series:
        return compute_spread(log_a, log_b, self.intercept, self.hedge_ratio)

    def zscore(self, log_a: pd.Series, log_b: pd.Series) -> pd.Series:
        return compute_zscore(self.spread(log_a, log_b), self.mean, self.std, min_std=1e-12)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def fit_spread_model(
    log_a_train: pd.Series,
    log_b_train: pd.Series,
    min_std: float = 1e-8,
    hedge: HedgeRatioResult | None = None,
) -> SpreadModel:
    """Fit hedge ratio (if not supplied) and spread normalisation on training data only.

    The standard deviation uses ``ddof=1``. With an OLS intercept the training mean is
    zero up to floating-point error, but it is estimated explicitly for transparency.
    """
    if hedge is None:
        hedge = estimate_hedge_ratio(log_a_train, log_b_train)
    spread = compute_spread(log_a_train, log_b_train, hedge.intercept, hedge.hedge_ratio)
    mean = float(spread.mean())
    std = float(spread.std(ddof=1))
    if not np.isfinite(std) or std < min_std:
        raise ValueError(
            f"Training spread standard deviation ({std:.3g}) is below {min_std:.1e}; the pair is degenerate "
            "(for example, one series is a scaled copy of the other)."
        )
    return SpreadModel(
        intercept=hedge.intercept,
        hedge_ratio=hedge.hedge_ratio,
        mean=mean,
        std=std,
        fit_start=hedge.fit_start,
        fit_end=hedge.fit_end,
        n_obs=hedge.n_obs,
    )


def half_life(spread: pd.Series) -> float:
    """Mean-reversion half-life (in bars) from an AR(1) fit ``d s_t = a + b s_{t-1} + e_t``.

    Returns ``nan`` when ``b >= 0`` (no estimated mean reversion) or the input is too short.
    """
    s = spread.dropna().to_numpy(float)
    if len(s) < 10:
        return float("nan")
    lagged, delta = s[:-1], np.diff(s)
    design = np.column_stack([np.ones_like(lagged), lagged])
    coef, *_ = np.linalg.lstsq(design, delta, rcond=None)
    b = float(coef[1])
    if b >= 0 or b <= -1:
        return float("nan")
    return float(-np.log(2.0) / np.log1p(b))


__all__ = ["SpreadModel", "compute_spread", "compute_zscore", "fit_spread_model", "half_life"]
