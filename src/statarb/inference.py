"""Uncertainty quantification for serially dependent daily returns and overlapping labels.

* **Stationary bootstrap** (Politis and Romano, 1994): resamples blocks of geometrically
  distributed length (mean ``mean_block_length``), which preserves short-range dependence such as
  volatility clustering and multi-day positions. Paired resampling (the same indices for two
  series on the same dates) is used for differences between strategies.
* **Effective sample size** of an autocorrelated series: ``n / (1 + 2 * sum of positive
  autocorrelations up to max_lag)``, a conservative approximation for overlapping labels.

Percentile intervals are reported. With few trades or short samples they are wide, which is the
point: a strategy difference is only evidence if its interval excludes zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .metrics import sharpe_ratio


def stationary_bootstrap_indices(n: int, mean_block_length: float, rng: np.random.Generator) -> np.ndarray:
    """One stationary-bootstrap resample of positions ``0..n-1`` (wrapping around the end)."""
    if n < 1:
        raise ValueError("n must be positive")
    if mean_block_length < 1:
        raise ValueError("mean_block_length must be at least 1")
    p_new = 1.0 / mean_block_length
    starts = rng.integers(0, n, size=n)
    new_block = rng.random(n) < p_new
    new_block[0] = True
    idx = np.empty(n, dtype=np.int64)
    current = 0
    for t in range(n):
        current = starts[t] if new_block[t] else (current + 1) % n
        idx[t] = current
    return idx


@dataclass(frozen=True)
class BootstrapInterval:
    point: float
    lower: float
    upper: float
    confidence: float
    samples: int
    share_positive: float

    def excludes_zero(self) -> bool:
        return bool(np.isfinite(self.lower) and (self.lower > 0 or self.upper < 0))

    def to_dict(self) -> dict[str, float]:
        return {"point": self.point, "lower": self.lower, "upper": self.upper, "confidence": self.confidence,
                "samples": self.samples, "share_positive": self.share_positive}


def _interval(point: float, draws: np.ndarray, confidence: float) -> BootstrapInterval:
    finite = draws[np.isfinite(draws)]
    if finite.size == 0 or not np.isfinite(point):
        return BootstrapInterval(point, np.nan, np.nan, confidence, int(finite.size), np.nan)
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(finite, [alpha, 1.0 - alpha])
    return BootstrapInterval(float(point), float(lower), float(upper), confidence, int(finite.size),
                             float(np.mean(finite > 0)))


def bootstrap_sharpe(returns: pd.Series, samples: int = 5000, mean_block_length: float = 20.0,
                     confidence: float = 0.95, seed: int = 0, periods_per_year: int = 252) -> BootstrapInterval:
    """Stationary-bootstrap percentile interval for the annualised Sharpe ratio (rf = 0)."""
    r = pd.Series(returns, dtype=float).dropna().to_numpy()
    point = sharpe_ratio(pd.Series(r), 0.0, periods_per_year)
    rng = np.random.default_rng(seed)
    draws = np.empty(samples)
    for b in range(samples):
        draws[b] = _sharpe(r[stationary_bootstrap_indices(len(r), mean_block_length, rng)], periods_per_year)
    return _interval(point, draws, confidence)


def bootstrap_sharpe_difference(returns_a: pd.Series, returns_b: pd.Series, samples: int = 5000,
                                mean_block_length: float = 20.0, confidence: float = 0.95, seed: int = 0,
                                periods_per_year: int = 252) -> BootstrapInterval:
    """Paired interval for Sharpe(a) - Sharpe(b) on identical dates (same resampled indices for both)."""
    a = pd.Series(returns_a, dtype=float)
    b = pd.Series(returns_b, dtype=float)
    if not a.index.equals(b.index):
        raise ValueError("paired bootstrap requires both return series on identical dates")
    ok = (a.notna() & b.notna()).to_numpy()
    x, y = a.to_numpy()[ok], b.to_numpy()[ok]
    point = _sharpe(x, periods_per_year) - _sharpe(y, periods_per_year)
    rng = np.random.default_rng(seed)
    draws = np.empty(samples)
    for k in range(samples):
        idx = stationary_bootstrap_indices(len(x), mean_block_length, rng)
        draws[k] = _sharpe(x[idx], periods_per_year) - _sharpe(y[idx], periods_per_year)
    return _interval(point, draws, confidence)


def _sharpe(values: np.ndarray, periods_per_year: int) -> float:
    if len(values) < 2:
        return np.nan
    sd = values.std(ddof=1)
    return np.nan if not np.isfinite(sd) or sd < 1e-12 else float(values.mean() / sd * np.sqrt(periods_per_year))


def effective_sample_size(values, max_lag: int = 10) -> float:
    """``n / (1 + 2 * sum_{k=1..max_lag} max(rho_k, 0))`` for an ordered series (NaN if n < 3)."""
    x = pd.Series(values, dtype=float).dropna().to_numpy()
    n = len(x)
    if n < 3:
        return float("nan")
    x = x - x.mean()
    denom = float(np.dot(x, x))
    if denom <= 0:
        return float(n)
    rho = [float(np.dot(x[:-k], x[k:]) / denom) for k in range(1, min(max_lag, n - 1) + 1)]
    return float(n / (1.0 + 2.0 * sum(r for r in rho if r > 0)))


__all__ = [
    "BootstrapInterval",
    "bootstrap_sharpe",
    "bootstrap_sharpe_difference",
    "effective_sample_size",
    "stationary_bootstrap_indices",
]
