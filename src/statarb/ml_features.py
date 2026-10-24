"""Leakage-safe ML dataset construction: reversion target, features and purged sample masks.

Notation
--------
``z_t`` is the spread z-score at the close of bar ``t``, computed with spread parameters
(intercept, hedge ratio, mean, std) that were frozen on a fitting window (see
:mod:`statarb.spread`). ``H`` is the prediction horizon and ``tau`` the tolerance.

Target (mean reversion over the next H sessions)
------------------------------------------------
::

    y_t = 1   if  |z_{t+H}| <= |z_t| - tau
    y_t = 0   otherwise
    y_t undefined (NaN) when bar t+H is not available

"Moves toward equilibrium" means the distance from the training-estimated equilibrium,
measured in training standard deviations, falls by at least ``tau``. The label is the
*only* object that looks into the future; every feature uses data up to and including ``t``.

Only *eligible* bars with ``|z_t| >= min_abs_z`` enter the training and evaluation sets.
Near equilibrium the target is trivially 0 (``|z_t| < tau`` makes a fall of ``tau``
impossible), and those bars would inflate accuracy without describing tradable situations.

Purging
-------
A label at ``t`` uses ``z_{t+H}``. A row may be used to fit or evaluate a model on a
window ``[start, end)`` only if ``t + H <= end - 1 - embargo``, so that no label reaches
into the following window. This stops validation or test prices from leaking into training
labels, and stops test prices from leaking into validation scores.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .spread import SpreadModel

FEATURE_DESCRIPTIONS: dict[str, str] = {
    "z": "Current spread z-score (training mean and std).",
    "abs_z": "Distance from the estimated equilibrium, |z_t|.",
    "z_lag_k": "Spread z-score k bars earlier.",
    "dz_k": "Change in z-score over k bars: z_t - z_{t-k}.",
    "reversion_speed_k": "Movement toward equilibrium over k bars: -sign(z_t) * (z_t - z_{t-k}).",
    "momentum_m": "Average z-score slope over m bars: (z_t - z_{t-m}) / m.",
    "spread_vol_ratio_w": "Trailing w-bar std of daily spread changes / the same std on the fitting window.",
    "ret_a_w": "Log return of asset A over the last w bars.",
    "ret_b_w": "Log return of asset B over the last w bars.",
    "corr_w": "Trailing w-bar correlation of the two assets' daily log returns.",
    "bars_beyond_entry": "Consecutive bars up to t with |z| >= the entry threshold (0 if inside).",
}


@dataclass(frozen=True)
class FeatureSpec:
    z_lags: tuple[int, ...] = (1, 2, 5)
    momentum_window: int = 10
    volatility_window: int = 20
    correlation_window: int = 20
    return_windows: tuple[int, ...] = (1, 5)
    entry_z: float = 2.0

    @classmethod
    def from_config(cls, config) -> "FeatureSpec":
        ml = config.ml
        return cls(ml.z_lags, ml.momentum_window, ml.volatility_window, ml.correlation_window,
                   ml.return_windows, config.strategy.entry_z)

    @property
    def columns(self) -> list[str]:
        cols = ["z", "abs_z"] + [f"z_lag_{k}" for k in self.z_lags]
        cols += ["dz_1", "dz_5", "reversion_speed_1", "reversion_speed_5", f"momentum_{self.momentum_window}",
                 f"spread_vol_ratio_{self.volatility_window}"]
        for w in self.return_windows:
            cols += [f"ret_a_{w}", f"ret_b_{w}"]
        return cols + [f"corr_{self.correlation_window}", "bars_beyond_entry"]


def training_spread_change_std(log_prices_fit: pd.DataFrame, model: SpreadModel) -> float:
    """Std (ddof=1) of daily spread changes on the fitting window: the volatility-feature scale."""
    spread = model.spread(log_prices_fit["A"], log_prices_fit["B"])
    value = float(spread.diff().std(ddof=1))
    if not np.isfinite(value) or value <= 0:
        raise ValueError("Spread changes on the fitting window have zero variance; volatility feature undefined.")
    return value


def consecutive_true(mask: pd.Series) -> pd.Series:
    """Length of the current run of True values ending at each bar (0 where False)."""
    values = mask.to_numpy(dtype=bool)
    counts = np.zeros(len(values), dtype=float)
    run = 0
    for i, flag in enumerate(values):
        run = run + 1 if flag else 0
        counts[i] = run
    return pd.Series(counts, index=mask.index)


def build_features(prices: pd.DataFrame, model: SpreadModel, spec: FeatureSpec, train_dspread_std: float) -> pd.DataFrame:
    """Feature matrix in which row ``t`` uses only prices up to and including bar ``t``.

    Rolling and lagged features are trailing (``rolling`` windows end at ``t``; ``shift``
    looks backwards). Rows whose look-back window is incomplete contain NaN and are excluded
    downstream rather than filled. The only training-dependent quantities are the frozen
    spread parameters in ``model`` and ``train_dspread_std``, both estimated on the fitting window.
    """
    logp = np.log(prices[["A", "B"]])
    spread = model.spread(logp["A"], logp["B"])
    z = (spread - model.mean) / model.std
    sign = np.sign(z)
    f = pd.DataFrame(index=prices.index)
    f["z"] = z
    f["abs_z"] = z.abs()
    for k in spec.z_lags:
        f[f"z_lag_{k}"] = z.shift(k)
    f["dz_1"] = z - z.shift(1)
    f["dz_5"] = z - z.shift(5)
    f["reversion_speed_1"] = -sign * f["dz_1"]
    f["reversion_speed_5"] = -sign * f["dz_5"]
    m = spec.momentum_window
    f[f"momentum_{m}"] = (z - z.shift(m)) / m
    v = spec.volatility_window
    f[f"spread_vol_ratio_{v}"] = spread.diff().rolling(v, min_periods=v).std(ddof=1) / train_dspread_std
    for w in spec.return_windows:
        f[f"ret_a_{w}"] = logp["A"] - logp["A"].shift(w)
        f[f"ret_b_{w}"] = logp["B"] - logp["B"].shift(w)
    daily = logp.diff()
    c = spec.correlation_window
    f[f"corr_{c}"] = daily["A"].rolling(c, min_periods=c).corr(daily["B"])
    f["bars_beyond_entry"] = consecutive_true(z.abs() >= spec.entry_z)
    return f[spec.columns]


def reversion_target(z: pd.Series, horizon: int, tolerance: float) -> pd.Series:
    """``y_t = 1{|z_{t+H}| <= |z_t| - tolerance}``; NaN where ``z_t`` or ``z_{t+H}`` is unavailable."""
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    future = z.shift(-horizon)
    y = (future.abs() <= z.abs() - tolerance).astype(float)
    y[future.isna() | z.isna()] = np.nan
    y.name = "target"
    return y


def eligible(z: pd.Series, min_abs_z: float) -> pd.Series:
    return (z.abs() >= min_abs_z) & z.notna()


def purged_window_mask(n_obs: int, start: int, end: int, horizon: int, embargo: int = 0) -> np.ndarray:
    """Positions in ``[start, end)`` whose label horizon ends at least ``embargo`` bars before ``end``."""
    if not 0 <= start <= end <= n_obs:
        raise ValueError("window bounds must satisfy 0 <= start <= end <= n_obs")
    pos = np.arange(n_obs)
    return (pos >= start) & (pos < end) & (pos + horizon <= end - 1 - embargo)


@dataclass
class MLDataset:
    """Features, target and eligibility on one contiguous block of bars, with a frozen spread model."""

    features: pd.DataFrame
    target: pd.Series
    eligible: pd.Series
    spec: FeatureSpec
    horizon: int
    tolerance: float

    @property
    def z(self) -> pd.Series:
        return self.features["z"]

    def rows(self, start: int, end: int, embargo: int = 0) -> np.ndarray:
        """Usable rows in ``[start, end)``: eligible, complete features, label inside the window (purged)."""
        mask = purged_window_mask(len(self.features), start, end, self.horizon, embargo)
        mask &= self.eligible.to_numpy(dtype=bool)
        mask &= self.target.notna().to_numpy()
        mask &= self.features.notna().all(axis=1).to_numpy()
        return mask

    def xy(self, mask: np.ndarray) -> tuple[pd.DataFrame, pd.Series]:
        return self.features.loc[mask], self.target.loc[mask].astype(int)


def build_dataset(prices: pd.DataFrame, model: SpreadModel, spec: FeatureSpec, fit_slice: slice,
                  horizon: int, tolerance: float, min_abs_z: float) -> MLDataset:
    """Build an :class:`MLDataset` over ``prices``; ``fit_slice`` locates the model's fitting window."""
    log_fit = np.log(prices.iloc[fit_slice][["A", "B"]])
    features = build_features(prices, model, spec, training_spread_change_std(log_fit, model))
    z = features["z"]
    return MLDataset(features, reversion_target(z, horizon, tolerance), eligible(z, min_abs_z), spec, horizon, tolerance)


__all__ = [
    "FEATURE_DESCRIPTIONS",
    "FeatureSpec",
    "MLDataset",
    "build_dataset",
    "build_features",
    "consecutive_true",
    "eligible",
    "purged_window_mask",
    "reversion_target",
    "training_spread_change_std",
]
