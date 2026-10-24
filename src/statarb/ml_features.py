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

Point-in-time training rows
---------------------------
By default (``ml.training_features: in_sample``) training rows use the spread parameters fitted on
the whole fitting window, so a training bar's z-score depends on prices after it, while every
prediction uses parameters frozen before the bar. :func:`point_in_time_rows` removes that mismatch:
each block of bars gets parameters estimated only on the bars before the block, and both its
features and its labels are computed with them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .hedge_ratio import estimate_hedge_ratio
from .spread import SpreadModel, fit_spread_model

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


@dataclass
class PointInTimeRows:
    """Training rows whose features and labels use spread parameters estimated strictly before them."""

    features: pd.DataFrame
    target: pd.Series
    positions: np.ndarray  # integer positions in the price frame that was passed in
    blocks: pd.DataFrame   # per block: dates, estimation window, frozen parameters, rows kept
    horizon: int

    def __len__(self) -> int:
        return len(self.target)

    def purged(self, end: int, embargo: int = 0) -> np.ndarray:
        """Rows whose label horizon ends at least ``embargo`` bars before position ``end``."""
        return self.positions + self.horizon <= end - 1 - embargo

    def xy(self, mask: np.ndarray | None = None) -> tuple[pd.DataFrame, pd.Series]:
        if mask is None:
            return self.features, self.target.astype(int)
        return self.features.loc[mask], self.target.loc[mask].astype(int)


def _day(value) -> str:
    return str(value.date()) if hasattr(value, "date") else str(value)


def point_in_time_rows(prices: pd.DataFrame, fit_start: int, fit_end: int, config, spec: FeatureSpec,
                       history_start: int = 0, embargo: int = 0) -> PointInTimeRows:
    """Point-in-time training rows on ``[fit_start, fit_end)`` (``ml.training_features: point_in_time``).

    The window is cut into blocks of ``ml.pit_block`` bars. For a block starting at bar ``b`` the hedge
    ratio and spread mean / std are estimated on the ``L`` bars ``[b - L, b)`` only
    (``L = ml.pit_estimation_window``, default ``walk_forward.train_window``), with the same estimator as
    the deployed model; the block's features *and* labels are computed with those frozen parameters. A
    training row is therefore built exactly like a live prediction. Blocks need ``L`` bars of history
    at or after ``history_start``, so the first ``L`` bars after it yield no rows. No price at or after
    ``fit_end`` is read, and rows are purged so that ``t + H <= fit_end - 1 - embargo``.
    """
    ml = config.ml
    L = ml.pit_estimation_window or config.walk_forward.train_window
    H = ml.horizon
    if not 0 <= history_start <= fit_start < fit_end <= len(prices):
        raise ValueError("point_in_time_rows needs 0 <= history_start <= fit_start < fit_end <= len(prices)")
    logp = np.log(prices[["A", "B"]])
    last = fit_end - H - embargo  # rows t < last keep their label inside the window
    frames, targets, positions, blocks = [], [], [], []
    for b0 in range(max(fit_start, history_start + L), min(fit_end, last), ml.pit_block):
        b1 = min(b0 + ml.pit_block, fit_end)
        est = logp.iloc[b0 - L:b0]
        record = {"block_start": _day(prices.index[b0]), "block_end": _day(prices.index[b1 - 1]),
                  "estimation_start": _day(est.index[0]), "estimation_end": _day(est.index[-1])}
        try:
            hedge = estimate_hedge_ratio(est["A"], est["B"], method=config.hedge_ratio.method)
            model = fit_spread_model(est["A"], est["B"], min_std=config.spread.min_std, hedge=hedge)
            features = build_features(prices.iloc[b0 - L:min(b1 + H, fit_end)], model, spec,
                                      training_spread_change_std(est, model))
        except ValueError as exc:
            blocks.append({**record, "eligible_bars": 0, "rows": 0, "note": str(exc)})
            continue
        z = features["z"]
        target = reversion_target(z, H, ml.tolerance_z)
        pos = np.arange(len(features)) + (b0 - L)
        in_block = (pos >= b0) & (pos < b1)
        elig = eligible(z, ml.min_abs_z).to_numpy(dtype=bool) & in_block
        keep = elig & (target.notna() & features.notna().all(axis=1)).to_numpy(dtype=bool) & (pos < last)
        frames.append(features.loc[keep])
        targets.append(target.loc[keep])
        positions.append(pos[keep])
        blocks.append({**record, "hedge_ratio": model.hedge_ratio, "intercept": model.intercept,
                       "spread_mean": model.mean, "spread_std": model.std, "eligible_bars": int(elig.sum()),
                       "rows": int(keep.sum()),
                       "positive_rate": float(target.loc[keep].mean()) if keep.any() else np.nan, "note": ""})
    if frames:
        X, y, pos = pd.concat(frames), pd.concat(targets), np.concatenate(positions)
    else:
        X = pd.DataFrame(columns=spec.columns, dtype=float)
        y, pos = pd.Series(dtype=float, name="target"), np.array([], dtype=int)
    return PointInTimeRows(X, y, pos.astype(int), pd.DataFrame(blocks), H)


__all__ = [
    "FEATURE_DESCRIPTIONS",
    "FeatureSpec",
    "MLDataset",
    "PointInTimeRows",
    "build_dataset",
    "build_features",
    "consecutive_true",
    "eligible",
    "point_in_time_rows",
    "purged_window_mask",
    "reversion_target",
    "training_spread_change_std",
]
