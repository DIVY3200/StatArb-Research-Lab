"""Target alignment, feature causality (no look-ahead) and purge masks."""

import numpy as np
import pandas as pd
import pytest

from statarb.config import ResearchConfig
from statarb.evaluation import chronological_split, fit_pair_model
from statarb.ml_features import (
    FeatureSpec,
    build_dataset,
    build_features,
    consecutive_true,
    eligible,
    purged_window_mask,
    reversion_target,
    training_spread_change_std,
)


def _z(values):
    return pd.Series(values, index=pd.bdate_range("2022-01-03", periods=len(values)), dtype=float)


# --------------------------------------------------------------------------- target
def test_target_hand_calculated():
    z = _z([2.0, 1.5, -1.0, 0.3, 2.5, 2.4])
    y = reversion_target(z, horizon=2, tolerance=0.25)
    # t=0: |z_2|=1.0 <= 2.0-0.25 -> 1 ; t=1: |z_3|=0.3 <= 1.25 -> 1 ; t=2: |z_4|=2.5 <= 0.75 -> 0
    # t=3: |z_5|=2.4 <= 0.05 -> 0 ; t=4, t=5: z_{t+2} unavailable -> NaN
    np.testing.assert_array_equal(y.to_numpy()[:4], [1.0, 1.0, 0.0, 0.0])
    assert y.iloc[4:].isna().all()


def test_target_tolerance_and_overshoot():
    z = _z([2.0, 0.0, -1.9])
    assert reversion_target(z, 2, 0.25).iloc[0] == 0.0  # overshoot to -1.9 is only 0.1 closer
    assert reversion_target(z, 2, 0.05).iloc[0] == 1.0
    assert reversion_target(z, 1, 0.25).iloc[0] == 1.0
    near = _z([0.1, 0.0, 0.0])
    assert reversion_target(near, 1, 0.25).iloc[0] == 0.0  # |z| < tolerance can never "revert"


def test_target_uses_exactly_the_horizon_bar():
    z = _z([2.0, 0.0, 0.0, 3.0])
    assert reversion_target(z, 3, 0.25).iloc[0] == 0.0  # only z_{t+3} matters, not the path


def test_target_invalid_arguments():
    with pytest.raises(ValueError):
        reversion_target(_z([1.0, 2.0]), 0, 0.1)
    with pytest.raises(ValueError):
        reversion_target(_z([1.0, 2.0]), 1, -0.1)


def test_eligibility_mask():
    z = _z([0.5, -1.2, np.nan, 1.0])
    assert eligible(z, 1.0).tolist() == [False, True, False, True]


# --------------------------------------------------------------------------- features
@pytest.fixture(scope="module")
def fitted(coint_pair):
    prices = coint_pair.prices
    splits = chronological_split(prices.index, 0.6, 0.2, 0.2)
    model = fit_pair_model(np.log(prices.iloc[splits["train"].slice]), ResearchConfig())
    return prices, splits, model.spread


def test_feature_values_by_hand(fitted):
    prices, splits, model = fitted
    spec = FeatureSpec(z_lags=(1, 2, 5), momentum_window=10, volatility_window=20, correlation_window=20,
                       return_windows=(1, 5), entry_z=2.0)
    scale = training_spread_change_std(np.log(prices.iloc[splits["train"].slice]), model)
    f = build_features(prices, model, spec, scale)
    logp = np.log(prices)
    z = model.zscore(logp["A"], logp["B"])
    t = 300
    assert f["z"].iloc[t] == pytest.approx(z.iloc[t])
    assert f["z_lag_5"].iloc[t] == pytest.approx(z.iloc[t - 5])
    assert f["dz_1"].iloc[t] == pytest.approx(z.iloc[t] - z.iloc[t - 1])
    assert f["reversion_speed_5"].iloc[t] == pytest.approx(-np.sign(z.iloc[t]) * (z.iloc[t] - z.iloc[t - 5]))
    assert f["momentum_10"].iloc[t] == pytest.approx((z.iloc[t] - z.iloc[t - 10]) / 10)
    assert f["ret_a_5"].iloc[t] == pytest.approx(logp["A"].iloc[t] - logp["A"].iloc[t - 5])
    spread = model.spread(logp["A"], logp["B"])
    expected_vol = spread.diff().iloc[t - 19:t + 1].std(ddof=1) / scale
    assert f["spread_vol_ratio_20"].iloc[t] == pytest.approx(expected_vol)
    daily = logp.diff()
    assert f["corr_20"].iloc[t] == pytest.approx(daily["A"].iloc[t - 19:t + 1].corr(daily["B"].iloc[t - 19:t + 1]))
    assert list(f.columns) == spec.columns
    assert f.iloc[:20].isna().any(axis=1).all()  # incomplete look-back windows stay NaN (no filling)


def test_features_are_causal_truncation_invariant(fitted):
    """Row t must be identical whether or not any later price exists or changes."""
    prices, splits, model = fitted
    spec = FeatureSpec()
    scale = training_spread_change_std(np.log(prices.iloc[splits["train"].slice]), model)
    full = build_features(prices, model, spec, scale)
    cut = 1100
    truncated = build_features(prices.iloc[:cut], model, spec, scale)
    pd.testing.assert_frame_equal(full.iloc[:cut], truncated)
    shocked = prices.copy()
    shocked.iloc[cut:] *= 1.3
    pd.testing.assert_frame_equal(full.iloc[:cut], build_features(shocked, model, spec, scale).iloc[:cut])


def test_volatility_scale_comes_from_fitting_window_only(fitted):
    prices, splits, model = fitted
    train = splits["train"].slice
    base = training_spread_change_std(np.log(prices.iloc[train]), model)
    shocked = prices.copy()
    shocked.iloc[splits["train"].end_idx:] *= np.linspace(1, 2, len(prices) - splits["train"].end_idx)[:, None]
    assert training_spread_change_std(np.log(shocked.iloc[train]), model) == base


def test_consecutive_beyond_threshold_counter():
    mask = pd.Series([False, True, True, False, True, True, True])
    assert consecutive_true(mask).tolist() == [0, 1, 2, 0, 1, 2, 3]


# --------------------------------------------------------------------------- purging
def test_purged_window_mask():
    mask = purged_window_mask(20, start=5, end=15, horizon=3)
    assert np.flatnonzero(mask).tolist() == list(range(5, 12))  # t + 3 <= 14
    embargoed = purged_window_mask(20, start=5, end=15, horizon=3, embargo=2)
    assert np.flatnonzero(embargoed).tolist() == list(range(5, 10))
    with pytest.raises(ValueError):
        purged_window_mask(10, 5, 12, 1)


def test_dataset_rows_never_use_labels_beyond_window(fitted):
    prices, splits, model = fitted
    train = splits["train"]
    ds = build_dataset(prices, model, FeatureSpec(), train.slice, horizon=5, tolerance=0.25, min_abs_z=1.0)
    rows = np.flatnonzero(ds.rows(train.start_idx, train.end_idx))
    assert rows.max() + 5 <= train.end_idx - 1
    assert ds.eligible.iloc[rows].all() and ds.features.iloc[rows].notna().all().all()
    # Changing prices after the training window leaves every training row (features and labels) unchanged.
    shocked = prices.copy()
    shocked.iloc[train.end_idx:] *= np.array([1.4, 0.7])
    ds2 = build_dataset(shocked, model, FeatureSpec(), train.slice, 5, 0.25, 1.0)
    rows2 = ds2.rows(train.start_idx, train.end_idx)
    np.testing.assert_array_equal(ds.rows(train.start_idx, train.end_idx), rows2)
    X1, y1 = ds.xy(ds.rows(train.start_idx, train.end_idx))
    X2, y2 = ds2.xy(rows2)
    pd.testing.assert_frame_equal(X1, X2)
    pd.testing.assert_series_equal(y1, y2)
    # Without the purge, the last training labels would have changed.
    tail = slice(train.end_idx - 5, train.end_idx)
    assert not ds.target.iloc[tail].equals(ds2.target.iloc[tail])
