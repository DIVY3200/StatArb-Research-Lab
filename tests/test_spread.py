import numpy as np
import pandas as pd
import pytest

from statarb.spread import compute_spread, compute_zscore, fit_spread_model, half_life


def _s(values):
    return pd.Series(values, index=pd.bdate_range("2021-01-01", periods=len(values)), dtype=float)


def test_spread_hand_calculated():
    log_a, log_b = _s([1.0, 1.2, 1.1]), _s([0.5, 0.6, 0.4])
    spread = compute_spread(log_a, log_b, intercept=0.2, hedge_ratio=1.5)
    # 1.0-0.2-0.75, 1.2-0.2-0.9, 1.1-0.2-0.6
    np.testing.assert_allclose(spread.to_numpy(), [0.05, 0.10, 0.30])


def test_zscore_hand_calculated_with_external_parameters():
    z = compute_zscore(_s([0.05, 0.10, 0.30]), mean=0.10, std=0.05)
    np.testing.assert_allclose(z.to_numpy(), [-1.0, 0.0, 4.0])


def test_degenerate_std_is_rejected():
    with pytest.raises(ValueError, match="below the minimum"):
        compute_zscore(_s([0.1, 0.2]), mean=0.0, std=0.0)
    with pytest.raises(ValueError, match="below the minimum"):
        compute_zscore(_s([0.1, 0.2]), mean=0.0, std=float("nan"))
    log_b = _s(np.log(np.linspace(10, 20, 100)))
    with pytest.raises(ValueError, match="degenerate"):
        fit_spread_model(0.1 + 0.9 * log_b, log_b)


def test_fit_uses_training_moments_and_evaluation_reuses_them():
    rng = np.random.default_rng(1)
    log_b = _s(3 + np.cumsum(0.01 * rng.standard_normal(400)))
    log_a = 0.5 + 0.9 * log_b + 0.02 * rng.standard_normal(400)
    train_a, train_b = log_a.iloc[:300], log_b.iloc[:300]
    model = fit_spread_model(train_a, train_b)
    train_spread = train_a - model.intercept - model.hedge_ratio * train_b
    assert model.mean == pytest.approx(train_spread.mean())
    assert model.std == pytest.approx(train_spread.std(ddof=1))
    assert model.mean == pytest.approx(0.0, abs=1e-12)  # OLS with intercept

    # Evaluation z-scores use the frozen training mean/std, not the evaluation sample's own.
    eval_a, eval_b = log_a.iloc[300:] + 0.05, log_b.iloc[300:]  # shift the evaluation spread upward
    z = model.zscore(eval_a, eval_b)
    expected = (eval_a - model.intercept - model.hedge_ratio * eval_b - model.mean) / model.std
    pd.testing.assert_series_equal(z, expected, check_names=False)
    assert z.mean() > 1.5  # a self-normalised z-score would hide the shift (mean ~ 0)


def test_half_life_of_ar1_process():
    rng = np.random.default_rng(2)
    phi, n = np.exp(-np.log(2) / 10), 20000
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.standard_normal()
    assert half_life(_s(x)) == pytest.approx(10, rel=0.15)
    assert np.isnan(half_life(_s(np.cumsum(np.ones(50)))))  # trending: no mean reversion
    assert np.isnan(half_life(_s([1.0, 2.0])))
