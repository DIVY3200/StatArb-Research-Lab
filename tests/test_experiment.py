"""The pre-registered v2 experiment machinery: pooling, repricing, pooled classification, decisions."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from statarb.config import ResearchConfig
from statarb.experiment import (
    CANDIDATES,
    cost_sensitivity,
    evaluate_decision,
    pooled_classification,
    pooled_returns,
    run_pair,
)
from statarb.ml_evaluation import reprice_walk_forward
from statarb.ml_strategy import VARIANT_LABELS

V2 = {
    "ml": {"rf_n_estimators": 30, "logistic_c_grid": [0.1, 1.0], "rf_max_depth_grid": [3],
           "rf_min_samples_leaf_grid": [10], "walk_forward_tuning": "nested", "class_weight": "none",
           "threshold_mode": "prevalence", "tuning_metric": "log_loss"},
    "walk_forward": {"train_window": 400, "test_window": 150},
}


@pytest.fixture(scope="module")
def pair_run(coint_assets):
    return run_pair(*coint_assets, ResearchConfig().with_overrides(V2))


def test_pooled_returns_union_calendar():
    d1 = {"traditional": pd.DataFrame({"net_return": [0.01, 0.02]}, index=pd.to_datetime(["2024-01-02", "2024-01-03"]))}
    d2 = {"traditional": pd.DataFrame({"net_return": [0.04, -0.02]}, index=pd.to_datetime(["2024-01-03", "2024-01-04"]))}
    pooled = pooled_returns([d1, d2], variants=["traditional"])
    np.testing.assert_allclose(pooled["traditional"].to_numpy(), [0.005, 0.03, -0.01])


def test_reprice_reproduces_walk_forward_exactly(pair_run):
    cfg = ResearchConfig().with_overrides(V2)
    again = reprice_walk_forward(pair_run.pair.prices, cfg, pair_run.wf)
    for variant, daily in pair_run.wf.daily.items():
        pd.testing.assert_frame_equal(again[variant], daily)


def test_cost_sensitivity_monotone(pair_run):
    cfg = ResearchConfig().with_overrides(V2)
    table = cost_sensitivity([pair_run], cfg, [0.0, 10.0, 40.0])
    for _, g in table.groupby("variant"):
        values = g.sort_values("cost_bps_per_leg_per_side")["cumulative_return"].to_numpy()
        assert np.all(np.diff(values) <= 1e-12)  # same signals, higher costs never help


def test_run_pair_requires_nested_tuning(coint_assets):
    with pytest.raises(ValueError, match="nested"):
        run_pair(*coint_assets, ResearchConfig())


def test_pooled_classification_against_prevalence_baseline():
    y = np.array([1, 0, 0, 0] * 25)
    pred = pd.DataFrame({"window": np.repeat([1, 2], 50), "model": "logistic_regression", "target": y,
                         "probability": np.where(y == 1, 0.6, 0.15), "fit_prevalence": 0.25},
                        index=pd.bdate_range("2023-01-02", periods=100))
    runs = [SimpleNamespace(label="X/Y", wf=SimpleNamespace(predictions=pred))]
    out = pooled_classification(runs, "logistic_regression", horizon=5)
    brier = np.mean((pred["probability"] - y) ** 2)
    brier_ref = np.mean((0.25 - y) ** 2)
    assert out["brier_skill_score"] == pytest.approx(1 - brier / brier_ref)
    assert out["roc_auc"] == pytest.approx(1.0)
    assert out["n_rows"] == 100 and 0 < out["effective_sample_size"] <= 100
    assert pooled_classification(runs, "random_forest", horizon=5) == {"n_rows": 0}


def _summary(sharpe_a, a_ci, diffs, auc=0.6, bss=0.05):
    sharpe = {"traditional": {"point": sharpe_a, "lower": a_ci[0], "upper": a_ci[1]}}
    diff = {v: {"ci_bonferroni": {"point": d[0], "lower": d[1], "upper": d[2]}} for v, d in diffs.items()}
    cls = {m: {"roc_auc": auc, "brier_skill_score": bss} for m in ("logistic_regression", "random_forest")}
    return {"bootstrap": {"sharpe": sharpe, "difference_vs_traditional": diff}, "classification": cls}


def _costs(winner=None):
    rows = [{"cost_bps_per_leg_per_side": 20.0, "variant": VARIANT_LABELS["traditional"], "cumulative_return": 0.0}]
    rows += [{"cost_bps_per_leg_per_side": 20.0, "variant": VARIANT_LABELS[v],
              "cumulative_return": 0.05 if v == winner else -0.01} for v in CANDIDATES]
    return pd.DataFrame(rows)


def test_decision_rules():
    neutral = {v: (0.1, -0.2, 0.4) for v in CANDIDATES}
    dev = _summary(0.5, (-0.1, 1.0), neutral)
    # Nothing passes -> reject (A: holdout point > 0 but CI includes 0 -> inconclusive, not retained).
    out = evaluate_decision(dev, _summary(0.3, (-0.2, 0.9), neutral), _costs())
    assert out["overall"] == "reject the tested hypothesis"
    assert out["per_strategy"]["traditional"]["decision"] == "inconclusive: not retained"
    # A's holdout CI above zero and positive development Sharpe -> retain the statistical strategy.
    out = evaluate_decision(dev, _summary(1.2, (0.2, 2.0), neutral), _costs())
    assert out["overall"] == "retain the statistical strategy"
    # One candidate passes all three gates -> retain an ML enhancement.
    strong = dict(neutral, hybrid_logistic_regression=(0.6, 0.1, 1.1))
    out = evaluate_decision(dev, _summary(0.3, (-0.2, 0.9), strong), _costs("hybrid_logistic_regression"))
    assert out["per_strategy"]["hybrid_logistic_regression"]["decision"] == "retain"
    assert out["overall"] == "retain an ML enhancement"
    # Same holdout evidence, but no development classification skill -> gate (a) fails -> reject.
    weak_dev = _summary(0.5, (-0.1, 1.0), neutral, auc=0.52, bss=-0.1)
    out = evaluate_decision(weak_dev, _summary(0.3, (-0.2, 0.9), strong), _costs("hybrid_logistic_regression"))
    assert out["per_strategy"]["hybrid_logistic_regression"]["decision"] == "reject"
