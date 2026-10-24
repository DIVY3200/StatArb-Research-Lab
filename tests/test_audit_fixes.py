"""Regression tests for the Phase 1 audit findings (docs/audit/PHASE1_AUDIT.md).

D1 nested walk-forward tuning, D2 special sessions, D3 threshold-selection metric, D4 exports,
M1 holdout guard, M3 probability quality (class weights, prevalence baseline), M7/M12 inference
and Sortino, and the walk-forward anchor used for holdout evaluation.
"""

import numpy as np
import pandas as pd
import pytest

from conftest import make_asset, make_frame
from statarb.config import ResearchConfig
from statarb.data_validation import detect_special_sessions, prepare_pair, prepare_pair_from_config
from statarb.evaluation import run_pair_research, walk_forward_schedule
from statarb.inference import (
    bootstrap_sharpe,
    bootstrap_sharpe_difference,
    effective_sample_size,
    stationary_bootstrap_indices,
)
from statarb.metrics import sortino_ratio
from statarb.ml_evaluation import MODEL_NAMES, ml_result_tables, run_ml_research, run_ml_walk_forward, select_threshold
from statarb.ml_models import classification_metrics, fit_model
from statarb.protocol import HoldoutAccessError, allowed_end_date, code_fingerprint
from statarb.signals import SignalRules

FAST_ML = {"rf_n_estimators": 30, "logistic_c_grid": [0.1, 1.0], "rf_max_depth_grid": [3],
           "rf_min_samples_leaf_grid": [10]}


def _v2_config(**ml):
    return ResearchConfig().with_overrides({
        "ml": {**FAST_ML, "walk_forward_tuning": "nested", "class_weight": "none", "threshold_mode": "prevalence",
               "tuning_metric": "log_loss", **ml},
        "walk_forward": {"train_window": 400, "test_window": 150},
    })


# --------------------------------------------------------------------------- D1: nested walk-forward
@pytest.fixture(scope="module")
def nested_run(coint_pair):
    cfg = _v2_config()
    return cfg, run_ml_walk_forward(coint_pair.prices, cfg, SignalRules(), None, None)


def test_nested_walk_forward_is_point_in_time(coint_pair, nested_run):
    """Changing every price after window k's trading window must not change windows 1..k at all."""
    cfg, base = nested_run
    k = 2
    cut = coint_pair.prices.index.get_loc(pd.Timestamp(base.windows.loc[k - 1, "eval_end"])) + 1
    shocked = coint_pair.prices.copy()
    shocked.iloc[cut:] = shocked.iloc[cut:] * np.array([1.35, 0.8])
    other = run_ml_walk_forward(shocked, cfg, SignalRules(), None, None)
    cols = [c for c in base.windows.columns if c.endswith(("_params", "_eval_roc_auc", "_threshold_hybrid"))
            or c.startswith(("return_", "trades_"))]
    pd.testing.assert_frame_equal(base.windows.iloc[:k][cols], other.windows.iloc[:k][cols])
    for variant, daily in base.daily.items():
        pd.testing.assert_frame_equal(daily.iloc[:cut - 400], other.daily[variant].iloc[:cut - 400])


def test_nested_mode_records_per_window_choices_and_prevalence_thresholds(nested_run):
    cfg, result = nested_run
    traded = result.windows[result.windows["traded"]]
    assert not traded.empty
    for short in ("LR", "RF"):
        assert traded[f"{short}_params"].notna().all()
        np.testing.assert_allclose(traded[f"{short}_threshold_hybrid"],
                                   np.clip(traded["fit_prevalence"] * cfg.ml.prevalence_multiplier, 0.01, 0.99))
    assert set(result.predictions["model"]) == set(MODEL_NAMES)
    assert result.predictions["probability"].between(0, 1).all()


def test_static_mode_still_requires_static_params_and_warns(coint_pair):
    cfg = ResearchConfig().with_overrides({"ml": FAST_ML, "walk_forward": {"train_window": 400, "test_window": 150}})
    with pytest.raises(ValueError, match="static"):
        run_ml_walk_forward(coint_pair.prices, cfg, SignalRules(), None, None)
    ml = run_ml_research(run_pair_research(coint_pair, cfg))
    assert any("audit finding D1" in w for w in ml.warnings)
    params = ml.walk_forward.windows.loc[ml.walk_forward.windows["traded"], "LR_params"].unique()
    assert len(params) == 1  # the single static choice reused everywhere (the leak D1 describes)


def test_unsafe_nested_threshold_combination_rejected():
    with pytest.raises(ValueError, match="look-ahead"):
        ResearchConfig().with_overrides({"ml": {"walk_forward_tuning": "nested", "threshold_source": "validation_selected"}})


def test_schedule_anchor_for_holdout():
    windows = walk_forward_schedule(1000, 300, 100, first_eval_idx=650)
    assert windows[0].eval_start_idx == 650 and windows[0].fit_start_idx == 350 and windows[0].fit_end_idx == 650
    assert windows[-1].eval_end_idx == 1000
    assert walk_forward_schedule(1000, 300, 100) == walk_forward_schedule(1000, 300, 100, first_eval_idx=300)
    with pytest.raises(ValueError, match="fitting window"):
        walk_forward_schedule(1000, 300, 100, first_eval_idx=200)


# --------------------------------------------------------------------------- D2: special sessions
def _frames_with_special_sessions():
    dates = list(pd.bdate_range("2021-01-04", periods=300))
    dates.insert(150, pd.Timestamp("2021-08-01"))  # a Sunday session
    close = 100 + np.arange(len(dates), dtype=float) * 0.1
    a = make_frame(close, dates=dates)
    b = make_frame(close * 0.5 + 3, dates=dates)
    a.loc[a.index[200], "Volume"] = 10.0  # thin session on leg A
    return a, b


def test_special_sessions_detected_and_kept_by_default():
    a, b = _frames_with_special_sessions()
    pair = prepare_pair(make_asset("AAA", a), make_asset("BBB", b), min_observations=100,
                        special_session_dates=("2021-03-01",), special_session_volume_ratio=0.1)
    flagged = {item["date"]: item["reasons"] for item in pair.report.special_sessions}
    assert "weekend" in flagged["2021-08-01"]
    assert "listed" in flagged["2021-03-01"]
    assert "leg A volume" in flagged[str(a.index[200].date())]
    assert len(pair.prices) == 301 and pair.report.special_sessions_dropped == 0  # keep = baseline behaviour


def test_special_sessions_dropped_and_never_traded():
    a, b = _frames_with_special_sessions()
    pair = prepare_pair(make_asset("AAA", a), make_asset("BBB", b), min_observations=100, special_sessions="drop",
                        special_session_dates=("2021-03-01",), special_session_volume_ratio=0.1)
    flagged = pd.DatetimeIndex([pd.Timestamp(x["date"]) for x in pair.report.special_sessions])
    assert len(flagged) == 3 and not pair.prices.index.isin(flagged).any()
    assert pair.report.special_sessions_dropped == 3 and len(pair.prices) == 298


def test_volume_rule_is_point_in_time():
    a, b = _frames_with_special_sessions()
    dates = a.index.intersection(b.index)
    base = detect_special_sessions(dates, (a, b), volume_ratio=0.1)
    later = a.copy()
    later.iloc[250:, later.columns.get_loc("Volume")] = 1e9  # only future volumes change
    again = detect_special_sessions(dates, (later, b), volume_ratio=0.1)
    cutoff = str(a.index[249].date())
    assert [x for x in base if x["date"] <= cutoff] == [x for x in again if x["date"] <= cutoff]


def test_prepare_pair_from_config_passes_session_options(coint_assets):
    cfg = ResearchConfig().with_overrides({"data": {"special_sessions": "drop", "special_session_dates": ["2016-03-01"]}})
    pair = prepare_pair_from_config(*coint_assets, cfg.data)
    assert pd.Timestamp("2016-03-01") not in pair.prices.index
    assert pair.report.special_sessions_dropped == 1


# --------------------------------------------------------------------------- D3, D4
def test_select_threshold_supports_calmar_and_rejects_unknown():
    table = pd.DataFrame({"threshold": [0.5, 0.6], "completed_trades": [5, 5], "sharpe_ratio": [2.0, 1.0],
                          "cumulative_return": [0.1, 0.2], "calmar_ratio": [0.5, 3.0]})
    assert select_threshold(table, "calmar", 3) == 0.6
    assert select_threshold(table, "sharpe", 3) == 0.5
    with pytest.raises(ValueError, match="Unsupported"):
        select_threshold(table, "sortino", 3)


def test_walk_forward_trade_logs_and_predictions_exported(coint_pair):
    cfg = _v2_config()
    ml = run_ml_research(run_pair_research(coint_pair, cfg))
    tables = ml_result_tables(ml)
    for variant in ml.walk_forward.daily:
        assert f"ml_walk_forward_trades_{variant}" in tables
    assert "ml_walk_forward_predictions" in tables


# --------------------------------------------------------------------------- M1: holdout guard
def test_holdout_guard():
    cfg = ResearchConfig().with_overrides({"holdout": {"enforce": True, "development_end": "2024-12-31",
                                                       "holdout_start": "2025-01-01", "holdout_end": "2026-09-30"}})
    assert allowed_end_date(cfg, "2024-12-31") == "2024-12-31"
    with pytest.raises(HoldoutAccessError, match="locked"):
        allowed_end_date(cfg, "2025-03-01")
    assert allowed_end_date(cfg, "2026-09-30", final_evaluation=True) == "2026-09-30"
    with pytest.raises(HoldoutAccessError, match="beyond"):
        allowed_end_date(cfg, "2026-10-15", final_evaluation=True)
    assert allowed_end_date(ResearchConfig(), "2030-01-01") == "2030-01-01"  # not enforced by default
    with pytest.raises(ValueError, match="holdout dates"):
        ResearchConfig().with_overrides({"holdout": {"enforce": True, "development_end": "2025-06-01",
                                                     "holdout_start": "2025-01-01", "holdout_end": "2026-09-30"}})
    assert len(code_fingerprint()) == 64


# --------------------------------------------------------------------------- M3: probability quality
def test_unweighted_logistic_regression_matches_base_rate(coint_pair):
    from statarb.evaluation import chronological_split, fit_pair_model
    from statarb.ml_features import FeatureSpec, build_dataset

    prices = coint_pair.prices
    splits = chronological_split(prices.index)
    model = fit_pair_model(np.log(prices.iloc[splits["train"].slice]), ResearchConfig()).spread
    ds = build_dataset(prices, model, FeatureSpec(), splits["train"].slice, 5, 0.25, 1.0)
    X, y = ds.xy(ds.rows(0, splits["train"].end_idx))
    unweighted = fit_model("logistic_regression", {"C": 10.0}, X, y, class_weight="none")
    weighted = fit_model("logistic_regression", {"C": 10.0}, X, y, class_weight="balanced")
    assert unweighted.predict_proba(X).mean() == pytest.approx(y.mean(), abs=0.01)
    assert abs(weighted.predict_proba(X).mean() - 0.5) < abs(weighted.predict_proba(X).mean() - y.mean())


def test_prevalence_baseline_skill_scores_by_hand():
    y = pd.Series([1, 0, 0, 0] * 10)
    const = classification_metrics(y, pd.Series([0.25] * 40), min_samples=30, reference_rate=0.25)
    assert const["brier_skill_score"] == pytest.approx(0.0) and const["log_loss_skill"] == pytest.approx(0.0)
    perfect = classification_metrics(y, y.astype(float).clip(0.001, 0.999), min_samples=30, reference_rate=0.25)
    assert perfect["brier_skill_score"] > 0.99
    worse = classification_metrics(y, pd.Series([0.5] * 40), min_samples=30, reference_rate=0.25)
    assert worse["brier_skill_score"] < 0  # forecasting 0.5 is worse than the 25% base rate
    assert np.isnan(classification_metrics(y, pd.Series([0.25] * 40), min_samples=30)["brier_skill_score"])


# --------------------------------------------------------------------------- M7 / M12: inference, Sortino
def test_sortino_by_hand():
    r = pd.Series([0.01, -0.02, 0.03, -0.01])
    downside = np.sqrt(np.mean([0.0, 0.0004, 0.0, 0.0001]))
    assert sortino_ratio(r) == pytest.approx(0.0025 / downside * np.sqrt(252))
    assert np.isnan(sortino_ratio(pd.Series([0.01, 0.02, 0.0])))  # no downside: undefined, not infinite


def test_stationary_bootstrap_indices():
    rng = np.random.default_rng(0)
    idx = stationary_bootstrap_indices(1000, 20, rng)
    assert len(idx) == 1000 and idx.min() >= 0 and idx.max() < 1000
    breaks = np.sum(np.diff(idx) != 1) - np.sum(np.diff(idx) == -999)  # wrap-arounds are continuations
    assert 25 < breaks < 80  # ~1000/20 = 50 block starts
    np.testing.assert_array_equal(idx, stationary_bootstrap_indices(1000, 20, np.random.default_rng(0)))


def test_bootstrap_intervals():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0.0005, 0.01, 1500))
    same = bootstrap_sharpe_difference(r, r, samples=300, seed=3)
    assert same.point == 0 and same.lower == 0 and same.upper == 0 and not same.excludes_zero()
    ci = bootstrap_sharpe(r, samples=500, seed=3)
    assert ci.lower < ci.point < ci.upper
    strong = bootstrap_sharpe(pd.Series(rng.normal(0.003, 0.01, 1500)), samples=500, seed=3)
    assert strong.excludes_zero() and strong.lower > 0
    with pytest.raises(ValueError, match="identical dates"):
        bootstrap_sharpe_difference(r, r.iloc[1:], samples=100)


def test_effective_sample_size():
    rng = np.random.default_rng(2)
    iid = rng.standard_normal(5000)
    assert effective_sample_size(iid) == pytest.approx(5000, rel=0.15)
    ar = np.zeros(5000)
    for t in range(1, 5000):
        ar[t] = 0.5 * ar[t - 1] + rng.standard_normal()
    assert effective_sample_size(ar, max_lag=20) == pytest.approx(5000 / 3, rel=0.25)  # n(1-rho)/(1+rho)
    assert np.isnan(effective_sample_size([1.0, 2.0]))


# --------------------------------------------------------------------------- amendments A1, A2
def test_exchange_specific_session_dates_only_affect_that_exchange():
    a, b = _frames_with_special_sessions()
    by_exchange = (("NS", ("2021-03-01",)),)
    us = prepare_pair(make_asset("KO", a), make_asset("PEP", b), min_observations=100,
                      special_session_dates_by_exchange=by_exchange)
    assert "2021-03-01" not in {x["date"] for x in us.report.special_sessions}
    nse = prepare_pair(make_asset("TCS.NS", a), make_asset("INFY.NS", b), min_observations=100,
                       special_session_dates_by_exchange=by_exchange)
    assert "2021-03-01" in {x["date"] for x in nse.report.special_sessions}
    cfg = ResearchConfig().with_overrides({"data": {"special_session_dates_by_exchange": {".ns": ["2021-03-01"]}}})
    assert cfg.data.special_session_dates_by_exchange == by_exchange  # mapping normalised to suffix "NS"


def test_prevalence_baseline_variant_b0(coint_pair, nested_run):
    from statarb.ml_strategy import MLSignalRules, ml_only_signals

    cfg, without = nested_run
    assert "ml_only_prevalence_baseline" not in without.daily  # off by default
    cfg_b0 = _v2_config(include_prevalence_baseline=True)
    with_b0 = run_ml_walk_forward(coint_pair.prices, cfg_b0, SignalRules(), None, None)
    for v in without.daily:  # adding B0 changes nothing else
        pd.testing.assert_frame_equal(without.daily[v], with_b0.daily[v])
    item = next(i for i, row in zip(with_b0.window_inputs, with_b0.windows.itertuples()) if row.traded)
    sig = item["signals"]["ml_only_prevalence_baseline"]
    z = sig["zscore"]
    expected = ml_only_signals(z, pd.Series(0.3, index=z.index), MLSignalRules(threshold=0.3, min_abs_z=1.0,
                                                                                exit_z=0.5, horizon=5))
    pd.testing.assert_series_equal(sig["target_position"], expected["target_position"])
