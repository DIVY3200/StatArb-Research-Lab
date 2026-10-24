"""End-to-end ML research: leakage prevention, identical comparison settings, walk-forward, snapshots."""

import json

import numpy as np
import pandas as pd
import pytest

from statarb.config import ResearchConfig
from statarb.data_validation import AlignedPair
from statarb.evaluation import run_pair_research, save_research_run
from statarb.ml_evaluation import (
    MODEL_NAMES,
    classification_summary,
    ml_result_tables,
    run_ml_research,
    select_threshold,
)

FAST = {
    "ml": {"rf_n_estimators": 30, "logistic_c_grid": [0.1, 1.0], "rf_max_depth_grid": [3],
           "rf_min_samples_leaf_grid": [10]},
    "walk_forward": {"train_window": 500, "test_window": 250},
}


def _config(**ml_overrides):
    overrides = {k: dict(v) for k, v in FAST.items()}
    overrides["ml"].update(ml_overrides)
    return ResearchConfig().with_overrides(overrides)


def _with_prices(pair: AlignedPair, prices: pd.DataFrame) -> AlignedPair:
    return AlignedPair(prices, pair.ticker_a, pair.ticker_b, pair.price_column_a, pair.price_column_b, pair.report,
                       pair.validation_a, pair.validation_b, pair.metadata_a, pair.metadata_b)


@pytest.fixture(scope="module")
def revealed(coint_pair):
    cfg = _config()
    base = run_pair_research(coint_pair, cfg, reveal_test=True)
    return base, run_ml_research(base)


def test_models_fitted_and_training_rows_purged(revealed):
    base, ml = revealed
    train_end = base.splits["train"].end_idx
    index = base.pair.prices.index
    for name in MODEL_NAMES:
        report = ml.models[name]
        assert report.ok, report.status
        last_train_row = index.get_loc(pd.Timestamp(report.fitted.train_end))
        assert last_train_row + ml.config.ml.horizon <= train_end - 1
        assert set(report.metrics) == {"train", "validation", "test"}
    samples = ml.sample_table
    assert (samples["rows_used"] <= samples["eligible_bars"]).all()
    val_last = index.get_loc(pd.Timestamp(samples.loc["validation", "last_row"]))
    assert val_last + ml.config.ml.horizon <= base.splits["validation"].end_idx - 1


def test_all_variants_share_dates_and_backtest_settings(revealed):
    base, ml = revealed
    for period, comp in ml.comparisons.items():
        expected_index = base.periods[period].backtest.daily.index
        assert len(comp.results) == 5
        configs = {r.config for r in comp.results.values()}
        assert len(configs) == 1  # identical lag, sizing, costs, borrow and end-of-period rules
        for result in comp.results.values():
            assert result.daily.index.equals(expected_index)
            assert result.daily["position"].iloc[-1] == 0
        pd.testing.assert_frame_equal(comp.results["traditional"].daily, base.periods[period].backtest.daily)


def test_test_period_hidden_until_revealed(coint_pair):
    cfg = _config()
    ml = run_ml_research(run_pair_research(coint_pair, cfg, reveal_test=False), include_walk_forward=False)
    assert set(ml.comparisons) == {"train", "validation"}
    assert all(set(r.metrics) == {"train", "validation"} for r in ml.models.values())
    assert len(ml.dataset.features) == run_pair_research(coint_pair, cfg).splits["validation"].end_idx


def test_validation_outputs_unaffected_by_test_period_prices(coint_pair):
    cfg = _config(threshold_source="validation_selected")
    base = run_pair_research(coint_pair, cfg, reveal_test=True)
    ml = run_ml_research(base, include_walk_forward=False)
    prices = coint_pair.prices.copy()
    start = base.splits["test"].start_idx
    prices.iloc[start:] = prices.iloc[start:] * np.array([1.6, 0.7])
    base2 = run_pair_research(_with_prices(coint_pair, prices), cfg, reveal_test=True)
    ml2 = run_ml_research(base2, include_walk_forward=False)
    for name in MODEL_NAMES:
        a, b = ml.models[name], ml2.models[name]
        assert a.tuning.best_params == b.tuning.best_params
        assert a.thresholds == b.thresholds
        pd.testing.assert_frame_equal(a.tuning.table, b.tuning.table)
        assert a.metrics["validation"] == b.metrics["validation"]
        pd.testing.assert_series_equal(a.probability.iloc[:start], b.probability.iloc[:start])
    pd.testing.assert_frame_equal(ml.comparisons["validation"].table, ml2.comparisons["validation"].table)


def test_training_fit_unaffected_by_validation_prices(coint_pair):
    """The H-bar purge keeps validation prices out of training labels, so the fitted model is identical."""
    cfg = _config()
    base = run_pair_research(coint_pair, cfg)
    ml = run_ml_research(base, include_walk_forward=False)
    prices = coint_pair.prices.copy()
    start = base.splits["validation"].start_idx
    prices.iloc[start:] = prices.iloc[start:] * np.array([1.3, 0.8])
    ml2 = run_ml_research(run_pair_research(_with_prices(coint_pair, prices), cfg), include_walk_forward=False)
    coef = ml.models["logistic_regression"].fitted.pipeline.named_steps["model"].coef_
    coef2 = ml2.models["logistic_regression"].fitted.pipeline.named_steps["model"].coef_
    np.testing.assert_array_equal(coef, coef2)


def test_walk_forward_traditional_matches_base_and_is_chained(revealed):
    base, ml = revealed
    wf = ml.walk_forward
    pd.testing.assert_frame_equal(wf.daily["traditional"], base.walk_forward.daily)
    windows = wf.windows
    assert (pd.to_datetime(windows["fit_end"]) < pd.to_datetime(windows["eval_start"])).all()
    for variant, daily in wf.daily.items():
        assert daily.index.is_monotonic_increasing and not daily.index.has_duplicates
        assert daily.index.equals(wf.daily["traditional"].index)
        growth = np.prod(1 + windows[f"return_{variant}"].to_numpy())
        assert daily["equity"].iloc[-1] / ml.config.portfolio.initial_capital == pytest.approx(growth)


def test_insufficient_data_is_reported_not_fitted(coint_pair):
    cfg = _config(min_train_samples=100_000)
    ml = run_ml_research(run_pair_research(coint_pair, cfg), include_walk_forward=False)
    for report in ml.models.values():
        assert not report.ok and "usable training rows" in report.status
    assert set(ml.comparisons["validation"].results) == {"traditional"}


def test_select_threshold_rule():
    table = pd.DataFrame({"threshold": [0.5, 0.6, 0.7], "completed_trades": [10, 2, 6],
                          "sharpe_ratio": [0.4, 2.0, 0.9], "cumulative_return": [0.1, 0.3, 0.2]})
    assert select_threshold(table, "sharpe", min_trades=3) == 0.7
    assert select_threshold(table, "sharpe", min_trades=50) is None


def test_exports_and_snapshot(revealed, tmp_path):
    base, ml = revealed
    tables = ml_result_tables(ml)
    assert {"ml_samples", "ml_classification_metrics", "ml_predictions", "strategy_comparison_test",
            "ml_walk_forward_windows"}.issubset(tables)
    summary = classification_summary(ml)
    assert {"LR | validation", "RF | test"}.issubset(summary.columns)
    run_dir = save_research_run(base, tmp_path, ml_result=ml)
    settings = json.loads((run_dir / "ml_settings.json").read_text())
    assert settings["horizon"] == 5 and settings["models"]["logistic_regression"]["status"] == "fitted"
    assert (run_dir / "strategy_comparison_validation.csv").exists()
