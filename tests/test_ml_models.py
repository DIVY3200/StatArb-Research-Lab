"""Pipelines, training-only preprocessing, prediction timestamps, metrics and tuning."""

import numpy as np
import pandas as pd
import pytest

from statarb.config import ResearchConfig
from statarb.evaluation import chronological_split, fit_pair_model
from statarb.ml_features import FeatureSpec, build_dataset
from statarb.ml_models import (
    InsufficientDataError,
    calibration_table,
    classification_metrics,
    feature_importance,
    fit_model,
    param_grid,
    tune_hyperparameters,
)


@pytest.fixture(scope="module")
def data(coint_pair):
    prices = coint_pair.prices
    splits = chronological_split(prices.index, 0.6, 0.2, 0.2)
    model = fit_pair_model(np.log(prices.iloc[splits["train"].slice]), ResearchConfig()).spread
    ds = build_dataset(prices, model, FeatureSpec(), splits["train"].slice, 5, 0.25, 1.0)
    tr, va = splits["train"], splits["validation"]
    return prices, splits, model, ds, ds.xy(ds.rows(tr.start_idx, tr.end_idx)), ds.xy(ds.rows(va.start_idx, va.end_idx))


def test_scaler_is_fitted_on_training_rows_only(data):
    *_, (X_train, y_train), (X_val, _) = data
    fitted = fit_model("logistic_regression", {"C": 1.0}, X_train, y_train)
    scaler = fitted.pipeline.named_steps["scaler"]
    np.testing.assert_allclose(scaler.mean_, X_train.mean().to_numpy())
    assert not np.allclose(scaler.mean_, pd.concat([X_train, X_val]).mean().to_numpy())
    assert fitted.n_train == len(X_train)
    assert fitted.train_end == str(X_train.index[-1].date())


def test_predictions_are_timestamped_and_causal(data):
    prices, splits, model, ds, (X_train, y_train), _ = data
    fitted = fit_model("random_forest", {"max_depth": 3, "min_samples_leaf": 10}, X_train, y_train, n_estimators=50)
    prob = fitted.predict_proba(ds.features)
    assert prob.index.equals(ds.features.index)
    assert prob[ds.features.isna().any(axis=1)].isna().all()  # incomplete features -> no prediction
    valid = prob.dropna()
    assert ((valid >= 0) & (valid <= 1)).all()
    # The prediction at t depends only on data up to t: perturbing later prices leaves it unchanged.
    cut = splits["validation"].start_idx + 50
    shocked = prices.copy()
    shocked.iloc[cut + 1:] *= 1.25
    ds2 = build_dataset(shocked, model, FeatureSpec(), splits["train"].slice, 5, 0.25, 1.0)
    pd.testing.assert_series_equal(prob.iloc[:cut + 1], fitted.predict_proba(ds2.features).iloc[:cut + 1])


def test_insufficient_samples_and_class_imbalance_guard(data):
    *_, (X_train, y_train), _ = data
    with pytest.raises(InsufficientDataError, match="usable training rows"):
        fit_model("logistic_regression", {"C": 1.0}, X_train.iloc[:20], y_train.iloc[:20], min_train_samples=60)
    one_class = y_train[y_train == 1]
    with pytest.raises(InsufficientDataError, match="class counts"):
        fit_model("logistic_regression", {"C": 1.0}, X_train.loc[one_class.index], one_class)
    with_gap = X_train.copy()
    with_gap.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        fit_model("logistic_regression", {"C": 1.0}, with_gap, y_train)


def test_classification_metrics_by_hand():
    y = pd.Series([1, 1, 0, 0, 1, 0, 1, 1, 0, 1] * 4)
    p = pd.Series([0.9, 0.4, 0.2, 0.6, 0.7, 0.1, 0.8, 0.3, 0.45, 0.55] * 4)
    m = classification_metrics(y, p, threshold=0.5, min_samples=30)
    # predictions: 1,0,0,1,1,0,1,0,0,1 -> TP=4, FN=2, FP=1, TN=3 per block of 10
    assert m["confusion_matrix"] == [[12, 4], [8, 16]]
    assert m["accuracy"] == pytest.approx(28 / 40)
    assert m["precision"] == pytest.approx(16 / 20)
    assert m["recall"] == pytest.approx(16 / 24)
    assert m["f1"] == pytest.approx(2 * (0.8 * 16 / 24) / (0.8 + 16 / 24))
    assert m["majority_class_accuracy"] == pytest.approx(24 / 40)
    assert 0.5 < m["roc_auc"] <= 1.0
    assert m["sufficient_sample"]


def test_metrics_undefined_cases():
    small = classification_metrics(pd.Series([1, 0, 1]), pd.Series([0.7, 0.2, 0.6]), min_samples=30)
    assert np.isnan(small["accuracy"]) and np.isnan(small["roc_auc"]) and not small["sufficient_sample"]
    single = classification_metrics(pd.Series([1] * 40), pd.Series(np.linspace(0.1, 0.9, 40)), min_samples=30)
    assert np.isnan(single["roc_auc"])
    assert "one class" in single["note"]
    assert calibration_table(pd.Series([1] * 40), pd.Series(np.linspace(0.1, 0.9, 40))).empty


def test_calibration_table_counts():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=500)
    y = (rng.uniform(size=500) < p).astype(int)  # perfectly calibrated by construction
    table = calibration_table(pd.Series(y), pd.Series(p), bins=5)
    assert table["count"].sum() == 500 and len(table) == 5
    np.testing.assert_allclose(table["observed_frequency"], table["mean_predicted"], atol=0.1)


def test_tuning_uses_validation_and_ignores_test_data(data, coint_pair):
    prices, splits, model, ds, (X_train, y_train), (X_val, y_val) = data
    grid = param_grid("logistic_regression", ResearchConfig().ml)
    result = tune_hyperparameters("logistic_regression", grid, X_train, y_train, X_val, y_val)
    assert result.best_params in grid
    assert len(result.table) == len(grid)
    # Altering test-period prices cannot change any training or validation row, hence not the choice.
    shocked = prices.copy()
    shocked.iloc[splits["test"].start_idx:] *= 1.5
    ds2 = build_dataset(shocked, model, FeatureSpec(), splits["train"].slice, 5, 0.25, 1.0)
    va = splits["validation"]
    X_val2, y_val2 = ds2.xy(ds2.rows(va.start_idx, va.end_idx))
    again = tune_hyperparameters("logistic_regression", grid, X_train, y_train, X_val2, y_val2)
    assert again.best_params == result.best_params
    pd.testing.assert_frame_equal(again.table, result.table)


def test_feature_importance_outputs(data):
    *_, (X_train, y_train), (X_val, y_val) = data
    lr = fit_model("logistic_regression", {"C": 1.0}, X_train, y_train)
    imp = feature_importance(lr, X_val, y_val, n_repeats=3)
    assert {"standardized_coefficient", "odds_ratio_per_sd", "permutation_importance"}.issubset(imp.columns)
    rf = fit_model("random_forest", {"max_depth": 3, "min_samples_leaf": 10}, X_train, y_train, n_estimators=50)
    imp = feature_importance(rf, X_val, y_val, n_repeats=3)
    assert imp["impurity_importance"].sum() == pytest.approx(1.0)
