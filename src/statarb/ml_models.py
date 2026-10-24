"""Classifiers for the reversion target: pipelines, fitting, validation-only tuning and diagnostics.

* **Logistic Regression** (interpretable baseline): ``StandardScaler -> LogisticRegression``.
* **Random Forest** (nonlinear comparison): ``RandomForestClassifier`` (no scaling needed).

Every transformer inside a pipeline (for example the scaler's means and variances) is fitted
only on the rows passed to :func:`fit_model`, which callers restrict to purged training rows.
By default (baseline) both models use balanced class weights, so a skewed class distribution does
not simply produce a majority-class predictor. Weighting distorts probabilities towards 0.5, so the
audit-driven protocol uses ``class_weight="none"`` to keep probabilities on the frequency scale.
Models are not fitted when there are too few rows or too few examples of either class.

The *prevalence baseline* forecasts the training base rate for every row. Brier and log-loss
skill scores (1 - model loss / baseline loss) are reported against it: a negative skill means the
model's probabilities are worse than not using a model at all.
"""

from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_LABELS = {"logistic_regression": "Logistic Regression", "random_forest": "Random Forest"}
SHORT_LABELS = {"logistic_regression": "LR", "random_forest": "RF"}


class InsufficientDataError(ValueError):
    """Raised when a model cannot be fitted responsibly on the available rows."""


def make_pipeline(name: str, params: dict, seed: int = 42, n_estimators: int = 200,
                  class_weight: str = "balanced") -> Pipeline:
    if class_weight not in {"balanced", "none"}:
        raise ValueError("class_weight must be balanced or none")
    balanced = class_weight == "balanced"
    if name == "logistic_regression":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(C=params.get("C", 1.0), class_weight="balanced" if balanced else None,
                                         max_iter=5000)),
        ])
    if name == "random_forest":
        return Pipeline([
            ("model", RandomForestClassifier(
                n_estimators=n_estimators, max_depth=params.get("max_depth"),
                min_samples_leaf=params.get("min_samples_leaf", 10),
                class_weight="balanced_subsample" if balanced else None, random_state=seed, n_jobs=1,
            )),
        ])
    raise ValueError(f"Unknown model: {name!r}")


def param_grid(name: str, ml_config) -> list[dict]:
    if name == "logistic_regression":
        return [{"C": c} for c in ml_config.logistic_c_grid]
    if name == "random_forest":
        return [{"max_depth": d, "min_samples_leaf": leaf}
                for d, leaf in itertools.product(ml_config.rf_max_depth_grid, ml_config.rf_min_samples_leaf_grid)]
    raise ValueError(f"Unknown model: {name!r}")


@dataclass
class FittedModel:
    name: str
    params: dict
    pipeline: Pipeline
    feature_columns: list[str]
    n_train: int
    class_counts: dict[int, int]
    train_start: str
    train_end: str

    def predict_proba(self, features: pd.DataFrame) -> pd.Series:
        """P(y=1) for every row with complete features; NaN elsewhere (never imputed)."""
        X = features[self.feature_columns]
        ok = X.notna().all(axis=1).to_numpy()
        prob = np.full(len(X), np.nan)
        if ok.any():
            classes = list(self.pipeline.classes_)
            prob[ok] = self.pipeline.predict_proba(X.loc[ok])[:, classes.index(1)]
        return pd.Series(prob, index=features.index, name=f"p_{self.name}")


def fit_model(name: str, params: dict, X: pd.DataFrame, y: pd.Series, seed: int = 42, n_estimators: int = 200,
              min_train_samples: int = 60, min_class_count: int = 10, class_weight: str = "balanced") -> FittedModel:
    """Fit one pipeline on the supplied (already purged, training-only) rows."""
    if X.isna().any().any():
        raise ValueError("training features contain NaN; drop incomplete rows before fitting")
    counts = {int(k): int(v) for k, v in y.value_counts().items()}
    if len(y) < min_train_samples:
        raise InsufficientDataError(
            f"{MODEL_LABELS[name]}: only {len(y)} usable training rows (minimum {min_train_samples})."
        )
    if len(counts) < 2 or min(counts.values()) < min_class_count:
        raise InsufficientDataError(
            f"{MODEL_LABELS[name]}: class counts {counts} - each class needs at least {min_class_count} training rows."
        )
    pipeline = make_pipeline(name, params, seed, n_estimators, class_weight)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        pipeline.fit(X, y)
    return FittedModel(name, dict(params), pipeline, list(X.columns), int(len(y)), counts,
                       _date(X.index[0]), _date(X.index[-1]))


def _date(value) -> str:
    return str(value.date()) if hasattr(value, "date") else str(value)


def classification_metrics(y_true: pd.Series, prob: pd.Series, threshold: float = 0.5,
                           min_samples: int = 30, reference_rate: float | None = None) -> dict[str, object]:
    """Out-of-sample classification diagnostics; rates are NaN when the sample is too small.

    ``majority_class_accuracy`` is the accuracy of always predicting the more frequent class.
    Accuracy should be judged against it, not against 50%. ``reference_rate`` is the prevalence
    baseline (normally the training base rate); when given, Brier and log-loss skill against a
    constant forecast of that rate are reported.
    """
    y = pd.Series(y_true).astype(int).to_numpy()
    p = pd.Series(prob).to_numpy(dtype=float)
    n = len(y)
    positives = int(y.sum()) if n else 0
    out: dict[str, object] = {
        "n_samples": n, "n_positive": positives, "n_negative": n - positives,
        "positive_rate": positives / n if n else np.nan,
        "sufficient_sample": bool(n >= min_samples and 0 < positives < n),
    }
    nan_keys = ("accuracy", "majority_class_accuracy", "precision", "recall", "f1", "roc_auc", "brier_score",
                "log_loss", "predicted_positive_rate", "brier_reference", "brier_skill_score", "log_loss_reference",
                "log_loss_skill")
    out["reference_rate"] = np.nan if reference_rate is None else float(reference_rate)
    if n < min_samples or n == 0:
        out.update({k: np.nan for k in nan_keys})
        out["confusion_matrix"] = confusion_matrix(y, (p >= threshold).astype(int), labels=[0, 1]).tolist() if n else [[0, 0], [0, 0]]
        out["note"] = f"Only {n} evaluation rows (minimum {min_samples}); metrics not reported."
        return out
    pred = (p >= threshold).astype(int)
    both = 0 < positives < n
    out.update(
        accuracy=float(accuracy_score(y, pred)),
        majority_class_accuracy=max(positives, n - positives) / n,
        precision=float(precision_score(y, pred, zero_division=np.nan)),
        recall=float(recall_score(y, pred, zero_division=np.nan)),
        f1=float(f1_score(y, pred, zero_division=np.nan)),
        roc_auc=float(roc_auc_score(y, p)) if both else np.nan,
        brier_score=float(brier_score_loss(y, p)),
        log_loss=float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1])),
        predicted_positive_rate=float(pred.mean()),
        confusion_matrix=confusion_matrix(y, pred, labels=[0, 1]).tolist(),
        note="" if both else "Only one class present: ROC-AUC undefined.",
    )
    out.update(_skill_scores(y, p, reference_rate))
    return out


def _skill_scores(y: np.ndarray, p: np.ndarray, reference_rate: float | None) -> dict[str, float]:
    """Brier / log-loss skill of ``p`` against a constant forecast ``reference_rate``."""
    keys = ("brier_reference", "brier_skill_score", "log_loss_reference", "log_loss_skill")
    if reference_rate is None or not 0.0 < reference_rate < 1.0 or len(y) == 0:
        return {k: np.nan for k in keys}
    ref = np.full(len(y), float(reference_rate))
    brier_ref = float(brier_score_loss(y, ref))
    ll_ref = float(log_loss(y, ref, labels=[0, 1]))
    brier = float(brier_score_loss(y, p))
    ll = float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1]))
    return {
        "brier_reference": brier_ref,
        "brier_skill_score": 1.0 - brier / brier_ref if brier_ref > 0 else np.nan,
        "log_loss_reference": ll_ref,
        "log_loss_skill": 1.0 - ll / ll_ref if ll_ref > 0 else np.nan,
    }


def calibration_table(y_true: pd.Series, prob: pd.Series, bins: int = 5) -> pd.DataFrame:
    """Reliability table: mean predicted probability vs observed frequency, quantile bins."""
    y = pd.Series(y_true).astype(int).to_numpy()
    p = pd.Series(prob).to_numpy(dtype=float)
    if len(y) < bins * 2 or len(np.unique(y)) < 2:
        return pd.DataFrame(columns=["mean_predicted", "observed_frequency", "count"])
    groups = pd.qcut(p, q=bins, duplicates="drop")
    table = pd.DataFrame({"p": p, "y": y}).groupby(groups, observed=True).agg(
        mean_predicted=("p", "mean"), observed_frequency=("y", "mean"), count=("y", "size"))
    return table.reset_index(drop=True)


def _score(metric: str, y: np.ndarray, p: np.ndarray) -> float:
    if len(y) == 0 or len(np.unique(y)) < 2:
        return np.nan
    if metric == "roc_auc":
        return float(roc_auc_score(y, p))
    return -float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1]))  # higher is better


@dataclass
class TuningResult:
    best_params: dict
    table: pd.DataFrame
    warnings: list[str] = field(default_factory=list)


def tune_hyperparameters(name: str, grid: list[dict], X_train: pd.DataFrame, y_train: pd.Series,
                         X_val: pd.DataFrame, y_val: pd.Series, metric: str = "roc_auc", seed: int = 42,
                         n_estimators: int = 200, min_train_samples: int = 60, min_class_count: int = 10,
                         min_eval_samples: int = 30, class_weight: str = "balanced") -> TuningResult:
    """Fit each grid point on training rows and score it on validation rows (never the test period)."""
    rows, warn = [], []
    for params in grid:
        fitted = fit_model(name, params, X_train, y_train, seed, n_estimators, min_train_samples, min_class_count,
                           class_weight)
        prob = fitted.predict_proba(X_val).to_numpy()
        score = _score(metric, y_val.to_numpy(), prob) if len(y_val) >= min_eval_samples else np.nan
        train_score = _score(metric, y_train.to_numpy(), fitted.predict_proba(X_train).to_numpy())
        rows.append({**{k: ("none" if v is None else v) for k, v in params.items()},
                     f"train_{metric}": train_score, f"validation_{metric}": score})
    table = pd.DataFrame(rows)
    scores = table[f"validation_{metric}"].to_numpy(dtype=float)
    if np.isfinite(scores).any():
        best = grid[int(np.nanargmax(scores))]
    else:
        best = grid[0]
        warn.append(f"{MODEL_LABELS[name]}: validation set too small or single-class for tuning; "
                    f"using the first grid point {best}.")
    return TuningResult(best, table, warn)


def feature_importance(fitted: FittedModel, X_eval: pd.DataFrame, y_eval: pd.Series, seed: int = 42,
                       n_repeats: int = 5) -> pd.DataFrame:
    """Model-specific importance plus permutation importance (ROC-AUC drop) on evaluation rows."""
    model = fitted.pipeline.named_steps["model"]
    frame = pd.DataFrame(index=fitted.feature_columns)
    if fitted.name == "logistic_regression":
        frame["standardized_coefficient"] = model.coef_[0]
        frame["odds_ratio_per_sd"] = np.exp(model.coef_[0])
    else:
        frame["impurity_importance"] = model.feature_importances_
    if len(y_eval) >= 20 and y_eval.nunique() == 2:
        perm = permutation_importance(fitted.pipeline, X_eval, y_eval, scoring="roc_auc", n_repeats=n_repeats,
                                      random_state=seed, n_jobs=1)
        frame["permutation_importance"] = perm.importances_mean
        frame["permutation_std"] = perm.importances_std
    else:
        frame["permutation_importance"] = np.nan
        frame["permutation_std"] = np.nan
    return frame


__all__ = [
    "FittedModel",
    "InsufficientDataError",
    "MODEL_LABELS",
    "SHORT_LABELS",
    "TuningResult",
    "calibration_table",
    "classification_metrics",
    "feature_importance",
    "fit_model",
    "make_pipeline",
    "param_grid",
    "tune_hyperparameters",
]
