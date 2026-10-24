"""ML research layer on top of :func:`statarb.evaluation.run_pair_research`.

Protocol (static workflow)
--------------------------
1. Reuse the statistical layer's chronological splits and its *training-fitted* spread model.
2. Build features and the reversion target over the visible sample (train + validation, plus
   test only once revealed). Features at ``t`` use data up to ``t`` only.
3. Training rows: eligible training bars whose label horizon ends inside the training window
   (purged by ``H`` bars, plus an optional embargo). Validation and test rows are purged the same
   way at their own right edge, so no label crosses into the next period.
4. Hyperparameters: each grid point is fitted on training rows and scored on validation rows
   (ROC-AUC or log loss). The best point is refitted on training rows only.
5. Probability thresholds for B and C: the configured threshold, or optionally the best
   validation-period Sharpe over ``threshold_grid``. The test period is never used.
6. Strategy comparison: A, B and C run on the *same* period dates with the same backtester
   settings (execution lag, sizing, costs, borrow, end-of-period liquidation), each starting flat.

Walk-forward: every window refits the spread model, re-checks cointegration, rebuilds features
with that window's frozen parameters, refits both classifiers on the window's purged fitting
rows, and trades the next window. All three variants share the same windows, cointegration gate and
capital-chaining convention.

Training features (``ml.training_features``): ``in_sample`` (baseline) builds the fitting rows with
the spread model fitted on the whole fitting window; ``point_in_time`` builds them block by block
with parameters estimated only on earlier bars (:func:`statarb.ml_features.point_in_time_rows`), so
training rows are constructed the same way as live predictions. Rows scored out of sample always use
the frozen model. With ``evaluation.static_refit_before_test`` the static test period uses the spread
model refitted on train + validation and classifiers refitted on the same span (hyperparameters and
thresholds stay those chosen on validation); with ``evaluation.static_require_cointegration`` B and C
are held flat wherever A is.

Walk-forward tuning modes (``ml.walk_forward_tuning``):

* ``static`` (baseline): every window reuses the hyperparameters (and thresholds) chosen once on the
  static validation period. **Audit finding D1:** windows evaluated before or during that period
  therefore use information from their future. Kept only to reproduce the baseline.
* ``nested`` (point-in-time): each window splits its own fitting window chronologically into an
  inner-training block and an inner-validation block (the last ``inner_validation_fraction``),
  both purged, tunes hyperparameters on the inner-validation block, then refits the chosen grid
  point on all purged fitting rows. Nothing from the trading window or later is used.

Thresholds (``ml.threshold_mode``): ``absolute`` uses ``probability_threshold`` (or the static
validation choice); ``prevalence`` uses ``prevalence_multiplier`` x the fitting rows' reversion rate,
i.e. "the model rates reversion as more likely than average", which needs no tuning.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .backtester import TRADE_COLUMNS, BacktestConfig, BacktestResult, run_backtest
from .evaluation import (
    PERIOD_LABELS,
    PairResearchResult,
    _date,
    _flat_signals,
    _sanitize,
    build_signals,
    fit_pair_model,
    rules_for_model,
    static_tradable,
    walk_forward_schedule,
)
from .metrics import strategy_metrics
from .ml_features import FEATURE_DESCRIPTIONS, FeatureSpec, MLDataset, PointInTimeRows, build_dataset, point_in_time_rows
from .ml_models import (
    MODEL_LABELS,
    SHORT_LABELS,
    FittedModel,
    InsufficientDataError,
    TuningResult,
    calibration_table,
    classification_metrics,
    feature_importance,
    fit_model,
    param_grid,
    tune_hyperparameters,
)
from .ml_strategy import VARIANT_LABELS, MLSignalRules, hybrid_signals, ml_only_signals
from .signals import SignalRules

MODEL_NAMES = ("logistic_regression", "random_forest")
VARIANTS = tuple(VARIANT_LABELS)
COMPARISON_KEYS = (
    "cumulative_return", "gross_cumulative_return", "cagr", "annualized_volatility", "sharpe_ratio", "sortino_ratio",
    "max_drawdown", "calmar_ratio",
    "completed_trades", "win_rate", "pct_time_invested", "total_turnover", "annualized_turnover",
    "total_transaction_costs", "total_borrow_costs", "final_equity",
)


@dataclass
class ModelReport:
    name: str
    status: str
    fitted: FittedModel | None
    tuning: TuningResult | None
    probability: pd.Series
    metrics: dict[str, dict] = field(default_factory=dict)
    calibration: dict[str, pd.DataFrame] = field(default_factory=dict)
    importance: pd.DataFrame | None = None
    thresholds: dict[str, float] = field(default_factory=dict)
    threshold_tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    test_refit: FittedModel | None = None        # refitted on train + validation (static_refit_before_test)
    test_thresholds: dict[str, float] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return MODEL_LABELS[self.name]

    @property
    def ok(self) -> bool:
        return self.fitted is not None


@dataclass
class VariantComparison:
    period: str
    results: dict[str, BacktestResult]
    table: pd.DataFrame

    @property
    def label(self) -> str:
        return PERIOD_LABELS[self.period]


@dataclass
class MLWalkForwardResult:
    windows: pd.DataFrame
    daily: dict[str, pd.DataFrame]
    trades: dict[str, pd.DataFrame]
    table: pd.DataFrame
    predictions: pd.DataFrame = field(default_factory=pd.DataFrame)
    window_inputs: list[dict] = field(default_factory=list)  # per window: eval bounds, hedge, signals per variant
    names: dict[str, str] = field(default_factory=dict)


@dataclass
class MLResearchResult:
    config: object
    names: dict[str, str]
    dataset: MLDataset
    sample_table: pd.DataFrame
    models: dict[str, ModelReport]
    comparisons: dict[str, VariantComparison]
    walk_forward: MLWalkForwardResult | None
    test_revealed: bool
    warnings: list[str] = field(default_factory=list)
    pit_blocks: pd.DataFrame | None = None  # point-in-time training blocks, when ml.training_features asks for them


# --------------------------------------------------------------------------- helpers
def _variant_signals(variant: str, z: pd.Series, prob: pd.Series | None, rules: SignalRules, thresholds: dict,
                     ml_cfg) -> pd.DataFrame | None:
    """Signals for B or C; ``None`` when the model is unavailable."""
    if prob is None:
        return None
    if variant.startswith("ml_only"):
        stop = rules.stop_loss_z if rules.stop_loss_z is not None and rules.stop_loss_z > ml_cfg.min_abs_z else None
        return ml_only_signals(z, prob, MLSignalRules(thresholds["ml_only"], ml_cfg.min_abs_z, rules.exit_z,
                                                      ml_cfg.horizon, stop))
    return hybrid_signals(z, prob, rules, thresholds["hybrid"])


def _flat(prices: pd.DataFrame, z: pd.Series | None) -> pd.DataFrame:
    frame = pd.DataFrame(index=prices.index)
    frame["zscore"] = z.to_numpy() if z is not None else np.nan
    frame["target_position"] = 0
    frame["event"] = ""
    frame["holding_bars"] = 0
    return frame


def _metrics_row(result: BacktestResult, config) -> dict[str, float]:
    m = config.metrics
    stats = strategy_metrics(result, m.risk_free, m.periods_per_year, m.min_years_for_cagr)
    return {k: stats[k] for k in COMPARISON_KEYS}


def _comparison_table(results: dict[str, BacktestResult], config) -> pd.DataFrame:
    return pd.DataFrame({VARIANT_LABELS[k]: _metrics_row(r, config) for k, r in results.items()})


def _threshold_grid(prices: pd.DataFrame, z: pd.Series, prob: pd.Series, rules: SignalRules, hedge: float,
                    bt_config: BacktestConfig, config, opens: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """Validation-period strategy metrics for each threshold in the grid, for B and C."""
    tables = {}
    for kind in ("ml_only", "hybrid"):
        rows = []
        for thr in config.ml.threshold_grid:
            sig = _variant_signals(kind, z, prob, rules, {"ml_only": thr, "hybrid": thr}, config.ml)
            res = run_backtest(prices, sig, hedge, bt_config, opens)
            row = _metrics_row(res, config)
            rows.append({"threshold": thr, **row})
        tables[kind] = pd.DataFrame(rows)
    return tables


_SELECTION_COLUMNS = {"sharpe": "sharpe_ratio", "cumulative_return": "cumulative_return", "calmar": "calmar_ratio"}


def select_threshold(table: pd.DataFrame, metric: str = "sharpe", min_trades: int = 3) -> float | None:
    """Best validation threshold by ``metric`` among rows with at least ``min_trades`` completed trades."""
    if metric not in _SELECTION_COLUMNS:
        raise ValueError(f"Unsupported selection metric {metric!r}; expected one of {sorted(_SELECTION_COLUMNS)}")
    column = _SELECTION_COLUMNS[metric]
    eligible = table.loc[(table["completed_trades"] >= min_trades) & np.isfinite(table[column].astype(float))]
    if eligible.empty:
        return None
    return float(eligible.loc[eligible[column].astype(float).idxmax(), "threshold"])


# --------------------------------------------------------------------------- static workflow
def run_ml_research(base: PairResearchResult, include_walk_forward: bool = True) -> MLResearchResult:
    cfg, ml = base.config, base.config.ml
    names = base.names
    prices = base.pair.prices.iloc[: base.research_end_idx]
    opens = None if base.pair.opens is None else base.pair.opens.iloc[: base.research_end_idx]
    splits = base.splits
    spread_model = base.model.spread
    spec = FeatureSpec.from_config(cfg)
    seed = cfg.reproducibility.random_seed
    warnings: list[str] = []
    if cfg.strategy.entry_z < ml.min_abs_z:
        warnings.append(f"Entry threshold {cfg.strategy.entry_z} is below ml.min_abs_z {ml.min_abs_z}: hybrid entries "
                        "can occur outside the range the models were trained on.")

    ds = build_dataset(prices, spread_model, spec, splits["train"].slice, ml.horizon, ml.tolerance_z, ml.min_abs_z)
    val_end = splits["validation"].end_idx
    refit = base.test_revealed and base.test_model is not None
    if refit:  # test-period features from the spread model refitted on train + validation
        ds_refit = build_dataset(prices, base.test_model.spread, spec, slice(0, val_end), ml.horizon, ml.tolerance_z,
                                 ml.min_abs_z)
        ds = _splice(ds, ds_refit, splits["test"].slice)
    periods = ["train", "validation"] + (["test"] if base.test_revealed else [])
    masks = {p: ds.rows(splits[p].start_idx, splits[p].end_idx, embargo=ml.embargo if p == "train" else 0)
             for p in periods}
    pit = None
    if ml.training_features == "point_in_time":
        pit = point_in_time_rows(prices, 0, splits["train"].end_idx, cfg, spec, embargo=ml.embargo)
        X_tr, y_tr = pit.xy()
        warnings.append(f"ML training rows use point-in-time features: {len(pit)} rows from {len(pit.blocks)} blocks "
                        f"of {ml.pit_block} bars, each with parameters from the "
                        f"{ml.pit_estimation_window or cfg.walk_forward.train_window} bars before it.")
    else:
        X_tr, y_tr = ds.xy(masks["train"])
    sample_table = _sample_table(ds, splits, periods, masks, pit)
    X_va, y_va = ds.xy(masks["validation"])
    refit_rows = None
    if refit:
        if pit is not None:
            refit_rows = point_in_time_rows(prices, 0, val_end, cfg, spec, embargo=ml.embargo).xy()
        else:
            refit_rows = ds_refit.xy(ds_refit.rows(0, val_end, embargo=ml.embargo))

    bt_config = BacktestConfig.from_research_config(cfg, tickers=(names["A"], names["B"]))
    val_split = splits["validation"]
    val_prices = prices.iloc[val_split.slice]
    val_z = ds.z.iloc[val_split.slice]
    models: dict[str, ModelReport] = {}
    for name in MODEL_NAMES:
        empty = pd.Series(np.nan, index=prices.index, name=f"p_{name}")
        try:
            tuning = tune_hyperparameters(name, param_grid(name, ml), X_tr, y_tr, X_va, y_va, ml.tuning_metric, seed,
                                          ml.rf_n_estimators, ml.min_train_samples, ml.min_class_count,
                                          ml.min_eval_samples, ml.class_weight)
            fitted = fit_model(name, tuning.best_params, X_tr, y_tr, seed, ml.rf_n_estimators, ml.min_train_samples,
                               ml.min_class_count, ml.class_weight)
        except InsufficientDataError as exc:
            models[name] = ModelReport(name, f"not fitted: {exc}", None, None, empty)
            warnings.append(str(exc))
            continue
        warnings.extend(tuning.warnings)
        prob = fitted.predict_proba(ds.features)
        report = ModelReport(name, "fitted", fitted, tuning, prob)
        prevalence = float(y_tr.mean())
        cls_threshold = _prevalence_threshold(prevalence, ml) if ml.threshold_mode == "prevalence" else ml.probability_threshold
        reference = {p: prevalence for p in periods}
        if refit_rows is not None:
            note = _refit_for_test(report, refit_rows, ds, splits["test"].slice, ml, seed)
            if note:
                warnings.append(note)
            else:
                reference["test"] = float(refit_rows[1].mean())
        for p in periods:
            thr = cls_threshold
            if ml.threshold_mode == "prevalence" and reference[p] != prevalence:
                thr = _prevalence_threshold(reference[p], ml)
            if p == "train" and pit is not None:
                y, pr = y_tr, fitted.predict_proba(X_tr)
            else:
                y, pr = ds.target[masks[p]].astype(int), prob[masks[p]]
            report.metrics[p] = classification_metrics(y, pr, thr, ml.min_eval_samples, reference_rate=reference[p])
            report.calibration[p] = calibration_table(y, pr, ml.calibration_bins)
        report.importance = feature_importance(fitted, X_va, y_va, seed)
        report.threshold_tables = _threshold_grid(val_prices, val_z, prob.iloc[val_split.slice],
                                                  base.periods["validation"].rules, spread_model.hedge_ratio,
                                                  bt_config, cfg, None if opens is None else opens.iloc[val_split.slice])
        report.thresholds = {"ml_only": ml.probability_threshold, "hybrid": ml.probability_threshold}
        if ml.threshold_mode == "prevalence":
            report.thresholds = {"ml_only": cls_threshold, "hybrid": cls_threshold}
        elif ml.threshold_source == "validation_selected":
            for kind, table in report.threshold_tables.items():
                chosen = select_threshold(table, cfg.evaluation.selection_metric, cfg.evaluation.min_trades_for_selection)
                if chosen is None:
                    warnings.append(f"{report.label} {kind}: no validation threshold met the minimum-trade rule; "
                                    f"using the configured threshold {ml.probability_threshold}.")
                else:
                    report.thresholds[kind] = chosen
        if report.test_refit is not None:
            report.test_thresholds = dict(report.thresholds)
            if ml.threshold_mode == "prevalence":
                report.test_thresholds = {k: _prevalence_threshold(reference["test"], ml) for k in report.thresholds}
        models[name] = report

    comparisons = {p: _compare_period(p, base, ds, models, bt_config, opens) for p in periods}
    walk_forward = None
    if include_walk_forward:
        # Static mode reproduces the baseline (audit D1: reuses validation-period choices in every window).
        params = {n: (r.tuning.best_params if r.ok else param_grid(n, ml)[0]) for n, r in models.items()}
        thresholds = {n: (r.thresholds or {"ml_only": ml.probability_threshold, "hybrid": ml.probability_threshold})
                      for n, r in models.items()}
        if ml.walk_forward_tuning == "nested":
            params, thresholds = None, None
            warnings.append("ML walk-forward uses nested per-window tuning (point-in-time).")
        else:
            warnings.append("ML walk-forward reuses hyperparameters tuned on the static validation period in every "
                            "window (baseline mode; audit finding D1: earlier windows see later data).")
        try:
            walk_forward = run_ml_walk_forward(base.pair.prices, cfg, base.periods["validation"].rules, params,
                                               thresholds, end_idx=base.research_end_idx, names=names,
                                               opens=base.pair.opens)
        except ValueError as exc:
            warnings.append(f"ML walk-forward skipped: {exc}")
    return MLResearchResult(cfg, names, ds, sample_table, models, comparisons, walk_forward, base.test_revealed, warnings,
                            pit.blocks if pit is not None else None)


def _splice(ds: MLDataset, other: MLDataset, sl: slice) -> MLDataset:
    """``ds`` with the bars in ``sl`` (features, target, eligibility) taken from ``other``."""
    features, target, elig = ds.features.copy(), ds.target.copy(), ds.eligible.copy()
    features.iloc[sl] = other.features.iloc[sl].to_numpy()
    target.iloc[sl] = other.target.iloc[sl].to_numpy()
    elig.iloc[sl] = other.eligible.iloc[sl].to_numpy()
    return MLDataset(features, target, elig, ds.spec, ds.horizon, ds.tolerance)


def _refit_for_test(report: ModelReport, rows: tuple[pd.DataFrame, pd.Series], ds: MLDataset, test_slice: slice,
                    ml_cfg, seed: int) -> str:
    """Refit one classifier on train + validation rows (same hyperparameters) for the test period."""
    X, y = rows
    try:
        refit = fit_model(report.name, report.tuning.best_params, X, y, seed, ml_cfg.rf_n_estimators,
                          ml_cfg.min_train_samples, ml_cfg.min_class_count, ml_cfg.class_weight)
    except InsufficientDataError as exc:
        return f"{report.label}: test-period refit impossible ({exc}); the training-fitted model is used."
    report.test_refit = refit
    report.probability.iloc[test_slice] = refit.predict_proba(ds.features.iloc[test_slice]).to_numpy()
    return ""


def _sample_table(ds: MLDataset, splits, periods, masks, pit: PointInTimeRows | None = None) -> pd.DataFrame:
    rows = []
    for p in periods:
        s = splits[p]
        window = np.zeros(len(ds.features), dtype=bool)
        window[s.start_idx:s.end_idx] = True
        elig = window & ds.eligible.to_numpy(dtype=bool)
        used = masks[p]
        y = ds.target[used]
        row = {
            "period": p, "start": s.start_date, "end": s.end_date, "bars": s.n_obs,
            "eligible_bars": int(elig.sum()), "rows_used": int(used.sum()),
            "excluded_label_crosses_boundary_or_missing": int((elig & ~used).sum()),
            "positive_rows": int((y == 1).sum()), "negative_rows": int((y == 0).sum()),
            "positive_rate": float(y.mean()) if len(y) else np.nan,
            "first_row": _date(ds.features.index[used][0]) if used.any() else None,
            "last_row": _date(ds.features.index[used][-1]) if used.any() else None,
        }
        if pit is not None:
            row["features"] = "frozen model (out of sample)"
            if p == "train":
                y = pit.target
                eligible_bars = int(pit.blocks["eligible_bars"].sum()) if len(pit.blocks) else 0
                row.update(eligible_bars=eligible_bars, rows_used=int(len(y)),
                           excluded_label_crosses_boundary_or_missing=eligible_bars - int(len(y)),
                           positive_rows=int((y == 1).sum()), negative_rows=int((y == 0).sum()),
                           positive_rate=float(y.mean()) if len(y) else np.nan,
                           first_row=_date(pit.features.index[0]) if len(y) else None,
                           last_row=_date(pit.features.index[-1]) if len(y) else None,
                           features="point in time (block-wise)")
        rows.append(row)
    return pd.DataFrame(rows).set_index("period")


def _compare_period(period: str, base: PairResearchResult, ds: MLDataset, models: dict[str, ModelReport],
                    bt_config: BacktestConfig, opens: pd.DataFrame | None = None) -> VariantComparison:
    split = base.splits[period]
    prices = base.pair.prices.iloc[split.slice]
    period_opens = None if opens is None else opens.iloc[split.slice]
    z = ds.z.iloc[split.slice]
    rules = base.periods[period].rules
    model = base.test_model if period == "test" and base.test_model is not None else base.model
    hedge = model.spread.hedge_ratio
    tradable = static_tradable(base, period)
    results: dict[str, BacktestResult] = {"traditional": base.periods[period].backtest}
    for name, report in models.items():
        prob = report.probability.iloc[split.slice] if report.ok else None
        thresholds = report.test_thresholds if period == "test" and report.test_thresholds else report.thresholds
        for kind in ("ml_only", "hybrid"):
            sig = _variant_signals(kind, z, prob, rules, thresholds, base.config.ml)
            if sig is not None:
                if not tradable:  # static cointegration gate failed: held flat like A
                    sig = _flat(prices, z)
                results[f"{kind}_{name}"] = run_backtest(prices, sig, hedge, bt_config, period_opens)
    return VariantComparison(period, results, _comparison_table(results, base.config))


# --------------------------------------------------------------------------- walk-forward
def _prevalence_threshold(prevalence: float, ml_cfg) -> float:
    return float(np.clip(prevalence * ml_cfg.prevalence_multiplier, 0.01, 0.99))


def _fit_sets(ds: MLDataset, fit_len: int, ml_cfg, pit: PointInTimeRows | None = None,
              offset: int = 0) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    """One window's purged fitting rows and, for nested tuning, its inner-training / inner-validation split.

    In-sample rows come from ``ds`` (features from the window's own fitted spread model); point-in-time
    rows come from ``pit``, whose positions are shifted by ``offset`` (the window's first bar).
    """
    nested = ml_cfg.walk_forward_tuning == "nested"
    inner_end = fit_len - int(round(fit_len * ml_cfg.inner_validation_fraction))
    if pit is None:
        sets = {"fit": ds.xy(ds.rows(0, fit_len, embargo=ml_cfg.embargo))}
        if nested:
            sets["inner"] = ds.xy(ds.rows(0, inner_end, embargo=ml_cfg.embargo))
            sets["inner_validation"] = ds.xy(ds.rows(inner_end, fit_len))
        return sets
    rel = pit.positions - offset
    sets = {"fit": pit.xy(rel + pit.horizon <= fit_len - 1 - ml_cfg.embargo)}
    if nested:
        sets["inner"] = pit.xy(rel + pit.horizon <= inner_end - 1 - ml_cfg.embargo)
        sets["inner_validation"] = pit.xy(rel >= inner_end)
    return sets


def _window_model(name: str, sets: dict[str, tuple[pd.DataFrame, pd.Series]], ml_cfg, seed: int,
                  static_params: dict | None, row: dict) -> FittedModel | None:
    """Choose hyperparameters (static or nested) and fit one model on the window's purged fitting rows."""
    short = SHORT_LABELS[name]
    grid = param_grid(name, ml_cfg)
    X_fit, y_fit = sets["fit"]
    if ml_cfg.walk_forward_tuning == "nested":
        (X_in, y_in), (X_iv, y_iv) = sets["inner"], sets["inner_validation"]
        try:
            tuning = tune_hyperparameters(name, grid, X_in, y_in, X_iv, y_iv, ml_cfg.tuning_metric, seed,
                                          ml_cfg.rf_n_estimators, ml_cfg.min_train_samples, ml_cfg.min_class_count,
                                          ml_cfg.min_eval_samples, ml_cfg.class_weight)
            chosen = tuning.best_params
            if tuning.warnings:
                row[f"{short}_tuning_note"] = "; ".join(tuning.warnings)
        except InsufficientDataError as exc:
            chosen = grid[0]
            row[f"{short}_tuning_note"] = f"inner tuning impossible ({exc}); first grid point used"
        row[f"{short}_inner_rows"] = f"{len(y_in)}/{len(y_iv)}"
    else:
        chosen = static_params[name] if static_params else grid[0]
    row[f"{short}_params"] = json.dumps(chosen, sort_keys=True)
    try:
        return fit_model(name, chosen, X_fit, y_fit, seed, ml_cfg.rf_n_estimators, ml_cfg.min_train_samples,
                         ml_cfg.min_class_count, ml_cfg.class_weight)
    except InsufficientDataError as exc:
        row[f"{short}_note"] = str(exc)
        return None


def run_ml_walk_forward(prices: pd.DataFrame, config, rules: SignalRules, params: dict[str, dict] | None,
                        thresholds: dict[str, dict[str, float]] | None, end_idx: int | None = None,
                        names: dict[str, str] | None = None, first_eval_idx: int | None = None,
                        opens: pd.DataFrame | None = None) -> MLWalkForwardResult:
    """A, B and C on identical walk-forward windows (see the module docstring for tuning modes).

    ``params`` / ``thresholds`` are the static-mode choices and are ignored in nested / prevalence mode.
    ``first_eval_idx`` anchors the first trading window (default: right after the first fitting window).
    ``opens`` (adjusted opening prices) are needed when ``strategy.execution_price`` is ``open``.
    """
    names = names or {"A": "A", "B": "B"}
    config = config.for_pair(names["A"], names["B"])
    tickers = (names["A"], names["B"])
    ml, wf = config.ml, config.walk_forward
    if ml.walk_forward_tuning == "static" and params is None:
        raise ValueError("static walk-forward tuning needs the static hyperparameters (params)")
    data = prices.iloc[:end_idx] if end_idx is not None else prices
    data_opens = None if opens is None else opens.iloc[: len(data)]
    logp = np.log(data)
    spec = FeatureSpec.from_config(config)
    seed = config.reproducibility.random_seed
    schedule = walk_forward_schedule(len(data), wf.train_window, wf.test_window, wf.expanding, wf.min_eval_observations,
                                     first_eval_idx=first_eval_idx)
    variants = ["traditional"] + [f"{k}_{n}" for n in MODEL_NAMES for k in ("ml_only", "hybrid")]
    if ml.include_prevalence_baseline:
        variants.append("ml_only_prevalence_baseline")
    equity = {v: float(config.portfolio.initial_capital) for v in variants}
    dailies: dict[str, list] = {v: [] for v in variants}
    trades: dict[str, list] = {v: [] for v in variants}
    rows, predictions, window_inputs = [], [], []

    for w in schedule:
        eval_prices = data.iloc[w.eval_start_idx:w.eval_end_idx]
        row = {"window": w.window_id, "fit_start": _date(data.index[w.fit_start_idx]),
               "fit_end": _date(data.index[w.fit_end_idx - 1]), "eval_start": _date(eval_prices.index[0]),
               "eval_end": _date(eval_prices.index[-1])}
        try:
            pair_model = fit_pair_model(logp.iloc[w.fit_start_idx:w.fit_end_idx], config, names["A"], names["B"])
        except ValueError as exc:
            pair_model, row["note"] = None, str(exc)
        traded = pair_model is not None and (pair_model.passes_gate(config) or not wf.require_cointegration)
        row.update(cointegrated=bool(pair_model is not None and pair_model.coint.reject_null), traded=traded,
                   eg_p_value=pair_model.coint.p_value if pair_model is not None else np.nan,
                   hedge_ratio=pair_model.spread.hedge_ratio if pair_model is not None else np.nan)
        if pair_model is not None and pair_model.suite is not None:
            row.update({f"suite_{k}": v for k, v in pair_model.suite.summary().items()})
        window_rules = rules
        if pair_model is not None and config.strategy.max_holding_half_lives:
            window_rules = rules_for_model(config, pair_model.half_life, rules.entry_z, rules.exit_z)
            row["max_holding_bars"] = window_rules.max_holding_bars
        hedge = pair_model.spread.hedge_ratio if pair_model is not None else 0.0
        signals: dict[str, pd.DataFrame] = {}
        if pair_model is None:
            signals["traditional"] = _flat_signals(eval_prices, None)
        elif traded:
            signals["traditional"] = build_signals(eval_prices, pair_model.spread, window_rules)
        else:
            signals["traditional"] = _flat_signals(eval_prices, pair_model.spread)

        z_eval = None
        if traded:
            block = data.iloc[w.fit_start_idx:w.eval_end_idx]
            fit_len = w.fit_end_idx - w.fit_start_idx
            ds = build_dataset(block, pair_model.spread, spec, slice(0, fit_len), ml.horizon, ml.tolerance_z, ml.min_abs_z)
            z_eval = ds.z.iloc[fit_len:]
            pit = None
            if ml.training_features == "point_in_time":
                pit = point_in_time_rows(data, w.fit_start_idx, w.fit_end_idx, config, spec)
            sets = _fit_sets(ds, fit_len, ml, pit, w.fit_start_idx)
            y_fit = sets["fit"][1]
            prevalence = float(y_fit.mean()) if len(y_fit) else np.nan
            row["fit_rows"], row["fit_prevalence"] = int(len(y_fit)), prevalence
            if pit is not None:
                row["pit_blocks"] = int(len(pit.blocks))
            eval_rows = ds.rows(fit_len, len(block))
            if ml.include_prevalence_baseline and np.isfinite(prevalence):
                # B0: B's rules driven by the constant prevalence forecast (no model).
                constant = pd.Series(prevalence, index=z_eval.index)
                thr = _prevalence_threshold(prevalence, ml) if ml.threshold_mode == "prevalence" else ml.probability_threshold
                signals["ml_only_prevalence_baseline"] = _variant_signals(
                    "ml_only", z_eval, constant, window_rules, {"ml_only": thr, "hybrid": thr}, ml)
            for name in MODEL_NAMES:
                short = SHORT_LABELS[name]
                fitted = _window_model(name, sets, ml, seed, params, row)
                if fitted is None:
                    continue
                if ml.threshold_mode == "prevalence" and np.isfinite(prevalence):
                    thr = _prevalence_threshold(prevalence, ml)
                    window_thresholds = {"ml_only": thr, "hybrid": thr}
                elif thresholds is not None:
                    window_thresholds = thresholds[name]
                else:
                    window_thresholds = {"ml_only": ml.probability_threshold, "hybrid": ml.probability_threshold}
                prob = fitted.predict_proba(ds.features)
                y_eval, p_eval = ds.target[eval_rows].astype(int), prob[eval_rows]
                cm = classification_metrics(y_eval, p_eval, window_thresholds["hybrid"], ml.min_eval_samples,
                                            reference_rate=prevalence)
                row.update({f"{short}_train_rows": fitted.n_train, f"{short}_eval_rows": cm["n_samples"],
                            f"{short}_eval_roc_auc": cm["roc_auc"], f"{short}_eval_brier_skill": cm["brier_skill_score"],
                            f"{short}_threshold_ml_only": window_thresholds["ml_only"],
                            f"{short}_threshold_hybrid": window_thresholds["hybrid"]})
                predictions.append(pd.DataFrame({"window": w.window_id, "model": name, "probability": p_eval,
                                                 "target": y_eval, "fit_prevalence": prevalence}))
                prob_eval = prob.iloc[fit_len:]
                for kind in ("ml_only", "hybrid"):
                    signals[f"{kind}_{name}"] = _variant_signals(kind, z_eval, prob_eval, window_rules,
                                                                 window_thresholds, ml)

        window_inputs.append({"eval": (w.eval_start_idx, w.eval_end_idx), "hedge": hedge, "signals": {}})
        eval_opens = None if data_opens is None else data_opens.iloc[w.eval_start_idx:w.eval_end_idx]
        for v in variants:
            sig = signals.get(v)
            if sig is None:
                sig = _flat(eval_prices, z_eval)
            window_inputs[-1]["signals"][v] = sig
            cfg_bt = BacktestConfig.from_research_config(config, tickers=tickers, initial_capital=equity[v])
            result = run_backtest(eval_prices, sig, hedge, cfg_bt, eval_opens, accrue_first_bar=w.window_id > 1)
            daily = result.daily.copy()
            daily.insert(0, "window", w.window_id)
            dailies[v].append(daily)
            if not result.trades.empty:
                t = result.trades.copy()
                t.insert(0, "window", w.window_id)
                trades[v].append(t)
            end_equity = float(result.equity.iloc[-1])
            row[f"return_{v}"] = end_equity / equity[v] - 1.0
            row[f"trades_{v}"] = int((result.trades["status"] == "closed").sum()) if not result.trades.empty else 0
            equity[v] = end_equity
        rows.append(row)

    combined_daily, combined_trades, results = {}, {}, {}
    for v in variants:
        daily = pd.concat(dailies[v])
        daily["drawdown"] = daily["equity"] / daily["equity"].cummax() - 1.0
        daily["cumulative_costs"] = (daily["trade_cost"] + daily["borrow_cost"]).cumsum()
        tr = pd.concat(trades[v], ignore_index=True) if trades[v] else pd.DataFrame(columns=["window"] + TRADE_COLUMNS)
        combined_daily[v], combined_trades[v] = daily, tr
        results[v] = BacktestResult(daily, tr, pd.DataFrame(index=daily.index), float("nan"),
                                    BacktestConfig.from_research_config(config, tickers=tickers))
    pred = pd.concat(predictions) if predictions else pd.DataFrame(
        columns=["window", "model", "probability", "target", "fit_prevalence"])
    return MLWalkForwardResult(pd.DataFrame(rows), combined_daily, combined_trades, _comparison_table(results, config),
                               pred, window_inputs, dict(names))


def reprice_walk_forward(prices: pd.DataFrame, config, wf: MLWalkForwardResult,
                         end_idx: int | None = None, opens: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """Re-run every window's stored signals through the backtester under ``config`` (e.g. other costs).

    Signals and model fits are reused unchanged, so only execution and cost assumptions differ.
    With the original config (and the same ``opens``) this reproduces ``wf.daily`` exactly.
    """
    data = prices.iloc[:end_idx] if end_idx is not None else prices
    data_opens = None if opens is None else opens.iloc[: len(data)]
    tickers = (wf.names["A"], wf.names["B"]) if wf.names else None
    equity: dict[str, float] = {}
    dailies: dict[str, list] = {}
    for k, item in enumerate(wf.window_inputs):
        start, end = item["eval"]
        eval_prices = data.iloc[start:end]
        eval_opens = None if data_opens is None else data_opens.iloc[start:end]
        for v, sig in item["signals"].items():
            capital = equity.get(v, float(config.portfolio.initial_capital))
            result = run_backtest(eval_prices, sig, item["hedge"],
                                  BacktestConfig.from_research_config(config, tickers=tickers, initial_capital=capital),
                                  eval_opens, accrue_first_bar=k > 0)
            daily = result.daily.copy()
            daily.insert(0, "window", len(dailies.get(v, [])) + 1)
            dailies.setdefault(v, []).append(daily)
            equity[v] = float(result.equity.iloc[-1])
    out = {}
    for v, frames in dailies.items():
        daily = pd.concat(frames)
        daily["drawdown"] = daily["equity"] / daily["equity"].cummax() - 1.0
        daily["cumulative_costs"] = (daily["trade_cost"] + daily["borrow_cost"]).cumsum()
        out[v] = daily
    return out


# --------------------------------------------------------------------------- exports
def ml_settings(result: MLResearchResult) -> dict[str, object]:
    ml = result.config.ml
    return {
        "target": f"y_t = 1 if |z_(t+{ml.horizon})| <= |z_t| - {ml.tolerance_z}, else 0; rows with |z_t| >= {ml.min_abs_z}",
        "horizon": ml.horizon, "tolerance_z": ml.tolerance_z, "min_abs_z": ml.min_abs_z, "embargo": ml.embargo,
        "features": result.dataset.spec.columns,
        "feature_definitions": FEATURE_DESCRIPTIONS,
        "threshold_source": ml.threshold_source, "configured_threshold": ml.probability_threshold,
        "threshold_mode": ml.threshold_mode, "prevalence_multiplier": ml.prevalence_multiplier,
        "tuning_metric": ml.tuning_metric, "class_weight": ml.class_weight,
        "walk_forward_tuning": ml.walk_forward_tuning, "inner_validation_fraction": ml.inner_validation_fraction,
        "training_features": ml.training_features,
        "pit_estimation_window": (ml.pit_estimation_window or result.config.walk_forward.train_window)
        if ml.training_features == "point_in_time" else None,
        "pit_block": ml.pit_block if ml.training_features == "point_in_time" else None,
        "models": {
            n: {"status": r.status, "selected_params": r.tuning.best_params if r.tuning else None,
                "thresholds": r.thresholds, "train_rows": r.fitted.n_train if r.ok else 0,
                "test_refit_rows": r.test_refit.n_train if r.test_refit is not None else None,
                "test_thresholds": r.test_thresholds or None,
                "train_class_counts": r.fitted.class_counts if r.ok else None,
                "train_rows_first": r.fitted.train_start if r.ok else None,
                "train_rows_last": r.fitted.train_end if r.ok else None}
            for n, r in result.models.items()
        },
        "test_revealed": result.test_revealed,
        "warnings": result.warnings,
    }


def classification_summary(result: MLResearchResult) -> pd.DataFrame:
    rows = {}
    for name, report in result.models.items():
        for period, m in report.metrics.items():
            rows[f"{SHORT_LABELS[name]} | {period}"] = {k: v for k, v in m.items() if k != "confusion_matrix"}
    return pd.DataFrame(rows)


def ml_result_tables(result: MLResearchResult) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {"ml_samples": result.sample_table, "ml_classification_metrics": classification_summary(result)}
    pred = pd.DataFrame({"z": result.dataset.z, "target": result.dataset.target, "eligible": result.dataset.eligible})
    for name, report in result.models.items():
        short = SHORT_LABELS[name]
        pred[f"p_{short}"] = report.probability
        if report.tuning is not None:
            tables[f"ml_tuning_{short}"] = report.tuning.table
        if report.importance is not None:
            tables[f"ml_importance_{short}"] = report.importance
        for kind, table in report.threshold_tables.items():
            tables[f"ml_threshold_grid_{kind}_{short}"] = table
        for period, table in report.calibration.items():
            tables[f"ml_calibration_{short}_{period}"] = table
    tables["ml_predictions"] = pred
    if result.pit_blocks is not None:
        tables["ml_point_in_time_blocks"] = result.pit_blocks
    for period, comp in result.comparisons.items():
        tables[f"strategy_comparison_{period}"] = comp.table
        tables[f"strategy_returns_{period}"] = pd.DataFrame({VARIANT_LABELS[k]: r.returns for k, r in comp.results.items()})
        for k, r in comp.results.items():
            if k != "traditional":
                tables[f"trades_{period}_{k}"] = r.trades
    if result.walk_forward is not None:
        tables["ml_walk_forward_windows"] = result.walk_forward.windows
        tables["ml_walk_forward_comparison"] = result.walk_forward.table
        tables["ml_walk_forward_returns"] = pd.DataFrame({VARIANT_LABELS[k]: d["net_return"]
                                                         for k, d in result.walk_forward.daily.items()})
        tables["ml_walk_forward_predictions"] = result.walk_forward.predictions
        for k, t in result.walk_forward.trades.items():
            tables[f"ml_walk_forward_trades_{k}"] = t
    return tables


def save_ml_tables(result: MLResearchResult, run_dir: str | Path) -> None:
    run_dir = Path(run_dir)
    for name, table in ml_result_tables(result).items():
        table.to_csv(run_dir / f"{name}.csv")
    payload = _sanitize(ml_settings(result))
    (run_dir / "ml_settings.json").write_text(json.dumps(payload, indent=2, default=str, allow_nan=False), encoding="utf-8")


__all__ = [
    "MLResearchResult",
    "MLWalkForwardResult",
    "MODEL_NAMES",
    "ModelReport",
    "VARIANTS",
    "VariantComparison",
    "classification_summary",
    "ml_result_tables",
    "ml_settings",
    "run_ml_research",
    "run_ml_walk_forward",
    "reprice_walk_forward",
    "save_ml_tables",
    "select_threshold",
]
