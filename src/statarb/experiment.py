"""Pre-registered v2 experiment: walk-forward A/B/C on several pairs, pooled inference, decision rules.

See ``docs/research/PREREGISTRATION.md``. The functions here only *evaluate*; data access, the holdout
guard and the freeze checks live in ``scripts/run_experiment.py``.

Primary portfolio: for each variant, the equal-weight average of the pairs' daily net returns on the
union of their trading calendars (a pair contributes 0 on days it does not trade).

Each pair runs under its own market's settings (``markets:`` overrides, e.g. NSE costs or an INR
risk-free rate). Sharpe ratios and their bootstrap intervals use *excess* returns: every pair's daily
net return minus its own market's daily risk-free rate, on that pair's trading days. With a zero
risk-free rate (v2) excess and net returns coincide.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from .data_loader import AssetData
from .data_validation import AlignedPair, prepare_pair_from_config
from .evaluation import signal_rules_from_config
from .inference import bootstrap_sharpe, bootstrap_sharpe_difference, effective_sample_size
from .metrics import daily_risk_free_series, return_metrics, sharpe_ratio, sortino_ratio
from .ml_evaluation import MLWalkForwardResult, reprice_walk_forward, run_ml_walk_forward
from .ml_strategy import VARIANT_LABELS

VARIANTS = ["traditional", "ml_only_logistic_regression", "hybrid_logistic_regression",
            "ml_only_random_forest", "hybrid_random_forest"]
CANDIDATES = VARIANTS[1:]
CANDIDATE_MODEL = {v: ("logistic_regression" if v.endswith("logistic_regression") else "random_forest")
                   for v in CANDIDATES}


@dataclass
class PairRun:
    label: str
    pair: AlignedPair
    first_eval_idx: int | None
    wf: MLWalkForwardResult
    eval_start: str = ""
    eval_end: str = ""
    notes: list[str] = field(default_factory=list)
    config: object = None  # the pair's config after market overrides


def run_pair(asset_a: AssetData, asset_b: AssetData, config, first_eval_date: str | None = None) -> PairRun:
    """Prepare one pair and run the A/B/C walk-forward (nested tuning required).

    The pair's market overrides (``config.for_pair``) are applied before anything is computed.
    """
    if config.ml.walk_forward_tuning != "nested":
        raise ValueError("The experiment requires ml.walk_forward_tuning: nested (point-in-time tuning).")
    config = config.for_pair(asset_a.ticker, asset_b.ticker)
    pair = prepare_pair_from_config(asset_a, asset_b, config.data)
    first_idx = None
    if first_eval_date is not None:
        first_idx = int(pair.prices.index.searchsorted(pd.Timestamp(first_eval_date)))
        if first_idx >= len(pair.prices):
            raise ValueError(f"No {asset_a.ticker}/{asset_b.ticker} observations on or after {first_eval_date}.")
    wf = run_ml_walk_forward(pair.prices, config, signal_rules_from_config(config), None, None,
                             names={"A": pair.ticker_a, "B": pair.ticker_b}, first_eval_idx=first_idx,
                             opens=pair.opens)
    index = wf.daily["traditional"].index
    return PairRun(f"{pair.ticker_a}/{pair.ticker_b}", pair, first_idx, wf, str(index[0].date()), str(index[-1].date()),
                   config=config)


def risk_free_daily(runs: list[PairRun], config=None) -> list[pd.Series]:
    """Each pair's daily risk-free rate on its own trading dates (its market's rate or year schedule).

    The first date is 0: the pair's account starts at that close, so no holding period precedes it.
    """
    out = []
    for r in runs:
        m = (r.config or config).metrics
        rate = daily_risk_free_series(m.risk_free, r.wf.daily["traditional"].index, m.periods_per_year)
        if len(rate):
            rate.iloc[0] = 0.0
        out.append(rate)
    return out


def uses_risk_free(rf_daily) -> bool:
    """Whether any pair has a non-zero risk-free rate (otherwise excess and net returns coincide)."""
    return any(bool(np.any(np.asarray(r, dtype=float) != 0.0)) for r in rf_daily or [])


# --------------------------------------------------------------------------- portfolio construction
def _variants(dailies, variants):
    if variants is not None:
        return variants
    order = VARIANTS + ["ml_only_prevalence_baseline"]
    return [v for v in order if v in dailies[0]]


def pooled_returns(dailies: list[dict[str, pd.DataFrame]], variants=None, rf_daily: list[float] | None = None
                   ) -> pd.DataFrame:
    """Equal-weight portfolio of pairs per variant on the union calendar (0 on non-trading days).

    ``rf_daily`` (one rate per pair) turns it into the pooled *excess* return: each pair's daily
    risk-free rate is subtracted on that pair's own trading days.
    """
    variants = _variants(dailies, variants)
    rf = rf_daily if uses_risk_free(rf_daily) else [0.0] * len(dailies)
    out = {}
    for v in variants:
        legs = [_minus(d[v]["net_return"], r) for d, r in zip(dailies, rf)]
        frame = pd.concat(legs, axis=1).sort_index().fillna(0.0)
        out[v] = frame.mean(axis=1)
    return pd.DataFrame(out)


def _minus(returns: pd.Series, rf) -> pd.Series:
    if isinstance(rf, pd.Series):
        return returns - rf.reindex(returns.index).fillna(0.0)
    return returns - rf if rf else returns


def gross_pooled_returns(dailies: list[dict[str, pd.DataFrame]], variants=None) -> pd.DataFrame:
    variants = _variants(dailies, variants)
    out = {}
    for v in variants:
        frame = pd.concat([d[v]["gross_return"] for d in dailies], axis=1).sort_index().fillna(0.0)
        out[v] = frame.mean(axis=1)
    return pd.DataFrame(out)


def activity(dailies: list[dict[str, pd.DataFrame]], trades: list[dict[str, pd.DataFrame]], v: str,
             initial_capital: float) -> dict:
    """Each variant's own activity, summed or averaged over pairs."""
    closed = sum(int((t[v]["status"] == "closed").sum()) if not t[v].empty else 0 for t in trades)
    invested = [float((d[v]["position"] != 0).mean()) for d in dailies]
    gross = [float(d[v]["gross_leverage"].mean()) for d in dailies]
    years = [len(d[v]) / 252 for d in dailies]
    turnover = [float(d[v]["turnover"].fillna(0).sum()) for d in dailies]
    costs = [float(d[v]["trade_cost"].sum() + d[v]["borrow_cost"].sum()) for d in dailies]
    return {
        "completed_trades": closed,
        "pct_time_invested_avg": float(np.mean(invested)),
        "avg_gross_exposure_avg": float(np.mean(gross)),
        "annualized_turnover_avg": float(np.mean([t / y for t, y in zip(turnover, years) if y > 0])),
        "total_costs_pct_of_capital_avg": float(np.mean(costs)) / initial_capital,
    }


def variant_table(pooled: pd.DataFrame, pooled_gross: pd.DataFrame, dailies, trades, config,
                  pooled_excess: pd.DataFrame | None = None) -> pd.DataFrame:
    """Pooled metrics per variant. With ``pooled_excess``, Sharpe and Sortino use those excess returns."""
    m = config.metrics
    rows = {}
    for v in pooled.columns:
        rf = m.risk_free if pooled_excess is None else 0.0
        stats = return_metrics(pooled[v], rf, m.periods_per_year, m.min_years_for_cagr)
        if pooled_excess is not None:
            stats["sharpe_ratio"] = sharpe_ratio(pooled_excess[v], 0.0, m.periods_per_year)
            stats["sortino_ratio"] = sortino_ratio(pooled_excess[v], 0.0, m.periods_per_year)
        stats["gross_cumulative_return"] = return_metrics(pooled_gross[v])["cumulative_return"]
        stats.update(activity(dailies, trades, v, config.portfolio.initial_capital))
        rows[VARIANT_LABELS[v]] = stats
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- classification
def pooled_classification(runs: list[PairRun], model: str, horizon: int, bins: int = 5) -> dict:
    """Out-of-sample classification pooled over windows and pairs, against the prevalence baseline."""
    frames = []
    for run in runs:
        pred = run.wf.predictions
        if not pred.empty:
            sel = pred[pred["model"] == model].copy()
            if not sel.empty:
                sel["pair"] = run.label
                frames.append(sel)
    if not frames:
        return {"n_rows": 0}
    data = pd.concat(frames)
    y = data["target"].astype(int).to_numpy()
    p = data["probability"].to_numpy(dtype=float)
    ref = data["fit_prevalence"].to_numpy(dtype=float)
    out: dict[str, object] = {"n_rows": int(len(y)), "positive_rate": float(y.mean()) if len(y) else np.nan,
                              "mean_predicted": float(p.mean()) if len(p) else np.nan,
                              "mean_reference": float(ref.mean()) if len(ref) else np.nan}
    ess = 0.0
    for _, g in data.groupby(["pair", "window"]):
        e = effective_sample_size(g["target"].to_numpy(dtype=float), max_lag=2 * horizon)
        ess += e if np.isfinite(e) else len(g)
    out["effective_sample_size"] = float(ess)
    if len(np.unique(y)) < 2:
        out.update(roc_auc=np.nan, brier=np.nan, brier_reference=np.nan, brier_skill_score=np.nan,
                   log_loss=np.nan, log_loss_reference=np.nan, log_loss_skill=np.nan)
        return out
    brier, brier_ref = brier_score_loss(y, p), brier_score_loss(y, ref)
    ll = log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1])
    ll_ref = log_loss(y, np.clip(ref, 1e-6, 1 - 1e-6), labels=[0, 1])
    out.update(roc_auc=float(roc_auc_score(y, p)), brier=float(brier), brier_reference=float(brier_ref),
               brier_skill_score=float(1 - brier / brier_ref), log_loss=float(ll), log_loss_reference=float(ll_ref),
               log_loss_skill=float(1 - ll / ll_ref))
    groups = pd.qcut(p, q=bins, duplicates="drop")
    cal = pd.DataFrame({"p": p, "y": y}).groupby(groups, observed=True).agg(
        mean_predicted=("p", "mean"), observed_frequency=("y", "mean"), count=("y", "size"))
    out["calibration"] = cal.reset_index(drop=True).to_dict(orient="records")
    per_window = []
    for (pair, window), g in data.groupby(["pair", "window"]):
        yy, pp = g["target"].astype(int).to_numpy(), g["probability"].to_numpy(dtype=float)
        per_window.append({"pair": pair, "window": int(window), "rows": int(len(g)),
                           "roc_auc": float(roc_auc_score(yy, pp)) if len(np.unique(yy)) == 2 else np.nan,
                           "positive_rate": float(yy.mean())})
    out["per_window"] = per_window
    return out


# --------------------------------------------------------------------------- inference
def bootstrap_summary(pooled: pd.DataFrame, inference_cfg, n_candidates: int = 4) -> dict:
    i = inference_cfg
    bonferroni = 1.0 - (1.0 - i.confidence) / n_candidates
    out = {"sharpe": {}, "difference_vs_traditional": {}, "bonferroni_confidence": bonferroni}
    for v in pooled.columns:
        out["sharpe"][v] = bootstrap_sharpe(pooled[v], i.bootstrap_samples, i.mean_block_length, i.confidence,
                                            i.seed).to_dict()
    for v in pooled.columns:
        if v == "traditional":
            continue
        args = (pooled[v], pooled["traditional"], i.bootstrap_samples, i.mean_block_length)
        out["difference_vs_traditional"][v] = {
            "ci": bootstrap_sharpe_difference(*args, i.confidence, i.seed).to_dict(),
            "ci_bonferroni": bootstrap_sharpe_difference(*args, bonferroni, i.seed).to_dict(),
        }
    if "ml_only_prevalence_baseline" in pooled.columns:
        out["ml_only_vs_prevalence_baseline"] = {
            v: bootstrap_sharpe_difference(pooled[v], pooled["ml_only_prevalence_baseline"], i.bootstrap_samples,
                                           i.mean_block_length, i.confidence, i.seed).to_dict()
            for v in ("ml_only_logistic_regression", "ml_only_random_forest") if v in pooled.columns
        }
    return out


# --------------------------------------------------------------------------- sensitivity and regimes
def at_cost_level(config, bps: float):
    """``config`` at one per-leg, per-side cost level.

    Generic costs with cash shorts (the baseline): commission = level, slippage = 0. Market profiles
    (e.g. NSE) or futures shorts: statutory charges stay as configured and the level is the slippage.
    """
    c = config.costs
    if c.profile == "generic" and c.short_instrument == "cash":
        return config.with_overrides({"costs": {"commission_bps": float(bps), "slippage_bps": 0.0}})
    return config.with_overrides({"costs": {"slippage_bps": float(bps)}})


def cost_sensitivity(runs: list[PairRun], config, cost_grid_bps) -> pd.DataFrame:
    """Re-price stored signals at each cost level (see :func:`at_cost_level`); models are not refitted."""
    rows = []
    rf = risk_free_daily(runs, config)
    for bps in cost_grid_bps:
        dailies = [reprice_walk_forward(r.pair.prices, at_cost_level(r.config or config, bps), r.wf, opens=r.pair.opens)
                   for r in runs]
        pooled = pooled_returns(dailies)
        excess = pooled_returns(dailies, rf_daily=rf) if uses_risk_free(rf) else pooled
        for v in pooled.columns:
            rows.append({"cost_bps_per_leg_per_side": float(bps), "variant": VARIANT_LABELS[v],
                         "cumulative_return": return_metrics(pooled[v])["cumulative_return"],
                         "sharpe_ratio": return_metrics(excess[v])["sharpe_ratio"]})
    return pd.DataFrame(rows)


#: The v2 execution and cost assumptions, used to measure what the v3 realism upgrades change.
V2_EXECUTION = {"strategy": {"execution_price": "close"},
                "costs": {"profile": "generic", "short_instrument": "cash", "commission_bps": 5.0, "slippage_bps": 5.0,
                          "borrow_bps_annual": 50.0},
                "portfolio": {"cash_interest": False}}


def assumption_checks(runs: list[PairRun], config) -> pd.DataFrame:
    """Re-price the stored signals under alternative assumptions (models, gates and signals unchanged).

    * ``as run``: the study's own assumptions (Sharpe on excess returns when a risk-free rate is set);
    * ``cash interest off``: no interest credit, Sharpe at rf = 0 (shows how much the rate level matters);
    * ``v2 execution and costs``: close fills, generic 5 + 5 bps costs, cash shorts with a 50 bps borrow fee and
      no interest, i.e. what the execution and cost upgrades change for the same trading decisions.
    """
    rf = risk_free_daily(runs, config)
    scenarios = {"as run": None, "cash interest off": {"portfolio": {"cash_interest": False}},
                 "v2 execution and costs": V2_EXECUTION}
    rows = []
    for name, overrides in scenarios.items():
        if overrides is None:
            dailies = [r.wf.daily for r in runs]
            pooled = pooled_returns(dailies)
            excess = uses_risk_free(rf)
            sharpe_input = pooled_returns(dailies, rf_daily=rf) if excess else pooled
            basis = "excess" if excess else "net"
        else:
            dailies = [reprice_walk_forward(r.pair.prices, (r.config or config).with_overrides(overrides), r.wf,
                                            opens=r.pair.opens) for r in runs]
            pooled = sharpe_input = pooled_returns(dailies)
            basis = "net"
        for v in pooled.columns:
            rows.append({"scenario": name, "variant": VARIANT_LABELS[v],
                         "cumulative_return": return_metrics(pooled[v])["cumulative_return"],
                         "sharpe_ratio": return_metrics(sharpe_input[v])["sharpe_ratio"], "sharpe_basis": basis})
    return pd.DataFrame(rows)


def calendar_breakdown(pooled: pd.DataFrame) -> pd.DataFrame:
    by_year = pooled.groupby(pooled.index.year).apply(lambda g: (1 + g).prod() - 1)
    by_year.columns = [VARIANT_LABELS[c] for c in by_year.columns]
    by_year.index.name = "year"
    return by_year


def volatility_regimes(runs: list[PairRun], pooled: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    """Descriptive (ex-post) terciles of average trailing leg volatility across pairs, on evaluation dates."""
    vols = []
    for r in runs:
        rets = np.log(r.pair.prices).diff().abs().mean(axis=1)
        vols.append(rets.rolling(window, min_periods=20).mean())
    level = pd.concat(vols, axis=1).mean(axis=1).reindex(pooled.index).ffill()
    tercile = pd.qcut(level, 3, labels=["low", "mid", "high"])
    rows = {}
    for name, g in pooled.groupby(tercile, observed=True):
        rows[str(name)] = {VARIANT_LABELS[v]: return_metrics(g[v])["sharpe_ratio"] for v in pooled.columns}
        rows[str(name)]["days"] = int(len(g))
    return pd.DataFrame(rows).T


# --------------------------------------------------------------------------- decision rules (pre-registered)
def evaluate_decision(development: dict, holdout: dict, holdout_costs: pd.DataFrame) -> dict:
    """Apply PREREGISTRATION.md section 6 to the development and holdout summaries."""
    decisions: dict[str, dict] = {}
    hb = holdout["bootstrap"]
    a_label = VARIANT_LABELS["traditional"]
    for v in CANDIDATES:
        model = CANDIDATE_MODEL[v]
        cls = development["classification"].get(model, {})
        auc, bss = cls.get("roc_auc", np.nan), cls.get("brier_skill_score", np.nan)
        gate_a = bool(np.isfinite(auc) and np.isfinite(bss) and auc > 0.55 and bss > 0)
        diff = hb["difference_vs_traditional"][v]["ci_bonferroni"]
        gate_b = bool(np.isfinite(diff["point"]) and diff["point"] > 0 and np.isfinite(diff["lower"]) and diff["lower"] > 0)
        at20 = holdout_costs[holdout_costs["cost_bps_per_leg_per_side"] == 20.0].set_index("variant")["cumulative_return"]
        gate_c = bool(at20.get(VARIANT_LABELS[v], np.nan) > at20.get(a_label, np.nan))
        decisions[v] = {"gate_a_development_classification": gate_a, "development_roc_auc": auc,
                        "development_brier_skill": bss, "gate_b_holdout_sharpe_difference": gate_b,
                        "holdout_sharpe_difference": diff, "gate_c_beats_A_at_20bps": gate_c,
                        "decision": "retain" if gate_a and gate_b and gate_c else "reject"}
    a_ci = hb["sharpe"]["traditional"]
    dev_sharpe = development["bootstrap"]["sharpe"]["traditional"]["point"]
    if np.isfinite(a_ci["lower"]) and a_ci["lower"] > 0 and np.isfinite(dev_sharpe) and dev_sharpe > 0:
        a_decision = "retain as research candidate"
    elif np.isfinite(a_ci["point"]) and a_ci["point"] > 0:
        a_decision = "inconclusive: not retained"
    else:
        a_decision = "reject"
    decisions["traditional"] = {"holdout_sharpe": a_ci, "development_sharpe_point": dev_sharpe, "decision": a_decision}
    any_ml = any(decisions[v]["decision"] == "retain" for v in CANDIDATES)
    if any_ml:
        overall = "retain an ML enhancement"
    elif a_decision.startswith("retain"):
        overall = "retain the statistical strategy"
    else:
        overall = "reject the tested hypothesis"
    return {"per_strategy": decisions, "overall": overall}


__all__ = [
    "CANDIDATES",
    "PairRun",
    "VARIANTS",
    "V2_EXECUTION",
    "activity",
    "assumption_checks",
    "at_cost_level",
    "bootstrap_summary",
    "calendar_breakdown",
    "cost_sensitivity",
    "evaluate_decision",
    "gross_pooled_returns",
    "pooled_classification",
    "pooled_returns",
    "risk_free_daily",
    "run_pair",
    "uses_risk_free",
    "variant_table",
    "volatility_regimes",
]
