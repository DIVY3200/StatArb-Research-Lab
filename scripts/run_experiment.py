"""Pre-registered v2 experiment runner (see docs/research/PREREGISTRATION.md).

Stages
------
1. Development (2015-2024 data only; the holdout guard refuses anything later)::

       python scripts/run_experiment.py --stage development

2. Freeze the specification after development (records fingerprints of the code, config,
   pre-registration and development results)::

       python scripts/run_experiment.py --freeze reports/experiments/<development-run-dir>

3. Final, one-time holdout evaluation (refuses if anything changed since the freeze, or if the
   holdout has already been evaluated)::

       python scripts/run_experiment.py --stage final

Outputs go to reports/experiments/<UTC timestamp>_<stage>/ and are never overwritten.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from statarb.config import load_config, resolve_path, save_config_snapshot  # noqa: E402
from statarb.data_loader import download_yfinance  # noqa: E402
from statarb.experiment import (  # noqa: E402
    CANDIDATES,
    bootstrap_summary,
    calendar_breakdown,
    cost_sensitivity,
    evaluate_decision,
    gross_pooled_returns,
    pooled_classification,
    pooled_returns,
    run_pair,
    variant_table,
    volatility_regimes,
)
from statarb.ml_evaluation import MODEL_NAMES  # noqa: E402
from statarb.ml_strategy import VARIANT_LABELS  # noqa: E402
from statarb.protocol import allowed_end_date, code_fingerprint, file_sha256  # noqa: E402

PREREG = ROOT / "docs" / "research" / "PREREGISTRATION.md"
FREEZE = ROOT / "docs" / "research" / "freeze.json"
HOLDOUT_MARKER = ROOT / "docs" / "research" / "holdout_evaluated.json"
DEFAULT_CONFIG = ROOT / "config" / "research_v2.yaml"


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _json(value):
    if isinstance(value, dict):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _environment() -> dict[str, str]:
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for module in ("numpy", "pandas", "scipy", "statsmodels", "sklearn", "yfinance"):
        try:
            out[module] = __import__(module).__version__
        except Exception:  # optional
            out[module] = "not installed"
    return out


def _fingerprints(config_path: Path) -> dict[str, str]:
    return {"code_fingerprint": code_fingerprint([config_path]), "config_sha256": file_sha256(config_path),
            "preregistration_sha256": file_sha256(PREREG)}


def verify_freeze(config_path: Path) -> dict:
    if not FREEZE.exists():
        raise SystemExit("REFUSED: no specification freeze found (docs/research/freeze.json). Run --freeze first.")
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    current = _fingerprints(config_path)
    mismatched = [k for k, v in current.items() if freeze.get(k) != v]
    if mismatched:
        raise SystemExit(f"REFUSED: {mismatched} changed since the freeze at {freeze.get('frozen_at')}. "
                         "The holdout may only be evaluated with the frozen specification.")
    dev_summary = ROOT / freeze["development_run"] / "summary.json"
    if file_sha256(dev_summary) != freeze["development_summary_sha256"]:
        raise SystemExit("REFUSED: the frozen development summary has been modified.")
    return freeze


def freeze(dev_dir: str, config_path: Path) -> None:
    dev = (ROOT / dev_dir).resolve() if not Path(dev_dir).is_absolute() else Path(dev_dir)
    manifest = json.loads((dev / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest["stage"] != "development":
        raise SystemExit("REFUSED: --freeze needs a development run directory.")
    current = _fingerprints(config_path)
    for key in ("code_fingerprint", "config_sha256"):
        if manifest[key] != current[key]:
            raise SystemExit(f"REFUSED: {key} differs from the development run; re-run development first.")
    if "Frozen:" not in PREREG.read_text(encoding="utf-8"):
        raise SystemExit("REFUSED: complete section 9 of PREREGISTRATION.md (add a 'Frozen:' line) before freezing.")
    record = {"frozen_at": _now(), **current, "config_path": str(config_path.relative_to(ROOT)).replace("\\", "/"),
              "development_run": str(dev.relative_to(ROOT)).replace("\\", "/"),
              "development_summary_sha256": file_sha256(dev / "summary.json")}
    FREEZE.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"Specification frozen: {FREEZE}")
    print(json.dumps(record, indent=2))


def run(stage: str, config_path: Path) -> Path:
    config = load_config(config_path)
    if not config.holdout.enforce:
        raise SystemExit("REFUSED: the experiment config must enforce the holdout (holdout.enforce: true).")
    final = stage == "final"
    freeze_record = None
    if final:
        if HOLDOUT_MARKER.exists():
            raise SystemExit(f"REFUSED: the holdout was already evaluated ({HOLDOUT_MARKER}). It is evaluated once.")
        freeze_record = verify_freeze(config_path)
    end = config.holdout.holdout_end if final else config.holdout.development_end
    end = allowed_end_date(config, end, final_evaluation=final)
    first_eval = config.holdout.holdout_start if final else None
    started = _now()

    cache = resolve_path(config.data.cache_dir)
    runs, pair_meta = [], []
    for a, b in config.data.candidate_pairs:
        asset_a = download_yfinance(a, config.data.start_date, end, cache)
        asset_b = download_yfinance(b, config.data.start_date, end, cache)
        result = run_pair(asset_a, asset_b, config, first_eval)
        runs.append(result)
        pair_meta.append({
            "pair": result.label, "evaluation": [result.eval_start, result.eval_end],
            "inputs": [asset_a.metadata(), asset_b.metadata()],
            "aligned_rows": int(len(result.pair.prices)),
            "special_sessions": result.pair.report.special_sessions,
            "special_sessions_dropped": result.pair.report.special_sessions_dropped,
            "windows": int(len(result.wf.windows)), "windows_traded": int(result.wf.windows["traded"].sum()),
        })
        print(f"{result.label}: {result.eval_start}..{result.eval_end}, windows {len(result.wf.windows)}, "
              f"traded {int(result.wf.windows['traded'].sum())}, special sessions dropped "
              f"{result.pair.report.special_sessions_dropped}")

    dailies = [r.wf.daily for r in runs]
    trades = [r.wf.trades for r in runs]
    pooled = pooled_returns(dailies)
    table = variant_table(pooled, gross_pooled_returns(dailies), dailies, trades, config)
    boot = bootstrap_summary(pooled, config.inference, n_candidates=len(CANDIDATES))
    classification = {m: pooled_classification(runs, m, config.ml.horizon, config.ml.calibration_bins) for m in MODEL_NAMES}
    costs = cost_sensitivity(runs, config, config.robustness.cost_grid_bps)
    years = calendar_breakdown(pooled)
    regimes = volatility_regimes(runs, pooled)
    per_pair = {}
    for r in runs:
        single = pooled_returns([r.wf.daily])
        per_pair[r.label] = variant_table(single, gross_pooled_returns([r.wf.daily]), [r.wf.daily], [r.wf.trades], config)

    out = ROOT / "reports" / "experiments" / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{stage}"
    out.mkdir(parents=True, exist_ok=False)
    save_config_snapshot(config, out / "config_snapshot.yaml")
    pooled.rename(columns=VARIANT_LABELS).to_csv(out / "pooled_net_returns.csv")
    table.to_csv(out / "variant_metrics_pooled.csv")
    costs.to_csv(out / "cost_sensitivity.csv", index=False)
    years.to_csv(out / "calendar_years.csv")
    regimes.to_csv(out / "volatility_regimes.csv")
    for r in runs:
        stem = r.label.replace("/", "_").replace(".", "-")
        per_pair[r.label].to_csv(out / f"variant_metrics_{stem}.csv")
        r.wf.windows.to_csv(out / f"windows_{stem}.csv", index=False)
        r.wf.predictions.to_csv(out / f"predictions_{stem}.csv")
        pd.DataFrame({VARIANT_LABELS[k]: d["net_return"] for k, d in r.wf.daily.items()}).to_csv(out / f"net_returns_{stem}.csv")
        for k, t in r.wf.trades.items():
            t.to_csv(out / f"trades_{stem}_{k}.csv", index=False)

    summary = {
        "stage": stage, "evaluation_dates": [str(pooled.index[0].date()), str(pooled.index[-1].date())],
        "pooled_metrics": table.to_dict(), "per_pair_metrics": {k: v.to_dict() for k, v in per_pair.items()},
        "bootstrap": boot, "classification": classification,
        "cost_sensitivity": costs.to_dict(orient="records"), "calendar_years": years.to_dict(),
        "volatility_regimes": regimes.to_dict(),
    }
    (out / "summary.json").write_text(json.dumps(_json(summary), indent=2), encoding="utf-8")
    manifest = {"stage": stage, "started_at": started, "finished_at": _now(), **_fingerprints(config_path),
                "config_path": str(config_path.relative_to(ROOT)).replace("\\", "/"), "data_end": end,
                "first_evaluation_date": first_eval, "pairs": pair_meta, "environment": _environment(),
                "random_seed": config.reproducibility.random_seed, "bootstrap_seed": config.inference.seed,
                "freeze": freeze_record}
    (out / "run_manifest.json").write_text(json.dumps(_json(manifest), indent=2, default=str), encoding="utf-8")

    if final:
        development = json.loads((ROOT / freeze_record["development_run"] / "summary.json").read_text(encoding="utf-8"))
        decision = evaluate_decision(development, _json(summary), costs)
        (out / "decision.json").write_text(json.dumps(_json(decision), indent=2), encoding="utf-8")
        HOLDOUT_MARKER.write_text(json.dumps({"evaluated_at": _now(), "run": str(out.relative_to(ROOT)).replace("\\", "/"),
                                              "decision": decision["overall"]}, indent=2), encoding="utf-8")
        print("DECISION:", decision["overall"])

    with pd.option_context("display.width", 250, "display.max_columns", None, "display.float_format", "{:,.4f}".format):
        print("\nPooled (equal-weight across pairs) net metrics:")
        print(table.loc[["cumulative_return", "gross_cumulative_return", "annualized_volatility", "sharpe_ratio",
                         "sortino_ratio", "max_drawdown", "completed_trades", "pct_time_invested_avg",
                         "annualized_turnover_avg"]])
    for v, ci in boot["sharpe"].items():
        print(f"Sharpe {VARIANT_LABELS[v]:28s} {ci['point']:7.3f}  95% CI [{ci['lower']:.3f}, {ci['upper']:.3f}]")
    for v, d in boot["difference_vs_traditional"].items():
        c, cb = d["ci"], d["ci_bonferroni"]
        print(f"Sharpe diff vs A {VARIANT_LABELS[v][:24]:24s} {c['point']:7.3f}  95% [{c['lower']:.3f}, {c['upper']:.3f}]  "
              f"{boot['bonferroni_confidence']:.2%} [{cb['lower']:.3f}, {cb['upper']:.3f}]")
    for v, c in boot.get("ml_only_vs_prevalence_baseline", {}).items():
        print(f"Sharpe diff vs B0 {VARIANT_LABELS[v][:23]:23s} {c['point']:7.3f}  95% [{c['lower']:.3f}, {c['upper']:.3f}]")
    for m, c in classification.items():
        keys = ("n_rows", "effective_sample_size", "positive_rate", "roc_auc", "brier_skill_score", "log_loss_skill")
        print(m, {k: (round(c[k], 4) if isinstance(c.get(k), float) else c.get(k)) for k in keys})
    print(f"\nOutputs: {out}")
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--stage", choices=["development", "final"])
    group.add_argument("--freeze", metavar="DEVELOPMENT_RUN_DIR")
    args = parser.parse_args(argv)
    config_path = Path(args.config).resolve()
    if args.freeze:
        freeze(args.freeze, config_path)
    else:
        run(args.stage, config_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
