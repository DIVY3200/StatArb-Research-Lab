"""Capture golden outputs of the ORIGINAL (pre-audit-fix) pipeline under the default config.

Run once, before any Phase-2+ change:  python tests/data/capture_golden_baseline.py
The JSON it writes is asserted by tests/test_baseline_preserved.py, so default behaviour can
never drift silently. Synthetic data only (offline, deterministic).
"""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from statarb.config import load_config  # noqa: E402
from statarb.data_loader import generate_synthetic_pair  # noqa: E402
from statarb.data_validation import prepare_pair  # noqa: E402
from statarb.evaluation import run_pair_research  # noqa: E402
from statarb.ml_evaluation import run_ml_research  # noqa: E402

BASE_KEYS = ["cumulative_return", "gross_cumulative_return", "sharpe_ratio", "max_drawdown", "completed_trades",
             "total_transaction_costs", "total_borrow_costs", "pct_time_invested"]
CLS_KEYS = ["n_samples", "positive_rate", "accuracy", "precision", "recall", "roc_auc", "brier_score", "log_loss"]


def _f(v):
    v = float(v)
    return None if not math.isfinite(v) else v


def capture() -> dict:
    config = load_config(ROOT / "config" / "default_config.yaml")
    pair = prepare_pair(*generate_synthetic_pair(n_obs=1500, seed=11, cointegrated=True))
    base = run_pair_research(pair, config, reveal_test=True)
    ml = run_ml_research(base)
    out = {
        "base_periods": {p: {k: _f(r.metrics[k]) for k in BASE_KEYS} for p, r in base.periods.items()},
        "base_walk_forward": {k: _f(base.walk_forward.metrics[k]) for k in BASE_KEYS},
        "base_walk_forward_windows": int(len(base.walk_forward.windows)),
        "model_params": {n: r.tuning.best_params for n, r in ml.models.items()},
        "classification": {n: {p: {k: _f(m[k]) for k in CLS_KEYS} for p, m in r.metrics.items()} for n, r in ml.models.items()},
        "comparisons": {p: {c: {k: _f(c_tab.loc[k, c]) for k in ("cumulative_return", "completed_trades", "total_transaction_costs")}
                            for c in c_tab.columns} for p, c_tab in ((p, c.table) for p, c in ml.comparisons.items())},
        "ml_walk_forward": {c: {k: _f(ml.walk_forward.table.loc[k, c]) for k in ("cumulative_return", "completed_trades", "total_transaction_costs")}
                            for c in ml.walk_forward.table.columns},
    }
    return out


if __name__ == "__main__":
    path = Path(__file__).with_name("golden_baseline_synthetic.json")
    path.write_text(json.dumps(capture(), indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {path}")
