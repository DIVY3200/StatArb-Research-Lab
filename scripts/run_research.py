"""Reproducible command-line research run (the notebook-free alternative).

Examples
--------
Offline synthetic demonstration (labelled synthetic data)::

    python scripts/run_research.py --source synthetic

Yahoo Finance download (cached under data/raw/), keeping the test period locked::

    python scripts/run_research.py --source yfinance --tickers KO PEP --start 2015-01-01 --end 2024-12-31

Reveal the final test period once the specification is fixed::

    python scripts/run_research.py --source yfinance --tickers KO PEP --reveal-test

CSV input (one long-format file with a Ticker column, or two single-asset files)::

    python scripts/run_research.py --source csv --csv data/sample/SYNTHETIC_cointegrated_pair.csv \
        --tickers SYNTH_COINT_A SYNTH_COINT_B

When ``ml.enabled`` is true in the config (default), the run also fits the Logistic Regression and
Random Forest reversion classifiers and compares strategies A (traditional), B (ML-only) and
C (hybrid) on identical dates and settings. Use ``--no-ml`` to skip this layer.

Every run writes a snapshot to reports/<timestamp>_<A>_<B>_<config-hash>/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from statarb.config import load_config, resolve_path  # noqa: E402
from statarb.data_loader import DataLoadError, download_yfinance, generate_synthetic_pair, load_csv  # noqa: E402
from statarb.data_validation import prepare_pair_from_config  # noqa: E402
from statarb.evaluation import PERIOD_LABELS, run_pair_research, save_research_run  # noqa: E402
from statarb.ml_evaluation import classification_summary, run_ml_research  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="YAML config (default: config/default_config.yaml)")
    parser.add_argument("--source", choices=["synthetic", "yfinance", "csv"], default=None)
    parser.add_argument("--tickers", nargs=2, metavar=("A", "B"), default=None)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--csv", nargs="+", default=None, help="CSV file(s) for --source csv")
    parser.add_argument("--independent", action="store_true", help="synthetic: generate a NON-cointegrated pair")
    parser.add_argument("--reveal-test", action="store_true", help="evaluate the final test period")
    parser.add_argument("--force-refresh", action="store_true", help="ignore the download cache")
    parser.add_argument("--no-save", action="store_true", help="do not write a reports/ snapshot")
    parser.add_argument("--no-ml", action="store_true", help="skip the machine-learning comparison layer")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    data_overrides = {}
    if args.source:
        data_overrides["source"] = args.source
    if args.tickers:
        data_overrides.update(asset_a=args.tickers[0], asset_b=args.tickers[1])
    if args.start:
        data_overrides["start_date"] = args.start
    if args.end:
        data_overrides["end_date"] = args.end
    if data_overrides:
        config = config.with_overrides({"data": data_overrides})
    d = config.data

    try:
        if d.source == "synthetic":
            s = config.synthetic
            asset_a, asset_b = generate_synthetic_pair(
                n_obs=s.n_observations, seed=s.seed, cointegrated=not args.independent, hedge_ratio=s.hedge_ratio,
                intercept=s.intercept, half_life=s.half_life, residual_vol=s.residual_volatility,
            )
        elif d.source == "yfinance":
            cache = resolve_path(d.cache_dir)
            asset_a = download_yfinance(d.asset_a, d.start_date, d.end_date, cache, args.force_refresh)
            asset_b = download_yfinance(d.asset_b, d.start_date, d.end_date, cache, args.force_refresh)
        else:
            if not args.csv:
                raise DataLoadError("--source csv requires --csv <file> [<file>]")
            assets = {}
            for path in args.csv:
                assets.update(load_csv(path))
            a, b = args.tickers or sorted(assets)[:2]
            asset_a, asset_b = assets[a], assets[b]
        pair = prepare_pair_from_config(asset_a, asset_b, d)
    except (DataLoadError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    result = run_pair_research(pair, config, reveal_test=args.reveal_test)
    print_report(result)
    ml_result = None
    if config.ml.enabled and not args.no_ml:
        ml_result = run_ml_research(result)
        print_ml_report(ml_result)
    if not args.no_save:
        print(f"\nSnapshot written to: {save_research_run(result, ml_result=ml_result)}")
    return 0


def print_report(result) -> None:
    pair, m, c = result.pair, result.model, result.model.coint
    print("=" * 78)
    print(f"Pair: {pair.ticker_a} (A, dependent) vs {pair.ticker_b} (B)   synthetic={pair.is_synthetic}")
    print(f"Common observations: {pair.report.n_common}  ({pair.report.first_common_date} to {pair.report.last_common_date})")
    for name, split in result.splits.items():
        status = "" if name != "test" or result.test_revealed else "  [locked]"
        print(f"  {name:<10} {split.start_date} to {split.end_date}  n={split.n_obs}{status}")
    print("-" * 78)
    print("Training-period estimates")
    print(f"  log A = {m.spread.intercept:.4f} + {m.spread.hedge_ratio:.4f} * log B   (R^2 {m.hedge.r_squared:.3f})")
    print(f"  Engle-Granger t = {c.test_statistic:.3f}, p = {c.p_value:.4f}  -> {c.decision}")
    print(f"  spread mean {m.spread.mean:.2e}, std {m.spread.std:.4f}, half-life {m.half_life:.1f} bars")
    for w in result.warnings:
        print(f"  WARNING: {w}")
    print("-" * 78)
    keys = ["cumulative_return", "gross_cumulative_return", "sharpe_ratio", "max_drawdown", "completed_trades",
            "win_rate", "pct_time_invested", "total_transaction_costs"]
    table = {PERIOD_LABELS[k].split(" (")[0]: {key: p.metrics[key] for key in keys} for k, p in result.periods.items()}
    if result.walk_forward is not None:
        table["Walk-forward OOS"] = {key: result.walk_forward.metrics[key] for key in keys}
    with pd.option_context("display.float_format", "{:,.4f}".format, "display.width", 120):
        print(pd.DataFrame(table))
    if result.selected_params:
        sp = result.selected_params
        print(f"\nValidation-best of {sp['n_candidates']} grid points: entry {sp['entry_z']}, exit {sp['exit_z']} "
              f"({sp['metric']} {sp['metric_value']:.3f}) - selection on validation only.")
    print(f"Test thresholds: {result.test_params}")


def print_ml_report(ml) -> None:
    cfg = ml.config.ml
    print("=" * 78)
    print(f"ML layer: target y_t = 1 if |z(t+{cfg.horizon})| <= |z(t)| - {cfg.tolerance_z}; rows with |z| >= {cfg.min_abs_z}")
    print(ml.sample_table[["start", "end", "rows_used", "positive_rate"]].to_string())
    for report in ml.models.values():
        params = report.tuning.best_params if report.tuning else None
        print(f"  {report.label}: {report.status}; params {params}; thresholds {report.thresholds}")
    for w in ml.warnings:
        print(f"  WARNING: {w}")
    summary = classification_summary(ml)
    if not summary.empty:
        keys = ["n_samples", "positive_rate", "majority_class_accuracy", "accuracy", "precision", "recall", "f1",
                "roc_auc", "brier_score"]
        with pd.option_context("display.float_format", "{:,.3f}".format, "display.width", 250, "display.max_columns", None):
            print("\nClassification (columns: model | period; 'train' is in-sample)")
            print(summary.loc[keys].astype(float))
    keys = ["cumulative_return", "gross_cumulative_return", "sharpe_ratio", "max_drawdown", "completed_trades",
            "total_turnover", "total_transaction_costs"]
    blocks = [(PERIOD_LABELS[p].split(" (")[0], c.table) for p, c in ml.comparisons.items() if p != "train"]
    if ml.walk_forward is not None:
        blocks.append(("Walk-forward OOS", ml.walk_forward.table))
    for title, table in blocks:
        with pd.option_context("display.float_format", "{:,.4f}".format, "display.width", 250, "display.max_columns", None):
            print(f"\nStrategy comparison - {title} (net of costs)")
            print(table.loc[keys])


if __name__ == "__main__":
    raise SystemExit(main())
