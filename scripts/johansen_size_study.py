"""Empirical size of the Johansen trace test versus Engle-Granger on independent random walks.

Reproduces the size caveat documented in ``statarb.cointegration.johansen_test`` and in the v3
pre-registration: with the weak drift typical of daily equity log prices, statsmodels' unrestricted-constant
Johansen test (``det_order=0``) rejects "no cointegration" far more often than its nominal level.

    python scripts/johansen_size_study.py [--runs 600] [--bars 756] [--seed 0]

Each run simulates two independent Gaussian random walks with drift (so no cointegrating relation exists)
and records whether each test rejects at 5%. 600 runs per drift level take a few minutes.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from statarb.cointegration import engle_granger_test, johansen_test  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=int, default=600)
    parser.add_argument("--bars", type=int, default=756)
    parser.add_argument("--vol", type=float, default=0.015, help="daily log-return volatility of each walk")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    rng = np.random.default_rng(args.seed)
    index = pd.bdate_range("2015-01-01", periods=args.bars)
    rows = []
    for drift in (0.0, 0.0002, 0.0005, 0.002):
        eg = jo = both = 0
        for _ in range(args.runs):
            steps = drift + args.vol * rng.standard_normal((args.bars, 2))
            logs = pd.DataFrame(np.cumsum(steps, axis=0) + np.log([60.0, 50.0]), index=index, columns=["A", "B"])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                eg_reject = engle_granger_test(logs["A"], logs["B"], check_integration=False).reject_null
                jo_reject = johansen_test(logs["A"], logs["B"]).reject_r0
            eg += eg_reject
            jo += jo_reject
            both += eg_reject and jo_reject
        rows.append({"drift_per_bar": drift, "annual_drift": (1 + drift) ** 252 - 1, "eg_size": eg / args.runs,
                     "johansen_size": jo / args.runs, "eg_and_johansen_size": both / args.runs})
        print(rows[-1], flush=True)
    table = pd.DataFrame(rows)
    se = np.sqrt(0.05 * 0.95 / args.runs)
    print(f"\nNominal level 5%; Monte Carlo standard error at 5% is {se:.3f} with {args.runs} runs.")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
