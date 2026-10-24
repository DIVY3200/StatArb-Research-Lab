"""Regenerate the labelled synthetic sample CSVs in data/sample/ (deterministic)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from statarb.data_loader import assets_to_long_csv, generate_synthetic_pair  # noqa: E402


def main() -> None:
    out = ROOT / "data" / "sample"
    coint = generate_synthetic_pair(n_obs=2520, seed=42, cointegrated=True)
    indep = generate_synthetic_pair(n_obs=2520, seed=42, cointegrated=False)
    for name, assets in (("SYNTHETIC_cointegrated_pair.csv", coint), ("SYNTHETIC_independent_pair.csv", indep)):
        path = assets_to_long_csv(list(assets), out / name)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
