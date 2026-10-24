"""Read-only audit diagnostics: Muhurat-session trades, forced exits, Brier skill vs prevalence baseline.

Usage: python -I docs/audit/diagnostics/03_execution_and_calibration.py <project_root> <snapshot_dir>
"""
import glob
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

root, snap = sys.argv[1], sys.argv[2]
muhurat = {"2017-10-19", "2018-11-07", "2019-10-27", "2020-11-14", "2021-11-04", "2022-10-24", "2023-11-12", "2024-11-01"}
prices = pd.read_csv(f"{snap}/aligned_prices.csv", index_col=0, parse_dates=True)
print("Muhurat dates present in aligned data:", sorted(d for d in muhurat if pd.Timestamp(d) in prices.index))
hits = []
for f in sorted(glob.glob(f"{snap}/trades_*.csv")) + [f"{snap}/walk_forward_trades.csv"]:
    t = pd.read_csv(f)
    for col in ("entry_date", "exit_date", "entry_signal_date", "exit_signal_date"):
        if col in t:
            hits += [(f.replace("\\", "/").split("/")[-1], col, d) for d in t[col].dropna().astype(str).str[:10] if d in muhurat]
print("Trades touching Muhurat dates:", hits or "none")

t = pd.read_csv(f"{snap}/walk_forward_trades.csv")
print("A walk-forward exit reasons:", t["exit_reason"].value_counts().to_dict(), "holding days:", t["holding_days"].tolist())
print("Forced end-of-window exits' net P&L / total net P&L:", round(t.loc[t.exit_reason == "end_of_period", "net_pnl"].sum() / t["net_pnl"].sum(), 3))
for p in ("validation", "test"):
    tr = pd.read_csv(f"{snap}/trades_{p}.csv")
    print(f"A {p}: exit reasons {tr['exit_reason'].value_counts().to_dict()}, holding days {tr['holding_days'].tolist()}")

pred = pd.read_csv(f"{snap}/ml_predictions.csv", index_col=0, parse_dates=True)
samples = pd.read_csv(f"{snap}/ml_samples.csv", index_col=0)
base = samples.loc["train", "positive_rate"]
for period, row in samples.iterrows():
    sel = pred.loc[row["first_row"]:row["last_row"]]
    sel = sel[sel["eligible"].astype(str).str.lower().eq("true") & sel["target"].notna() & sel["p_LR"].notna() & sel["p_RF"].notna()]
    y = sel["target"].astype(int).to_numpy()
    assert len(y) == row["rows_used"]
    ref = brier_score_loss(y, np.full(len(y), base))
    out = {"n": len(y), "prevalence": round(float(y.mean()), 3), "Brier(prevalence baseline)": round(ref, 4)}
    for m in ("p_LR", "p_RF"):
        b = brier_score_loss(y, sel[m])
        out[f"Brier {m[2:]}"] = round(b, 4)
        out[f"BSS {m[2:]}"] = round(1 - b / ref, 3)
        out[f"AUC {m[2:]}"] = round(roc_auc_score(y, sel[m]), 3)
    print(period, out)
