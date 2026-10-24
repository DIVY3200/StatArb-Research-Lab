"""Read-only audit diagnostics: in-sample vs OOS label base rates, effective sample size, bootstrap CIs."""
import sys, warnings
root = sys.argv[1]; snap = sys.argv[2]
sys.path.insert(0, root + "/src")
warnings.simplefilter("ignore")
import numpy as np, pandas as pd
from statarb.config import load_config, resolve_path
from statarb.data_loader import download_yfinance
from statarb.data_validation import prepare_pair
from statarb.evaluation import chronological_split, fit_pair_model, walk_forward_schedule
from statarb.ml_features import reversion_target

cfg = load_config(); ml = cfg.ml
cache = resolve_path(cfg.data.cache_dir)
pair = prepare_pair(download_yfinance("TCS.NS", "2015-01-01", "2024-12-31", cache),
                    download_yfinance("INFY.NS", "2015-01-01", "2024-12-31", cache))
p = pair.prices; logp = np.log(p)
splits = chronological_split(p.index, 0.6, 0.2, 0.2)
tr = splits["train"]

# (1) In-sample vs out-of-sample label base rate on the SAME training-period dates
static = fit_pair_model(logp.iloc[tr.slice], cfg).spread
z_in = static.zscore(logp["A"], logp["B"])
y_in = reversion_target(z_in, ml.horizon, ml.tolerance_z)
oos_rows = []
for w in walk_forward_schedule(len(p), 504, 126):
    if w.eval_start_idx >= tr.end_idx:
        break
    m = fit_pair_model(logp.iloc[w.fit_start_idx:w.fit_end_idx], cfg).spread
    z = m.zscore(logp["A"], logp["B"])
    y = reversion_target(z, ml.horizon, ml.tolerance_z)
    for i in range(w.eval_start_idx, min(w.eval_end_idx, tr.end_idx - ml.horizon)):
        oos_rows.append((p.index[i], z.iloc[i], y.iloc[i]))
oos = pd.DataFrame(oos_rows, columns=["date", "z", "y"]).set_index("date")
oos = oos[oos["z"].abs() >= ml.min_abs_z]
ins = pd.DataFrame({"z": z_in, "y": y_in}).loc[oos.index.min():p.index[tr.end_idx - 1 - ml.horizon]]
ins = ins[ins["z"].abs() >= ml.min_abs_z]
print(f"(1) Training-period dates {oos.index.min().date()}..{p.index[tr.end_idx-1-ml.horizon].date()}:")
print(f"    in-sample z (static params fit on 2015-2020):  eligible rows {len(ins)}, positive rate {ins['y'].mean():.3f}, mean |z| {ins['z'].abs().mean():.2f}")
print(f"    out-of-sample z (rolling 504-bar params):        eligible rows {len(oos)}, positive rate {oos['y'].mean():.3f}, mean |z| {oos['z'].abs().mean():.2f}")
print(f"    share of bars eligible: in-sample {len(ins)/len(pd.DataFrame({'z': z_in}).loc[oos.index.min():p.index[tr.end_idx-1-ml.horizon]]):.2f}")

# (2) Effective sample size of overlapping labels (training rows, in-sample design)
y = y_in[(z_in.abs() >= ml.min_abs_z)].iloc[: tr.end_idx].dropna()
y = y.loc[: p.index[tr.end_idx - 1 - ml.horizon]]
x = y.to_numpy() - y.mean()
rho = [np.dot(x[:-k], x[k:]) / np.dot(x, x) for k in range(1, 2 * ml.horizon + 1)]
n_eff = len(x) / (1 + 2 * sum(r for r in rho if r > 0))
print(f"(2) Training labels: n={len(x)}, lag-1..5 autocorr {np.round(rho[:5], 2).tolist()}, n_eff ~ {n_eff:.0f} (n/H = {len(x)/ml.horizon:.0f})")

# (3) Stationary bootstrap of walk-forward daily net returns (paired), Sharpe and Sharpe difference
r = pd.read_csv(f"{snap}/ml_walk_forward_returns.csv", index_col=0, parse_dates=True)
cols = list(r.columns)
rng = np.random.default_rng(20261021)
def sharpe(v):
    sd = v.std(ddof=1)
    return np.nan if sd < 1e-12 else v.mean() / sd * np.sqrt(252)
def stationary_idx(n, mean_block, rng):
    idx = np.empty(n, dtype=int); i = rng.integers(n)
    for t in range(n):
        idx[t] = i
        i = rng.integers(n) if rng.random() < 1 / mean_block else (i + 1) % n
    return idx
R = r.to_numpy()
B = 2000
boot = np.empty((B, len(cols)))
for b_ in range(B):
    ix = stationary_idx(len(R), 20, rng)
    boot[b_] = [sharpe(R[ix, j]) for j in range(len(cols))]
print("(3) Walk-forward Sharpe, stationary bootstrap (mean block 20 days, 2000 resamples), 95% CI:")
for j, c in enumerate(cols):
    lo, hi = np.nanpercentile(boot[:, j], [2.5, 97.5])
    print(f"    {c:28s} point {sharpe(R[:, j]):6.3f}  CI [{lo:6.3f}, {hi:6.3f}]")
for j, c in enumerate(cols[1:], start=1):
    d = boot[:, j] - boot[:, 0]
    lo, hi = np.nanpercentile(d, [2.5, 97.5])
    print(f"    Sharpe difference {c} - A: point {sharpe(R[:, j]) - sharpe(R[:, 0]):6.3f}  CI [{lo:6.3f}, {hi:6.3f}]  P(diff>0)={np.nanmean(d > 0):.2f}")

# (4) Fold-by-fold: traded windows where each variant beat A
w = pd.read_csv(f"{snap}/ml_walk_forward_windows.csv", index_col=0)
traded = w[w["traded"]]
for v in ("ml_only_logistic_regression", "hybrid_logistic_regression", "ml_only_random_forest", "hybrid_random_forest"):
    better = (traded[f"return_{v}"] > traded["return_traditional"] + 1e-12).sum()
    worse = (traded[f"return_{v}"] < traded["return_traditional"] - 1e-12).sum()
    print(f"(4) {v}: better than A in {better}/{len(traded)} traded windows, worse in {worse}")

# (5) Hyperparameter-selection overlap: WF eval windows relative to the static validation period
val_start, val_end = splits["validation"].start_date, splits["validation"].end_date
before = (pd.to_datetime(w["eval_end"]) < pd.Timestamp(val_start)).sum()
overlap = ((pd.to_datetime(w["eval_start"]) <= pd.Timestamp(val_end)) & (pd.to_datetime(w["eval_end"]) >= pd.Timestamp(val_start))).sum()
print(f"(5) Static validation period {val_start}..{val_end}: WF windows entirely before it: {before}; overlapping it: {overlap}; of {len(w)}")

# (6) Cost sensitivity of A (from the snapshot)
for per in ("validation", "test"):
    c = pd.read_csv(f"{snap}/cost_sensitivity_{per}.csv", index_col=0)
    print(f"(6) A cost sensitivity {per}:", {f"{b:g}bps": round(v, 4) for b, v in zip(c['cost_bps_per_leg_per_side'], c['cumulative_return'])})
