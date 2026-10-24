"""Read-only audit diagnostics: data calendar, corporate actions, EG asymmetry, Johansen."""
import sys, warnings
sys.path.insert(0, sys.argv[1] + "/src")
warnings.simplefilter("ignore")
import numpy as np
from statarb.config import load_config, resolve_path
from statarb.data_loader import download_yfinance
from statarb.data_validation import prepare_pair
from statarb.evaluation import chronological_split
from statarb.cointegration import engle_granger_test
from statsmodels.tsa.vector_ar.vecm import coint_johansen

cfg = load_config()
cache = resolve_path(cfg.data.cache_dir)
a = download_yfinance("TCS.NS", "2015-01-01", "2024-12-31", cache)
b = download_yfinance("INFY.NS", "2015-01-01", "2024-12-31", cache)
print("sources:", a.source, b.source)
for asset in (a, b):
    f = asset.frame
    dow = f.index.dayofweek
    wk = f.index[dow >= 5]
    print(f"\n{asset.ticker}: rows {len(f)}; weekend-dated rows: {len(wk)} -> {[str(d.date()) for d in wk]}")
    vol = f["Volume"]
    low = f.loc[vol < vol.rolling(60, min_periods=20).median() * 0.1]
    print(f"  rows with volume < 10% of 60d median: {len(low)} -> {[str(d.date()) for d in low.index][:12]}")
    r_close = np.log(f["Close"]).diff().abs()
    r_adj = np.log(f["Adj Close"]).diff().abs()
    print(f"  max |log ret| Close {r_close.max():.3f} on {r_close.idxmax().date()}, Adj Close {r_adj.max():.3f} on {r_adj.idxmax().date()}")
    ratio = (f["Adj Close"] / f["Close"])
    steps = ratio.pct_change().abs()
    print(f"  Adj/Close factor steps > 0.5%: {int((steps > 0.005).sum())} (dividend adjustments); largest {steps.max():.3%} on {steps.idxmax().date()}")

pair = prepare_pair(a, b)
p = pair.prices
dow = p.index.dayofweek
print("\naligned rows:", len(p), "| weekend-dated aligned rows:", [str(d.date()) for d in p.index[dow >= 5]])
gaps = p.index.to_series().diff().dt.days
print("calendar gaps > 4 days:", [(str(i.date()), int(g)) for i, g in gaps[gaps > 4].items()])

splits = chronological_split(p.index, 0.6, 0.2, 0.2)
tr = np.log(p.iloc[splits["train"].slice])
ab = engle_granger_test(tr["A"], tr["B"], name_a="TCS", name_b="INFY")
ba = engle_granger_test(tr["B"], tr["A"], name_a="INFY", name_b="TCS")
print(f"\nEG training TCS~INFY: t={ab.test_statistic:.3f} p={ab.p_value:.4f} beta={ab.hedge_ratio:.4f}")
print(f"EG training INFY~TCS: t={ba.test_statistic:.3f} p={ba.p_value:.4f} beta={ba.hedge_ratio:.4f} (1/beta={1/ba.hedge_ratio:.4f})")
print("ADF (levels p, diff p):", ab.adf_levels)
for k_ar in (1, 2, 5):
    jo = coint_johansen(tr[["A", "B"]].to_numpy(), det_order=0, k_ar_diff=k_ar)
    print(f"Johansen (k_ar_diff={k_ar}): trace r=0 stat {jo.lr1[0]:.2f} vs 5% cv {jo.cvt[0,1]:.2f} -> {'reject' if jo.lr1[0] > jo.cvt[0,1] else 'fail'}; "
          f"max-eig {jo.lr2[0]:.2f} vs {jo.cvm[0,1]:.2f}")
for name in ("validation", "test"):
    sl = np.log(p.iloc[splits[name].slice])
    r = engle_granger_test(sl["A"], sl["B"], check_integration=False)
    print(f"EG within {name} only (diagnostic, not for selection): p={r.p_value:.4f} beta={r.hedge_ratio:.3f}")
