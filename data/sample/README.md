# Sample data: SYNTHETIC ONLY

The CSV files in this folder are **generated**, not market history. They exist for offline
demonstrations and deterministic tests, and must never be presented as real results.

| File | Contents |
|---|---|
| `SYNTHETIC_cointegrated_pair.csv` | `SYNTH_COINT_A`, `SYNTH_COINT_B`: `log A = 0.4 + 0.8 log B + u`, with `u` a stationary AR(1) process (half-life 15 bars). |
| `SYNTHETIC_independent_pair.csv` | `SYNTH_INDEP_A`, `SYNTH_INDEP_B`: two independent random walks (no cointegration). |

Both files are long format (`Date, Ticker, Open, High, Low, Close, Adj Close, Volume`), with 2,520
business days starting 2015-01-02 and seed 42. The dates are business days, not an exchange calendar.
Regenerate them with:

```bash
python scripts/generate_sample_data.py
```

Any ticker starting with `SYNTH_`, or any file whose name contains `synthetic`, is flagged as
synthetic by the loader. The dashboard then shows a SYNTHETIC DATA banner.
