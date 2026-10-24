# v3 development report

**Date:** 2026-10-25.
**Protocol:** [`PREREGISTRATION.md`](PREREGISTRATION.md) (registered 2026-10-24, before this run).
**Config:** `config/research_v3.yaml`.
**Status:** development complete; **not frozen**. The holdout (2026-10-01 to 2028-09-30) does not exist yet and has
not been touched.

This report covers development data only. Under the pre-registration, development results may be used to find
defects but not to change the specification. **No holdout conclusion can be drawn from it.**

## 1. Run

| Item | Value |
|---|---|
| Run | `reports/experiments/20261024T183817Z_development_v3` (committed copy: [`results/`](results/)) |
| Data | Yahoo Finance via yfinance, 2015-01-01 to 2026-09-30 (cached download; input hashes in `run_manifest.json`) |
| Evaluation windows | 2017-01-03 to 2026-09-30: 20 windows for KO/PEP, 19 for each NSE pair (126-bar trading windows after 504-bar fits) |
| Special sessions dropped | KO/PEP 0, HDFCBANK/ICICIBANK 12, TCS/INFY 11 |
| Code fingerprint | `047aa5dc1b14455ca5f807cf5e85bff3b4e5b078507b6f0f9903a822df5fd4b3` |
| Config SHA-256 | `91651ecd376e618f152d75095a479cec4c79232f3927102842d6d509a5acd760` |
| Pre-registration SHA-256 | `831908abf30c75cab9fb6a00561ac51cb256b96ef2fe0be5362cb1736e8a9e68` |
| Environment | Python 3.13.7, numpy 2.5.4, pandas 3.0.6, scipy 1.18.1, statsmodels 0.15.0, scikit-learn 1.9.1, Windows 11 |

**Disclosure.** An earlier v3 development run (2026-10-24 17:09 UTC) exposed an accounting defect.

* **The defect:** the first bar of each walk-forward window earned no cash interest while the risk-free rate was
  still subtracted.
* **Its effect:** a pair that only ever held cash showed a Sharpe ratio of −1.19 instead of an undefined one.
* **The fix:**
  * chained windows now accrue a flat day's interest on their first bar;
  * the starting placeholder of a backtest or benchmark series is no longer charged the rate.
* **Tests:**
  * `tests/test_realism.py::test_flat_period_earns_exactly_the_risk_free_rate`;
  * `::test_flat_chained_walk_forward_has_zero_excess_return`.
* **Scope:** neither the hypotheses nor any setting changed. The run was repeated with the fix (the run above),
  and the defective run was discarded.

## 2. Results (development, pooled equal-weight portfolio of the three pairs)

Sharpe ratios are annualised and computed on daily **excess** returns (each pair minus its market's risk-free rate),
with 95% stationary-bootstrap intervals (5,000 resamples, mean block 20).

| Variant | Sharpe [95% CI] | Net cumulative return | Completed trades | Time invested (avg) |
|---|---|---|---|---|
| A. Traditional | 0.023 [−0.650, 0.628] | 52.5% | 10 | 4.0% |
| B. ML-only (LR) | 0.385 [−0.205, 0.888] | 63.0% | 35 | 4.1% |
| C. Hybrid (LR) | 0.220 [−0.506, 0.825] | 58.0% | 9 | 3.5% |
| B. ML-only (RF) | 0.172 [−0.394, 0.712] | 57.1% | 30 | 4.8% |
| C. Hybrid (RF) | 0.218 [−0.491, 0.799] | 57.7% | 8 | 3.3% |
| B0. B's rules, no model | 0.062 [−0.501, 0.615] | 53.9% | 23 | 6.6% |

**The cumulative returns are almost entirely cash interest.** The book is flat about 96% of the time, and idle cash
earns the T-bill rate (averaged over 2017–2026: US about 2.5% a year, India about 5.6%). With cash interest switched off, A's
cumulative return over the same period is −0.1% (§4).

Paired Sharpe differences against A, with the pre-registered Bonferroni level for the four candidates (98.75%):

| Candidate | Difference | 95% CI | 98.75% CI |
|---|---|---|---|
| B-LR | +0.362 | [−0.049, 0.822] | [−0.135, 0.952] |
| C-LR | +0.197 | [−0.024, 0.501] | [−0.061, 0.586] |
| B-RF | +0.149 | [−0.173, 0.498] | [−0.249, 0.596] |
| C-RF | +0.195 | [−0.126, 0.544] | [−0.207, 0.628] |

Against the B0 control: B-LR − B0 = +0.323 [−0.051, 0.779]; B-RF − B0 = +0.109 [−0.167, 0.432].

**Classification (pooled out-of-sample rows, 407 rows, effective sample size about 112):**

| Model | ROC-AUC | Brier skill vs prevalence | Log-loss skill | Mean predicted vs realised rate |
|---|---|---|---|---|
| Logistic Regression | 0.502 | −0.085 | −0.077 | 0.46 vs 0.35 |
| Random Forest | 0.500 | −0.008 | −0.004 | 0.46 vs 0.35 |

## 3. Consequences under the pre-registered rules

* **Gate 6.1(a) fails for both models**: development ROC-AUC is not above 0.55, and Brier skill is negative. Under
  §6, therefore, **no ML candidate can be retained, whatever the holdout shows**. This is disclosed now, before any
  holdout data exists, as v2 did before its holdout.
  * Point-in-time training rows did not create predictive skill. Both models rank reversions no better than chance.
  * B-LR's higher Sharpe point estimate therefore cannot be attributed to prediction. Its intervals include zero,
    both against A and against B0.
* **Strategy A:** the development Sharpe point estimate is above 0 (0.023), so the development half of rule 6.2 is
  met. A is retained as a research candidate only if its holdout Sharpe has a 95% CI lower bound above 0.

## 4. Diagnostics

### Cointegration gate

The Engle-Granger + Johansen gate traded **6 of 58** pair-windows:

| Pair | Windows | EG (A on B) passes | Johansen (r = 0) passes | Traded (both) |
|---|---|---|---|---|
| KO / PEP | 20 | 0 | 0 | 0 |
| HDFCBANK.NS / ICICIBANK.NS | 19 | 1 | 4 | 1 |
| TCS.NS / INFY.NS | 19 | 9 | 7 | 5 |

Engle-Granger alone would have traded 10 pair-windows; the Johansen requirement removed 4 TCS/INFY windows. Five
windows passed Johansen but not Engle-Granger. Four of them reported rank 2 ("both series stationary"), consistent
with the test's oversizing (§4 of the pre-registration). Every window's statistics are in `windows_*.csv`.

### Johansen size (`scripts/johansen_size_study.py`, 600 independent random-walk pairs per row, 756 bars)

| Drift per bar | Engle-Granger size | Johansen size | EG and Johansen size |
|---|---|---|---|
| 0 | 4.8% | 11.8% | 2.8% |
| 0.0002 (about 5% a year) | 5.0% | 11.2% | 2.8% |
| 0.0005 (about 13% a year) | 5.5% | 11.8% | 3.8% |
| 0.002 | 5.8% | 5.8% | 3.3% |

The nominal level is 5%, with a Monte Carlo standard error of 0.9 percentage points.

### Assumption checks (same signals and models, re-priced; `assumption_checks.csv`)

| Scenario | A Sharpe | A cumulative | B-LR Sharpe | B-LR cumulative |
|---|---|---|---|---|
| As run (excess returns) | 0.023 | 52.5% | 0.385 | 63.0% |
| Cash interest off (net, rf = 0) | 0.003 | −0.1% | 0.355 | 6.6% |
| v2 execution and costs (close fills, 5 + 5 bps, cash shorts, no interest) | 0.021 | 0.2% | 0.523 | 9.8% |

* **The rate schedule barely moves the excess-return Sharpe ratios.** Interest income and the subtracted rate cancel
  for idle capital, as the pre-registration anticipated.
* **The execution and cost upgrades matter for the active strategies.** Next-open fills, NSE statutory costs and
  futures rolls lower B-LR's Sharpe from 0.52 to 0.36–0.39. A trades too rarely for them to change its result.

### Cost sensitivity (excess-return Sharpe; US: commission + slippage; NSE: slippage on top of statutory charges)

| bps per leg per side | 0 | 10 | 20 | 40 |
|---|---|---|---|---|
| A. Traditional | 0.049 | −0.003 | −0.055 | −0.157 |
| B. ML-only (LR) | 0.455 | 0.316 | 0.176 | −0.101 |
| C. Hybrid (LR) | 0.246 | 0.194 | 0.142 | 0.039 |
| B. ML-only (RF) | 0.229 | 0.114 | 0.000 | −0.224 |
| C. Hybrid (RF) | 0.242 | 0.194 | 0.146 | 0.051 |

## 5. Limitations

* **Very low power.** The strategies are invested about 4% of the time, and 6 traded pair-windows cannot confirm
  or reject a modest edge.
* **Development data are not independent of the researcher.** They include the spent v2 holdout (2025-01-01 to
  2026-09-30).
* **Approximations:**
  * risk-free rates are approximate, researcher-entered annual averages;
  * NSE charges use the late-2024 schedule for the whole sample;
  * futures lot sizes and basis are not modelled.
* **Scope:** three hand-picked large-cap pairs. Results do not generalise, and this is not financial advice.

## 6. Next steps

1. Optionally record amendments in §8 of the pre-registration (none so far).
2. Freeze the specification: add a `Frozen:` line, then run
   `python scripts/run_experiment.py --config config/research_v3.yaml --freeze reports/experiments/20261024T183817Z_development_v3`.
   Any later change to the engine code or config would require a new development run first.
3. After 2028-09-30, run `--stage final` once. The runner refuses to do so earlier, a second time, or after any
   change since the freeze.
