# Pre-registration: v2 research protocol

**Registered:** 2026-10-21, before any v2 code was written and before any v2 experiment was run.
**Status:** registered. Amendments are allowed only in §8, with a timestamp and reason, and only *before* the
specification freeze in §9. Every v2 run records the SHA-256 of this file.

## 1. Why a new protocol

The Phase 1 audit (`docs/audit/PHASE1_AUDIT.md`) found that:

* the 2022-12-28 to 2024-12-31 test period has been viewed repeatedly (M1);
* the ML walk-forward reused hyperparameters tuned on later data (D1);
* the ML probabilities were miscalibrated by class weighting (M3);
* special NSE sessions were traded as normal bars (D2);
* no result was accompanied by uncertainty estimates (M7).

All 2015–2024 data is therefore treated as **development data**. Conclusions rest on a new holdout that has not
been downloaded or inspected by this project.

## 2. Hypotheses

* **H-STAT.** The traditional z-score pairs strategy (A), traded only in walk-forward windows whose fitting period
  passes the Engle-Granger test at 5%, earns a positive risk-adjusted return after costs out of sample.
* **H-ML.** Adding a reversion classifier (Logistic Regression or Random Forest), either as a stand-alone signal
  (B) or as an entry filter (C), improves the net risk-adjusted return over A on the same dates under the same
  execution assumptions.

## 3. Data

| Item | Specification |
|---|---|
| Pairs | KO/PEP, HDFCBANK.NS/ICICIBANK.NS, TCS.NS/INFY.NS: the three candidate pairs named in the original project specification, before any data was seen. All three are analysed; none may be dropped after seeing results. |
| Prices | Yahoo Finance `Adj Close` via yfinance (`auto_adjust=False`), as delivered |
| Development data | 2015-01-01 to 2024-12-31 |
| **Holdout (untouched)** | **2025-01-01 to 2026-09-30**. Downloaded and evaluated **once**, only after the freeze in §9. |
| Special sessions | Dropped (`data.special_sessions: drop`): weekend-dated bars, NSE Muhurat dates listed in `config/research_v2.yaml`, and bars where either leg's volume is below 10% of its trailing 60-bar median volume. Every dropped date is reported. |

## 4. Protocol (all variants)

* **Walk-forward:** rolling 504-bar fitting window and 126-bar trading windows. Development evaluation windows run
  from bar 504 to 2024-12-31. Holdout windows start on the first holdout bar, with fitting windows that end
  immediately before it (point-in-time: they use only data available at the time).
* **Gate:** a window is traded only if its fitting window passes Engle-Granger (log A on log B, constant,
  AIC lags) at 5%. The same gate applies to every variant.
* **Spread:** OLS hedge ratio and intercept, with spread mean and std estimated on the fitting window and frozen
  for its trading window.
* **A. Traditional:** entry |z| ≥ 2.0, exit |z| ≤ 0.5 (crossing included), no stop, no maximum holding. Unchanged
  from the baseline.
* **Execution and costs:** decision at close *t*, fill at close *t+1*; 5 + 5 bps per leg per side; borrow 50 bps
  per year on short notional; gross exposure 1.0× equity at entry with drift rebalance at +10%; flat at each window
  start and liquidated at each window end; capital chained across windows; idle cash earns 0.
* **ML target:** y_t = 1 if |z_{t+5}| ≤ |z_t| − 0.25, on rows with |z_t| ≥ 1.0, using the window's frozen spread
  parameters. Labels are purged so that t + 5 ≤ (end of the window that uses them) − 1.
* **Features:** the 17 point-in-time features of `statarb.ml_features` (unchanged).
* **Models:** Logistic Regression (StandardScaler + LR) and Random Forest (200 trees), **unweighted**
  (`class_weight: none`) so that probabilities estimate the reversion frequency.
* **Nested tuning:** in each window, the last 30% of the fitting window is the inner validation block. Grid points
  are fitted on the purged inner-training rows and scored by **log loss** on the purged inner-validation rows. The
  chosen point is refitted on all purged fitting-window rows. No data from the trading window or later is used.
* **B / C threshold:** enter only when p_t ≥ π_fit, the reversion prevalence in that window's purged fitting rows
  ("the model rates reversion as more likely than average"). This is fixed, not tuned.
* **B. ML-only and C. Hybrid:** as implemented in `statarb.ml_strategy` (unchanged rules).

## 5. Metrics and inference

* **Primary portfolio:** for each variant, the equal-weight average of the three pairs' daily net returns on the
  union calendar (0 on a pair's non-trading days).
* **Primary metric:** annualised Sharpe ratio of net daily returns (252 periods, rf = 0).
* **Uncertainty:** stationary bootstrap (mean block 20 bars, 5,000 resamples, seed 20261021) for Sharpe ratios and
  paired Sharpe differences against A.
* **Also reported:** net and gross cumulative return, CAGR, volatility, Sortino, max drawdown, turnover, trades,
  % time invested, average exposure, per-window results, cost sensitivity (0 / 10 / 20 / 40 bps per leg per
  side), calendar-year breakdown.
* **Classification (pooled over windows and pairs, trading-window rows only):** ROC-AUC, Brier score, Brier skill
  score against the prevalence baseline (constant forecast π_fit per window), log loss, calibration bins, positive
  rate, and effective sample size.

## 6. Decision rules (applied to the holdout, after the freeze)

The four ML candidates are B-LR, B-RF, C-LR and C-RF. Bonferroni-adjusted level: 0.05 / 4, i.e. a **98.75% CI**.

1. **Retain an ML enhancement** (a candidate) only if **all** of the following hold:
   * (a) **development** pooled out-of-sample ROC-AUC > 0.55 **and** Brier skill score > 0 for its model;
   * (b) **holdout** primary-portfolio Sharpe(candidate) − Sharpe(A) > 0, with the lower bound of the 98.75%
     bootstrap CI > 0;
   * (c) the holdout net cumulative return of the candidate exceeds A's at **20 bps** per leg per side.
2. **Retain the statistical strategy (A)** as a research candidate (not for deployment) only if the holdout
   primary-portfolio Sharpe has a 95% bootstrap CI lower bound > 0 **and** the development walk-forward Sharpe
   point estimate is > 0.
   * If the holdout point estimate is > 0 but the CI includes 0, the result is "inconclusive: not retained".
3. **Reject the tested hypothesis** for any strategy that meets neither rule.

No decision rule, threshold, pair or metric may be changed after any v2 result has been seen.

## 7. What development results may and may not be used for

* **May:** find and fix implementation defects (logged in §8 with evidence), and check that outputs are complete.
* **May not:** change hypotheses, decision rules, pairs, thresholds, horizons, grids, costs or the holdout dates.

## 8. Amendments (before the freeze only)

**A1, 2026-10-21 (after development run `reports/experiments/20261021T161656Z_development`). Implementation
defect.** The NSE Muhurat date list was applied to every pair, so six regular NYSE sessions were dropped from
KO/PEP (2015-11-11, 2017-10-19, 2018-11-07, 2021-11-04, 2022-10-24, 2024-11-01). Fix: listed special-session
dates become exchange-specific (`data.special_session_dates_by_exchange`, keyed by ticker suffix, e.g. `NS`) and
apply only to pairs that include that exchange. The weekend and volume rules are unchanged. The development run
is repeated with the fix.

**A2, 2026-10-21 (same development run). Omitted baseline restored.** The project specification (Phase 5)
requires evaluation against "a simple prevalence-based prediction baseline", which §4–§6 omitted at the strategy
level. B differs from A in its rules (entry at |z| ≥ 1, horizon exits) as well as in its use of a model, so B's
results cannot be attributed to the model without this control. A diagnostic variant **B0** is added: B's exact
rules driven by the constant prevalence forecast p_t = π_fit, so every eligible bar qualifies. Each B candidate's
paired Sharpe difference against B0 is reported. (For C, the prevalence-forecast counterpart is exactly A, already
the comparator.)

**B0 is not a decision candidate. §6 is unchanged:** the four candidates, the Bonferroni level (0.05/4) and all
gates stay as registered. A2 adds information for interpretation only. Both amendments were made *after* seeing
development results, which this entry discloses.

## 9. Specification freeze

Frozen: 2026-10-21, after development run `reports/experiments/20261021T162712Z_development` (which includes
amendments A1 and A2). The machine-checked record is `docs/research/freeze.json`: SHA-256 of the engine code plus
`config/research_v2.yaml`, of the config, of this file, and of the development summary. `scripts/run_experiment.py
--stage final` refuses to run if any of these change, and refuses a second holdout evaluation.

Disclosure before the holdout is opened: on development data, the pooled out-of-sample ROC-AUC is 0.523 (LR) and
0.515 (RF), and the Brier skill against the prevalence baseline is −0.109 (LR) and −0.016 (RF). Gate 6.1(a)
therefore already fails for both models, so under §6 no ML candidate can be retained whatever the holdout shows.
The holdout still decides rule 6.2 for strategy A and provides the out-of-sample comparison for the report.
