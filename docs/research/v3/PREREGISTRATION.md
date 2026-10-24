# Pre-registration: v3 research protocol

**Registered:** 2026-10-24, after the v3 engine options were implemented and tested on synthetic data, and
**before any v3 run on market data**. Configuration: `config/research_v3.yaml`.
**Status:** registered, not frozen. Amendments are allowed only in §8, with a timestamp and reason, and only
before the specification freeze in §9. Every v3 run records the SHA-256 of this file.

## 1. Why a new protocol

The v2 study is closed: its holdout (2025-01-01 to 2026-09-30) was evaluated once, and the decision was "reject
the tested hypothesis" (`docs/research/RESEARCH_REPORT.md`). v2 also left these modelling gaps, now implemented
as opt-in engine options:

* ML training rows used spread parameters fitted on the whole fitting window, including bars after each row,
  while every prediction uses parameters frozen before the bar (train/serve skew).
* No time stop based on the estimated mean-reversion speed.
* Engle-Granger was run in one direction only (log A on log B), and Johansen was not used.
* Fills at the close of *t + 1*, a generic cost model for every market, and cash shorts for NSE stocks. Most
  Indian investors cannot hold multi-day cash shorts, so single-stock futures are the realistic instrument.
* Idle cash earned nothing, and Sharpe ratios used rf = 0.

v3 asks the v2 questions again with these upgrades, on a holdout that **does not exist yet** on the registration
date.

**Disclosure.** The v3 development span (2015-01-01 to 2026-09-30) includes the spent v2 holdout, which the
researcher has seen. v3 development results are therefore not independent of the researcher's knowledge. Only
the v3 holdout is untouched evidence.

## 2. Hypotheses

* **H-STAT.** The traditional z-score pairs strategy (A) earns a positive risk-adjusted excess return after
  costs out of sample. A is traded only in walk-forward windows whose fitting period passes Engle-Granger (log A
  on log B) **and** the Johansen trace test (r = 0) at 5%.
* **H-ML.** A reversion classifier (Logistic Regression or Random Forest) trained on point-in-time rows improves
  A's net risk-adjusted return on the same dates under the same execution assumptions. It is used either as a
  stand-alone signal (B) or as an entry filter (C).
* **Descriptive (development only, no decision):** how much next-open fills, market-specific costs, futures
  shorts and cash interest change A relative to the v2 assumptions.

## 3. Data

| Item | Specification |
|---|---|
| Pairs | KO/PEP, HDFCBANK.NS/ICICIBANK.NS, TCS.NS/INFY.NS (unchanged from v2). All three are analysed; none may be dropped. |
| Prices | Yahoo Finance via yfinance (`auto_adjust=False`), as delivered. `Adj Close` for signals and marks. The adjusted open (`Open × Adj Close / Close`) is used for next-open fills. |
| Development data | 2015-01-01 to 2026-09-30 (includes the spent v2 holdout; see §1) |
| **Holdout (untouched)** | **2026-10-01 to 2028-09-30**. Downloaded and evaluated **once**, only after the freeze in §9 and after 2028-09-30. |
| Special sessions | Dropped as in v2 (weekend-dated bars, the listed NSE Muhurat dates, and bars with either leg's volume below 10% of its trailing 60-bar median). |

## 4. Protocol (all variants)

Unchanged from v2 unless marked **v3**.

* **Walk-forward:** rolling 504-bar fitting windows and 126-bar trading windows. Development windows start at
  bar 504. Holdout windows start on the first holdout bar, with fitting windows that end immediately before it.
* **Gate (v3):** a window is traded only if its fitting window passes Engle-Granger (log A on log B, constant,
  AIC lags) **and** the Johansen trace test for r = 0 (constant, one lagged difference) at 5%. Reverse-direction
  EG is recorded for every window but does not gate. The same gate applies to every variant.
  * Johansen is not used alone. In the project's own simulation, the 5% Johansen test rejected 11–12% of
    independent, weakly drifting random walks (`scripts/johansen_size_study.py`).
* **Spread:** OLS hedge ratio and intercept, with spread mean and std frozen on the fitting window.
* **A. Traditional:**
  * entry |z| ≥ 2.0, exit |z| ≤ 0.5, no stop-loss;
  * **v3 time stop:** exit after ⌈3 × half-life⌉ bars, where the half-life is the fitting window's AR(1)
    half-life (no time stop when it is undefined).
* **Execution (v3):** decision at the close of *t*, fill at the **open** of *t + 1*. The adjusted open is used;
  if an open is unusable, the fill falls back to that bar's close and the run reports it. Gross exposure is
  1.0× equity at entry, with a drift rebalance at +10%. Each window starts flat and is liquidated at the window
  end. Capital is chained across windows.
* **Costs, US pair:** 5 + 5 bps per leg per side; cash shorts with a 50 bps per year borrow fee.
* **Costs, NSE pairs (v3):**
  * Statutory schedule as configured in `markets.NS` of the config:
    * STT: 10 bps on both sides of delivery trades; 2 bps on futures sells;
    * exchange charges: 0.297 bps cash, 0.173 bps futures;
    * SEBI fee: 0.01 bps;
    * stamp duty on buys: 1.5 bps delivery, 0.2 bps futures;
    * GST: 18% on brokerage, exchange and SEBI charges.
  * Other charges:
    * Rs 20 per futures order (before GST);
    * Rs 15.93 depository charge per delivery sell;
    * 5 bps slippage.
  * **Shorts are single-stock futures:** no borrow fee, a roll every 21 bars (futures buy + sell + 2 bps
    calendar spread), and margin at 20% of notional, reported but not binding.
  * The late-2024 schedule is applied to the whole sample. This is conservative for futures STT before
    October 2024.
  * Lot sizes are **not** modelled. NSE revised them several times during the sample (and HDFCBANK had a bonus
    issue), so one constant lot size would misstate historical contract values.
* **Cash and risk-free rate (v3):** cash interest is credited daily on equity − long notional + short notional
  at the market's calendar-year rate (`metrics.risk_free_by_year`):
  * US: approximate 3-month T-bill averages;
  * NSE pairs: approximate 91-day T-bill averages.
  * Years after 2026 reuse the 2026 rate.
  * These values are approximate and researcher-entered. Because both the interest credit and the Sharpe
    benchmark use them, excess-return Sharpe ratios are insensitive to their level; §5 reports a check.
* **ML target, features, models, nested tuning, prevalence threshold, B, C and B0:** as in v2 (§4 and amendment
  A2 of `docs/research/PREREGISTRATION.md`), with one change:
  * **v3 point-in-time training rows:** each 21-bar block of a fitting window gets spread parameters estimated
    only on the 252 bars before the block. That block's features and labels are computed with those
    parameters. Labels are purged at the window edge as before.
  * Scored rows (inner validation in nested tuning, and the trading window) use the frozen parameters as before.

## 5. Metrics and inference

* **Primary portfolio:** for each variant, the equal-weight average of the three pairs' daily net returns on the
  union calendar (0 on a pair's non-trading days).
* **Primary metric (v3):** the annualised Sharpe ratio of the primary portfolio's daily **excess** returns. Each
  pair's daily net return has its own market's daily risk-free rate subtracted on that pair's trading days
  before pooling.
* **Uncertainty:** stationary bootstrap (mean block 20 bars, 5,000 resamples, seed 20261024) for Sharpe ratios
  and paired Sharpe differences against A.
* **Also reported:**
  * net and gross cumulative return, CAGR, volatility, Sortino (on excess returns), max drawdown;
  * turnover, trades, % time invested, average exposure;
  * per-window results, including every window's EG (both directions) and Johansen statistics;
  * cost sensitivity at 0 / 10 / 20 / 40 bps per leg per side (US: commission + slippage; NSE: slippage on top
    of statutory charges);
  * calendar-year breakdown.
* **Classification:** as in v2 (pooled ROC-AUC, Brier and log-loss skill against the prevalence baseline,
  calibration, effective sample size).
* **Development-only checks (descriptive):**
  * A re-priced with cash interest switched off, to show the risk-free check above;
  * A re-run under the v2 execution and cost assumptions, to measure what the realism upgrades change.

## 6. Decision rules (applied to the holdout, after the freeze)

Identical to v2 §6. The only differences: the Sharpe ratios are the excess-return Sharpe ratios of §5, and the
20 bps level in 1(c) is read per market as defined in §5.

The four ML candidates are B-LR, B-RF, C-LR and C-RF. The Bonferroni-adjusted level is 0.05 / 4, i.e. a
**98.75% CI**.

1. **Retain an ML enhancement** (a candidate) only if **all** of the following hold:
   * (a) its model's **development** pooled out-of-sample ROC-AUC is > 0.55 **and** its Brier skill score is > 0;
   * (b) on the **holdout**, Sharpe(candidate) − Sharpe(A) for the primary portfolio is > 0, and the lower bound
     of its 98.75% bootstrap CI is > 0;
   * (c) on the holdout, the candidate's net cumulative return exceeds A's at **20 bps** per leg per side.
2. **Retain the statistical strategy (A)** as a research candidate (not for deployment) only if both hold:
   * its holdout primary-portfolio Sharpe has a 95% bootstrap CI lower bound > 0;
   * its development walk-forward Sharpe point estimate is > 0.

   If the holdout point estimate is > 0 but the CI includes 0, the result is "inconclusive: not retained".
3. **Reject the tested hypothesis** for any strategy that meets neither rule.

No decision rule, threshold, pair, metric, cost or rate may be changed after any v3 result has been seen.

## 7. What development results may and may not be used for

* **May:** find and fix implementation defects (logged in §8 with evidence), and check that outputs are complete.
* **May not:** change hypotheses, decision rules, pairs, thresholds, horizons, grids, costs, rates, the gate, the
  time stop, the point-in-time settings or the holdout dates.

## 8. Amendments (before the freeze only)

None.

## 9. Specification freeze

Not frozen yet. To freeze after a complete development run:

1. Record any amendments in §8.
2. Add a line starting `Frozen:` here, with the date and the development run directory.
3. Run `python scripts/run_experiment.py --config config/research_v3.yaml --freeze reports/experiments/<run>`.
   This writes `docs/research/v3/freeze.json`.

The final stage (`--stage final`) refuses to run until the holdout end date has passed, if anything has changed
since the freeze, or if the holdout was already evaluated.
