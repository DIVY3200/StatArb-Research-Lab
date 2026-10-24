# Phase 1 audit: StatArb Research Lab (statistical + ML layers)

**Date:** 2026-10-21. **Scope:** the whole repository as of 2026-10-20 22:51 local time, before any changes.
**Rule followed:** no source, test, config or notebook file was modified for this audit. Only `docs/audit/` was
added, and one new run snapshot was written to `reports/` (no existing snapshot was overwritten).

Evidence files in this folder:

| File | Contents |
|---|---|
| `baseline_pytest.txt` | Verbatim output of the baseline test run |
| `baseline_tcs_infy_2015_2024.json` | Machine-readable baseline: inputs and hashes, splits, config, environment, warnings, model status, every A/B/C result |
| `diagnostics/01_data_and_statistics.py` | Calendar and corporate-action checks, Engle-Granger in both directions, Johansen |
| `diagnostics/02_ml_evaluation.py` | In-sample vs out-of-sample label rates, effective sample size, stationary bootstrap, fold-by-fold comparison, tuning overlap, cost sensitivity |
| `diagnostics/03_execution_and_calibration.py` | Muhurat-session trades, forced exits, Brier skill vs a prevalence baseline |

Run the diagnostics with `python -I docs/audit/diagnostics/<script> <project_root> [<snapshot_dir>]`, using the
snapshot `reports/20261021T152215Z_TCS-NS_INFY-NS_0d0c9cc8b701`.

---

## 1. Baseline

### 1.1 Environment and tests

* Python 3.13.7 (Windows 11); numpy 2.5.4, pandas 3.0.6, scipy 1.18.1, statsmodels 0.15.0, scikit-learn 1.9.1,
  yfinance 1.7.0, plotly 7.1.0, streamlit 1.66.0. The full lock is in `requirements-lock.txt`.
* `pytest -v`: **175 passed, 0 failed, 0 errors** in 145 s (`baseline_pytest.txt`).

### 1.2 Reproduction of the reported TCS.NS / INFY.NS experiment

Command, with settings unchanged from the original run:

```
python scripts/run_research.py --source yfinance --tickers TCS.NS INFY.NS --start 2015-01-01 --end 2024-12-31 --reveal-test
```

* Data came from the local cache, originally retrieved 2026-10-20T14:02:09Z. The SHA-256 of both input files and
  of the aligned prices matches the original run, and the config fingerprint is the same (`0d0c9cc8b701`).
* **The reproduction is exact:** all 62 output tables match the original snapshot
  (`reports/20261020T171002Z_...`) within rtol 1e-12.

| Item | Value |
|---|---|
| Aligned sample | 2,467 common bars, 2015-01-01 to 2024-12-31, `Adj Close` for both legs, 0 rows dropped |
| Train / validation / test | 2015-01-01–2020-12-31 (1,480) / 2021-01-01–2022-12-27 (493) / 2022-12-28–2024-12-31 (494) |
| Training fit | log TCS = 0.5059 + 1.0727 · log INFY, R² 0.894; spread std 0.0990; half-life 48.3 bars |
| Engle-Granger (training) | t = −3.124, **p = 0.0835 → fails at 5%** (warning issued) |
| Strategy | entry \|z\| 2.0, exit 0.5, no stop, no max holding; execution lag 1 bar (close t → close t+1) |
| Costs and sizing | 5 bps commission + 5 bps slippage per leg per side; borrow 50 bps/yr; gross 1.0× equity; drift rebalance at +10%; liquidate at each period end |
| Walk-forward | rolling 504-bar fit, 126-bar trade windows, trade only if the fit window passes EG; 16 windows, 9 traded |
| ML | H = 5, tau = 0.25, rows with \|z\| ≥ 1; LR selected C = 10; RF selected max_depth 3, leaf 10; threshold 0.5 (pre-specified); 560 / 486 / 400 rows; no model skipped; no ML warnings |

Net cumulative return, with completed trades in brackets:

| Period | A. Traditional | B. ML-only (LR) | C. Hybrid (LR) | B. ML-only (RF) | C. Hybrid (RF) |
|---|---|---|---|---|---|
| Training (in-sample) | +18.4% (3) | +74.1% (41) | +18.4% (3) | +63.6% (38) | +18.4% (3) |
| Validation | −7.6% (1) | −7.5% (7) | −4.3% (1) | −6.0% (16) | −4.3% (1) |
| Final test | +6.9% (2) | +6.8% (9) | +6.8% (2) | −2.0% (15) | +6.8% (2) |
| Walk-forward | +21.5% (10) | +33.1% (25) | +24.3% (10) | +25.4% (24) | +22.2% (10) |

Classification: test ROC-AUC LR 0.564, RF 0.533. Test accuracy LR 0.588, RF 0.588, against a majority-class
accuracy of 0.860. Walk-forward windows 7 and 9 have no defined ROC-AUC (fewer than 30 rows or a single class).

---

## 2. How each quantity is computed (trace)

| Stage | Function | Data window | Verified |
|---|---|---|---|
| Prices | `download_yfinance` (`auto_adjust=False`) → `Adj Close` used as delivered | full download | Splits and bonuses adjusted (no jumps in Close or Adj Close); adjustment-factor steps above 0.5%: 12 (TCS) / 21 (INFY), consistent with dividends |
| Cleaning, alignment | `validate_price_frame`, `align_pair` (inner join, no fill) | full | 0 dropped rows; holiday gaps ≤ 5 days |
| Split | `chronological_split` (60/20/20 by bar count) | — | Contiguous, non-overlapping (tests) |
| Hedge ratio, intercept | `estimate_hedge_ratio` (OLS log A on log B) | training (static) or each fit window (WF) | Unchanged when later prices change (tests) |
| Cointegration | `engle_granger_test` → `statsmodels.coint` (MacKinnon) | same as hedge ratio | Correct residual-based critical values |
| Spread, z-score | `SpreadModel.spread/zscore` (frozen mean and std, ddof 1) | applied forward | Hand-calculated tests |
| Signals A | `generate_signals` finite-state machine | per period, starts flat | Tests for direction, no duplicates, no same-bar reversals |
| Features | `build_features`: trailing windows and lags only | point-in-time given frozen spread parameters | Truncation-invariance test |
| Labels | `reversion_target`: 1{\|z_{t+5}\| ≤ \|z_t\| − 0.25} | needs t+5 | Hand-calculated tests |
| Fit rows | `MLDataset.rows`: t + 5 ≤ window_end − 1 − embargo | training / fit window | Purge tests; LR coefficients identical when validation prices change |
| Models | sklearn pipelines; scaler fitted on fit rows | training / fit window | Scaler-mean test |
| Signals B, C | `ml_only_signals`, `hybrid_signals` (decide at close t) | per period | Unit tests |
| Execution, accounting | `run_backtest`: mark-to-market on prior units, then trade at close t + lag | per period / window | Accounting identity, both-leg cost, borrow and timing tests |
| Metrics | `strategy_metrics`, `comparison_table` | per period | Hand-calculated tests |

---

## 3. Findings

Severity: **High** means it can change a conclusion; **Medium** means it biases results or misleads
interpretation; **Low** is a correctness or hygiene issue with small impact.

### 3.1 Confirmed defects (the code does not do what the methodology requires)

| ID | Severity | Finding | Evidence |
|---|---|---|---|
| D1 | **High** | **Hyperparameter look-ahead in the ML walk-forward.** `run_ml_walk_forward` reuses hyperparameters tuned once on the static validation period (2021-01-01–2022-12-27) in every window. 7 of 16 windows are evaluated entirely *before* that period and 5 overlap it. Thresholds were the pre-specified 0.5 in the baseline, so only hyperparameters leak; under `threshold_source: validation_selected` thresholds would leak too. | `ml_evaluation.py` (`params = r.tuning.best_params`); `02_ml_evaluation.py` §5 |
| D2 | Medium | **Diwali Muhurat sessions are treated as normal tradable bars.** These special ~1-hour sessions (7 in the aligned data, including Sunday 2019-10-27 and Saturday 2020-11-14) are used for signals, labels, horizons and fills. One trade (training, B-LR) entered at the 2017-10-19 Muhurat close. Validation flags neither weekend dates nor these sessions. | `01_data_and_statistics.py`; `03_execution_and_calibration.py` |
| D3 | Low | **`select_threshold` silently treats `selection_metric: calmar` as Sharpe** (its column map has no `calmar`), although the config accepts `calmar`. Inactive in the baseline (`threshold_source: config`). | `ml_evaluation.py` `_SELECTION_COLUMNS` |
| D4 | Low | **ML walk-forward trade logs per variant are not exported.** Daily returns and per-window summaries are saved, but not the trade lists, so B/C walk-forward activity cannot be audited from a snapshot. | Snapshot file list |

### 3.2 Methodological weaknesses

| ID | Severity | Finding | Evidence |
|---|---|---|---|
| M1 | **High** | **The final test period is no longer untouched.** The 2022-12-28–2024-12-31 test was revealed on 2026-10-20 (statistical layer, then ML layer) and today, and the ML layer's design and defaults were chosen *after* strategy A's test results had been seen. Any decision based on this test period is contaminated by researcher knowledge. | Snapshot timestamps 14:02Z and 17:10Z on 2026-10-20 |
| M2 | **High** | **TCS/INFY has no cointegration evidence in training by any test.** EG gives p = 0.084 (TCS dependent) and p = 0.119 (INFY dependent). Johansen trace statistics are 9.0–10.0 against a 5% critical value of 15.5 for lag orders 1, 2 and 5. The static workflow trades it anyway (labelled exploratory), and the walk-forward gate passed 9 of 16 heavily overlapping windows. | `01_data_and_statistics.py` |
| M3 | **High** | **ML probabilities are miscalibrated by design.** `class_weight="balanced"` shifts probabilities toward 0.5. The Brier skill against a constant forecast of the training base rate (0.261) is negative even in-sample (LR −0.18, RF −0.02) and strongly negative out of sample (validation LR −3.88, RF −0.87; test LR −2.07, RF −0.75). "p ≥ 0.5" therefore does not mean "more likely than not", and B/C thresholds have no probabilistic meaning. | `03_execution_and_calibration.py` |
| M4 | **High** | **Stale static parameters with no time or risk limit.** The validation/test design trades 2021–2024 with a spread fitted on 2015–2020 and no stop or maximum holding. A's validation "trade" is one position held 483 of 493 bars and closed by the period end (−7.6%). Test trades last 311 and 110 bars, against an estimated half-life of 48 bars. These are long-duration bets on a drifting spread, not mean-reversion trades. | Trade logs; `03_...py` |
| M5 | Medium | **Feature-distribution shift between training and deployment.** Training rows use in-sample z (spread parameters fitted on the same window). On the same 2017–2020 dates, out-of-sample z from rolling fits has mean \|z\| 2.68 against 1.28 in-sample. Models trained on a narrow \|z\| range then extrapolate (the LR probabilities saturate near 1 when \|z\| is large). *The hypothesis that in-sample z inflates the reversion base rate was **refuted**: the out-of-sample rate on those dates is 0.328 against 0.278 in-sample.* | `02_ml_evaluation.py` §1 |
| M6 | Medium | **Effective sample sizes are small.** Overlapping 5-bar labels have lag-1 autocorrelation 0.49. The 560 training rows are worth about 158 independent observations. Walk-forward fits use 144–172 rows (about 40–50 effective). Neither is reported. | `02_...py` §2 |
| M7 | Medium | **No uncertainty quantification.** With a stationary bootstrap (20-day mean blocks, 2,000 resamples), every walk-forward Sharpe 95% CI includes 0 (A: −0.41 to 1.20) and every ML − A Sharpe difference CI includes 0 (B-LR: −0.18 to 0.53). Fold-by-fold, B-LR beats A in 6 of 9 traded windows, B-RF in 4 of 9, and C-LR and C-RF beat A in 3 and 2 windows (differing in only 4 and 3). | `02_...py` §3–4 |
| M8 | Medium | **Forced period-end liquidation drives results.** 3 of A's 10 walk-forward trades are closed by window ends, and those exits' net P&L equals −78.5% of A's total net P&L. Validation and test outcomes are mostly forced exits. This is a convention, not a strategy decision, and its effect differs by variant. | `03_...py` |
| M9 | Medium | **Multiple testing and researcher degrees of freedom are not accounted for at the strategy level.** For each pair: 16 entry/exit combinations, 5 costs, 4 + 4 hyperparameter settings, 20 threshold backtests, 16 walk-forward EG tests, and 5 variants × 3 periods compared. Across 3 pairs that is 45 strategy cells. Benjamini-Hochberg is used only in the universe screen. | Config grids |
| M10 | Medium | **Execution and cost realism for NSE.** Overnight shorting of Indian cash equities is generally not permitted outside securities lending, so a realistic implementation would use single-stock futures (lot sizes, margin, monthly rolls, different taxes), none of which are modelled. The generic 10 bps per side per leg is likely optimistic for cash delivery trades, where securities transaction tax alone is charged on both purchase and sale (0.1% each side in the schedule known to me; verify current rates). A trades rarely, so it is insensitive to costs (test +6.9% → +5.6% at 40 bps); B's turnover is 2.4–14.7× A's on TCS/INFY, which makes it far more sensitive. | `cost_sensitivity_*.csv`; README |
| M11 | Low | **Execution at the close of t+1**, one bar later than the next executable price (the open of t+1). This is conservative, but it changes holding periods relative to a 5-bar label horizon. No next-open option exists. | `backtester.py` |
| M12 | Low | **Reporting gaps against the Phase 4/5 requirements:** no Sortino ratio, no prevalence-baseline or Brier-skill reporting, no effective sample size, no confidence intervals, and no explicit separation of common-period portfolio performance from each strategy's own activity. | `metrics.py`, `ml_evaluation.py` |
| M13 | Low | **Engle-Granger is asymmetric and has no structural-break diagnostics.** The dependent leg is fixed by ticker order. The hedge ratio ranges from 0.12 to 1.28 across walk-forward windows. | Walk-forward window table |
| M14 | Low | **Idle capital earns 0, with rf = 0.** The strategy and cash are treated consistently, but in India (policy rates around 6%) a realistic rf would lower every excess-return Sharpe ratio, and collateral would in practice earn interest. Comparisons with cash and long-only benchmarks should use a realistic rf. | `metrics.py`, config |

### 3.3 Checked and found correct

* **Execution timing:** no fill before close t+1; positions earn only from the next bar; final-bar signals are
  never executed; truncation invariance holds (tests, and re-reading `run_backtest`).
* **Two-leg accounting:** equity identity, cash + notional = equity, costs on both legs at execution prices, borrow
  on prior-close short notional, drift rebalance timing. The suspected off-by-one in "% time invested" is **not** a
  defect: end-of-day positions count the same number of days as return-bearing days, shifted by one.
* **Point-in-time statistics:** hedge ratio, intercept, mean and std come from the fitting window only; the
  volatility-feature scale comes from the fitting window; the scaler is fitted on training rows only.
* **Label purging:** every training row has t + 5 ≤ train_end − 1 (last training row 2020-12-23); validation rows
  are purged at the test boundary (last row 2022-12-20).
* **Corporate actions:** adjusted for the 2018 bonus issues; the INFY −17.7% day (2019-10-22) is a real event.
* **Determinism:** fixed seeds and single-threaded Random Forest; identical reproduction.

### 3.4 Assumptions in force (unchanged, documented)

Adjusted closes approximate a total-return index (dividends received long and paid short). Shorting is available
at the close with a flat borrow fee. Costs are linear. No market impact, partial fills or margin calls. Each
period or window starts flat and liquidates at its end. Static spread parameters are held fixed per period.
OLS uses ticker A as the dependent leg.

---

## 4. Prioritised implementation plan

**Principles:** the baseline stays reproducible. Every behavioural change is behind a config option whose
default reproduces the baseline, and a new `config/research_v2.yaml` turns on the redesigned protocol. Each fix
comes with a regression test. No historical snapshot is overwritten.

| # | Priority | Change | Addresses | Regression test |
|---|---|---|---|---|
| 1 | P0 | **Pre-registration and a fresh holdout.** Write `docs/research/PREREGISTRATION.md` (hypotheses, primary metric, decision rule, all settings) *before* running v2. Reserve 2025-01-01 onward as an untouched final test, never downloaded or viewed until the specification is frozen. Treat 2015–2024 as development data only. | M1, M9 | Config guard: holdout dates cannot overlap the development range; holdout data is refused unless an explicit `--final-evaluation` flag is given |
| 2 | P0 | **Nested walk-forward for ML.** In each window, split the fit window chronologically into inner-train and inner-validation (purged), tune hyperparameters (and thresholds, if enabled) there, refit on the full purged fit window, then trade the next window. | D1 | Changing prices in window k+1 (or later) leaves window ≤ k hyperparameters, probabilities and returns unchanged |
| 3 | P0 | **Special-session handling.** Detect weekend-dated bars and a configurable list or volume heuristic for Muhurat sessions. Option `data.special_sessions: keep` (baseline) \| `no_trade` \| `drop`, with v2 using `no_trade`. | D2 | No entry or exit is ever executed on a flagged bar under `no_trade`; `keep` reproduces the baseline |
| 4 | P0 | Fix `select_threshold` for `calmar` (or reject unsupported metrics explicitly); export per-variant ML walk-forward trade logs. | D3, D4 | Calmar selection test; snapshot contains `ml_walk_forward_trades_<variant>.csv` |
| 5 | P1 | **Probability quality.** Add the prevalence baseline (constant training base rate) with Brier skill and log-loss skill; offer unweighted models and calibration (Platt or isotonic) fitted on inner-validation rows; use prevalence-relative thresholds in v2. | M3, M12 | Brier skill computed by hand on a toy set; calibration never sees evaluation rows |
| 6 | P1 | **Point-in-time training features (candidate).** Build ML training rows from out-of-sample z (spread parameters re-estimated on data before each sub-block), so training and deployment features come from the same process. Compare against the current design inside development data only. | M5 | Each training feature row depends only on prices before its own block |
| 7 | P1 | **Research-grade reporting.** Sortino ratio; exposure; fold-by-fold tables for every variant; stationary-bootstrap CIs for Sharpe and Sharpe differences; effective sample size; activity statistics reported separately from common-period portfolio results. | M6, M7, M12 | Sortino and bootstrap tested on deterministic inputs; ESS against a known AR(1) case |
| 8 | P1 | **Static-design options:** require cointegration to trade in the static workflow, optional max holding (for example 3× half-life) and stop, and refit on train+validation before the test (`static_refit`). Baseline defaults unchanged. | M2, M4, M8 | Option-specific tests; defaults reproduce the baseline exactly |
| 9 | P1 | **Market profiles and cost sensitivity for all variants:** `generic` (baseline) and an `nse_cash` profile with documented, user-verifiable statutory charges; a cost grid for A/B/C; regime breakdown by calendar year and volatility tercile. | M10 | Cost profile arithmetic tests; sensitivity grids monotone in cost |
| 10 | P2 | Next-open execution option using adjusted opens (Open × AdjClose / Close). | M11 | Fill price equals adjusted open of t+1; no fill on the signal bar |
| 11 | P2 | Symmetric checks: EG in both directions plus Johansen reported side by side, and rolling-β stability statistics. | M2, M13 | Matches statsmodels on synthetic data |

Order: 1 → 4 (P0) before any new result is produced; then 5, 7, 8 and 9; then 6 and 10–11. Deep learning and
reinforcement learning are out of scope: the simpler models show no discrimination (ROC-AUC ≈ 0.5) and negative
probabilistic skill, which gives no credible reason to add complexity (Phase 5 instruction).

---

## 5. Preliminary assessment (not the final recommendation)

On the current evidence, which includes a contaminated test period and a leaking ML walk-forward:

* **ML enhancement:** not supported. Out-of-sample ROC-AUC is about 0.5, Brier skill is negative everywhere, and
  no Sharpe difference against A is distinguishable from zero. The best-looking result (B-LR walk-forward +33.1%)
  is exactly the configuration affected by defect D1.
* **Statistical strategy on TCS/INFY:** not supported as a cointegration trade. There is no training-period
  cointegration evidence, the positive test and walk-forward returns have Sharpe CIs that include zero, and
  outcomes are dominated by a few long, force-closed positions.

The final recommendation (retain the statistical strategy, retain an ML enhancement, or reject the hypothesis)
will be made only after the P0/P1 fixes, under the pre-registered protocol, on development data with walk-forward
evidence, and then once on the untouched 2025+ holdout.

---

## 6. Resolution status (added 2026-10-21, after Phases 2–5)

Fixes are opt-in. The default config still reproduces the baseline exactly (`tests/test_baseline_preserved.py`,
and a real-data re-run whose 62 original tables are identical on every original row and column). The fixes are
enabled together in `config/research_v2.yaml`. Results: `docs/research/RESEARCH_REPORT.md`.

| ID | Status | Where | Regression tests (`tests/`) |
|---|---|---|---|
| D1 | **Fixed** (`ml.walk_forward_tuning: nested`); the baseline mode now warns | `ml_evaluation.run_ml_walk_forward`, `_window_model` | `test_audit_fixes.py::test_nested_walk_forward_is_point_in_time`, `::test_static_mode_still_requires_static_params_and_warns`, `::test_unsafe_nested_threshold_combination_rejected` |
| D2 | **Fixed** (`data.special_sessions: drop`; weekend, exchange-specific list, trailing-volume rule) | `data_validation.detect_special_sessions`, `prepare_pair` | `::test_special_sessions_*`, `::test_volume_rule_is_point_in_time`, `::test_exchange_specific_session_dates_only_affect_that_exchange` |
| D3 | **Fixed** | `ml_evaluation.select_threshold` | `::test_select_threshold_supports_calmar_and_rejects_unknown` |
| D4 | **Fixed** | `ml_evaluation.ml_result_tables` | `::test_walk_forward_trade_logs_and_predictions_exported` |
| M1 | **Addressed**: pre-registration, fresh 2025-01-01–2026-09-30 holdout, code-level guard, fingerprinted freeze, one-time marker | `protocol.py`, `scripts/run_experiment.py` | `::test_holdout_guard` |
| M2 | **Partly**: the v2 evidence is walk-forward with the EG gate on every variant; the static workflow still trades failed pairs (labelled exploratory) | — | — |
| M3 | **Addressed**: unweighted models, prevalence threshold, Brier and log-loss skill vs the prevalence baseline. Probabilities remain poor out of sample (holdout LR Brier skill −1.50) | `ml_models` | `::test_unweighted_logistic_regression_matches_base_rate`, `::test_prevalence_baseline_skill_scores_by_hand` |
| M4 | **Not implemented** (static-design options); v2 relies on walk-forward | — | — |
| M5 | **Not implemented**; confirmed again on the holdout (LR mean p 0.80 vs 0.33 realised) | — | — |
| M6, M7 | **Addressed**: effective sample size and stationary-bootstrap CIs for Sharpe and paired differences | `inference.py`, `experiment.py` | `::test_effective_sample_size`, `::test_bootstrap_intervals`, `::test_stationary_bootstrap_indices` |
| M8 | **Unchanged** (period-end liquidation is part of the protocol); reported | — | — |
| M9 | **Addressed** for the registered candidates (Bonferroni, 4 candidates) | `experiment.bootstrap_summary` | `test_experiment.py::test_decision_rules` |
| M10 | **Partly**: a cost grid (0/10/20/40 bps) for every variant; no NSE futures or statutory-cost model | `experiment.cost_sensitivity` | `test_experiment.py::test_cost_sensitivity_monotone` |
| M11, M13, M14 | **Not implemented** (next-open execution; EG both ways / Johansen in the engine; realistic rf and collateral interest). EG-both-ways and Johansen are available in `diagnostics/01_*` | — | — |
| M12 | **Addressed**: Sortino, Brier/log-loss skill, B0 control, activity vs common-period reporting | `metrics.py`, `experiment.py` | `::test_sortino_by_hand`, `::test_prevalence_baseline_variant_b0`, `test_experiment.py::*` |

New defect found during development and fixed before the freeze: listed NSE session dates were applied to the US
pair (amendment A1; `::test_exchange_specific_session_dates_only_affect_that_exchange`). A second edge case was
caught by a new test: `pooled_classification` returned NaN-filled output for a model with no predictions.

## 7. Resolution status update (2026-10-25, engine options for the remaining items)

The items left open in §6 are now implemented as **opt-in** options. The defaults are unchanged, so the default
config still reproduces the baseline exactly (`tests/test_baseline_preserved.py`). `config/research_v2.yaml` also
reproduces the frozen v2 development run exactly: every number in its `summary.json` and every common column of its
CSV tables, re-run on 2026-10-24. The new options are evaluated in the separately pre-registered v3 study
(`docs/research/v3/`), never on the spent v2 holdout.

| ID | Status | Option | Where | Regression tests (`tests/`) |
|---|---|---|---|---|
| M2 | **Addressed**: the static workflow can hold periods flat when the training fit fails the gate | `evaluation.static_require_cointegration` | `evaluation.run_pair_research`, `static_tradable`; ML comparisons follow A | `test_research_options.py::test_static_gate_holds_periods_flat`, `::test_static_ml_follows_the_refit_and_the_gate` |
| M4 | **Addressed**: a half-life time stop in static and walk-forward runs (per window); a test-period refit on train + validation | `strategy.max_holding_half_lives`, `evaluation.static_refit_before_test` | `evaluation.rules_for_model`, `run_pair_research`; ML test refit in `ml_evaluation` | `::test_rules_for_model_half_life_cap`, `::test_static_half_life_stop_caps_every_trade`, `::test_walk_forward_half_life_stop_is_per_window`, `::test_static_refit_before_test` |
| M5 | **Addressed**: point-in-time ML training rows (block-wise parameters from earlier bars only). On v3 development data the models still show no skill (ROC-AUC 0.50) | `ml.training_features: point_in_time`, `ml.pit_estimation_window`, `ml.pit_block` | `ml_features.point_in_time_rows`, `ml_evaluation` | `::test_point_in_time_rows_use_only_earlier_data`, `::test_point_in_time_training_in_static_ml`, `::test_point_in_time_walk_forward` |
| M10 | **Addressed**: NSE statutory costs (STT, exchange, SEBI, stamp, GST, brokerage, DP), single-stock-futures shorts (no borrow fee, rolls, margin reporting, optional lot rounding), per-market overrides | `costs.profile: nse`, `costs.short_instrument: futures`, `markets:` | `portfolio.MarketCostModel`, `backtester`, `config.ResearchConfig.for_pair` | `test_realism.py::test_nse_rates_by_hand`, `::test_futures_short_round_trip_costs_by_hand`, `::test_futures_rolls_charged_on_schedule`, `::test_futures_lot_sizes_round_both_legs_and_block_small_positions`, `test_research_options.py::test_market_overrides_reach_the_backtests` |
| M11 | **Addressed**: next-open execution with adjusted opens and a reported close fallback | `strategy.execution_price: open` | `data_validation.adjusted_open`, `backtester` | `test_realism.py::test_open_fill_earns_only_after_the_fill`, `test_research_options.py::test_next_open_execution_end_to_end` |
| M13 | **Addressed**: Engle-Granger in both directions and Johansen in the engine, recorded per window, with selectable gates. The Johansen trace test is oversized with weakly drifting prices (11–12% at 5%, `scripts/johansen_size_study.py`), so v3 requires EG **and** Johansen | `cointegration.gate`, `cointegration.record_all_tests` | `cointegration.johansen_test`, `cointegration_suite` | `test_cointegration_suite.py`, `test_research_options.py::test_walk_forward_gate_variants` |
| M14 | **Addressed**: cash interest on idle capital, short proceeds and futures carry; constant or calendar-year risk-free rates per market; Sharpe on excess returns | `portfolio.cash_interest`, `metrics.risk_free_by_year` | `backtester`, `metrics.period_excess_returns`, `experiment.risk_free_daily` | `test_realism.py::test_schedule_config_and_cash_interest_by_year`, `::test_flat_chained_walk_forward_has_zero_excess_return` |

Still open: M8 (period-end liquidation stays part of the protocol). Not modelled: futures basis, time-varying lot
sizes, and a daily risk-free series from an official source; v3 uses approximate annual averages.
