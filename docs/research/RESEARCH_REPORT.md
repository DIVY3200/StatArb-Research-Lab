# Research report: does ML improve cointegration pairs trading?

**Date:** 2026-10-21. **Protocol:** pre-registered (`docs/research/PREREGISTRATION.md`, frozen, with two disclosed
amendments). **Pairs:** KO/PEP, HDFCBANK.NS/ICICIBANK.NS, TCS.NS/INFY.NS. **Development:** 2015–2024 (trading
2017–2024). **Untouched holdout:** 2025-01-01 to 2026-09-30, evaluated once.

## Recommendation

**Reject the tested hypothesis.** This is the outcome of the pre-registered decision rules, applied
mechanically ([`results/20261021T162902Z_final/decision.json`](results/20261021T162902Z_final/decision.json)).
Committed copies of every run are in [`results/`](results/); `reports/` itself is not version-controlled.

* **No ML enhancement is retained.** Neither model has out-of-sample predictive skill, on development or holdout
  data. Pooled ROC-AUC is 0.52–0.54 and the Brier skill against a constant base-rate forecast is negative
  everywhere. No ML variant beats the traditional strategy on the holdout. On development data, the best variant's
  apparent edge (B-LR) comes mainly from its trading *rules*, not its *model*.
* **The statistical strategy is not retained.** Its development walk-forward Sharpe is 0.33 (95% CI −0.38 to
  1.03). On the holdout it lost money (Sharpe −0.38, CI −1.18 to 0.53) on its only trade.

"Reject" means the evidence does not support the hypothesis. It does not prove there is no edge: the holdout is
short and the strategy rarely trades (see §6). What the evidence does show is that nothing in this framework
justifies deploying or extending these strategies, and adding model complexity (deep learning, reinforcement
learning) would not be justified.

---

## 1. Question and hypotheses

* **H-STAT:** the z-score pairs strategy (A), traded only where the fitting window passes Engle-Granger at 5%,
  earns a positive risk-adjusted return after costs out of sample.
* **H-ML:** a Logistic Regression or Random Forest reversion classifier, used alone (B) or as an entry filter (C),
  improves on A's net risk-adjusted return on the same dates under the same execution assumptions.

## 2. What changed relative to the baseline

The baseline (the original implementation, reproduced exactly in Phase 1) remains available. Its defaults are
unchanged, which is enforced by `tests/test_baseline_preserved.py` and re-checked on real data (all 62 original
output tables identical on every original row and column). The v2 protocol (`config/research_v2.yaml`) turns on
the audit fixes:

| Audit finding | v2 change | Tests |
|---|---|---|
| D1: ML walk-forward reused hyperparameters tuned on later data | Nested, per-window tuning on a purged inner-validation block (log loss) | `test_nested_walk_forward_is_point_in_time` (changing later prices leaves earlier windows identical), `test_static_mode_still_requires_static_params_and_warns`, `test_unsafe_nested_threshold_combination_rejected` |
| D2: special NSE sessions traded as normal bars | Weekend-dated bars, exchange-specific listed sessions and very thin bars are dropped | `test_special_sessions_*`, `test_volume_rule_is_point_in_time`, `test_exchange_specific_session_dates_only_affect_that_exchange` |
| D3: `calmar` silently treated as Sharpe | Supported explicitly; unknown metrics rejected | `test_select_threshold_supports_calmar_and_rejects_unknown` |
| D4: ML walk-forward trade logs not exported | Exported per variant, along with predictions | `test_walk_forward_trade_logs_and_predictions_exported` |
| M1: test period viewed repeatedly | Pre-registration, a fresh holdout, a code-level holdout guard, a fingerprinted freeze, and a one-time marker | `test_holdout_guard`; `scripts/run_experiment.py` refusals |
| M3: class-weighted probabilities miscalibrated | Unweighted models, prevalence threshold, Brier and log-loss skill against a prevalence baseline | `test_unweighted_logistic_regression_matches_base_rate`, `test_prevalence_baseline_skill_scores_by_hand` |
| M6, M7, M12: no sample sizes or uncertainty | Effective sample size, stationary-bootstrap CIs for Sharpe and paired differences, Sortino | `test_effective_sample_size`, `test_bootstrap_intervals`, `test_stationary_bootstrap_indices`, `test_sortino_by_hand` |
| (spec Phase 5) prevalence baseline at strategy level | B0: B's rules driven by the constant prevalence forecast (amendment A2) | `test_prevalence_baseline_variant_b0` |

Strategy A, the execution model (decide at close *t*, fill at close *t+1*), costs (5 + 5 bps per leg per side,
borrow 50 bps per year), sizing (1.0× gross) and the walk-forward schedule (504-bar fit, 126-bar trade, EG gate)
are **unchanged**.

## 3. Phase 1 baseline (for reference)

The original TCS/INFY 2015–2024 static split, reproduced exactly: A returned −7.6% on validation, +6.9% on the
test and +21.5% in walk-forward. B-LR returned +33.1% in walk-forward, but that figure is affected by D1. The test
period used there is contaminated (M1), and the baseline is not evidence for either hypothesis. Details are in
`docs/audit/PHASE1_AUDIT.md` and `docs/audit/baseline_tcs_infy_2015_2024.json`.

## 4. Development results (2017-01-03 to 2024-12-31)

Run [`results/20261021T162712Z_development`](results/20261021T162712Z_development). The primary portfolio is the equal-weight average of the
three pairs' daily net returns.

| Variant | Net return | Gross return | Sharpe [95% CI] | Sortino | Max DD | Trades | % time invested | Turnover / yr |
|---|---|---|---|---|---|---|---|---|
| A. Traditional | +6.4% | +7.4% | 0.33 [−0.38, 1.03] | 0.50 | −8.4% | 11 | 6.6% | 0.95× |
| B. ML-only (LR) | +15.6% | +18.1% | 0.74 [0.08, 1.33] | 1.27 | −5.1% | 29 | 7.4% | 2.49× |
| C. Hybrid (LR) | +8.0% | +9.0% | 0.42 [−0.31, 1.13] | 0.66 | −7.6% | 11 | 6.2% | 0.95× |
| B. ML-only (RF) | +8.8% | +11.9% | 0.45 [−0.27, 1.12] | 0.72 | −7.6% | 39 | 7.3% | 3.35× |
| C. Hybrid (RF) | +7.3% | +8.2% | 0.39 [−0.35, 1.13] | 0.61 | −8.3% | 11 | 5.4% | 0.94× |
| B0. B rules, no model | +11.2% | +14.1% | 0.48 [−0.16, 1.11] | 0.74 | −8.4% | 35 | 10.9% | 3.02× |

Paired Sharpe differences (stationary bootstrap, 5,000 resamples, 20-bar mean blocks):

| Comparison | Difference | 95% CI | 98.75% CI (Bonferroni, 4 candidates) |
|---|---|---|---|
| B-LR − A | +0.42 | [0.02, 0.85] | [−0.08, 0.95] |
| C-LR − A | +0.09 | [−0.03, 0.25] | [−0.06, 0.31] |
| B-RF − A | +0.12 | [−0.25, 0.51] | [−0.35, 0.62] |
| C-RF − A | +0.07 | [−0.11, 0.28] | [−0.16, 0.35] |
| **B-LR − B0** (model's contribution within B) | +0.26 | [−0.09, 0.65] | — |
| **B-RF − B0** | −0.03 | [−0.32, 0.24] | — |

Activity by pair (net return, with trades in brackets): KO/PEP never traded, because no fitting window passed the
EG test. HDFCBANK/ICICIBANK had one traded window: A −5.8% (1), B-LR +7.2% (4), B0 −5.3% (3). TCS/INFY had 9 traded
windows: A +26.2% (10), B-LR +42.3% (25), B0 +42.5% (32), C-LR +28.3% (10). No window traded in 2023 or 2024, so
the development evidence is effectively 2017–2022.

**Classification (pooled out of sample, 552 rows, effective sample size ≈ 164):**

| Model | ROC-AUC | Brier (prevalence baseline) | Brier skill | Log-loss skill | Mean p vs realised rate |
|---|---|---|---|---|---|
| Logistic Regression | 0.523 | 0.290 (0.262) | −0.109 | −0.145 | 0.574 vs 0.384 |
| Random Forest | 0.515 | 0.266 (0.262) | −0.016 | −0.015 | 0.531 vs 0.384 |

Per-window ROC-AUC ranges from 0.42 to 0.80 (LR) and 0.35 to 0.73 (RF) across 10 traded windows: no stable
discrimination. Even the prevalence baseline was optimistic: the fitting windows' in-sample reversion rate averaged
0.52, against 0.38 realised. Across development and holdout, nested tuning fell back to the first grid point in 2 of
24 model-window fits, because inner-validation blocks were too small.

**Costs (Sharpe at 0 / 10 / 20 / 40 bps per leg per side):** A 0.37 / 0.33 / 0.29 / 0.22; B-LR 0.84 / 0.74 / 0.65 /
0.45; B-RF 0.58 / 0.45 / 0.31 / 0.05; B0 0.58 / 0.48 / 0.38 / 0.17; C-LR 0.46 / 0.42 / 0.38 / 0.30. The
high-turnover B variants lose their edge fastest as costs rise.

**Regimes:** by calendar year, returns are concentrated in 2019 (A: +8.5% in 2019, against −4.8% in 2018 and +6.4%
over the whole period). In descriptive volatility terciles, every variant lost money in the low-volatility third
(A Sharpe −0.49).

**What development shows:** the only development difference that clears even an unadjusted 95% interval (B-LR vs A)
is mostly a rule effect. B0, which uses no model at all, captures more than a third of it, and the model's own
contribution (B-LR − B0) is not distinguishable from zero. On TCS/INFY, which supplies nearly all the activity, B0
and B-LR perform the same. Pre-registered gate 6.1(a) (development ROC-AUC > 0.55 and Brier skill > 0) fails for
both models. This was disclosed in the freeze before the holdout was opened.

## 5. Holdout results (2025-01-01 to 2026-09-30, evaluated once)

Run [`results/20261021T162902Z_final`](results/20261021T162902Z_final). The data were downloaded for the first time at
2026-10-21T16:28Z, after the freeze, which was verified by fingerprint.

| Variant | Net return | Sharpe [95% CI] | Max DD | Trades |
|---|---|---|---|---|
| A. Traditional | −0.9% | −0.38 [−1.18, 0.53] | −1.7% | 1 |
| B. ML-only (LR) | −2.1% | −0.63 [−1.56, 0.40] | −3.0% | 6 |
| C. Hybrid (LR) | −0.9% | identical to A | −1.7% | 1 |
| B. ML-only (RF) | −1.8% | −0.53 [−1.46, 0.45] | −3.1% | 6 |
| C. Hybrid (RF) | −0.9% | identical to A | −1.7% | 1 |
| B0. B rules, no model | −2.2% | −0.64 [−1.54, 0.35] | −3.1% | 6 |

* **Activity was very low.** Only 2 of 12 pair-windows passed the EG gate (HDFCBANK/ICICIBANK 2026-01-09 to
  2026-07-17; TCS/INFY 2026-07-20 to 2026-09-30). KO/PEP never passed (p = 0.26–0.93). A made one trade (−2.8% on
  HDFCBANK/ICICIBANK). The hybrid filter never bound, so C equals A.
* **The ML models again showed no skill.** Pooled holdout ROC-AUC: LR 0.524, RF 0.536 (108 rows, effective
  sample size ≈ 38). Brier skill: LR −1.50, RF −0.10. LR's mean predicted probability was 0.80 against a 0.33
  realised rate, with predictions above 0.96 on rows that reverted 14–50% of the time. Removing class weights did
  not cure the extrapolation problem identified in audit finding M5.
* **Model vs no model within B:** B-LR − B0 = +0.004 [−0.18, 0.22]; B-RF − B0 = +0.11 [−0.03, 0.36].
* **Costs:** A's return is −0.8% at 0 bps and −1.1% at 40 bps. Every B variant (and B0) is worse than A at every
  cost level; C equals A.
* **Special sessions dropped in the holdout:** 2025-02-01 (Saturday Budget-day session, weekend rule), 2025-10-21
  (listed Muhurat), and 2025-03-18 and 2026-01-15 (both legs below 10% of trailing volume).

## 6. Pre-registered decision (§6 of the pre-registration)

| Strategy | Gate (a): dev classification skill | Gate (b): holdout Sharpe diff, 98.75% CI > 0 | Gate (c): beats A at 20 bps | Decision |
|---|---|---|---|---|
| B-LR | fail (AUC 0.523, BSS −0.109) | fail (−0.25) | fail | reject |
| C-LR | fail | fail (0.00) | fail | reject |
| B-RF | fail (AUC 0.515, BSS −0.016) | fail (−0.15) | fail | reject |
| C-RF | fail | fail (0.00) | fail | reject |
| A | holdout Sharpe 95% CI lower bound > 0 and dev Sharpe > 0? Holdout point −0.38, lower bound −1.18 | — | — | reject |

**Overall: reject the tested hypothesis.**

## 7. Interpretation, limitations and caveats

* **Statistical power is low.** Over 21 months the gate allowed trading in 2 of 12 pair-windows, and A traded once.
  The holdout alone could not have confirmed a modest edge. The conclusion rests on all the evidence together:
  development Sharpe CIs that include zero, no classification skill in either period, and no consistent improvement
  from any ML variant.
* **Cointegration is rarely present.** Across 48 development and 12 holdout pair-windows, the EG gate passed 10 and
  2 times. KO/PEP never passed. A cointegration-gated strategy on these pairs is mostly flat.
* **Amendments made after development results.** A1 (exchange-specific session dates) fixed a real defect. A2 (the B0
  control) restored a baseline the specification required. Neither touched the hypotheses, candidates or decision
  rules, and both were logged before the freeze. The gate-(a) failure was known before the holdout and disclosed in
  the freeze.
* **Multiple testing:** 4 registered candidates (Bonferroni), plus B0 and A as references. The earlier 2015–2024 test
  period was contaminated (M1) and is used here only as development data.
* **Known open items, not implemented:** point-in-time (out-of-sample) features for ML training rows (M5), static-
  design options such as max holding, refitting and a static cointegration gate (M4, M8), a single-stock-futures
  model for NSE shorting and India-specific statutory costs (M10; only a cost grid is reported), next-open
  execution (M11), Johansen/EG-both-directions reporting in the engine (M13; available in the audit diagnostics),
  and a realistic risk-free rate and collateral interest (M14).
* **Data:** Yahoo Finance adjusted closes as delivered, with input hashes recorded per run. Free data can be revised.
* **Scope:** three pre-specified large-cap pairs. Results do not generalise to other pairs, universes or execution
  styles, and this is not financial advice.

## 8. Reproducing this report

```bash
python -m pytest                                              # 203 tests, offline
python scripts/run_research.py --source yfinance --tickers TCS.NS INFY.NS --reveal-test   # baseline (default config)
python scripts/run_experiment.py --stage development          # v2 development (refuses data after 2024-12-31)
# python scripts/run_experiment.py --freeze <dev run>         # already done: docs/research/freeze.json
# python scripts/run_experiment.py --stage final              # one-time holdout; now refuses (holdout_evaluated.json)
```

Artifacts (written to `reports/experiments/`, committed as byte-identical copies in [`results/`](results/)):
`20261021T161656Z_development` (first development run, before A1/A2), `20261021T162712Z_development` (frozen
development) and `20261021T162902Z_final` (holdout, `decision.json`).
Each holds the config snapshot, input hashes, environment, seeds, per-pair windows, trades, predictions, and
pooled tables. `docs/research/freeze.json` and `docs/research/holdout_evaluated.json` record the protocol state.
