# StatArb Research Lab

[![tests](https://github.com/DIVY3200/StatArb-Research-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/DIVY3200/StatArb-Research-Lab/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A reproducible research platform for studying **cointegration-based pairs trading** on daily stock prices, with
a **machine-learning layer** that tests whether reversion classifiers improve on the traditional rules. It
includes a modular Python package (`statarb`), a Streamlit dashboard, a command-line research script, a notebook,
and an offline test suite.

> **Research questions.** Can a pairs-trading strategy based on statistically justified relationships between
> two assets generate robust out-of-sample returns after transaction costs and realistic trading assumptions?
> And does adding a Logistic Regression or Random Forest prediction of spread reversion improve it when every
> variant is evaluated on the same out-of-sample dates under the same execution and cost assumptions?

The platform is built to answer these questions honestly. Parameters are fitted on training data only, and ML
labels are purged at period boundaries. Trades execute after a delay. Both legs pay costs. The final test period
stays locked until the specification is fixed. **It is a research and education tool. It does not trade, and
nothing it produces is financial advice.**

> **Current answer (2026-10-21).** After an audit of the original implementation and a pre-registered v2
> experiment with a one-time holdout (2025-01-01 to 2026-09-30), the pre-registered decision is **reject the
> tested hypothesis**:
> * the ML models show no out-of-sample predictive skill (ROC-AUC about 0.52, negative Brier skill);
> * no ML variant beats the traditional strategy;
> * the traditional strategy's Sharpe interval includes zero in development and was negative on the holdout.
>
> See [Audit and v2 research protocol](#audit-and-v2-research-protocol),
> [`docs/research/RESEARCH_REPORT.md`](docs/research/RESEARCH_REPORT.md) and
> [`docs/audit/PHASE1_AUDIT.md`](docs/audit/PHASE1_AUDIT.md).
>
> **Next study (v3, registered 2026-10-24).** v3 adds:
> * next-open fills;
> * NSE statutory costs with single-stock-futures shorts;
> * cash interest at calendar-year T-bill rates;
> * a half-life time stop and an Engle-Granger + Johansen gate;
> * point-in-time ML training rows.
>
> Its holdout runs from 2026-10-01 to 2028-09-30. On development data the ML models again show no skill
> (ROC-AUC 0.50), so under v3's rules no ML variant can be retained whatever the holdout shows. See
> [v3 protocol](#v3-protocol-realism-and-robustness-upgrades).

---

## Contents

1. [Quick start](#quick-start)
2. [Architecture](#architecture)
3. [Data](#data)
4. [Methodology](#methodology)
5. [Machine-learning layer](#machine-learning-layer)
6. [Audit and v2 research protocol](#audit-and-v2-research-protocol)
7. [v3 protocol: realism and robustness upgrades](#v3-protocol-realism-and-robustness-upgrades)
8. [Dashboard](#dashboard)
9. [Reproducibility](#reproducibility)
10. [Testing](#testing)
11. [Example results (KO/PEP, HDFCBANK/ICICIBANK, TCS/INFY)](#example-results)
12. [Research integrity and limitations](#research-integrity-and-limitations)
13. [Troubleshooting](#troubleshooting)
14. [Future improvements](#future-improvements)
15. [Resume summary](#resume-summary)

---

## Quick start

Requires Python 3.10+. Developed on Python 3.13.7 (Windows 11). CI runs the full test suite on Linux, Windows and
macOS with Python 3.10, 3.11, 3.12, 3.13 and 3.14 (`.github/workflows/tests.yml`).

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 2. Install dependencies (exact tested versions are in requirements-lock.txt)
pip install -r requirements.txt
pip install -e .                 # optional: makes `import statarb` work from anywhere

# 3. Run the dashboard
streamlit run app.py

# 4. Run the tests (offline, no network needed)
pytest

# 5. Reproducible command-line research run (writes a snapshot to reports/)
python scripts/run_research.py --source synthetic
python scripts/run_research.py --source yfinance --tickers KO PEP --start 2015-01-01 --end 2024-12-31
python scripts/run_research.py --source yfinance --tickers KO PEP --reveal-test   # after fixing the specification
python scripts/run_research.py --source synthetic --no-ml                         # statistical layer only
```

The ML comparison runs by default (`ml.enabled: true`) and adds about 5–10 s per pair.

```bash
# Audited v2 protocol (pre-registered; see docs/research/)
python scripts/run_experiment.py --stage development   # 2015-2024 only; refuses later data
# --freeze and --stage final have already been run once; the final stage now refuses to run again

# v3 protocol (pre-registered; see docs/research/v3/)
python scripts/run_experiment.py --config config/research_v3.yaml --stage development   # 2015-2026-09 only
# --stage final refuses to run before the holdout ends (2028-09-30), without a freeze, or a second time
python scripts/johansen_size_study.py                   # reproduces the Johansen size caveat (a few minutes)
```

The dashboard opens on a **labelled synthetic dataset**, so it works offline. Switch the sidebar data source to
Yahoo Finance or CSV upload to analyse real prices.

---

## Architecture

```
app.py                      Streamlit dashboard (UI only; every number comes from src/statarb)
config/default_config.yaml  All research parameters; a snapshot is saved with every run
src/statarb/
  config.py                 Typed, validated configuration (unknown keys and invalid values are rejected)
  data_loader.py            yfinance download + cache, CSV loading, labelled synthetic data, SHA-256 hashing
  data_validation.py        Cleaning, validation reports, two-asset date alignment (no forward fill)
  diagnostics.py            Exploratory statistics (returns, rolling volatility/correlation, summaries)
  hedge_ratio.py            Training-window OLS hedge ratio on log prices
  cointegration.py          Engle-Granger (both directions), Johansen, trading gates, ADF checks, BH correction
  spread.py                 Spread, training-normalised z-score, AR(1) half-life
  signals.py                Finite-state entry/exit/stop/max-holding signal machine
  portfolio.py              Hedge ratio -> dollar leg weights; generic and NSE cost schedules, lot rounding
  backtester.py             Two-leg event backtester: close or next-open fills, costs, borrow, futures rolls and
                            margin, cash interest, trade log
  metrics.py                Metrics, benchmarks, comparison tables; constant or calendar-year risk-free rates
  evaluation.py             Chronological splits, research pipeline, parameter grid, walk-forward, snapshots
  universe.py               Training-only pair screen with false-discovery control
  ml_features.py            Reversion target, causal features, purged masks, point-in-time training rows
  ml_models.py              scikit-learn pipelines (LR, RF), guarded fitting, validation tuning, metrics, importance
  ml_strategy.py            Signal rules for B (ML-only) and C (hybrid)
  ml_evaluation.py          ML research run, A/B/C comparison, ML walk-forward (static or nested tuning), repricing
  inference.py              Stationary bootstrap (Sharpe and paired differences), effective sample size
  experiment.py             Pre-registered experiments: per-market settings, pooling, excess-return inference,
                            decision rules, assumption checks
  protocol.py               Holdout guard and code fingerprints
  visualization.py          Plotly figures (colour-vision-deficiency-safe palette, single y-axis per panel)
scripts/run_research.py     Notebook-free reproducible research run (statistical + ML layers)
scripts/run_experiment.py   Experiments (v2, v3): development / freeze / one-time final holdout
scripts/johansen_size_study.py  Monte Carlo size of Engle-Granger vs Johansen on independent random walks
config/research_v2.yaml     The audited, pre-registered v2 protocol (the baseline stays in default_config.yaml)
config/research_v3.yaml     The v3 protocol: v2 plus the realism and robustness options, with NSE market overrides
docs/audit/                 Phase 1 audit report, baseline record, diagnostics scripts
docs/research/              v2: pre-registration, freeze record, holdout marker, research report, results
docs/research/v3/           v3: pre-registration, development report, results
scripts/generate_sample_data.py  Regenerates the synthetic sample CSVs
notebooks/pairs_research.ipynb   Step-by-step research walkthrough
tests/                      265 offline pytest tests (see Testing)
.github/workflows/tests.yml CI: the test suite on Linux, Windows and macOS, Python 3.10 to 3.14
data/raw/                   Download cache (git-ignored)   data/sample/  SYNTHETIC CSVs only
reports/                    Timestamped research-run snapshots (git-ignored)
```

Data flows one way: **load → validate → align → split → fit (train) → signals → backtest → metrics → report**.
The ML layer sits on top. It reuses the statistical layer's splits and training-fitted spread model, builds features
and labels, fits classifiers, and sends its signals through the *same* backtester. Each stage is a separate module
with no UI code, so the dashboard, CLI, notebook and tests all run the same code.

---

## Data

### Sources

| Source | How | Notes |
|---|---|---|
| Yahoo Finance | `download_yfinance()` / sidebar "Yahoo Finance" | Optional, needs network. Cached as CSV plus a metadata file (retrieval time, SHA-256) in `data/raw/`. NSE symbols need `.NS` (e.g. `TCS.NS`). |
| CSV upload | `load_csv()` / sidebar "CSV upload" | One long-format file with a `Ticker` column, or one file per asset. Columns: `Date, Open, High, Low, Close, Adj Close (optional), Volume`. Common aliases such as `adj_close` and `date` are recognised. |
| Synthetic | `generate_synthetic_pair()` | Deterministic, **clearly labelled** (`SYNTH_` tickers, banner in the UI). For demos and tests only, never presented as market data. |

Suggested candidate pairs for exploration (**not** assumed to be cointegrated): `HDFCBANK.NS / ICICIBANK.NS`,
`TCS.NS / INFY.NS`, `KO / PEP`. Aim for 5–10 years of daily data. Coverage, corporate-action handling and history
vary by provider and instrument.

### Price conventions (dividends and splits)

* yfinance is called with `auto_adjust=False`. `Adj Close` (split- and dividend-adjusted by Yahoo) is used **as
  delivered**, with no further adjustment, so prices are never double-adjusted. If `Adj Close` is unavailable,
  `Close` is used and a warning is shown.
* Back-adjusted prices are proportional to a total-return index. The proportionality constant depends on
  dividends paid *after* each date, but it shifts all log prices by a constant, which the regression intercept
  absorbs. Z-scores are therefore unaffected.
* Using adjusted returns means dividends are implicitly **received on the long leg and paid on the short leg**.
* CSV prices are used exactly as supplied.

### Cleaning and validation (`data_validation.py`)

* Rows are sorted chronologically. Unparseable dates are removed.
* **Duplicates:** exact duplicate rows are collapsed to one. Conflicting duplicates (same date, different values)
  are *all* removed, because neither can be trusted.
* Missing, infinite and non-positive prices are **removed, never filled**. Every removed date is reported.
* OHLC inconsistencies, negative volume, calendar gaps longer than 7 days and daily log moves above 25% (a
  possible unadjusted split) are flagged as warnings.
* Too little history raises an actionable error.
* **Alignment** is an inner join on dates where both assets have valid prices, so every return spans the same
  interval for both legs. Dates present for only one asset are listed. Pairs listed on different exchanges trigger
  a warning about non-synchronous closing prices.

---

## Methodology

### Chronological splits

The aligned sample is split, without shuffling, into **training (60%) → validation (20%) → final test (20%)**.
The proportions are configurable.

| Period | Used for |
|---|---|
| Training | Hedge ratio, intercept, spread mean/std, Engle-Granger test, universe screening |
| Validation | Comparing entry/exit thresholds (parameter grid) |
| Final test | One evaluation of the fixed specification. Locked in the dashboard until you reveal it. |

Each period's backtest starts flat. While the test period is locked, the exploratory charts and the walk-forward
analysis stop at the end of validation.

### Cointegration (Engle-Granger two-step)

1. On the training window, regress `log P_A = α + β log P_B + ε` (OLS).
2. Run an augmented Dickey-Fuller regression on the residuals, with the lag order chosen by AIC.
3. Compare the statistic with **Engle-Granger / MacKinnon** critical values (`statsmodels.tsa.stattools.coint`).
   Plain ADF critical values would be wrong, because the residuals come from an estimated regression.

The dashboard reports the statistic, p-value, 1/5/10% critical values, the chosen significance level (default 5%)
and the decision. It also runs ADF checks that each log price is I(1). Pairs that fail are shown with a prominent
warning and are never described as cointegrated opportunities.

**Both directions and Johansen.** The engine can also run Engle-Granger with the legs swapped and the Johansen
trace and maximum-eigenvalue tests on each fitting window. `cointegration.record_all_tests` records them, and
`cointegration.gate` chooses which must pass before a window is traded:

* `eg`: Engle-Granger, A on B (the default);
* `eg_both`: Engle-Granger in both directions;
* `johansen`: Johansen, rank 0 rejected;
* `eg_and_johansen`: Engle-Granger and Johansen.

The dashboard's Cointegration tab shows all of them. `statsmodels`' Johansen test with a constant over-rejects
when prices drift only weakly: in this project's simulation (`scripts/johansen_size_study.py`, 600 pairs of
independent random walks per drift level), the 5% trace test rejected 11–12% of the time. Johansen is therefore
best used as confirmation. The `eg_and_johansen` gate had a measured size of 2.8–3.8%.

**Interpretation.** H0 is *no cointegration*. Rejecting H0 is in-sample statistical evidence that the spread was
mean-reverting during the training window. It is not proof that the relation will persist, and the p-value is
not the probability that the strategy is profitable. Results depend on the sample period and on which asset is
the dependent variable. Structural breaks can invalidate the relation. Testing many pairs inflates false
discoveries: the test suite includes a fixed-seed pair of *independent* random walks that passes at 5% purely by
chance. Statistical significance is not economic significance.

### Hedge ratio

`β` is estimated by OLS on training log prices and held fixed within each fitting/trading window. It is an
**elasticity**, and it maps to a **dollar** ratio, not a share ratio:
`d spread ≈ r_A − β r_B`, so long $1 of A is hedged by short $β of B. The implied share ratio is
`units_B / units_A = −β · P_A / P_B`. OLS standard errors are not reported because they are invalid with
non-stationary regressors. Regressing B on A would give a different, non-reciprocal β.

### Spread and z-score

```
spread_t = log P_A,t − α − β · log P_B,t
z_t      = (spread_t − mean_train) / std_train        (std with ddof = 1)
```

All four parameters come from the fitting window and stay frozen during the following evaluation window. Since
`z_t` depends only on prices at `t` and on frozen parameters, there is no look-ahead. A near-zero training standard
deviation (a degenerate pair) raises an error instead of producing unstable z-scores. A stationary-looking spread
does not guarantee stable mean reversion later.

### Signals (finite-state machine, `signals.py`)

| Condition at close t | Action |
|---|---|
| Flat and `z ≤ −entry` | **Long spread**: long A, short B (A cheap relative to B) |
| Flat and `z ≥ +entry` | **Short spread**: short A, long B (A expensive) |
| Long and `z ≥ −exit` / short and `z ≤ +exit` | Exit (reverted to within the exit band, or crossed it) |
| `\|z\| ≥ stop` against the position (optional) | Stop-loss exit; no re-entry until `\|z\| < entry` |
| Held `max_holding` bars (optional) | Time exit; no re-entry until `\|z\| < entry` |
| Held ⌈k × half-life⌉ bars (optional, `strategy.max_holding_half_lives`) | Time exit at k times the fitting window's AR(1) half-life (per walk-forward window); combined with `max_holding`, the shorter wins |

Defaults: entry 2.0, exit 0.5, stop-loss off, max holding off. A same-direction signal while a position is open
is ignored (no pyramiding). An opposite signal first triggers the exit, and a new position can only open on a
**later** bar, so there are no same-bar reversals or conflicting positions. Entries beyond the stop level are
refused.

### Execution timing

A signal from the close of bar *t* is executed at the **close of bar t + lag** (default `lag = 1`; `lag = 0` is
rejected). With `strategy.execution_price: open` it fills at the **open** of bar *t + lag* instead. Opens are
adjusted with the close's factor (`Open × Adj Close / Close`). Units carried overnight are marked from the previous
close to the fill price, and units held after trading from the fill to the close, so the overnight gap after a
signal is not earned. A bar without a usable open falls back to its close, and the run reports it. This follows the "wait one day" convention of Gatev, Goetzmann & Rouwenhorst (2006). A new position
earns returns from the next bar onward, so a signal can never earn the price move that produced it. On the last
bar of a period, open positions are liquidated (paying exit costs) and no new positions open, so the final
observation creates no artificial return. Tests check all of these rules, including that results up to bar *k*
are unchanged by any price after bar *k*.

### Portfolio construction and costs

* **Sizing:** gross-exposure normalised. At entry, `|w_A| + |w_B| = max_gross_exposure` (default 1.0× equity), with
  `w_A = ±G/(1+|β|)` and `w_B = ∓β·G/(1+|β|)`. The size is set so the limit holds *after* paying entry costs.
* **Holding:** units stay fixed between trades, so weights drift. If gross leverage observed at the previous close
  exceeds the limit by more than 10%, the position is scaled back at the next close, paying costs.
* **Transaction costs:** commission + slippage in bps of traded notional, on **both legs**, for entries, exits and
  rebalances (defaults 5 + 5 bps per leg per side). A zero-cost setting exists for diagnostics only.
* **Borrow fee:** annual bps on prior-close short notional of cash shorts, accrued daily (default 50 bps per year).
* **Short selling:** an explicit assumption. If it is disabled, the backtester refuses to trade, because every
  pairs position needs one short leg.
* **NSE cost schedule (optional, `costs.profile: nse`):**
  * per leg and side: STT (10 bps on delivery buys and sells, 2 bps on futures sells), exchange charges, the SEBI
    fee, stamp duty on buys, and 18% GST on brokerage + exchange + SEBI charges;
  * plus brokerage (bps and/or per order), depository charges and slippage.
  * The values are the researcher's understanding of the late-2024 schedule; verify them before relying on them.
* **Futures shorts (optional, `costs.short_instrument: futures`):** short legs are single-stock futures:
  * no borrow fee;
  * a roll every `futures_roll_every_bars` bars, costed as a futures buy + sell plus a calendar spread;
  * initial margin reported (with a warning if long exposure plus margin exceeds equity);
  * optional rounding down to whole lots. Most Indian investors cannot hold multi-day cash shorts.
* **Cash interest (optional, `portfolio.cash_interest`):**
  * the cash balance (equity − long notional + short notional) earns the risk-free rate daily: idle capital,
    short-sale proceeds, and the carry of short futures;
  * the rate is `metrics.risk_free_rate`, or the calendar-year schedule `metrics.risk_free_by_year`;
  * chained walk-forward windows also accrue it on their first bar.
* **Per-market settings:** a `markets:` section applies overrides by the ticker suffix of leg A (for example `NS`),
  so one config can price US pairs with generic costs and NSE pairs with statutory costs and futures.
* **Not modelled:** margin calls, financing spreads, futures basis, time-varying lot sizes, market impact beyond
  linear slippage, partial fills, taxes. These are research assumptions, not any broker's terms.

The backtester records signal and execution timestamps, units and notional for each leg, long/short/gross/net
exposure, cash, mark-to-market P&L, costs, turnover, equity, drawdown, realised and unrealised P&L, and a trade
log (entry/exit dates, direction, z-scores, holding period, gross P&L, costs, net P&L and return).

### Metrics and benchmarks

* Net and gross cumulative return, CAGR (n/a for periods under 1 year), annualised volatility, Sharpe ratio
  (on excess returns over a constant or calendar-year risk-free rate, converted to a daily geometric rate; 252
  periods per year; the starting bar of a period, which has no holding period, is not charged the rate), max
  drawdown, Calmar, win rate, number of completed trades, average trade return, average holding period, turnover,
  total costs, average gross and net exposure, and % of time invested.
* Undefined metrics (zero volatility, no drawdown, no trades, too few observations) are **NaN / "n/a", never ±∞**.
* Benchmarks on identical dates: buy-and-hold A, buy-and-hold B, a 50/50 buy-and-hold of both, and cash at the
  risk-free rate. A capped long-short strategy is not like-for-like with a fully invested long-only stock, so
  compare risk-adjusted figures.

### Walk-forward evaluation

Each window fits on the previous `train_window` bars (default 504, about 2 years; rolling, or optionally
expanding). If configured, it requires the fit window to pass the cointegration gate (EG by default). It then trades the next `test_window`
bars (default 126, about 6 months) with frozen parameters and closes positions at the window end. The next window
starts where the previous one ended. Evaluation windows never overlap, capital is carried forward, and results
are chained in time order. The dashboard shows each window's fit and evaluation dates, β, p-value, whether it
traded, and its return.

### Robustness

* **Parameter grid** (entry × exit) on the **validation period only**, shown as heatmaps with the number of
  combinations tried. A "validation-selected" option exists, but it never looks at the test period.
* **Cost sensitivity:** net return over a range of per-leg costs (0–40 bps).
* **Walk-forward** results by window and by calendar year.
* **Universe screen:** Engle-Granger on training data only for all pairs, with raw and Benjamini-Hochberg-adjusted
  p-values, the number of tests, and warnings about expected false positives and survivorship bias.

---

## Machine-learning layer

The ML layer asks whether a classifier that predicts spread reversion can improve on the z-score rules. It makes
no assumption that it will. Code: `ml_features.py`, `ml_models.py`, `ml_strategy.py`, `ml_evaluation.py`.
Settings: the `ml:` section of `config/default_config.yaml`.

### Prediction target

With `z_t` the spread z-score at the close of bar *t*, computed with the **training-frozen** spread parameters:

```
y_t = 1   if  |z_{t+H}| <= |z_t| - tau        (the spread moved toward equilibrium by at least tau std devs)
y_t = 0   otherwise
y_t undefined when bar t+H is not available
```

Defaults: horizon **H = 5** sessions and tolerance **tau = 0.25** training standard deviations. "Toward its
estimated equilibrium" means toward the training-period spread mean. Only the value at *t + H* counts, not the path.
An overshoot through the mean counts only if |z| still ends at least tau closer. Rows are restricted to **eligible**
bars with `|z_t| >= min_abs_z` (default 1.0). Near equilibrium the label is trivially 0 (|z| < tau cannot fall by
tau), and such rows would inflate accuracy without describing tradable situations.

The label is the **only** quantity that uses future data. It is never a feature.

### Features (all computed from data up to and including bar *t*)

| Feature | Definition |
|---|---|
| `z` | Current spread z-score (training mean and std) |
| `abs_z` | Distance from the estimated equilibrium, \|z_t\| |
| `z_lag_1`, `z_lag_2`, `z_lag_5` | Lagged z-scores |
| `dz_1`, `dz_5` | Change in z over 1 and 5 bars |
| `reversion_speed_1`, `reversion_speed_5` | Movement toward equilibrium: −sign(z_t)·(z_t − z_{t−k}) |
| `momentum_10` | Average z slope over 10 bars: (z_t − z_{t−10}) / 10 |
| `spread_vol_ratio_20` | Trailing 20-bar std of daily spread changes ÷ the same std on the **fitting window** |
| `ret_a_1`, `ret_b_1`, `ret_a_5`, `ret_b_5` | Log returns of each asset over 1 and 5 bars |
| `corr_20` | Trailing 20-bar correlation of the two assets' daily log returns |
| `bars_beyond_entry` | Consecutive bars up to *t* with \|z\| ≥ the entry threshold |

Training-dependent quantities are the spread parameters (intercept, hedge ratio, mean, std) and the volatility
scale, all estimated on the fitting window. Every rolling window is trailing and every lag looks backwards. Rows
whose look-back is incomplete are excluded, not filled. A test checks that row *t* is identical whether or not
any later price exists or changes.

### Models

| Model | Pipeline | Hyperparameter grid (chosen on validation) |
|---|---|---|
| Logistic Regression (interpretable baseline) | `StandardScaler → LogisticRegression(class_weight="balanced")` | C ∈ {0.01, 0.1, 1, 10} |
| Random Forest (nonlinear comparison) | `RandomForestClassifier(200 trees, class_weight="balanced_subsample")` | max_depth ∈ {3, 6} × min_samples_leaf ∈ {10, 30} |

* All preprocessing (the scaler's means and variances) is fitted on purged **training rows only**.
* **Class imbalance:** balanced class weights. Class distributions are reported for every period.
* **Insufficient data:** a model is not fitted with fewer than 60 usable training rows, or fewer than 10 rows of
  either class. The dashboard reports the reason instead. Classification rates are reported as n/a when an
  evaluation set has fewer than 30 rows. ROC-AUC is n/a when only one class is present.

### Validation protocol (leakage controls)

| Step | Data used | Safeguard |
|---|---|---|
| Spread parameters and volatility scale | Training period | Frozen for validation and test |
| Training rows (optional `ml.training_features: point_in_time`) | Each `pit_block`-bar block of the fitting window | Spread parameters estimated only on the `pit_estimation_window` bars **before** the block, so training features and labels are built exactly like live predictions. The baseline uses the whole fitting window's parameters, including bars after each row |
| Classifier fit (including scaler) | Training rows with `t + H <= train_end − 1 − embargo` | **Purge:** no training label uses a validation price. Optional extra embargo (default 0). |
| Hyperparameters | Validation rows (purged at the validation/test boundary) | Selected by validation ROC-AUC (or log loss) |
| Probability threshold for B and C | Config (default 0.5), or best validation Sharpe over `threshold_grid` | The test period is never used |
| Final test | Test rows and dates, once revealed | Evaluated once with the fixed specification |
| Walk-forward | Each window's own fitting window | Spread model and both classifiers refitted on purged rows; hyperparameters and thresholds fixed from the steps above; the same cointegration gate applies to every variant |

Tests verify each safeguard. Altering validation prices leaves the fitted Logistic Regression coefficients
identical, which is what the purge guarantees. Altering test prices leaves every validation output unchanged: the
tuned hyperparameters, thresholds, probabilities, metrics and strategy results.

### Strategy variants

All variants decide at the close of bar *t* and trade at the close of *t + 1*. They share the same gross-exposure
sizing, two-leg commission and slippage, borrow fee, short-selling assumption and end-of-period liquidation, and
each period starts flat.

* **A. Traditional:** the z-score state machine (entry 2.0, exit 0.5, optional stop and maximum holding).
* **B. ML-only research signal:** when flat, enter if `p_t ≥ threshold` and `|z_t| ≥ min_abs_z`, betting on
  reversion (long spread if z < 0, short if z > 0). Exit, in priority order, on the stop-loss (if enabled), on
  reversion into the exit band, or after *H* bars. At expiry the position is renewed without trading only if the
  latest prediction still supports it. Missing predictions never open positions.
* **C. Hybrid:** A's entry condition *and* `p_t ≥ threshold` on the entry bar. If the model disagrees, the bar is
  skipped; the trade can still open later while |z| stays beyond the entry level. A's exits, stop-loss, maximum
  holding and re-entry rules are unchanged.

### ML evaluation

For training (in-sample), validation (used for tuning, so optimistic) and test (out-of-sample), the dashboard and
exports report: class distribution, accuracy **next to the majority-class accuracy**, precision, recall, F1,
ROC-AUC, confusion matrix, Brier score, log loss, a reliability (calibration) table, predicted probability against
the realised target, Random Forest impurity and permutation importance, and Logistic Regression standardised
coefficients. Strategy results are reported net of costs: cumulative return, CAGR (when at least 1 year),
annualised volatility, Sharpe ratio, max drawdown, turnover, trade count, and gross vs net.

**Why classification accuracy does not guarantee profitability:** accuracy weighs every row equally, but P&L
depends on the *size* of the moves the model gets right or wrong. Predictions on bars where no trade is taken add
nothing. A model that trades often can be right more often than not and still lose to costs. Under class
imbalance, "always predict the majority class" scores well while carrying no information. Consecutive rows also
share most of their *H*-bar label window, so the effective sample size is closer to rows / *H* than to the raw count.

---

## Audit and v2 research protocol

A Phase 1 audit (`docs/audit/PHASE1_AUDIT.md`) reproduced the original results exactly, then found four defects
and a set of methodological weaknesses. The defects:

* **D1:** the ML walk-forward reused hyperparameters tuned on a later validation period (look-ahead in 12 of 16
  windows).
* **D2:** NSE Muhurat special sessions, including weekend-dated bars, were traded as normal bars.
* **D3:** a selection metric was silently substituted.
* **D4:** trade logs were missing from snapshots.

The main weaknesses: a test period viewed repeatedly; TCS/INFY not cointegrated by any test; class-weighted
probabilities worse than a constant forecast; stale static parameters; no uncertainty estimates.

**How the fixes are delivered.** Every fix is an **opt-in config option**, so `config/default_config.yaml` still
reproduces the original baseline exactly (`tests/test_baseline_preserved.py`). `config/research_v2.yaml` enables
them all:

* nested, per-window hyperparameter tuning;
* unweighted models with a prevalence threshold, plus Brier and log-loss skill against a prevalence baseline;
* special sessions dropped (exchange-specific);
* a B0 control (B's rules without a model);
* stationary-bootstrap confidence intervals, Bonferroni-adjusted for the four ML candidates;
* a code-level holdout guard.

**Protocol.** `scripts/run_experiment.py` enforces it in three stages, recorded in
`docs/research/PREREGISTRATION.md`:

1. **Development** on 2015–2024. The guard refuses any later data.
2. **Freeze.** SHA-256 fingerprints of the engine, config, pre-registration and development results are stored in
   `docs/research/freeze.json`.
3. **One-time final evaluation** on the untouched 2025-01-01 to 2026-09-30 holdout. It refuses to run if any
   fingerprint changed, or a second time.

**Outcome** (`docs/research/RESEARCH_REPORT.md`; raw run outputs committed in `docs/research/results/`):

| | Development 2017–2024, pooled Sharpe [95% CI] | Holdout 2025-01-01 to 2026-09-30, pooled Sharpe [95% CI] |
|---|---|---|
| A. Traditional | 0.33 [−0.38, 1.03] | −0.38 [−1.18, 0.53] (1 trade) |
| B. ML-only (LR) / (RF) | 0.74 [0.08, 1.33] / 0.45 [−0.27, 1.12] | −0.63 / −0.53 |
| C. Hybrid (LR) / (RF) | 0.42 / 0.39 | identical to A (the filter never bound) |
| B0. B's rules, no model | 0.48 [−0.16, 1.11] | −0.64 |
| Pooled ROC-AUC, LR / RF | 0.523 / 0.515 (Brier skill −0.11 / −0.02) | 0.524 / 0.536 (Brier skill −1.50 / −0.10) |

The pre-registered decision is **reject the tested hypothesis**. No ML candidate passes the development
classification gate, and none beats A on the holdout. A's holdout Sharpe is negative. Most of B-LR's development
edge comes from its rules: B-LR − B0 = +0.26 [−0.09, 0.65]. Power is low (the cointegration gate allowed trading in
2 of 12 holdout pair-windows), so "reject" means the evidence does not support the hypothesis, not proof that no
edge exists.

---

## v3 protocol: realism and robustness upgrades

The audit's remaining open items are now engine options. Each is off by default, so the baseline and v2 results
are unchanged. `config/research_v2.yaml` still reproduces the frozen v2 development run exactly.
`config/research_v3.yaml` turns them on together:

| Option | v3 setting |
|---|---|
| Execution | Fill at the **open** of t + 1 (`strategy.execution_price: open`) |
| Time stop | Exit after ⌈3 × fitting-window half-life⌉ bars (`strategy.max_holding_half_lives: 3`) |
| Cointegration gate | Engle-Granger **and** Johansen at 5% (`cointegration.gate: eg_and_johansen`); reverse EG recorded |
| NSE pairs (`markets.NS`) | Statutory cost schedule, single-stock-futures shorts with monthly rolls, Rs 20 per futures order, 5 bps slippage |
| US pair | Generic 5 + 5 bps, cash shorts with a 50 bps borrow fee (as in v2) |
| Cash and Sharpe | Cash interest at approximate calendar-year T-bill averages (US 3-month, India 91-day); Sharpe on excess returns |
| ML training rows | Point in time: 21-bar blocks, parameters from the 252 bars before each block |
| Static workflow | Gate on the training fit; refit on train + validation before the test period |

The study is pre-registered in `docs/research/v3/PREREGISTRATION.md`:

* development data from 2015-01-01 to 2026-09-30 (this includes the spent v2 holdout, as disclosed there);
* a holdout from **2026-10-01 to 2028-09-30**, which does not exist yet;
* the v2 decision rules.

The runner refuses the final stage before the holdout ends, without a freeze, or a second time.

**Development results** (2017-01-03 to 2026-09-30; [`docs/research/v3/DEVELOPMENT_REPORT.md`](docs/research/v3/DEVELOPMENT_REPORT.md)):

| | Pooled excess-return Sharpe [95% CI] | Trades |
|---|---|---|
| A. Traditional | 0.02 [−0.65, 0.63] | 10 |
| B. ML-only (LR) / (RF) | 0.39 [−0.21, 0.89] / 0.17 [−0.39, 0.71] | 35 / 30 |
| C. Hybrid (LR) / (RF) | 0.22 / 0.22 | 9 / 8 |
| B0. B's rules, no model | 0.06 [−0.50, 0.62] | 23 |
| Pooled ROC-AUC, LR / RF | 0.502 / 0.500 (Brier skill −0.09 / −0.01) | |

* **No ML skill.** Point-in-time training did not create predictive skill. Gate (a) of the decision rules fails
  for both models, so no ML candidate can be retained whatever the holdout shows. This is disclosed before any
  holdout data exists.
* **Rarely traded.** The stricter gate traded 6 of 58 pair-windows (Engle-Granger alone: 10), and the book is flat
  about 96% of the time.
* **The headline returns are mostly interest.** The 52–63% cumulative returns are mostly T-bill interest on idle
  cash. With cash interest off, A returns −0.1%.
* **The realism upgrades bite for active strategies.** Re-pricing the same signals under v2's assumptions raises
  B-LR's Sharpe from 0.39 to 0.52 but leaves A's near zero.

The specification is not frozen yet. Freezing (`--freeze`) and the one-time holdout evaluation after 2028-09-30
are the remaining steps.

---

## Dashboard

`streamlit run app.py` opens at http://localhost:8501. The sidebar holds the data source, tickers, date range,
split proportions, significance level, entry/exit/stop/max-holding settings, execution lag, commission, slippage,
borrow fee, short-selling assumption, capital, gross-exposure limit, risk-free rate, walk-forward settings, a
**research protocol** selector, the realism and robustness options (all off in the baseline), ML settings, the
source of the test-period thresholds, and the **"Reveal final test-period results"** switch:

* **Protocols:** Baseline; v2 audited (`config/research_v2.yaml`'s data and ML settings); v3 realistic
  (`config/research_v3.yaml`).
* **Realism and robustness options:** cointegration gate, static gate and refit, half-life time stop, close or
  open fills, generic or NSE cost model, cash or futures shorts, cash interest with a constant or calendar-year
  rate.
* **ML settings:** on/off, horizon, tolerance, minimum |z|, probability threshold and its source, embargo, and
  in-sample or point-in-time training rows.

Under v3 the defaults follow the protocol file and the market of leg A (`.NS`), and every value stays editable. Under the baseline protocol the
ML tabs warn that walk-forward hyperparameters come from the static validation period (audit finding D1).

| Tab | Contents |
|---|---|
| A. Overview | Assets, range, observation count, data quality, provenance (source, retrieval time, hash), split table |
| B. Exploratory analysis | Prices, normalised prices, returns, rolling volatility and correlation, log-price scatter with training fit, summary statistics, missing data |
| C. Cointegration | Training-period regression, β, α, EG statistic, p-value, critical values, I(1) checks, interpretation; EG with the legs swapped, Johansen trace and max-eigenvalue tests, the three hedge ratios and every gate's verdict |
| D. Spread and signals | Spread and z-score with period shading, thresholds, signal markers, and signal → execution log |
| E. Portfolio and backtest | Leg weights, exposure, equity vs benchmarks, drawdown, gross vs net, cumulative costs, positions, trades |
| F. Performance report | Metrics by period (training / validation / test / walk-forward), strategy vs benchmarks, trade returns |
| G. Robustness | Validation heatmaps, cost sensitivity, walk-forward windows, yearly returns, universe screen |
| H. Assumptions | Price adjustment, short selling (cash or futures), borrow and cash interest, costs (generic or the NSE schedule with all-in rates), fill timing, risk controls, cointegration gate, estimation windows, data source |
| I. ML models | Target definition, samples and evaluation dates per period, model settings and tuning tables, classification metrics (training / validation / test clearly labelled), predicted probability vs actual target, confusion matrix, calibration, RF importance, LR coefficients |
| J. Strategy comparison (A/B/C) | Metrics table, equity curves, drawdowns and transaction costs for A, B and C (LR and RF) on identical dates. Period switch: training, validation, test, walk-forward. Walk-forward per-window heatmap and table. Validation threshold grids. |
| Export | CSV downloads (returns, equity, positions, trades, signals, metrics, ML predictions, metrics, calibration, importance, comparisons), statistical tests JSON, config YAML, run snapshot |

Errors are actionable: invalid or delisted tickers, failed downloads, insufficient history, degenerate pairs, and
invalid parameter combinations (for example exit ≥ entry) each produce a specific message.

---

## Reproducibility

* All parameters live in `config/default_config.yaml` (sources, tickers, dates, splits, significance, hedge-ratio
  method, thresholds, costs, capital and exposure limits, annualisation, walk-forward settings, random seed).
  Unknown keys are rejected.
* `save_research_run()` (dashboard Export tab, or `scripts/run_research.py`) writes
  `reports/<UTC timestamp>_<A>_<B>_<config hash>/` containing `config_snapshot.yaml`, `run_metadata.json` (data
  source, retrieval timestamp, SHA-256 of the inputs and of the aligned prices, validation and alignment reports,
  splits, package versions), `statistical_tests.json`, and every table as CSV. With the ML layer on, it also
  writes `ml_settings.json` (target definition, features, selected hyperparameters, thresholds, training-row date
  ranges) plus `ml_*` and `strategy_comparison_*` tables.
* Cached downloads keep their original hash. If a cached file changes, a warning is raised. **Free providers may
  revise or remove historical data**, so a later re-download can differ; compare the hashes.
* Synthetic data and tests are deterministic (fixed seeds). Exact package versions are in `requirements-lock.txt`.
* Results are bit-identical on x86-64 Windows and Linux across Python 3.10 to 3.14 and the supported library ranges
  (checked in CI). On Apple Silicon (arm64), Random Forest fits differ in the last digits (the golden test-period RF
  Brier score is 0.2510946 instead of 0.2510814), so ML results there can differ slightly from the published runs.
* Each pre-registered study keeps its pre-registration, freeze record and holdout marker in its own
  `holdout.protocol_dir` (`docs/research` for v2, `docs/research/v3` for v3). Every experiment run records the
  SHA-256 of the engine code, config and pre-registration. Committed result copies use `-text` in `.gitattributes`
  so the hashes verify on every platform.

---

## Testing

```bash
pytest            # 265 tests, about 4-6 min, fully offline
```

| File | Covers |
|---|---|
| `test_data_validation.py` | Sorting, missing/invalid/non-positive prices, exact vs conflicting duplicates, insufficient history, gaps, extreme moves, CSV and yfinance-cache loading (offline), hash-change detection, synthetic labelling |
| `test_alignment.py` | Inner-join alignment, reporting of dropped dates, no forward fill, same-interval returns for both legs, cross-exchange warning |
| `test_cointegration.py` | Known cointegrated vs independent series, empirical test size about 5% over 100 seeds, a documented false positive, output structure, edge cases (short, misaligned, NaN, constant, collinear), Benjamini-Hochberg vs statsmodels |
| `test_hedge_ratio.py` | Exact and noisy known relations, invalid inputs, **fitted parameters unchanged when post-training prices are altered** |
| `test_spread.py` | Hand-calculated spread and z-score, degenerate std, training moments reused out of sample, half-life |
| `test_signals.py` | Entry/exit for both signs, no duplicate entries, exit-then-later re-entry, stop-loss and max-holding blocks, NaN handling, no same-bar reversals on random paths |
| `test_backtester.py` | Execution lag, **signals never earn their own move**, **truncation invariance (no look-ahead)**, position direction and dollar ratio, both-leg P&L alignment, both-leg costs by hand, borrow accrual, accounting identities, drift rebalance, final-bar liquidation, last-bar signals ignored |
| `test_metrics.py` | Known CAGR/volatility/Sharpe values, drawdowns by hand, undefined metrics → NaN, trade statistics, benchmarks |
| `test_evaluation.py` | Chronological splits, walk-forward schedule and capital chaining, untraded windows, test period hidden until revealed, **validation and grid results unaffected by test-period data**, run snapshots |
| `test_universe_and_config.py` | Screen counts and BH correction, training-only screening, config round-trip and validation |
| `test_ml_features.py` | **Target alignment** by hand (horizon, tolerance, overshoot, boundary NaNs), eligibility, feature values by hand, **feature causality (truncation invariance)**, training-only volatility scale, **purge masks**, training rows unchanged when later prices change |
| `test_ml_models.py` | **Scaler fitted on training rows only**, **prediction timestamps and causality**, insufficient-sample and class guards, metrics against a hand-computed confusion matrix, undefined metrics, calibration table, tuning unaffected by test data, importance outputs |
| `test_ml_strategy.py` | Hybrid = traditional when the model always agrees, hybrid never enters without confirmation, ML-only direction, entry rules, horizon exit vs renewal, exit band and stop-loss, no same-bar reversals, **next-bar execution**, end-of-sample liquidation, both-leg costs |
| `test_ml_evaluation.py` | Purged training and validation rows, **identical dates and backtest settings for all variants**, test hidden until revealed, **validation outputs unaffected by test prices**, **training fit unaffected by validation prices**, ML walk-forward reproduces the original walk-forward for A, chaining, insufficient-data handling, exports and snapshots |
| `test_app.py` | Headless Streamlit runs of the real `app.py` (synthetic data, revealed test, failed pair, invalid input, ML tabs, ML switched off, v2 protocol with B0) |
| `test_baseline_preserved.py` | **The default config reproduces the pre-audit golden outputs exactly** (`tests/data/golden_baseline_synthetic.json`, captured before any fix). The statistical layer is compared on every platform; the ML outputs on x86-64, because Random Forest fits differ in the last digits on Apple Silicon |
| `test_audit_fixes.py` | One or more regression tests per audit fix: **nested walk-forward is point-in-time** (later prices never change earlier windows), static-mode disclosure, unsafe option combination rejected, holdout schedule anchor, special-session detection / drop / point-in-time volume rule / exchange-specific dates, calmar selection, exported trade logs, holdout guard, unweighted LR matches the base rate, prevalence-baseline skill by hand, Sortino by hand, stationary bootstrap, bootstrap intervals, effective sample size, B0 control |
| `test_experiment.py` | Union-calendar pooling, **repricing reproduces the walk-forward exactly**, cost sensitivity monotone, nested tuning required, pooled classification vs prevalence baseline by hand, **every branch of the pre-registered decision rules** |
| `test_realism.py` | NSE rates by hand, generic model = original costs, lot rounding, **next-open fills earn only after the fill**, open-mode truncation invariance, close fallback, adjusted opens, futures round trip / rolls / lots by hand, cash interest base, risk-free schedules, **a flat book earns exactly the risk-free rate (zero excess)** in single periods and chained walk-forward windows |
| `test_cointegration_suite.py` | Every gate on cointegrated and independent pairs, EG direction asymmetry, Johansen input checks and recorded failures, gate validation, per-market config overrides |
| `test_research_options.py` | Half-life time stop (static and per window), static gate and pre-test refit, gate variants in walk-forward, next-open execution end to end, NSE costs with futures shorts and interest, untouched US pair, excess-return pooling, **repricing reproduces open-execution NSE runs exactly**, **point-in-time rows read no later data**, point-in-time static and walk-forward ML, ML following the refit and the gate |

Statistical tests are probabilistic, so fixed seeds are used and the size check uses generous binomial bounds.
GitHub Actions runs the suite on Linux, Windows and macOS with Python 3.10 to 3.14 on every push to `main` and on
pull requests.

---

## Example results

*These are the **baseline-protocol** results, kept for reference. The B/C walk-forward figures below are affected
by audit finding D1, and the 2022–2024 test period was later found to be contaminated (M1). The audited v2
results are in [Audit and v2 research protocol](#audit-and-v2-research-protocol).*

Run with the default configuration (entry 2.0, exit 0.5, 5 + 5 bps per leg per side, 50 bps per year borrow,
1.0× gross, execution lag 1). Data: Yahoo Finance `Adj Close`, 2015-01-01 to 2024-12-31, retrieved
2026-10-20. Thresholds were fixed in advance, so the test period was evaluated once with no tuning. These are
**historical research results on three hand-picked candidate pairs, not evidence of future profitability.**

| Pair | Train EG p-value | Training (in-sample) | Validation | Final test | Walk-forward OOS |
|---|---|---|---|---|---|
| KO / PEP | 0.080 (fails 5%) | +20.2% (4 trades) | 0.0% (no trades) | −3.3% (1 trade) | 0.0% (all 16 fit windows fail EG; no trades) |
| HDFCBANK.NS / ICICIBANK.NS | 0.037 (passes) | +43.3% (2 trades) | −5.4% (1 trade) | −14.2% (1 trade, held 99.6% of the period) | −8.5% (1 of 16 windows traded) |
| TCS.NS / INFY.NS | 0.084 (fails 5%) | +18.4% (3 trades) | −7.6% (1 trade) | +6.9% (2 trades) | +21.5% (10 trades in 9 traded windows) |

What these show:

* **In-sample results flatter.** Every pair is profitable on its own training data and much weaker out of sample.
* **Passing the test is not enough.** HDFCBANK/ICICIBANK is the only pair that passes on training data, and it
  loses money in both out-of-sample periods. Its walk-forward β ranges from −0.10 to 1.39, a sign of an unstable
  relationship.
* **Few trades.** With 1–10 out-of-sample trades per pair, none of these figures is statistically meaningful.
* With three pairs and several periods examined, at least one positive result was likely by chance.

These runs therefore **do not** support a claim that the strategy is robustly profitable after costs.

### Traditional vs ML-only vs hybrid

Same data and default settings, with ML defaults fixed in the config before any run: H = 5, tau = 0.25,
min |z| = 1.0, probability threshold 0.5 (pre-specified, not tuned), hyperparameters chosen on validation ROC-AUC.
Net cumulative return, with completed trades in brackets:

| Pair | Period | A. Traditional | B. ML-only (LR) | C. Hybrid (LR) | B. ML-only (RF) | C. Hybrid (RF) |
|---|---|---|---|---|---|---|
| KO / PEP | Validation | +0.0% (0) | +2.5% (5) | +0.0% (0) | +0.1% (8) | +0.0% (0) |
| KO / PEP | Final test | −3.3% (1) | −1.9% (3) | −3.3% (1) | −3.3% (5) | −3.3% (1) |
| KO / PEP | Walk-forward | +0.0% (0) | +0.0% (0) | +0.0% (0) | +0.0% (0) | +0.0% (0) |
| HDFCBANK.NS / ICICIBANK.NS | Validation | −5.4% (1) | −24.9% (18) | −5.4% (1) | −3.8% (10) | −5.4% (1) |
| HDFCBANK.NS / ICICIBANK.NS | Final test | −14.2% (1) | −14.1% (5) | −14.2% (1) | −7.2% (16) | −13.0% (1) |
| HDFCBANK.NS / ICICIBANK.NS | Walk-forward | −8.5% (1) | −6.2% (5) | −8.1% (1) | −11.6% (9) | −8.5% (1) |
| TCS.NS / INFY.NS | Validation | −7.6% (1) | −7.5% (7) | −4.3% (1) | −6.0% (16) | −4.3% (1) |
| TCS.NS / INFY.NS | Final test | +6.9% (2) | +6.8% (9) | +6.8% (2) | −2.0% (15) | +6.8% (2) |
| TCS.NS / INFY.NS | Walk-forward | +21.5% (10) | +33.1% (25) | +24.3% (10) | +25.4% (24) | +22.2% (10) |

Classification quality:

| Pair | Positive rate (train / val / test) | LR ROC-AUC (val / test) | RF ROC-AUC (train / val / test) | Test accuracy LR / RF vs majority class |
|---|---|---|---|---|
| KO / PEP | 23% / 28% / 24% | 0.67 / 0.44 | 0.81 / 0.58 / 0.48 | 0.31 / 0.27 vs 0.76 |
| HDFCBANK.NS / ICICIBANK.NS | 21% / 5% / 3% | 0.58 / 0.59 | 0.88 / 0.58 / 0.53 | 0.07 / 0.82 vs 0.97 |
| TCS.NS / INFY.NS | 26% / 16% / 14% | 0.50 / 0.56 | 0.82 / 0.48 / 0.53 | 0.59 / 0.59 vs 0.86 |

What these show:

* **Little predictive power out of sample.** Test ROC-AUC ranges from 0.44 to 0.59 (mean 0.52), and accuracy is
  below the always-predict-the-majority-class baseline in all 12 out-of-sample model/period cells. The Random Forest
  reaches 0.81–0.88 ROC-AUC on its own training rows, which is a clear sign of overfitting.
* **The base rate moves.** For HDFCBANK/ICICIBANK, the share of rows that reverted fell from 21% (train) to 5% and
  3%, so models trained on a mean-reverting regime faced a trending one. For TCS/INFY, Logistic Regression
  probabilities saturate near 1 when |z| drifts far beyond its training range, while the spread does not revert.
  This is linear extrapolation outside the training distribution (visible in tab I).
* **The hybrid filter rarely binds at 0.5.** C matched A exactly in 7 of 12 validation/test cells. Where it
  differed, it delayed an entry; the differences are small and go both ways.
* **ML-only trades much more.** B made 3–18 trades per validation or test period where A made 0–2, and its
  transaction costs were 2.6×–15.1× A's where A traded. Its results are mixed: the best walk-forward result for TCS/INFY (+33.1% vs +21.5%), the worst
  validation result for HDFCBANK/ICICIBANK (−24.9%), and a loss on the TCS/INFY test (RF −2.0% vs A +6.9%).
* **No variant is consistently better** across pairs and across validation, test and walk-forward. With 5 variants
  × 3 pairs × 3 periods compared, some cells look better by chance. These runs provide **no evidence that the ML
  models improve the traditional strategy**.

Reproduce with `python scripts/run_research.py --source yfinance --tickers TCS.NS INFY.NS --reveal-test` (and the
other pairs). Results can change if the provider revises its data.

---

## Research integrity and limitations

* **Cointegration does not guarantee profitable trading.** A significant test is in-sample evidence. The spread
  may revert too slowly, too little, or not at all after costs.
* **Historical relationships break.** Mergers, regulation, business-model changes and index events can end a
  relationship. In the example above, HDFCBANK/ICICIBANK passed the training test and then lost money out of
  sample. This platform does not diagnose *why* a relationship failed.
* **Free historical data may be incomplete or revised.** Yahoo data can change, be back-filled or disappear. Hashes
  are recorded so changes can be detected.
* **Corporate actions affect prices.** Adjusted closes depend on the provider's adjustment methodology; errors in
  split or dividend adjustment produce false signals. Suspicious jumps are flagged but not corrected.
* **Short selling may be unavailable or expensive.** Borrow can be scarce, recalled or banned. Indian cash
  equities generally cannot be shorted overnight outside the securities lending scheme (SLB) or derivatives.
* **Borrowing, financing, liquidity and market impact** can reduce or eliminate returns. They are modelled by
  linear costs, a flat borrow fee or futures rolls, and cash interest at approximate annual rates. Market impact,
  futures basis and margin calls are not modelled.
* **Parameter searches and pair selection create multiple-testing bias.** Every pair, threshold and window that is
  tried, formally or informally, weakens the evidence of the result that is finally reported. Benjamini-Hochberg
  only corrects for the tests it is given.
* **Survivorship bias:** universes built from today's listed companies leave out failed and delisted firms.
* **Simplified execution:** fills at the next bar's close or open, no partial fills, no intraday risk, no
  non-synchronous closes across exchanges.
* **ML-specific limitations:**
  * Samples are small and overlapping. A few hundred eligible rows per period, with neighbouring rows sharing
    most of their 5-bar label window, means effective sample sizes are much smaller.
  * Classifiers assume the training distribution persists. Regime shifts in the base rate and extrapolation
    beyond the training range of |z| break that assumption, as the examples show.
  * Hyperparameters are tuned on validation, so validation classification and strategy results are optimistic.
    Walk-forward windows that overlap the validation period are not independent of that tuning.
  * Static and walk-forward models use one pair's history only. Pooling many pairs would give more data but
    introduces cross-sectional leakage questions not handled here.
  * Feature importance describes the fitted model, not causal drivers of reversion. Correlated features share and
    swap importance.
* **This project does not execute live trades and is not financial advice.**

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `No data returned for 'XYZ'` | Check the symbol (NSE needs `.NS`, e.g. `INFY.NS`), the date range, and your network. Yahoo is sometimes rate-limited; retry later or upload a CSV. |
| `Insufficient history` / `Only N overlapping observations` | Extend the date range or choose assets with a longer common history (minimum 252 by default). |
| `Training spread standard deviation ... degenerate` | One series is (almost) a scaled copy of the other; the pair cannot be traded as a spread. |
| `Invalid settings: strategy.exit_z must satisfy ...` | The exit threshold must be below the entry threshold, and the stop-loss above it. |
| `Short selling is disabled ...` | Enable the short-selling assumption, or accept that this strategy cannot be run. |
| Stale or changed data | Press "Force re-download", or delete the files in `data/raw/`. |
| `ModuleNotFoundError: statarb` (notebook or scripts) | Run from the project root, or `pip install -e .`. |
| `... not fitted: only N usable training rows` / `class counts ...` | Too few eligible rows or too few reversions: lower `min_abs_z`, lengthen the training window, or accept that the ML layer cannot be evaluated on this pair. |
| Dashboard slow after changing ML settings | The ML layer makes about 42 model fits per run: 10 for the static tuning grids and final fits, plus 2 per walk-forward window. Results are cached per setting; switch "Run ML comparison" off while exploring the statistical layer. |
| Streamlit port in use | `streamlit run app.py --server.port 8502` |

---

## Future improvements

* Kalman-filter or rolling-OLS dynamic hedge ratios, with tests for look-ahead.
* Trading on symmetric hedge ratios (total least squares, or the Johansen vector already reported), and a
  bootstrap Johansen test with correct size.
* Volume-based market-impact and capacity models; futures basis and time-varying NSE lot sizes.
* Daily risk-free series from official sources (FRED, RBI) instead of approximate annual averages.
* Volatility-targeted position sizing, and portfolio-level aggregation across many pairs.
* Structural-break diagnostics (Chow, CUSUM) and regime filters.
* The deflated Sharpe ratio for multiple testing.
* Point-in-time universes to reduce survivorship bias.
* Pool training rows across many pairs (with pair-level purging) to give the classifiers enough data.
* Probability calibration (Platt or isotonic) fitted on a separate calibration split, and abstention when |z| is
  outside the training range.
* Meta-labelling targets defined on actual trade outcomes net of costs, rather than on spread moves.

---

## Resume summary

Accurate wording, based on what this repository actually does:

> Built a Python statistical-arbitrage and machine-learning research platform (pandas, statsmodels,
> scikit-learn, Plotly, Streamlit) for cointegration-based pairs trading. Features: training-only Engle-Granger
> testing and hedge-ratio estimation, a finite-state z-score signal engine, and a two-leg backtester with delayed
> execution and transaction, slippage and borrow costs. Added Logistic Regression and Random Forest classifiers of
> spread mean reversion with causal features, purged chronological labels and validation-only tuning, and compared
> traditional, ML-only and hybrid strategies on identical out-of-sample dates with walk-forward re-estimation.
> Audited the platform, found and fixed a hyperparameter look-ahead and other defects behind opt-in, regression-
> tested options (265 offline pytest tests, including golden-baseline, look-ahead and label-leakage checks,
> run in CI on Linux, Windows and macOS). Ran a pre-registered experiment with nested walk-forward tuning,
> bootstrap confidence intervals and a one-time 2025–2026 holdout. The pre-registered decision rejected the
> hypothesis: the ML models showed no out-of-sample skill and the pairs strategy showed no reliable edge after
> costs. Extended the engine with next-open execution, an NSE statutory cost model with single-stock-futures
> shorting, cash interest at calendar-year rates, Engle-Granger in both directions plus Johansen gates and
> point-in-time ML training rows, and pre-registered a v3 study with a 2026–2028 holdout.

Do not claim the strategy, or the ML enhancement, is profitable or superior: the evidence above does not support
that.

---

*References:* Engle & Granger (1987), *Econometrica* 55(2); MacKinnon (2010), *Critical Values for Cointegration
Tests*, Queen's Economics Dept. WP 1227; Gatev, Goetzmann & Rouwenhorst (2006), *Review of Financial Studies*
19(3); Benjamini & Hochberg (1995), *JRSS B* 57(1); López de Prado (2018), *Advances in Financial Machine
Learning*, ch. 7 (purging and embargoing).
