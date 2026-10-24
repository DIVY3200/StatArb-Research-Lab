"""StatArb Research Lab - Streamlit dashboard.

Run with:  streamlit run app.py

Every number shown is computed by the ``statarb`` package from the selected data at run
time; nothing is hard-coded. Research and education only: no live trading, not advice.
"""

from __future__ import annotations

import dataclasses
import io
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
import yaml  # noqa: E402

from statarb import diagnostics as dx  # noqa: E402
from statarb import visualization as viz  # noqa: E402
from statarb.cointegration import ASSUMPTIONS, LIMITATIONS  # noqa: E402
from statarb.config import ResearchConfig, load_config, resolve_path  # noqa: E402
from statarb.data_loader import AssetData, DataLoadError, download_yfinance, generate_synthetic_pair, load_csv  # noqa: E402
from statarb.data_validation import AlignedPair, prepare_pair_from_config, truncate_list, validate_price_frame  # noqa: E402
from statarb.evaluation import (  # noqa: E402
    PERIOD_LABELS,
    PairResearchResult,
    result_tables,
    run_pair_research,
    save_research_run,
    statistical_tests_payload,
)
from statarb.ml_evaluation import (  # noqa: E402
    MLResearchResult,
    classification_summary,
    ml_result_tables,
    run_ml_research,
)
from statarb.ml_features import FEATURE_DESCRIPTIONS  # noqa: E402
from statarb.ml_models import MODEL_LABELS, SHORT_LABELS  # noqa: E402
from statarb.ml_strategy import VARIANT_LABELS  # noqa: E402
from statarb.portfolio import target_leg_weights  # noqa: E402
from statarb.universe import screen_pairs, synthetic_universe  # noqa: E402

st.set_page_config(page_title="StatArb Research Lab", page_icon=":material/query_stats:", layout="wide")

BASE = load_config()
V2 = load_config(ROOT / "config" / "research_v2.yaml")  # audited protocol; the dashboard reuses its exact settings
V2_ML_KEYS = ("walk_forward_tuning", "inner_validation_fraction", "class_weight", "threshold_mode",
              "prevalence_multiplier", "tuning_metric", "include_prevalence_baseline")
V2_DATA_KEYS = ("special_sessions", "special_session_dates", "special_session_dates_by_exchange",
                "special_session_volume_ratio")
SOURCES = {"synthetic": "Synthetic demo (offline)", "yfinance": "Yahoo Finance (yfinance)", "csv": "CSV upload"}
PCT_KEYS = ("return", "drawdown", "volatility", "cagr", "win_rate", "pct", "leverage")
COUNT_KEYS = ("observations", "trades", "rebalances", "rows")


# =========================================================================== cached computations
@st.cache_data(show_spinner=False, ttl=6 * 3600)
def cached_download(ticker: str, start: str, end: str, refresh_token: int) -> AssetData:
    return download_yfinance(ticker, start, end, cache_dir=resolve_path(BASE.data.cache_dir),
                             force_refresh=refresh_token > 0)


@st.cache_data(show_spinner=False)
def cached_synthetic(cointegrated: bool, seed: int, n_obs: int) -> tuple[AssetData, AssetData]:
    s = BASE.synthetic
    return generate_synthetic_pair(n_obs=n_obs, seed=seed, cointegrated=cointegrated, hedge_ratio=s.hedge_ratio,
                                   intercept=s.intercept, half_life=s.half_life, residual_vol=s.residual_volatility)


@st.cache_data(show_spinner=False)
def cached_csv(payload: bytes, name: str) -> dict[str, AssetData]:
    return load_csv(payload, source_name=name)


@st.cache_data(show_spinner=False, max_entries=24)
def cached_research(pair_key: str, config_json: str, reveal_test: bool, _pair: AlignedPair) -> PairResearchResult:
    return run_pair_research(_pair, ResearchConfig.from_dict(json.loads(config_json)), reveal_test=reveal_test)


@st.cache_data(show_spinner=False, max_entries=12)
def cached_ml(pair_key: str, config_json: str, reveal_test: bool, _base: PairResearchResult) -> MLResearchResult:
    config = ResearchConfig.from_dict(json.loads(config_json))
    return run_ml_research(dataclasses.replace(_base, config=config))


# =========================================================================== formatting helpers
def fmt_value(name: str, value) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        key = name.lower()
        if any(k in key for k in COUNT_KEYS) and float(value).is_integer():
            return f"{int(value):,}"
        if "equity" in key or ("costs" in key and "pct" not in key):
            return f"{value:,.0f}"
        if "days" in key:
            return f"{value:.1f}"
        if "turnover" in key:
            return f"{value:.2f}x"  # traded notional as a multiple of equity
        if any(k in key for k in PCT_KEYS):
            return f"{value:.2%}"
        return f"{value:.3f}"
    return str(value)


def fmt_table(table: pd.DataFrame) -> pd.DataFrame:
    """Format a metrics table whose index holds metric names."""
    return pd.DataFrame({col: [fmt_value(str(idx), v) for idx, v in table[col].items()] for col in table.columns},
                        index=[str(i).replace("_", " ") for i in table.index])


def tidy(frame: pd.DataFrame, decimals: int = 6) -> pd.DataFrame:
    """Display copy: dates without times, floats rounded (removes 1e-16 noise and negative zeros)."""
    out = frame.copy()
    if isinstance(out.index, pd.DatetimeIndex):
        out.index = out.index.strftime("%Y-%m-%d")
    for col in out.columns:
        series = out[col]
        if pd.api.types.is_datetime64_any_dtype(series):
            out[col] = series.dt.strftime("%Y-%m-%d").fillna("")
        elif pd.api.types.is_float_dtype(series):
            out[col] = series.round(decimals) + 0.0
    return out


def fmt_p(value: float) -> str:
    return "< 0.0001" if value < 1e-4 else f"{value:.4f}"


def table_height(n_rows: int) -> int:
    return int(35 * (n_rows + 1) + 3)


def is_dark() -> bool:
    theme = getattr(st.context, "theme", None)
    return getattr(theme, "type", None) == "dark"


def chart(fig, key: str) -> None:
    st.plotly_chart(fig, key=key, config={"displaylogo": False})


def csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv().encode("utf-8")


def show_list(items: list[str], kind: str = "warning") -> None:
    for item in items:
        getattr(st, kind)(item)


# =========================================================================== sidebar
def sidebar() -> dict:
    sb = st.sidebar
    sb.title("StatArb Research Lab")
    sb.caption("Research and education only. No live trading. Not financial advice.")

    sb.header("Data")
    source = sb.radio("Data source", list(SOURCES), format_func=SOURCES.get, index=0)
    params: dict = {"source": source}
    if source == "yfinance":
        options = [f"{a} / {b}" for a, b in BASE.data.candidate_pairs] + ["Custom"]
        choice = sb.selectbox("Candidate pair (not assumed cointegrated)", options, index=len(options) - 2)
        default_a, default_b = (BASE.data.asset_a, BASE.data.asset_b) if choice == "Custom" else choice.split(" / ")
        params["ticker_a"] = sb.text_input("Asset A ticker (dependent leg)", default_a).strip().upper()
        params["ticker_b"] = sb.text_input("Asset B ticker", default_b).strip().upper()
        params["start"] = sb.date_input("Start date", date.fromisoformat(BASE.data.start_date), min_value=date(1990, 1, 1))
        params["end"] = sb.date_input("End date (inclusive)", date.fromisoformat(BASE.data.end_date))
        if sb.button("Force re-download", help="Ignore the local cache and download again (provider data can change)."):
            st.session_state["refresh_token"] = st.session_state.get("refresh_token", 0) + 1
        sb.caption("NSE symbols need the `.NS` suffix (e.g. `TCS.NS`). Downloads are cached in `data/raw/`.")
    elif source == "csv":
        params["files"] = sb.file_uploader(
            "Upload CSV file(s)", type=["csv"], accept_multiple_files=True,
            help="Either one long-format file with a Ticker column, or one file per asset (named after the ticker). "
                 "Columns: Date, Open, High, Low, Close, Adj Close (optional), Volume.",
        )
    else:
        params["cointegrated"] = sb.radio("Synthetic relation", ["Cointegrated", "Independent random walks"]) == "Cointegrated"
        params["seed"] = int(sb.number_input("Random seed", 0, 10_000, BASE.synthetic.seed))
        params["n_obs"] = int(sb.slider("Observations", 600, 5000, BASE.synthetic.n_observations, step=60))
        sb.caption("Synthetic data is generated, clearly labelled, and never market history.")
    params["prefer_adjusted"] = sb.toggle("Prefer adjusted close", BASE.data.prefer_adjusted)

    sb.header("Research design")
    train = sb.slider("Training share", 0.40, 0.80, BASE.splits.train, 0.05)
    validation = sb.slider("Validation share", 0.10, 0.30, BASE.splits.validation, 0.05)
    test = round(1.0 - train - validation, 4)
    sb.caption(f"Final test share: **{test:.0%}** (remainder, chronologically last)")
    significance = sb.select_slider("Cointegration significance level", [0.01, 0.05, 0.10], BASE.cointegration.significance)

    sb.header("Strategy")
    col1, col2 = sb.columns(2)
    entry = col1.number_input("Entry |z|", 0.5, 5.0, BASE.strategy.entry_z, 0.25)
    exit_ = col2.number_input("Exit |z|", 0.0, 4.5, BASE.strategy.exit_z, 0.25)
    use_stop = sb.toggle("Stop-loss", BASE.strategy.stop_loss_z is not None)
    stop = sb.number_input("Stop-loss |z|", 1.0, 10.0, BASE.strategy.stop_loss_z or 4.0, 0.25) if use_stop else None
    use_hold = sb.toggle("Maximum holding period", BASE.strategy.max_holding_days is not None)
    max_hold = int(sb.number_input("Max holding (trading days)", 1, 500, BASE.strategy.max_holding_days or 60)) if use_hold else None
    lag = int(sb.number_input("Execution lag (bars, >= 1)", 1, 5, BASE.strategy.execution_lag,
                              help="Signal at close t is traded at close t + lag."))

    sb.header("Costs and portfolio")
    col1, col2 = sb.columns(2)
    commission = col1.number_input("Commission (bps/leg/side)", 0.0, 100.0, BASE.costs.commission_bps, 0.5)
    slippage = col2.number_input("Slippage (bps/leg/side)", 0.0, 100.0, BASE.costs.slippage_bps, 0.5)
    borrow = sb.number_input("Short borrow fee (bps per year)", 0.0, 2000.0, BASE.costs.borrow_bps_annual, 5.0)
    shorting = sb.toggle("Short selling available (assumption)", BASE.costs.short_selling_allowed)
    capital = sb.number_input("Initial capital", 1_000.0, 1e10, BASE.portfolio.initial_capital, 10_000.0, format="%.0f")
    gross = sb.slider("Gross exposure limit (x equity)", 0.25, 3.0, BASE.portfolio.max_gross_exposure, 0.25)
    rf = sb.number_input("Annual risk-free rate", 0.0, 0.20, BASE.metrics.risk_free_rate, 0.005, format="%.3f")

    sb.header("Walk-forward")
    col1, col2 = sb.columns(2)
    wf_train = int(col1.number_input("Fit window (bars)", 120, 2520, BASE.walk_forward.train_window, 21))
    wf_test = int(col2.number_input("Trade window (bars)", 21, 756, BASE.walk_forward.test_window, 21))
    wf_expanding = sb.toggle("Expanding fit window", BASE.walk_forward.expanding)
    wf_require = sb.toggle("Trade only if fit window passes EG test", BASE.walk_forward.require_cointegration)

    sb.header("Research protocol")
    protocol = sb.radio(
        "Protocol", ["Baseline (as originally built)", "v2 audited (config/research_v2.yaml)"],
        help="v2 drops special sessions (weekend-dated bars, NSE Muhurat, very thin volume), tunes ML hyperparameters "
             "inside each walk-forward window, uses unweighted models with a prevalence threshold, and adds the "
             "prevalence-forecast control B0. See docs/audit and docs/research.")
    v2 = protocol.startswith("v2")

    sb.header("Machine learning")
    ml_cfg = BASE.ml
    ml_on = sb.toggle("Run ML comparison (A vs B vs C)", ml_cfg.enabled,
                      help="Fits Logistic Regression and Random Forest reversion classifiers (adds a few seconds).")
    col1, col2 = sb.columns(2)
    horizon = int(col1.number_input("Horizon H (bars)", 1, 30, ml_cfg.horizon))
    tolerance = col2.number_input("Tolerance (z)", 0.0, 2.0, ml_cfg.tolerance_z, 0.05)
    col1, col2 = sb.columns(2)
    min_abs_z = col1.number_input("Min |z| for ML rows", 0.0, 4.0, ml_cfg.min_abs_z, 0.25)
    prob_thr = col2.number_input("Probability threshold", 0.05, 0.95, ml_cfg.probability_threshold, 0.05,
                                 disabled=v2, help="v2 uses the fitting rows' reversion rate instead.")
    embargo = int(sb.number_input("Extra embargo (bars)", 0, 60, ml_cfg.embargo,
                                  help="Dropped from the end of each fit set in addition to the H-bar purge."))
    thr_label = sb.radio("ML probability threshold source", ["Pre-specified (above)", "Validation-selected"],
                         disabled=v2,
                         help="Validation-selected picks the best validation Sharpe over the configured grid. "
                              "The test period is never used.")

    sb.header("Final test period")
    source_label = sb.radio("Thresholds used on the test period",
                            ["Pre-specified (sidebar values)", "Validation-selected (best validation grid point)"])
    reveal = sb.toggle("Reveal final test-period results", False,
                       help="Keep this off while choosing parameters. Turning it on shows the untouched test period.")

    params["overrides"] = {
        "data": {"source": source, "prefer_adjusted": params["prefer_adjusted"]},
        "splits": {"train": train, "validation": validation, "test": test},
        "cointegration": {"significance": significance},
        "strategy": {"entry_z": entry, "exit_z": exit_, "stop_loss_z": stop, "max_holding_days": max_hold,
                     "execution_lag": lag},
        "costs": {"commission_bps": commission, "slippage_bps": slippage, "borrow_bps_annual": borrow,
                  "short_selling_allowed": shorting},
        "portfolio": {"initial_capital": capital, "max_gross_exposure": gross},
        "metrics": {"risk_free_rate": rf},
        "walk_forward": {"train_window": wf_train, "test_window": wf_test, "expanding": wf_expanding,
                         "require_cointegration": wf_require},
        "evaluation": {"test_parameter_source": "config" if source_label.startswith("Pre") else "validation_selected"},
        "ml": {"enabled": ml_on, "horizon": horizon, "tolerance_z": tolerance, "min_abs_z": min_abs_z,
               "probability_threshold": prob_thr, "embargo": embargo,
               "threshold_source": "config" if thr_label.startswith("Pre") else "validation_selected"},
    }
    if v2:
        v2_ml, v2_data = V2.to_dict()["ml"], V2.to_dict()["data"]
        params["overrides"]["ml"].update({k: v2_ml[k] for k in V2_ML_KEYS}, threshold_source="config")
        params["overrides"]["data"].update({k: v2_data[k] for k in V2_DATA_KEYS})
    if source == "yfinance":
        params["overrides"]["data"].update(asset_a=params["ticker_a"], asset_b=params["ticker_b"],
                                           start_date=str(params["start"]), end_date=str(params["end"]))
    params["reveal"] = reveal
    return params


# =========================================================================== data loading
def load_assets(params: dict, config: ResearchConfig) -> tuple[AssetData, AssetData]:
    source = params["source"]
    if source == "synthetic":
        return cached_synthetic(params["cointegrated"], params["seed"], params["n_obs"])
    if source == "yfinance":
        a, b = params["ticker_a"], params["ticker_b"]
        if not a or not b:
            raise DataLoadError("Enter both ticker symbols.")
        token = st.session_state.get("refresh_token", 0)
        with st.spinner(f"Loading {a} and {b} ..."):
            return (cached_download(a, config.data.start_date, config.data.end_date, token),
                    cached_download(b, config.data.start_date, config.data.end_date, token))
    files = params.get("files") or []
    if not files:
        raise DataLoadError("Upload one long-format CSV (with a Ticker column) or two single-asset CSV files.")
    assets: dict[str, AssetData] = {}
    for f in files:
        assets.update(cached_csv(f.getvalue(), f.name))
    if len(assets) < 2:
        raise DataLoadError(f"Found {len(assets)} asset(s) in the uploaded files; two are required.")
    names = sorted(assets)
    col1, col2 = st.columns(2)
    a = col1.selectbox("Asset A (dependent leg)", names, index=0)
    b = col2.selectbox("Asset B", names, index=1)
    return assets[a], assets[b]


# =========================================================================== sections
def section_overview(res: PairResearchResult) -> None:
    pair, names = res.pair, res.names
    va, vb = pair.validation_a, pair.validation_b
    issues = len(va.warnings) + len(vb.warnings) + len(pair.report.warnings)
    cols = st.columns(3)
    cols[0].metric("Asset A (dependent leg)", names["A"])
    cols[1].metric("Asset B", names["B"])
    cols[2].metric("Data quality", "PASS" if issues == 0 else f"PASS, {issues} warning(s)")
    cols = st.columns(3)
    cols[0].metric("Common observations", f"{pair.report.n_common:,}")
    cols[1].metric("First common date", pair.report.first_common_date)
    cols[2].metric("Last common date", pair.report.last_common_date)

    st.subheader("Data provenance")
    prov = pd.DataFrame([
        {"asset": names[k], "source": m.get("source"), "retrieved_at (UTC)": m.get("retrieved_at"),
         "price column": col, "synthetic": m.get("is_synthetic"), "sha256": (m.get("sha256") or "")[:16] + "..."}
        for k, m, col in (("A", pair.metadata_a, pair.price_column_a), ("B", pair.metadata_b, pair.price_column_b))
    ])
    st.dataframe(prov, hide_index=True)
    for note in pair.metadata_a.get("notes", []) + pair.metadata_b.get("notes", []):
        st.caption(note)

    st.subheader("Validation and alignment")
    st.dataframe(pd.DataFrame({names["A"]: va.summary(), names["B"]: vb.summary()}).astype(str))
    st.dataframe(pd.DataFrame({"alignment": pair.report.summary()}).astype(str))
    for label, report in ((names["A"], va), (names["B"], vb)):
        if report.warnings:
            with st.expander(f"{label}: {len(report.warnings)} data-quality warning(s)"):
                show_list(report.warnings)
                if report.large_gaps:
                    st.dataframe(pd.DataFrame(report.large_gaps), hide_index=True)
                if report.extreme_return_dates:
                    st.dataframe(pd.DataFrame(report.extreme_return_dates), hide_index=True)
    if pair.report.warnings:
        show_list(pair.report.warnings)
    if pair.report.dates_only_in_a or pair.report.dates_only_in_b:
        with st.expander("Dates excluded because only one asset traded"):
            st.write(f"Only in {names['A']}: {truncate_list(pair.report.dates_only_in_a)}")
            st.write(f"Only in {names['B']}: {truncate_list(pair.report.dates_only_in_b)}")

    st.subheader("Chronological research splits")
    rows = []
    for name, split in res.splits.items():
        visible = name != "test" or res.test_revealed
        rows.append({"period": name, "role": PERIOD_LABELS[name], "start": split.start_date, "end": split.end_date,
                     "observations": split.n_obs, "status": "visible" if visible else "locked (not yet evaluated)"})
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.info(
        "Parameters (hedge ratio, intercept, spread mean and standard deviation) are fitted on the training period "
        "only. Thresholds may be compared on the validation period. The final test period stays locked until you "
        "reveal it in the sidebar, after fixing the specification."
    )


def section_eda(res: PairResearchResult, dark: bool) -> None:
    names = res.names
    prices = res.pair.prices.iloc[: res.research_end_idx]
    if not res.test_revealed:
        st.caption("Exploratory charts cover the training and validation periods only; the test period is locked.")
    returns = dx.simple_returns(prices)
    window = st.slider("Rolling window (trading days)", 21, 252, 63, 21)
    chart(viz.price_history_figure(prices, names, dark), "eda_prices")
    chart(viz.normalized_price_figure(prices, names, dark), "eda_norm")
    chart(viz.returns_figure(returns, names, dark), "eda_returns")
    col1, col2 = st.columns(2)
    with col1:
        chart(viz.rolling_volatility_figure(dx.rolling_volatility(returns, window), names, window, dark), "eda_vol")
    with col2:
        chart(viz.rolling_correlation_figure(dx.rolling_correlation(returns, window), window, dark), "eda_corr")
    hedge = res.model.hedge
    chart(viz.log_price_scatter_figure(np.log(prices), names, hedge.intercept, hedge.hedge_ratio,
                                       res.splits["train"].end_idx, dark), "eda_scatter")

    st.subheader("Summary statistics")
    st.dataframe(fmt_table(dx.summary_statistics(prices, names)))
    corr = dx.correlation_summary(prices)
    cols = st.columns(4)
    for col, (k, v) in zip(cols, corr.items()):
        col.metric(k.replace("_", " "), f"{v:.3f}")
    st.subheader("Missing and excluded data")
    st.dataframe(dx.missing_data_summary(res.pair))
    st.info(
        "**Correlation is not cointegration.** Correlation measures whether returns move together day to day; "
        "cointegration asks whether a linear combination of *price levels* is stationary (mean-reverting). Two "
        "independent random walks can show very high price correlation by chance, and highly correlated returns "
        "can coexist with a spread that drifts away permanently. None of these charts proves a tradable relation."
    )


def section_cointegration(res: PairResearchResult) -> None:
    c, h, sm = res.model.coint, res.model.hedge, res.model.spread
    st.subheader(f"Training-period estimates ({c.sample_start} to {c.sample_end}, {c.n_obs:,} observations)")
    st.latex(r"\log P_{A,t} = \hat\alpha + \hat\beta \, \log P_{B,t} + \varepsilon_t")
    cols = st.columns(5)
    cols[0].metric("Hedge ratio (beta)", f"{h.hedge_ratio:.4f}")
    cols[1].metric("Intercept (alpha)", f"{h.intercept:.4f}")
    cols[2].metric("EG test statistic", f"{c.test_statistic:.3f}")
    cols[3].metric("p-value", fmt_p(c.p_value))
    cols[4].metric("Significance level", f"{c.significance:.0%}")
    if c.reject_null:
        st.success(f"{c.decision} (training period only)")
    else:
        st.error(
            f"{c.decision} This pair does **not** pass the training-period cointegration test. Results below are "
            "exploratory and must not be described as a cointegrated opportunity."
        )
    st.write(c.interpretation())
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Engle-Granger critical values (MacKinnon)**")
        st.dataframe(pd.DataFrame({"critical value": c.critical_values}).T)
        st.markdown("**Fit diagnostics**")
        st.dataframe(pd.DataFrame({"value": {
            "R-squared (levels; not evidence of cointegration)": f"{h.r_squared:.4f}",
            "Training spread mean": f"{sm.mean:.6f}",
            "Training spread std": f"{sm.std:.6f}",
            "AR(1) half-life (bars)": fmt_value("half_life", res.model.half_life),
            "Implied share ratio units_B/units_A at last training close":
                f"{-h.hedge_ratio * res.pair.prices['A'].iloc[res.splits['train'].end_idx - 1] / res.pair.prices['B'].iloc[res.splits['train'].end_idx - 1]:.4f}",
        }}))
    with col2:
        st.markdown("**Integration-order checks (ADF p-values, H0: unit root)**")
        adf = pd.DataFrame(c.adf_levels).T.rename(columns={"level_p_value": "levels", "diff_p_value": "first differences"})
        st.dataframe(adf.map(fmt_p))
        st.caption("Engle-Granger assumes both log prices are I(1): unit root in levels (high p), stationary in differences (low p).")
    show_list([w for w in c.warnings if "does not pass" not in w])
    with st.expander("Test assumptions and limitations"):
        st.markdown("**Assumptions**\n" + "\n".join(f"- {a}" for a in ASSUMPTIONS))
        st.markdown("**Limitations**\n" + "\n".join(f"- {a}" for a in LIMITATIONS))
        st.markdown("- The p-value is the probability of a test statistic this extreme *if there were no "
                    "cointegration*; it is not the probability that the strategy is profitable.")


def section_signals(res: PairResearchResult, dark: bool) -> None:
    sm, rules = res.model.spread, res.periods["validation"].rules
    chart(viz.spread_figure(res.spread, res.splits, sm.mean, sm.std, dark), "sig_spread")
    frames = [p.backtest.signals for p in res.periods.values()]
    chart(viz.zscore_figure(res.zscore, res.splits, rules.entry_z, rules.exit_z, rules.stop_loss_z, frames, dark), "sig_z")
    st.caption(
        "Markers sit on the bar whose close produced the signal. Each trade is executed "
        f"{res.config.strategy.execution_lag} bar(s) later. Every period starts flat. A positive z-score means A is "
        "expensive relative to B (short spread: short A, long B); a negative z-score means A is cheap (long spread)."
    )
    period = st.segmented_control("Signal and execution log", list(res.periods), default="validation",
                                  format_func=lambda k: PERIOD_LABELS[k].split(" (")[0], key="signal_period")
    period = period or "validation"
    trades = res.periods[period].backtest.trades
    if trades.empty:
        st.info("No trades were opened in this period.")
    else:
        st.dataframe(tidy(trades[["trade_id", "direction", "entry_signal_date", "entry_date", "entry_z", "exit_signal_date",
                                  "exit_date", "exit_z", "exit_reason", "holding_days"]], 4), hide_index=True)
    with st.expander("Full signal history for this period"):
        st.dataframe(tidy(res.periods[period].backtest.signals))
    st.warning("A stationary-looking spread in one sample does not guarantee stable mean reversion later.")


def _portfolio_choices(res: PairResearchResult) -> dict[str, tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    choices = {k: (PERIOD_LABELS[k], p.backtest.daily, p.backtest.trades, p.benchmarks) for k, p in res.periods.items()}
    if res.walk_forward is not None:
        wf = res.walk_forward
        choices["walk_forward"] = ("Walk-forward (combined out-of-sample windows)", wf.daily, wf.trades, wf.benchmarks)
    return choices


def section_portfolio(res: PairResearchResult, dark: bool) -> None:
    names, cfg = res.names, res.config
    beta = res.model.spread.hedge_ratio
    wa, wb = target_leg_weights(1, beta, cfg.portfolio.max_gross_exposure)
    st.markdown(
        f"**Leg weights at entry** (training hedge ratio beta = {beta:.4f}, gross limit {cfg.portfolio.max_gross_exposure:.2f}x): "
        f"long spread = **{wa:+.1%}** of equity in {names['A']} and **{wb:+.1%}** in {names['B']}; short spread is the "
        "mirror image. The log-price beta maps to a *dollar* ratio, not a share ratio. Units are then held fixed, so "
        "weights drift until exit or a drift rebalance."
    )
    choices = _portfolio_choices(res)
    key = st.segmented_control("Period", list(choices), default="validation", format_func=lambda k: choices[k][0],
                               key="portfolio_period") or "validation"
    label, daily, trades, bench = choices[key]
    if key == "train":
        st.warning("In-sample period: parameters were fitted on these same data. This is not evidence of performance.")
    chart(viz.equity_vs_benchmarks_figure(daily["net_return"], bench, f"Growth of 1: {label}", dark), f"pf_eq_{key}")
    col1, col2 = st.columns(2)
    with col1:
        chart(viz.drawdown_figure({"Pairs strategy (net)": daily["equity"]}, dark), f"pf_dd_{key}")
    with col2:
        chart(viz.gross_net_figure(daily, dark), f"pf_gn_{key}")
    chart(viz.exposure_figure(daily, names, dark), f"pf_exp_{key}")
    st.subheader("Position history (both legs)")
    st.dataframe(tidy(daily[["position", "event", "units_a", "units_b", "notional_a", "notional_b", "gross_leverage",
                             "net_leverage", "pnl", "trade_cost", "borrow_cost", "equity", "net_return", "drawdown"]], 4))
    st.subheader("Trade log")
    if trades.empty:
        st.info("No trades in this period.")
    else:
        st.dataframe(tidy(trades, 4), hide_index=True)


def section_performance(res: PairResearchResult, dark: bool) -> None:
    rows = {}
    for name, p in res.periods.items():
        rows[PERIOD_LABELS[name]] = p.metrics
    if res.walk_forward is not None:
        rows["Walk-forward (combined out-of-sample)"] = res.walk_forward.metrics
    keys = ["observations", "cumulative_return", "gross_cumulative_return", "cagr", "annualized_volatility",
            "sharpe_ratio", "max_drawdown", "calmar_ratio", "completed_trades", "win_rate", "avg_trade_net_return",
            "avg_holding_days", "pct_time_invested", "avg_gross_leverage_when_invested", "avg_net_leverage",
            "annualized_turnover", "total_transaction_costs", "total_borrow_costs", "final_equity"]
    summary = pd.DataFrame(rows).loc[keys]
    st.subheader("Strategy results by chronological period")
    st.dataframe(fmt_table(summary), height=table_height(len(summary)))
    if not res.test_revealed:
        st.info("Final test period is locked. Reveal it in the sidebar once the specification is fixed.")
    st.caption(
        f"Sharpe uses the {res.config.metrics.risk_free_rate:.2%} annual risk-free rate (daily geometric) and "
        f"{res.config.metrics.periods_per_year} periods per year. CAGR and Calmar are shown as n/a for periods "
        f"shorter than {res.config.metrics.min_years_for_cagr:g} year(s). Undefined ratios are n/a, never infinite."
    )
    for name, p in res.periods.items():
        with st.expander(f"Strategy vs benchmarks: {PERIOD_LABELS[name]}", expanded=name != "train"):
            st.dataframe(fmt_table(p.comparison), height=table_height(len(p.comparison)))
            chart(viz.trade_return_figure(p.backtest.trades, dark), f"perf_trades_{name}")
    st.warning(
        "The pairs strategy is a long-short position capped at a gross exposure of "
        f"{res.config.portfolio.max_gross_exposure:.2f}x equity and is often flat. Buy-and-hold benchmarks are fully "
        "invested and long-only, so raw returns are not like-for-like; compare risk-adjusted metrics and drawdowns. "
        "Benchmarks pay no transaction costs. Cash earns the risk-free rate; idle strategy capital earns zero."
    )


def section_robustness(res: PairResearchResult, dark: bool, params: dict) -> None:
    grid = res.validation_grid
    st.subheader("Parameter sensitivity (validation period only)")
    st.warning(
        f"{len(grid)} threshold combinations were evaluated on the validation period. Picking the best of many "
        "combinations overstates expected performance (multiple-testing / data-snooping bias). The test period is "
        "never used here."
    )
    if not grid.empty:
        col1, col2 = st.columns(2)
        with col1:
            chart(viz.parameter_heatmap(grid, "sharpe_ratio", "Validation Sharpe ratio", dark), "rob_heat_sharpe")
        with col2:
            chart(viz.parameter_heatmap(grid, "cumulative_return", "Validation net cumulative return", dark), "rob_heat_ret")
        with st.expander("Grid table"):
            st.dataframe(grid, hide_index=True)
    if res.selected_params:
        sp = res.selected_params
        st.write(f"Best validation point by {sp['metric']}: entry {sp['entry_z']:g}, exit {sp['exit_z']:g} "
                 f"({sp['metric']} = {sp['metric_value']:.3f}, minimum {res.config.evaluation.min_trades_for_selection} trades).")
    else:
        st.write("No grid point met the minimum-trade requirement for selection.")
    st.write(f"Thresholds used on the final test period: **{res.test_params['source']}** "
             f"(entry {res.test_params['entry_z']:g}, exit {res.test_params['exit_z']:g}).")

    st.subheader("Performance before and after costs")
    chart(viz.cost_sensitivity_figure(res.cost_sensitivity, dark), "rob_costs")
    with st.expander("Cost sensitivity tables"):
        for name, frame in res.cost_sensitivity.items():
            st.markdown(f"**{name}**")
            st.dataframe(frame, hide_index=True)

    st.subheader("Walk-forward evaluation")
    wf = res.walk_forward
    if wf is None:
        show_list([w for w in res.warnings if "Walk-forward" in w])
    else:
        wfc = res.config.walk_forward
        st.markdown(
            f"Each window fits on the previous **{wfc.train_window}** bars ({'expanding' if wfc.expanding else 'rolling'}), "
            f"trades the next **{wfc.test_window}** bars with frozen parameters, closes positions at the window end, "
            f"then rolls forward. Windows whose fit period fails the EG test are "
            f"{'not traded' if wfc.require_cointegration else 'still traded'}. Thresholds: sidebar values. "
            f"Coverage ends at {'the end of the data' if res.test_revealed else 'the end of the validation period'}."
        )
        chart(viz.walk_forward_figure(wf.windows, dark), "rob_wf_bars")
        chart(viz.equity_vs_benchmarks_figure(wf.returns, wf.benchmarks, "Walk-forward out-of-sample growth of 1", dark), "rob_wf_eq")
        st.dataframe(tidy(wf.windows, 4), hide_index=True)
        yearly = wf.daily["net_return"].groupby(wf.daily.index.year).apply(lambda r: float(np.prod(1 + r) - 1))
        st.markdown("**Walk-forward net return by calendar year**")
        st.dataframe(pd.DataFrame({"net return": yearly.map(lambda v: f"{v:.2%}")}).T)
        if res.config.evaluation.test_parameter_source == "validation_selected":
            st.caption("Walk-forward windows overlapping the validation period are not independent of the threshold choice.")

    with st.expander("Universe screen (training data only, with Benjamini-Hochberg correction)"):
        section_universe(res, params, dark)


def section_universe(res: PairResearchResult, params: dict, dark: bool) -> None:
    cfg = res.config
    if params["source"] == "yfinance":
        default = ", ".join(sorted({t for pair in cfg.data.candidate_pairs for t in pair}))
        tickers_text = st.text_input("Tickers to screen (comma separated)", default)
    else:
        st.caption("Without Yahoo Finance, the screen runs on a labelled synthetic universe (two planted cointegrated pairs).")
        tickers_text = ""
    fdr = st.select_slider("False-discovery rate (BH)", [0.01, 0.05, 0.10], 0.05)
    if not st.button("Run screen"):
        return
    try:
        if params["source"] == "yfinance":
            tickers = [t.strip().upper() for t in tickers_text.split(",") if t.strip()]
            series = {}
            for t in tickers:
                asset = cached_download(t, cfg.data.start_date, cfg.data.end_date, st.session_state.get("refresh_token", 0))
                clean, report = validate_price_frame(asset.frame, t, cfg.data.prefer_adjusted, cfg.data.min_observations)
                if report.passed:
                    series[t] = clean[report.price_column]
                else:
                    st.warning(" ".join(report.errors))
        else:
            series = {t: a.frame["Adj Close"] for t, a in synthetic_universe(seed=7, n_obs=1500).items()}
        result = screen_pairs(series, cfg.splits.train, cfg.cointegration.significance, fdr, cfg.data.min_observations)
    except (DataLoadError, ValueError) as exc:
        st.error(str(exc))
        return
    st.write(f"{result.n_pairs_tested} pairs tested on {result.train_start} to {result.train_end}. {result.criterion}")
    show_list(result.warnings)
    chart(viz.screen_pvalue_figure(result.table, result.significance, dark), "univ_p")
    st.dataframe(result.table, hide_index=True)


def section_assumptions(res: PairResearchResult) -> None:
    c = res.config
    st.markdown(f"""
**Price adjustments.** Research uses `{res.pair.price_column_a}` for {res.names['A']} and `{res.pair.price_column_b}` for
{res.names['B']}. Yahoo `Adj Close` is split- and dividend-adjusted by the provider and used as delivered (no further
adjustment, so no double adjustment). Back-adjusted prices are proportional to a total-return index; the constant of
proportionality is absorbed by the regression intercept. Adjusted returns imply dividends are received on long legs and
paid on short legs. CSV data is used exactly as supplied.

**Short selling.** Assumed {'available' if c.costs.short_selling_allowed else 'unavailable (backtests refuse to trade)'}
for both assets at the closing price. In practice shares may be hard to borrow, recalled, or banned from shorting;
Indian cash-equity shorting is generally limited to intraday or securities lending (SLB), so multi-day shorts of `.NS`
shares may be infeasible without derivatives.

**Borrow and financing.** Short borrow fee {c.costs.borrow_bps_annual:g} bps per year on prior-close short notional,
accrued daily. No interest is earned on idle capital or short-sale proceeds, no margin interest is charged, and margin
calls are not modelled. Gross exposure is capped at {c.portfolio.max_gross_exposure:g}x equity at entry and scaled back
if it drifts more than {c.portfolio.rebalance_tolerance or 0:.0%} above the cap.

**Transaction costs.** {c.costs.commission_bps:g} bps commission + {c.costs.slippage_bps:g} bps slippage per leg, per
side, on traded notional (entries, exits and rebalances, on both legs). These are generic assumptions, not a broker
model; market impact is assumed linear and small.

**Execution timing.** Signals use the close of bar t; trades fill at the close of bar t + {c.strategy.execution_lag}.
P&L is marked close to close on units held. The last bar of each period liquidates open positions; no signal from the
final bars is executed beyond the data.

**Estimation windows.** Static workflow: hedge ratio, intercept, spread mean and std are estimated on the training
period ({res.splits['train'].start_date} to {res.splits['train'].end_date}) and frozen. Walk-forward: re-estimated every
{c.walk_forward.test_window} bars on a {'expanding' if c.walk_forward.expanding else f'rolling {c.walk_forward.train_window}-bar'} window.

**Data source.** {res.pair.metadata_a.get('source')} / {res.pair.metadata_b.get('source')}, retrieved
{res.pair.metadata_a.get('retrieved_at')}. Free providers may revise, back-fill or remove history; the SHA-256 hash of each
input is recorded with saved runs so changes can be detected.

**Not modelled:** intraday execution, partial fills, capacity, corporate-action timing risk, taxes, currency effects, and
survivorship bias in the choice of instruments. This is a research tool, not a trading system, and nothing here is
financial advice.
""")


ML_PERIOD_NOTES = {
    "train": "training rows (in-sample: the models were fitted on them)",
    "validation": "validation rows (used to tune hyperparameters, so these scores are optimistic)",
    "test": "final test rows (untouched out-of-sample)",
}


def section_ml_models(ml: MLResearchResult, dark: bool) -> None:
    cfg = ml.config.ml
    ds = ml.dataset
    st.subheader("Prediction target")
    st.latex(r"y_t = \mathbf{1}\left\{\,|z_{t+H}| \le |z_t| - \tau\,\right\}, \qquad H = %d,\ \ \tau = %.2f"
             % (cfg.horizon, cfg.tolerance_z))
    st.markdown(
        f"A row is a bar *t* with |z<sub>t</sub>| ≥ **{cfg.min_abs_z:g}**. Its label is 1 when, **{cfg.horizon}** sessions later, the "
        f"spread's distance from the training-estimated equilibrium has fallen by at least **{cfg.tolerance_z:g}** training "
        "standard deviations. Labels are the only quantity that looks ahead. Every feature uses prices up to *t*, and the "
        f"spread parameters are frozen on the training period. A row is used only if its label horizon ends inside its own "
        f"period (an {cfg.horizon}-bar purge, plus {cfg.embargo} embargo bar(s) on the fit set), so no label crosses into the "
        "next period. The last bars of the data have no label and are excluded.", unsafe_allow_html=True)

    st.subheader("Samples and evaluation dates")
    st.dataframe(tidy(ml.sample_table.reset_index(), 4), hide_index=True)
    st.caption("rows_used counts eligible bars with complete features and an in-period label. Positive rate is the "
               "share of rows that reverted, so accuracy should be compared with the majority-class rate.")

    st.subheader("Model settings")
    rows = []
    for name, report in ml.models.items():
        rows.append({"model": report.label, "status": report.status,
                     "selected hyperparameters": json.dumps(report.tuning.best_params) if report.tuning else "",
                     "training rows": report.fitted.n_train if report.ok else 0,
                     "training class counts": json.dumps(report.fitted.class_counts) if report.ok else "",
                     "training rows from": report.fitted.train_start if report.ok else "",
                     "training rows to": report.fitted.train_end if report.ok else "",
                     "threshold B (ML-only)": report.thresholds.get("ml_only"),
                     "threshold C (hybrid)": report.thresholds.get("hybrid")})
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    weighting = ("balanced class weights", "balanced subsample weights") if cfg.class_weight == "balanced" \
        else ("unweighted", "unweighted")
    threshold_rule = ("the fitting rows' reversion rate (prevalence threshold)" if cfg.threshold_mode == "prevalence"
                      else f"**{cfg.threshold_source}**")
    st.caption(f"Pipelines: StandardScaler + LogisticRegression ({weighting[0]}), and RandomForestClassifier "
               f"({cfg.rf_n_estimators} trees, {weighting[1]}). Preprocessing is fitted on training rows only. "
               f"Hyperparameters are chosen by validation {cfg.tuning_metric}. Probability threshold: {threshold_rule}.")
    with st.expander("Hyperparameter tuning tables (fit on training, scored on validation)"):
        for report in ml.models.values():
            if report.tuning is not None:
                st.markdown(f"**{report.label}**")
                st.dataframe(report.tuning.table, hide_index=True)
    with st.expander("Feature definitions (all use data up to and including bar t)"):
        st.dataframe(pd.DataFrame({"definition": FEATURE_DESCRIPTIONS}), )
        st.write("Columns used: " + ", ".join(ds.spec.columns))
    show_list(ml.warnings)

    st.subheader("Classification metrics")
    summary = classification_summary(ml)
    if summary.empty:
        st.warning("No model could be fitted on this data.")
        return
    keys = ["n_samples", "positive_rate", "majority_class_accuracy", "accuracy", "precision", "recall", "f1", "roc_auc",
            "brier_score", "reference_rate", "brier_skill_score", "log_loss", "log_loss_skill", "predicted_positive_rate"]
    shown = summary.loc[keys]
    shown.columns = [c.replace("| train", "| train (in-sample)").replace("| validation", "| validation (tuning)")
                     .replace("| test", "| test (OOS)") for c in shown.columns]
    st.dataframe(fmt_table(shown), height=table_height(len(keys)))
    threshold_text = ("the training reversion rate (prevalence threshold)" if cfg.threshold_mode == "prevalence"
                      else f"the pre-specified threshold {cfg.probability_threshold:g}")
    st.caption(f"Classification uses {threshold_text}. The prevalence baseline forecasts the training reversion rate "
               "(reference_rate) for every row; a negative Brier or log-loss skill means the model's probabilities are "
               f"worse than that constant. With fewer than {cfg.min_eval_samples} rows, or only one class, rates are "
               f"reported as n/a. Consecutive rows share most of their {cfg.horizon}-bar label window, so the effective "
               f"sample size is closer to rows / {cfg.horizon}.")
    st.info(
        "**Accuracy is not profitability.** A model can be right on most rows and still lose money: it may be right "
        "on small moves and wrong on large ones, fire on bars where no trade is taken, or trade so often that costs eat "
        "the edge. Under class imbalance, high accuracy can come from always predicting the majority class. Judge "
        "ROC-AUC against 0.5, accuracy against the majority-class rate, and the strategy comparison net of costs."
    )

    fitted = {n: r for n, r in ml.models.items() if r.ok}
    col1, col2 = st.columns(2)
    model = col1.segmented_control("Model", list(fitted), default=next(iter(fitted)), key="ml_model",
                                   format_func=lambda n: MODEL_LABELS[n]) or next(iter(fitted))
    periods = list(fitted[model].metrics)
    period = col2.segmented_control("Period", periods, default="validation", key="ml_period",
                                    format_func=lambda p: ML_PERIOD_NOTES[p].split(" (")[0].replace(" rows", "")) or "validation"
    report = fitted[model]
    labelled = ds.eligible & ds.target.notna()
    chart(viz.probability_figure(report.probability, ds.target, labelled, cfg.probability_threshold,
                                 {k: v for k, v in _visible_splits(ml).items()},
                                 f"{report.label}: predicted reversion probability vs actual outcome", dark), "ml_prob")
    col1, col2 = st.columns(2)
    with col1:
        chart(viz.confusion_figure(report.metrics[period]["confusion_matrix"],
                                   f"Confusion matrix: {ML_PERIOD_NOTES[period].split(' (')[0]}", dark), "ml_cm")
        note = report.metrics[period].get("note")
        if note:
            st.caption(note)
    with col2:
        chart(viz.calibration_figure({fitted[n].label: fitted[n].calibration[period] for n in fitted}, dark), "ml_cal")
    st.caption(f"Showing {ML_PERIOD_NOTES[period]}.")

    st.subheader("What the models use")
    col1, col2 = st.columns(2)
    if "random_forest" in fitted:
        imp = fitted["random_forest"].importance
        with col1:
            chart(viz.importance_figure(imp["impurity_importance"], "Random Forest: impurity importance (training fit)",
                                        dark=dark), "ml_rf_imp")
        with col2:
            chart(viz.importance_figure(imp["permutation_importance"], "Random Forest: permutation importance "
                                        "(validation ROC-AUC drop)", imp["permutation_std"], dark=dark), "ml_rf_perm")
    if "logistic_regression" in fitted:
        imp = fitted["logistic_regression"].importance
        with col1:
            chart(viz.importance_figure(imp["standardized_coefficient"], "Logistic Regression: coefficient per 1 SD "
                                        "of the feature", dark=dark, diverging=True), "ml_lr_coef")
        with col2:
            chart(viz.importance_figure(imp["permutation_importance"], "Logistic Regression: permutation importance "
                                        "(validation ROC-AUC drop)", imp["permutation_std"], dark=dark), "ml_lr_perm")
    st.caption("Coefficients are log-odds per one training standard deviation. Correlated features (for example z and "
               "its lags) share and can swap weight, so individual coefficients are not causal effects. Impurity "
               "importance favours high-cardinality features. Permutation importance on validation rows is the more "
               "reliable guide, but it is noisy with few rows.")


def _visible_splits(ml: MLResearchResult) -> dict:
    from statarb.evaluation import DataSplit

    rows = ml.sample_table
    index = ml.dataset.features.index
    splits = {}
    for name, row in rows.iterrows():
        start = index.get_loc(pd.Timestamp(row["start"]))
        end = index.get_loc(pd.Timestamp(row["end"])) + 1
        splits[name] = DataSplit(name, start, end, row["start"], row["end"])
    return splits


def section_strategy_comparison(ml: MLResearchResult, dark: bool) -> None:
    cfg = ml.config.ml
    thr = {n: r.thresholds for n, r in ml.models.items() if r.ok}
    thr_text = "; ".join(f"{SHORT_LABELS[n]}: B {t['ml_only']:g}, C {t['hybrid']:g}" for n, t in thr.items()) or "no model fitted"
    st.markdown(
        f"""
- **A. Traditional:** z-score entry, exit, optional stop and maximum holding (the strategy in tabs D–G).
- **B. ML-only:** enter when P(reversion) ≥ threshold and |z| ≥ {cfg.min_abs_z:g}, betting on reversion (long spread
  if z < 0). Exit on the stop-loss, on reversion into the exit band, or after {cfg.horizon} bars unless the latest
  prediction still supports the position (then it is renewed without trading).
- **C. Hybrid:** A's entry rule *and* P(reversion) ≥ threshold on the entry bar. A's exits and risk controls are
  unchanged.
- **B0** (v2 only): B's rules driven by the constant prevalence forecast (no model). B minus B0 isolates what the
  model adds beyond B's rule changes.

Thresholds: {thr_text}. All variants use the same dates, execution lag
({ml.config.strategy.execution_lag} bar), gross-exposure sizing, two-leg commission and slippage, borrow fee,
short-selling assumption and end-of-period liquidation, and each period starts flat.
""")
    options = {p: PERIOD_LABELS[p] for p in ml.comparisons}
    if ml.walk_forward is not None:
        options["walk_forward"] = "Walk-forward (combined out-of-sample windows)"
    key = st.segmented_control("Period", list(options), default="validation", key="cmp_period",
                               format_func=lambda k: options[k]) or "validation"
    if key == "train":
        st.warning("In-sample: the models and spread parameters were fitted on these dates. ML-only results here are "
                   "heavily overfitted and are not evidence of anything.")
    elif key == "validation":
        st.warning("The validation period was used to tune hyperparameters (and thresholds, if validation-selected), "
                   "so these results are optimistic.")
    elif key == "test":
        st.success("Final test period: untouched by every fitting and tuning step.")
    if not ml.test_revealed and key != "train":
        st.info("The final test period is locked. Reveal it in the sidebar once the specification is fixed.")

    if key == "walk_forward":
        wf = ml.walk_forward
        returns = {VARIANT_LABELS[k]: d["net_return"] for k, d in wf.daily.items()}
        table = wf.table
        if cfg.walk_forward_tuning == "static":
            st.warning("Baseline protocol: every walk-forward window reuses hyperparameters tuned on the static "
                       "validation period, so windows before or during that period use later information (audit "
                       "finding D1). Select the v2 protocol in the sidebar for point-in-time tuning.")
    else:
        comp = ml.comparisons[key]
        returns = {VARIANT_LABELS[k]: r.returns for k, r in comp.results.items()}
        table = comp.table
    st.dataframe(fmt_table(table), height=table_height(len(table)))
    chart(viz.variant_equity_figure(returns, f"Growth of 1 by strategy variant: {options[key]}", dark), f"cmp_eq_{key}")
    col1, col2 = st.columns(2)
    with col1:
        chart(viz.variant_drawdown_figure(returns, dark), f"cmp_dd_{key}")
    with col2:
        chart(viz.variant_costs_figure(table.loc["total_transaction_costs"], dark), f"cmp_cost_{key}")

    if key == "walk_forward":
        wf = ml.walk_forward
        present = {k: VARIANT_LABELS[k] for k in wf.daily}
        chart(viz.walk_forward_variant_heatmap(wf.windows, present, dark), "cmp_wf_heat")
        st.dataframe(tidy(wf.windows, 4), hide_index=True)
        tuning_text = ("Hyperparameters are tuned inside each window on its own inner-validation block (nested)."
                       if cfg.walk_forward_tuning == "nested"
                       else "Hyperparameters and thresholds stay fixed from the static training/validation step.")
        st.caption("Each window refits the spread model, re-checks cointegration (the same gate applies to every "
                   "variant), rebuilds features and refits both classifiers on that window's purged rows. "
                   + tuning_text)

    with st.expander("Validation threshold grids (exploratory; the test period is never used)"):
        for name, report in ml.models.items():
            for kind, grid in report.threshold_tables.items():
                st.markdown(f"**{report.label}: {'B. ML-only' if kind == 'ml_only' else 'C. Hybrid'}**")
                st.dataframe(grid[["threshold", "cumulative_return", "sharpe_ratio", "max_drawdown", "completed_trades",
                                   "total_transaction_costs"]], hide_index=True)
    n_variants = len(table.columns)
    st.warning(
        f"{n_variants} variants are compared here, on top of every pair, threshold and setting tried elsewhere. Some "
        "will look better than others by chance. A difference is only interesting if it holds across validation, test "
        "and walk-forward, after costs, with enough trades. This dashboard does not declare a winner."
    )


def section_export(res: PairResearchResult, ml: MLResearchResult | None = None) -> None:
    st.markdown("Download any table as CSV, or save a complete, reproducible snapshot of this run to `reports/`.")
    tables = result_tables(res)
    ml_tables = ml_result_tables(ml) if ml is not None else {}
    groups = {
        "Daily portfolio returns and equity": [k for k in tables if k.startswith("daily_portfolio")],
        "Position history (both legs)": [k for k in tables if k.startswith("positions")],
        "Trade logs": [k for k in tables if k.startswith("trades") or k == "walk_forward_trades"],
        "Signal history": [k for k in tables if k.startswith("signals")],
        "Performance metrics": [k for k in tables if "metrics" in k],
        "Robustness": [k for k in tables if k.startswith(("validation_parameter", "cost_sensitivity", "walk_forward_windows", "walk_forward_daily"))],
    }
    if ml_tables:
        tables = {**tables, **ml_tables}
        groups["ML models (predictions, metrics, calibration, importance, tuning)"] = [
            k for k in ml_tables if k.startswith(("ml_samples", "ml_classification", "ml_predictions", "ml_calibration",
                                                  "ml_importance", "ml_tuning", "ml_threshold"))]
        groups["Strategy comparison A/B/C"] = [
            k for k in ml_tables if k.startswith(("strategy_", "trades_", "ml_walk_forward"))]
    for title, keys in groups.items():
        st.markdown(f"**{title}**")
        cols = st.columns(4)
        for i, k in enumerate(keys):
            cols[i % 4].download_button(k, csv_bytes(tables[k]), f"{k}.csv", "text/csv", key=f"dl_{k}")
    st.markdown("**Statistical tests and configuration**")
    col1, col2, col3 = st.columns(3)
    col1.download_button("statistical_tests.json", json.dumps(statistical_tests_payload(res), indent=2, default=str),
                         "statistical_tests.json", "application/json", key="dl_tests")
    col2.download_button("config_snapshot.yaml", yaml.safe_dump(res.config.to_dict(), sort_keys=False),
                         "config_snapshot.yaml", "text/yaml", key="dl_cfg")
    buffer = io.StringIO()
    res.pair.prices.rename(columns=res.names).to_csv(buffer)
    col3.download_button("aligned_prices.csv", buffer.getvalue(), "aligned_prices.csv", "text/csv", key="dl_prices")
    if st.button("Save run snapshot to reports/", type="primary"):
        try:
            path = save_research_run(res, ml_result=ml)
            st.success(f"Saved to {path}")
        except OSError as exc:
            st.error(f"Could not save the snapshot: {exc}")


# =========================================================================== main
def main() -> None:
    params = sidebar()
    st.title("StatArb Research Lab")
    st.caption("Can a statistically justified pairs-trading strategy earn robust out-of-sample returns after costs? "
               "This dashboard tests that question on the data you select. Research and education only.")
    try:
        config = BASE.with_overrides(params["overrides"])
    except ValueError as exc:
        st.error(f"Invalid settings: {exc}")
        st.stop()

    try:
        asset_a, asset_b = load_assets(params, config)
        pair = prepare_pair_from_config(asset_a, asset_b, config.data)
    except DataLoadError as exc:
        st.error(str(exc))
        st.stop()
    except ValueError as exc:
        st.error(f"Data problem: {exc}")
        st.stop()

    if pair.is_synthetic:
        st.warning("SYNTHETIC DATA: these series are generated for demonstration. Results illustrate the method only "
                   "and are not historical market evidence.", icon=":material/science:")
    pair_key = f"{pair.ticker_a}|{pair.ticker_b}|{pair.metadata_a.get('sha256')}|{pair.metadata_b.get('sha256')}|" \
               f"{pd.util.hash_pandas_object(pair.prices).sum()}"
    full = config.to_dict()
    base_json = json.dumps({k: v for k, v in full.items() if k != "ml"}, sort_keys=True)  # ML settings don't affect it
    try:
        with st.spinner("Running training-only estimation, backtests and walk-forward analysis ..."):
            res = dataclasses.replace(cached_research(pair_key, base_json, params["reveal"], pair), config=config)
    except ValueError as exc:
        st.error(f"Research could not be completed: {exc}")
        st.stop()

    ml = None
    if config.ml.enabled:
        try:
            with st.spinner("Fitting reversion classifiers and comparing strategies A / B / C ..."):
                ml = cached_ml(pair_key, json.dumps(full, sort_keys=True), params["reveal"], res)
        except ValueError as exc:
            st.warning(f"ML comparison could not be completed: {exc}")

    if not res.model.coint.reject_null:
        st.error(f"{pair.ticker_a} / {pair.ticker_b} does not pass the training-period Engle-Granger test at the "
                 f"{config.cointegration.significance:.0%} level (p = {res.model.coint.p_value:.3f}). All backtests "
                 "below are exploratory.", icon=":material/warning:")
    dark = is_dark()
    tabs = st.tabs(["A. Overview", "B. Exploratory analysis", "C. Cointegration", "D. Spread and signals",
                    "E. Portfolio and backtest", "F. Performance report", "G. Robustness", "H. Assumptions",
                    "I. ML models", "J. Strategy comparison (A/B/C)", "Export"])
    with tabs[0]:
        section_overview(res)
    with tabs[1]:
        section_eda(res, dark)
    with tabs[2]:
        section_cointegration(res)
    with tabs[3]:
        section_signals(res, dark)
    with tabs[4]:
        section_portfolio(res, dark)
    with tabs[5]:
        section_performance(res, dark)
    with tabs[6]:
        section_robustness(res, dark, params)
    with tabs[7]:
        section_assumptions(res)
    for tab, section in ((tabs[8], section_ml_models), (tabs[9], section_strategy_comparison)):
        with tab:
            if ml is None:
                st.info("The ML comparison is switched off. Enable it in the sidebar under Machine learning.")
            else:
                section(ml, dark)
    with tabs[10]:
        section_export(res, ml)


main()
