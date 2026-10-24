"""Chronological evaluation: splits, research pipeline, walk-forward analysis and run snapshots.

Estimation and trading schedule
-------------------------------
Static research workflow (``run_pair_research``):

1. Split the aligned sample chronologically into train / validation / test.
2. Fit the hedge ratio, intercept, spread mean and spread std on **train only** and run
   the Engle-Granger test on train only.
3. Apply that frozen model to each period. Every period is backtested separately and
   starts flat. The training-period backtest is *in-sample* and labelled as such.
4. Entry/exit grids are evaluated on **validation only**. The test period is never used
   to choose parameters.
5. The test period is evaluated once with either the pre-specified (config) thresholds or
   the validation-selected thresholds, and only when the caller explicitly reveals it.

Walk-forward workflow (``run_walk_forward``): for each window, fit on the preceding
``train_window`` bars (rolling) or all preceding bars (expanding), optionally require the
fit window to pass the Engle-Granger test, trade the next ``test_window`` bars with the
frozen parameters, close any position at the window end, then roll forward by
``test_window`` bars. Out-of-sample windows never overlap and are chained in time order.
"""

from __future__ import annotations

import json
import math
import platform
import sys
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .backtester import BacktestConfig, BacktestResult, run_backtest
from .cointegration import CointegrationResult, CointegrationSuite, cointegration_suite, engle_granger_test
from .config import ResearchConfig, resolve_path, save_config_snapshot
from .data_validation import AlignedPair
from .hedge_ratio import HedgeRatioResult, estimate_hedge_ratio
from .metrics import benchmark_returns, comparison_table, strategy_metrics
from .signals import SignalRules, generate_signals
from .spread import SpreadModel, fit_spread_model, half_life

PERIOD_LABELS = {
    "train": "Training (in-sample)",
    "validation": "Validation (out-of-sample, used for parameter selection)",
    "test": "Final test (out-of-sample, untouched until specification fixed)",
}


# --------------------------------------------------------------------------- splits
@dataclass(frozen=True)
class DataSplit:
    name: str
    start_idx: int
    end_idx: int  # exclusive
    start_date: str
    end_date: str

    @property
    def n_obs(self) -> int:
        return self.end_idx - self.start_idx

    @property
    def slice(self) -> slice:
        return slice(self.start_idx, self.end_idx)


def chronological_split(
    index: pd.Index, train: float = 0.6, validation: float = 0.2, test: float = 0.2, min_obs: int = 20
) -> dict[str, DataSplit]:
    """Split an ordered index into contiguous train / validation / test blocks (no shuffling)."""
    if abs(train + validation + test - 1.0) > 1e-9:
        raise ValueError("split proportions must sum to 1")
    if min(train, validation, test) <= 0:
        raise ValueError("every split proportion must be positive")
    if not index.is_monotonic_increasing or index.has_duplicates:
        raise ValueError("index must be strictly increasing (sort and de-duplicate first)")
    n = len(index)
    n_train = int(math.floor(n * train))
    n_val = int(math.floor(n * validation))
    bounds = {"train": (0, n_train), "validation": (n_train, n_train + n_val), "test": (n_train + n_val, n)}
    splits = {}
    for name, (start, end) in bounds.items():
        if end - start < min_obs:
            raise ValueError(
                f"The {name} split would contain only {end - start} observations (minimum {min_obs}). "
                "Use more data or different proportions."
            )
        splits[name] = DataSplit(name, start, end, _date(index[start]), _date(index[end - 1]))
    return splits


def _date(value) -> str:
    return str(value.date()) if hasattr(value, "date") else str(value)


# --------------------------------------------------------------------------- model fitting
@dataclass
class PairModel:
    hedge: HedgeRatioResult
    coint: CointegrationResult
    spread: SpreadModel
    half_life: float
    suite: CointegrationSuite | None = None  # reverse EG + Johansen, when the gate or config asks for them

    def passes_gate(self, config: ResearchConfig) -> bool:
        """Whether this fitting window may be traded under ``cointegration.gate`` (baseline: EG, A on B)."""
        gate = config.cointegration.gate
        if gate == "eg" or self.suite is None:
            return self.coint.reject_null
        return self.suite.passes(gate)

    def summary(self) -> dict[str, object]:
        return {
            "fit_start": self.spread.fit_start,
            "fit_end": self.spread.fit_end,
            "n_obs": self.spread.n_obs,
            "intercept": self.spread.intercept,
            "hedge_ratio": self.spread.hedge_ratio,
            "spread_mean": self.spread.mean,
            "spread_std": self.spread.std,
            "r_squared": self.hedge.r_squared,
            "eg_statistic": self.coint.test_statistic,
            "eg_p_value": self.coint.p_value,
            "cointegrated_at_alpha": self.coint.reject_null,
            "half_life_bars": self.half_life,
        }


def fit_pair_model(log_prices: pd.DataFrame, config: ResearchConfig, name_a: str = "A", name_b: str = "B") -> PairModel:
    """Fit hedge ratio, spread normalisation and Engle-Granger test on one fitting window."""
    log_a, log_b = log_prices["A"], log_prices["B"]
    hedge = estimate_hedge_ratio(log_a, log_b, method=config.hedge_ratio.method)
    coint = engle_granger_test(
        log_a,
        log_b,
        significance=config.cointegration.significance,
        trend=config.cointegration.trend,
        autolag=config.cointegration.autolag,
        min_obs=config.cointegration.min_observations,
        name_a=name_a,
        name_b=name_b,
    )
    spread_model = fit_spread_model(log_a, log_b, min_std=config.spread.min_std, hedge=hedge)
    hl = half_life(spread_model.spread(log_a, log_b))
    suite = None
    c = config.cointegration
    if c.gate != "eg" or c.record_all_tests:
        suite = cointegration_suite(log_a, log_b, c.significance, c.trend, c.autolag, c.min_observations,
                                    c.johansen_det_order, c.johansen_k_ar_diff, name_a, name_b, eg_ab=coint)
    return PairModel(hedge=hedge, coint=coint, spread=spread_model, half_life=hl, suite=suite)


def signal_rules_from_config(config: ResearchConfig, entry_z: float | None = None, exit_z: float | None = None) -> SignalRules:
    return SignalRules(
        entry_z=config.strategy.entry_z if entry_z is None else entry_z,
        exit_z=config.strategy.exit_z if exit_z is None else exit_z,
        stop_loss_z=config.strategy.stop_loss_z,
        max_holding_bars=config.strategy.max_holding_days,
    )


def rules_for_model(config: ResearchConfig, half_life_bars: float | None = None, entry_z: float | None = None,
                    exit_z: float | None = None) -> SignalRules:
    """Signal rules for one fitted model: adds the half-life time stop when ``max_holding_half_lives`` is set.

    The cap is ``ceil(max_holding_half_lives x half-life)`` bars (at least 1), combined with any fixed
    ``max_holding_days`` (the smaller wins). Without the option, or with an undefined half-life, the
    rules are exactly :func:`signal_rules_from_config`.
    """
    rules = signal_rules_from_config(config, entry_z, exit_z)
    k = config.strategy.max_holding_half_lives
    if k is None or half_life_bars is None or not np.isfinite(half_life_bars):
        return rules
    cap = max(1, int(math.ceil(k * half_life_bars)))
    fixed = rules.max_holding_bars
    return replace(rules, max_holding_bars=cap if fixed is None else min(fixed, cap))


def build_signals(prices: pd.DataFrame, model: SpreadModel, rules: SignalRules) -> pd.DataFrame:
    """Spread, z-score and state-machine signals for one period using a frozen model."""
    logp = np.log(prices)
    spread = model.spread(logp["A"], logp["B"])
    signals = generate_signals(model.zscore(logp["A"], logp["B"]), rules)
    signals.insert(0, "spread", spread)
    return signals


def backtest_period(prices: pd.DataFrame, model: SpreadModel, rules: SignalRules, bt_config: BacktestConfig,
                    opens: pd.DataFrame | None = None) -> BacktestResult:
    return run_backtest(prices, build_signals(prices, model, rules), model.hedge_ratio, bt_config, opens)


def _slice(frame: pd.DataFrame | None, sl: slice) -> pd.DataFrame | None:
    return None if frame is None else frame.iloc[sl]


# --------------------------------------------------------------------------- period evaluation
@dataclass
class PeriodEvaluation:
    name: str
    label: str
    split: DataSplit
    rules: SignalRules
    backtest: BacktestResult
    metrics: dict[str, float]
    benchmarks: pd.DataFrame
    comparison: pd.DataFrame

    @property
    def in_sample(self) -> bool:
        return self.name == "train"


def evaluate_period(
    prices: pd.DataFrame,
    split: DataSplit,
    model: SpreadModel,
    rules: SignalRules,
    bt_config: BacktestConfig,
    config: ResearchConfig,
    names: dict[str, str],
    opens: pd.DataFrame | None = None,
    trade: bool = True,
) -> PeriodEvaluation:
    period_prices = prices.iloc[split.slice]
    period_opens = _slice(opens, split.slice)
    if trade:
        result = backtest_period(period_prices, model, rules, bt_config, period_opens)
    else:  # static cointegration gate failed: the period is held flat
        result = run_backtest(period_prices, _flat_signals(period_prices, model), model.hedge_ratio, bt_config,
                              period_opens)
    m = config.metrics
    metrics = strategy_metrics(result, m.risk_free, m.periods_per_year, m.min_years_for_cagr)
    bench = benchmark_returns(period_prices, names, m.risk_free, m.periods_per_year)
    comp = comparison_table(result.returns, result.gross_returns, bench, m.risk_free, m.periods_per_year, m.min_years_for_cagr)
    return PeriodEvaluation(split.name, PERIOD_LABELS.get(split.name, split.name), split, rules, result, metrics, bench, comp)


def parameter_grid_search(
    prices: pd.DataFrame,
    model: SpreadModel,
    config: ResearchConfig,
    bt_config: BacktestConfig,
    entry_grid=None,
    exit_grid=None,
    opens: pd.DataFrame | None = None,
    half_life_bars: float | None = None,
) -> pd.DataFrame:
    """Evaluate every valid (entry, exit) pair on the given period (validation or training only)."""
    entry_grid = entry_grid or config.robustness.entry_grid
    exit_grid = exit_grid or config.robustness.exit_grid
    m = config.metrics
    rows = []
    for entry in entry_grid:
        for exit_ in exit_grid:
            if exit_ >= entry:
                continue
            if config.strategy.stop_loss_z is not None and config.strategy.stop_loss_z <= entry:
                continue
            rules = rules_for_model(config, half_life_bars, entry_z=entry, exit_z=exit_)
            result = backtest_period(prices, model, rules, bt_config, opens)
            stats = strategy_metrics(result, m.risk_free, m.periods_per_year, m.min_years_for_cagr)
            rows.append(
                {
                    "entry_z": entry,
                    "exit_z": exit_,
                    "completed_trades": stats["completed_trades"],
                    "cumulative_return": stats["cumulative_return"],
                    "gross_cumulative_return": stats["gross_cumulative_return"],
                    "sharpe_ratio": stats["sharpe_ratio"],
                    "max_drawdown": stats["max_drawdown"],
                    "calmar_ratio": stats["calmar_ratio"],
                    "win_rate": stats["win_rate"],
                    "pct_time_invested": stats["pct_time_invested"],
                    "total_transaction_costs": stats["total_transaction_costs"],
                }
            )
    return pd.DataFrame(rows)


_SELECTION_COLUMNS = {"sharpe": "sharpe_ratio", "cumulative_return": "cumulative_return", "calmar": "calmar_ratio"}


def select_parameters(grid: pd.DataFrame, metric: str = "sharpe", min_trades: int = 3) -> dict[str, float] | None:
    """Pick the best grid point by ``metric`` among points with at least ``min_trades`` trades."""
    if grid.empty:
        return None
    column = _SELECTION_COLUMNS[metric]
    eligible = grid.loc[(grid["completed_trades"] >= min_trades) & np.isfinite(grid[column])]
    if eligible.empty:
        return None
    best = eligible.loc[eligible[column].idxmax()]
    return {"entry_z": float(best["entry_z"]), "exit_z": float(best["exit_z"]), "metric": metric,
            "metric_value": float(best[column]), "n_candidates": int(len(grid))}


def cost_sensitivity(
    prices: pd.DataFrame,
    model: SpreadModel,
    rules: SignalRules,
    config: ResearchConfig,
    bt_config: BacktestConfig,
    cost_grid_bps=None,
    opens: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Re-run one period at different per-leg, per-side trading costs.

    Generic profile: the grid value is commission + slippage. Market profiles (e.g. NSE): statutory
    charges are fixed and the grid value is the slippage on top of them.
    """
    cost_grid_bps = cost_grid_bps or config.robustness.cost_grid_bps
    m = config.metrics
    rows = []
    for bps in cost_grid_bps:
        if bt_config.simple_costs:
            cfg = replace(bt_config, commission_bps=float(bps), slippage_bps=0.0)
        else:
            cfg = replace(bt_config, slippage_bps=float(bps))
        result = backtest_period(prices, model, rules, cfg, opens)
        stats = strategy_metrics(result, m.risk_free, m.periods_per_year, m.min_years_for_cagr)
        rows.append(
            {
                "cost_bps_per_leg_per_side": float(bps),
                "cumulative_return": stats["cumulative_return"],
                "sharpe_ratio": stats["sharpe_ratio"],
                "max_drawdown": stats["max_drawdown"],
                "completed_trades": stats["completed_trades"],
                "total_transaction_costs": stats["total_transaction_costs"],
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- walk-forward
@dataclass(frozen=True)
class WalkForwardWindow:
    window_id: int
    fit_start_idx: int
    fit_end_idx: int  # exclusive
    eval_start_idx: int
    eval_end_idx: int  # exclusive


def walk_forward_schedule(
    n_obs: int, train_window: int, test_window: int, expanding: bool = False, min_eval_obs: int = 20,
    first_eval_idx: int | None = None,
) -> list[WalkForwardWindow]:
    """Non-overlapping evaluation windows, each preceded by its fitting window.

    By default the first trading window starts right after the first full fitting window
    (bar ``train_window``). ``first_eval_idx`` anchors it later, for example at the first bar of a
    holdout period, with the fitting window ending immediately before it.
    """
    if train_window < 2 or test_window < 1:
        raise ValueError("train_window and test_window must be positive")
    windows: list[WalkForwardWindow] = []
    fit_end = train_window if first_eval_idx is None else int(first_eval_idx)
    if fit_end < train_window:
        raise ValueError(f"first_eval_idx {fit_end} leaves less than one fitting window ({train_window} bars) before it")
    while fit_end < n_obs:
        eval_end = min(fit_end + test_window, n_obs)
        if eval_end - fit_end < min_eval_obs:
            break
        fit_start = 0 if expanding else fit_end - train_window
        windows.append(WalkForwardWindow(len(windows) + 1, fit_start, fit_end, fit_end, eval_end))
        fit_end = eval_end
    if not windows:
        raise ValueError(
            f"Not enough data for walk-forward analysis: {n_obs} observations, but the first fit window "
            f"needs {train_window} plus at least {min_eval_obs} evaluation bars."
        )
    return windows


@dataclass
class WalkForwardResult:
    windows: pd.DataFrame
    daily: pd.DataFrame
    trades: pd.DataFrame
    signals: pd.DataFrame
    metrics: dict[str, float]
    benchmarks: pd.DataFrame
    comparison: pd.DataFrame
    rules: SignalRules

    @property
    def returns(self) -> pd.Series:
        return self.daily["net_return"]


def run_walk_forward(
    prices: pd.DataFrame,
    config: ResearchConfig,
    rules: SignalRules | None = None,
    end_idx: int | None = None,
    names: dict[str, str] | None = None,
    opens: pd.DataFrame | None = None,
    first_eval_idx: int | None = None,
) -> WalkForwardResult:
    """Walk-forward evaluation over ``prices[:end_idx]`` with periodic re-estimation."""
    names = names or {"A": "A", "B": "B"}
    config = config.for_pair(names["A"], names["B"])
    rules = rules or signal_rules_from_config(config)
    wf = config.walk_forward
    data = prices.iloc[:end_idx] if end_idx is not None else prices
    data_opens = None if opens is None else opens.iloc[: len(data)]
    logp = np.log(data)
    schedule = walk_forward_schedule(len(data), wf.train_window, wf.test_window, wf.expanding, wf.min_eval_observations,
                                     first_eval_idx=first_eval_idx)
    equity = float(config.portfolio.initial_capital)
    rows, dailies, trade_frames, signal_frames = [], [], [], []

    for window in schedule:
        fit_slice = slice(window.fit_start_idx, window.fit_end_idx)
        eval_prices = data.iloc[window.eval_start_idx:window.eval_end_idx]
        row = {
            "window": window.window_id,
            "fit_start": _date(data.index[window.fit_start_idx]),
            "fit_end": _date(data.index[window.fit_end_idx - 1]),
            "eval_start": _date(eval_prices.index[0]),
            "eval_end": _date(eval_prices.index[-1]),
            "fit_obs": window.fit_end_idx - window.fit_start_idx,
            "eval_obs": len(eval_prices),
        }
        bt_config = BacktestConfig.from_research_config(config, tickers=(names["A"], names["B"]), initial_capital=equity)
        eval_opens = None if data_opens is None else data_opens.iloc[window.eval_start_idx:window.eval_end_idx]
        model: PairModel | None = None
        try:
            model = fit_pair_model(logp.iloc[fit_slice], config, names["A"], names["B"])
            row.update(
                hedge_ratio=model.spread.hedge_ratio, intercept=model.spread.intercept,
                spread_mean=model.spread.mean, spread_std=model.spread.std,
                eg_p_value=model.coint.p_value, cointegrated=model.coint.reject_null,
                half_life_bars=model.half_life,
            )
            if model.suite is not None:
                row.update({f"suite_{k}": v for k, v in model.suite.summary().items()})
        except ValueError as exc:
            row.update(hedge_ratio=np.nan, intercept=np.nan, spread_mean=np.nan, spread_std=np.nan,
                       eg_p_value=np.nan, cointegrated=False, half_life_bars=np.nan, note=str(exc))

        trade = model is not None and (model.passes_gate(config) or not wf.require_cointegration)
        window_rules = rules
        if model is not None and config.strategy.max_holding_half_lives:
            window_rules = rules_for_model(config, model.half_life, rules.entry_z, rules.exit_z)
            row["max_holding_bars"] = window_rules.max_holding_bars
        if trade:
            signals = build_signals(eval_prices, model.spread, window_rules)
        else:
            signals = _flat_signals(eval_prices, model.spread if model is not None else None)
            failed = ("the Engle-Granger test" if config.cointegration.gate == "eg"
                      else f"the {config.cointegration.gate} cointegration gate")
            row.setdefault("note", f"Not traded: fit window failed {failed}." if model is not None else row.get("note"))
        result = run_backtest(eval_prices, signals, model.spread.hedge_ratio if model is not None else 0.0, bt_config,
                              eval_opens, accrue_first_bar=window.window_id > 1)
        m = config.metrics
        stats = strategy_metrics(result, m.risk_free, m.periods_per_year, min_years=math.inf)
        row.update(
            traded=trade, completed_trades=stats["completed_trades"], window_return=stats["cumulative_return"],
            window_gross_return=stats["gross_cumulative_return"], window_max_drawdown=stats["max_drawdown"],
            transaction_costs=stats["total_transaction_costs"],
            start_equity=equity, end_equity=float(result.equity.iloc[-1]),
        )
        rows.append(row)
        daily = result.daily.copy()
        daily.insert(0, "window", window.window_id)
        dailies.append(daily)
        if not result.trades.empty:
            trades = result.trades.copy()
            trades.insert(0, "window", window.window_id)
            trade_frames.append(trades)
        sig = signals.copy()
        sig.insert(0, "window", window.window_id)
        signal_frames.append(sig)
        equity = float(result.equity.iloc[-1])

    daily = pd.concat(dailies)
    daily["drawdown"] = daily["equity"] / daily["equity"].cummax() - 1.0
    daily["cumulative_costs"] = (daily["trade_cost"] + daily["borrow_cost"]).cumsum()
    trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    if not trades.empty:
        trades["trade_id"] = np.arange(1, len(trades) + 1)
    combined = BacktestResult(daily, trades if not trades.empty else _empty_trades(), pd.concat(signal_frames),
                              float("nan"), BacktestConfig.from_research_config(config))
    m = config.metrics
    metrics = strategy_metrics(combined, m.risk_free, m.periods_per_year, m.min_years_for_cagr)
    oos_prices = data.loc[daily.index]
    bench = benchmark_returns(oos_prices, names, m.risk_free, m.periods_per_year)
    comp = comparison_table(combined.returns, combined.gross_returns, bench, m.risk_free, m.periods_per_year, m.min_years_for_cagr)
    return WalkForwardResult(pd.DataFrame(rows), daily, combined.trades, combined.signals, metrics, bench, comp, rules)


def _flat_signals(prices: pd.DataFrame, model: SpreadModel | None) -> pd.DataFrame:
    frame = pd.DataFrame(index=prices.index)
    if model is not None:
        logp = np.log(prices)
        frame["spread"] = model.spread(logp["A"], logp["B"])
        frame["zscore"] = model.zscore(logp["A"], logp["B"])
    else:
        frame["spread"] = np.nan
        frame["zscore"] = np.nan
    frame["target_position"] = 0
    frame["event"] = ""
    frame["holding_bars"] = 0
    return frame


def _empty_trades() -> pd.DataFrame:
    from .backtester import TRADE_COLUMNS

    return pd.DataFrame(columns=TRADE_COLUMNS)


# --------------------------------------------------------------------------- full research run
@dataclass
class PairResearchResult:
    config: ResearchConfig
    pair: AlignedPair
    splits: dict[str, DataSplit]
    model: PairModel
    research_end_idx: int
    spread: pd.Series
    zscore: pd.Series
    periods: dict[str, PeriodEvaluation]
    validation_grid: pd.DataFrame
    selected_params: dict[str, float] | None
    test_params: dict[str, object]
    cost_sensitivity: dict[str, pd.DataFrame]
    walk_forward: WalkForwardResult | None
    test_revealed: bool
    warnings: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).replace(microsecond=0).isoformat())
    cointegration_suite: CointegrationSuite | None = None  # training window: EG both ways + Johansen
    test_model: PairModel | None = None                    # set when evaluation.static_refit_before_test is on

    @property
    def names(self) -> dict[str, str]:
        return {"A": self.pair.ticker_a, "B": self.pair.ticker_b}


def run_pair_research(
    pair: AlignedPair,
    config: ResearchConfig,
    reveal_test: bool = False,
    include_walk_forward: bool = True,
) -> PairResearchResult:
    """End-to-end static research workflow for one aligned pair (see module docstring).

    Static-workflow options (all off in the baseline): ``evaluation.static_require_cointegration`` holds a
    period flat when its model fails ``cointegration.gate``; ``evaluation.static_refit_before_test`` refits
    the spread model on train + validation before the test period (data available at that time);
    ``strategy.max_holding_half_lives`` adds a time stop at k x the fitted half-life. The pair's market
    overrides (``markets:``) are applied first, and the result carries the resolved config.
    """
    config = config.for_pair(pair.ticker_a, pair.ticker_b)  # market overrides (no-op without ``markets``)
    prices = pair.prices
    opens = pair.opens
    names = {"A": pair.ticker_a, "B": pair.ticker_b}
    s = config.splits
    splits = chronological_split(prices.index, s.train, s.validation, s.test)
    logp = np.log(prices)
    warnings: list[str] = []
    if pair.is_synthetic:
        warnings.append("SYNTHETIC DATA: results illustrate the methodology only and are not market evidence.")

    train = splits["train"]
    model = fit_pair_model(logp.iloc[train.slice], config, names["A"], names["B"])
    warnings.extend(model.coint.warnings)
    suite = _training_suite(logp.iloc[train.slice], config, names, model)

    research_end = len(prices) if reveal_test else splits["validation"].end_idx
    visible = prices.iloc[:research_end]
    spread = model.spread.spread(np.log(visible["A"]), np.log(visible["B"]))
    zscore = model.spread.zscore(np.log(visible["A"]), np.log(visible["B"]))

    bt_config = BacktestConfig.from_research_config(config, tickers=(names["A"], names["B"]))
    base_rules = rules_for_model(config, model.half_life)
    ev = config.evaluation
    tradable = model.passes_gate(config) if ev.static_require_cointegration else True
    if not tradable:
        warnings.append(f"Static cointegration gate ({config.cointegration.gate}) failed on the training fit: "
                        "the training and validation periods are held flat.")
    periods = {
        "train": evaluate_period(prices, train, model.spread, base_rules, bt_config, config, names, opens, tradable),
        "validation": evaluate_period(prices, splits["validation"], model.spread, base_rules, bt_config, config, names,
                                      opens, tradable),
    }

    val_slice = splits["validation"].slice
    val_prices, val_opens = prices.iloc[val_slice], _slice(opens, val_slice)
    grid = parameter_grid_search(val_prices, model.spread, config, bt_config, opens=val_opens,
                                 half_life_bars=model.half_life)
    selected = select_parameters(grid, ev.selection_metric, ev.min_trades_for_selection)

    if ev.test_parameter_source == "validation_selected" and selected is not None:
        test_rules = rules_for_model(config, model.half_life, selected["entry_z"], selected["exit_z"])
        test_params = {"source": "validation_selected", "entry_z": test_rules.entry_z, "exit_z": test_rules.exit_z}
    else:
        test_rules = base_rules
        test_params = {"source": "config", "entry_z": base_rules.entry_z, "exit_z": base_rules.exit_z}
        if ev.test_parameter_source == "validation_selected":
            warnings.append(
                "No validation grid point met the selection criteria; the pre-specified thresholds are used on the test period."
            )

    costs = {"validation": cost_sensitivity(val_prices, model.spread, base_rules, config, bt_config, opens=val_opens)}
    test_model = None
    if reveal_test:
        test_spread, test_tradable = model.spread, tradable
        if ev.static_refit_before_test:
            test_model = fit_pair_model(logp.iloc[: splits["validation"].end_idx], config, names["A"], names["B"])
            test_rules = rules_for_model(config, test_model.half_life, test_rules.entry_z, test_rules.exit_z)
            test_spread = test_model.spread
            test_tradable = test_model.passes_gate(config) if ev.static_require_cointegration else True
            test_params.update(model="refit on train + validation", refit_hedge_ratio=test_model.spread.hedge_ratio,
                               refit_eg_p_value=test_model.coint.p_value, refit_fit_end=test_model.spread.fit_end)
            sl = splits["test"].slice
            log_test = np.log(prices.iloc[sl])
            spread.iloc[sl] = test_spread.spread(log_test["A"], log_test["B"]).to_numpy()
            zscore.iloc[sl] = test_spread.zscore(log_test["A"], log_test["B"]).to_numpy()
            if not test_tradable:
                warnings.append("Static cointegration gate failed on the refit model: the test period is held flat.")
        periods["test"] = evaluate_period(prices, splits["test"], test_spread, test_rules, bt_config, config, names,
                                          opens, test_tradable)
        test_slice = splits["test"].slice
        costs["test"] = cost_sensitivity(prices.iloc[test_slice], test_spread, test_rules, config, bt_config,
                                         opens=_slice(opens, test_slice))

    walk_forward = None
    if include_walk_forward:
        try:
            walk_forward = run_walk_forward(prices, config, base_rules, end_idx=research_end, names=names, opens=opens)
        except ValueError as exc:
            warnings.append(f"Walk-forward analysis skipped: {exc}")

    return PairResearchResult(
        config=config, pair=pair, splits=splits, model=model, research_end_idx=research_end, spread=spread,
        zscore=zscore, periods=periods, validation_grid=grid, selected_params=selected, test_params=test_params,
        cost_sensitivity=costs, walk_forward=walk_forward, test_revealed=reveal_test, warnings=warnings,
        cointegration_suite=suite, test_model=test_model,
    )


def static_tradable(result: PairResearchResult, period: str) -> bool:
    """Whether the static workflow trades ``period`` (``evaluation.static_require_cointegration``)."""
    if not result.config.evaluation.static_require_cointegration:
        return True
    model = result.test_model if period == "test" and result.test_model is not None else result.model
    return model.passes_gate(result.config)


def _training_suite(log_train: pd.DataFrame, config: ResearchConfig, names: dict[str, str],
                    model: PairModel) -> CointegrationSuite | None:
    """Reverse EG and Johansen on the training window (reuses the model's suite when it has one)."""
    if model.suite is not None:
        return model.suite
    c = config.cointegration
    try:
        return cointegration_suite(log_train["A"], log_train["B"], c.significance, c.trend, c.autolag,
                                   c.min_observations, c.johansen_det_order, c.johansen_k_ar_diff,
                                   names["A"], names["B"], eg_ab=model.coint)
    except ValueError:
        return None


# --------------------------------------------------------------------------- exports and snapshots
def result_tables(result: PairResearchResult) -> dict[str, pd.DataFrame]:
    """All exportable tables keyed by file stem."""
    tables: dict[str, pd.DataFrame] = {}
    for name, period in result.periods.items():
        bt = period.backtest
        tables[f"daily_portfolio_{name}"] = bt.daily
        tables[f"positions_{name}"] = bt.daily[[
            "position", "units_a", "units_b", "notional_a", "notional_b", "long_exposure", "short_exposure",
            "gross_exposure", "net_exposure", "gross_leverage", "cash", "equity",
        ]]
        tables[f"trades_{name}"] = bt.trades
        tables[f"signals_{name}"] = bt.signals
        tables[f"metrics_comparison_{name}"] = period.comparison
    tables["strategy_metrics"] = pd.DataFrame({name: p.metrics for name, p in result.periods.items()})
    tables["validation_parameter_grid"] = result.validation_grid
    for name, frame in result.cost_sensitivity.items():
        tables[f"cost_sensitivity_{name}"] = frame
    if result.walk_forward is not None:
        tables["walk_forward_windows"] = result.walk_forward.windows
        tables["walk_forward_daily"] = result.walk_forward.daily
        tables["walk_forward_trades"] = result.walk_forward.trades
        tables["walk_forward_metrics_comparison"] = result.walk_forward.comparison
    return tables


def statistical_tests_payload(result: PairResearchResult) -> dict[str, object]:
    return {
        "label": "Training-period estimates",
        "hedge_ratio": result.model.hedge.to_dict(),
        "cointegration": result.model.coint.to_dict(),
        "spread_model": result.model.spread.to_dict(),
        "half_life_bars": result.model.half_life,
        "cointegration_suite": result.cointegration_suite.to_dict() if result.cointegration_suite else None,
        "test_model_refit": result.test_model.summary() if result.test_model else None,
    }


def run_metadata(result: PairResearchResult) -> dict[str, object]:
    pair = result.pair
    return {
        "created_at": result.created_at,
        "config_fingerprint": result.config.fingerprint(),
        "asset_a": pair.metadata_a,
        "asset_b": pair.metadata_b,
        "price_columns": {"A": pair.price_column_a, "B": pair.price_column_b},
        "alignment": pair.report.summary(),
        "validation": {
            "A": pair.validation_a.summary() if pair.validation_a else None,
            "B": pair.validation_b.summary() if pair.validation_b else None,
        },
        "splits": {k: asdict(v) | {"n_obs": v.n_obs} for k, v in result.splits.items()},
        "test_revealed": result.test_revealed,
        "test_parameters": result.test_params,
        "validation_selected_parameters": result.selected_params,
        "n_validation_grid_points": int(len(result.validation_grid)),
        "warnings": result.warnings,
        "environment": _environment(),
    }


def _environment() -> dict[str, str]:
    versions = {"python": sys.version.split()[0], "platform": platform.platform()}
    for module in ("numpy", "pandas", "scipy", "statsmodels", "yfinance", "plotly", "streamlit"):
        try:
            versions[module] = __import__(module).__version__
        except Exception:  # optional dependency missing
            versions[module] = "not installed"
    return versions


def save_research_run(result: PairResearchResult, output_dir: str | Path | None = None, ml_result=None) -> Path:
    """Write a reproducible snapshot of one research run to ``reports/<run_id>/``.

    When ``ml_result`` (from :func:`statarb.ml_evaluation.run_ml_research`) is given, the ML
    settings, classification metrics, predictions and strategy comparisons are saved too.
    """
    root = resolve_path(output_dir or result.config.reproducibility.reports_dir)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe = lambda t: "".join(ch if ch.isalnum() else "-" for ch in t)  # noqa: E731
    run_dir = root / f"{stamp}_{safe(result.pair.ticker_a)}_{safe(result.pair.ticker_b)}_{result.config.fingerprint()}"
    run_dir.mkdir(parents=True, exist_ok=False)

    save_config_snapshot(result.config, run_dir / "config_snapshot.yaml")
    prices_path = run_dir / "aligned_prices.csv"
    result.pair.prices.rename(columns=result.names).to_csv(prices_path)
    meta = run_metadata(result)
    from .data_loader import file_sha256

    meta["aligned_prices_sha256"] = file_sha256(prices_path)
    _write_json(run_dir / "run_metadata.json", meta)
    _write_json(run_dir / "statistical_tests.json", statistical_tests_payload(result))
    for name, table in result_tables(result).items():
        table.to_csv(run_dir / f"{name}.csv")
    if ml_result is not None:
        from .ml_evaluation import save_ml_tables  # local import: ml_evaluation depends on this module

        save_ml_tables(ml_result, run_dir)
    return run_dir


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(_sanitize(payload), indent=2, default=_json_default, allow_nan=False), encoding="utf-8")


def _sanitize(value):
    """Recursively replace non-finite floats with ``None`` so the output is strict JSON."""
    if isinstance(value, dict):
        return {str(k): _sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return str(value)


__all__ = [
    "DataSplit",
    "PERIOD_LABELS",
    "PairModel",
    "PairResearchResult",
    "PeriodEvaluation",
    "WalkForwardResult",
    "WalkForwardWindow",
    "backtest_period",
    "build_signals",
    "chronological_split",
    "cost_sensitivity",
    "evaluate_period",
    "fit_pair_model",
    "parameter_grid_search",
    "result_tables",
    "run_metadata",
    "run_pair_research",
    "rules_for_model",
    "run_walk_forward",
    "save_research_run",
    "static_tradable",
    "select_parameters",
    "signal_rules_from_config",
    "statistical_tests_payload",
    "walk_forward_schedule",
]
