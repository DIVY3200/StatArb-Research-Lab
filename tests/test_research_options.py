"""Opt-in engine options end to end (all off by default; tests/test_baseline_preserved.py guards that).

Half-life time stop, static cointegration gate and pre-test refit, gate variants in walk-forward,
next-open execution, market costs with futures shorts and cash interest, per-market risk-free rates,
and point-in-time ML training rows.
"""

import dataclasses
import math

import numpy as np
import pandas as pd
import pytest

from statarb.config import ResearchConfig
from statarb.data_validation import prepare_pair
from statarb.evaluation import (
    rules_for_model,
    run_pair_research,
    run_walk_forward,
    signal_rules_from_config,
    static_tradable,
    statistical_tests_payload,
)
from statarb.experiment import at_cost_level, pooled_returns, risk_free_daily, run_pair
from statarb.metrics import daily_risk_free
from statarb.ml_evaluation import MODEL_NAMES, reprice_walk_forward, run_ml_research, run_ml_walk_forward
from statarb.ml_features import FeatureSpec, point_in_time_rows

FAST = {"ml": {"rf_n_estimators": 30, "logistic_c_grid": [0.1, 1.0], "rf_max_depth_grid": [3],
               "rf_min_samples_leaf_grid": [10]},
        "walk_forward": {"train_window": 500, "test_window": 250}}
NESTED = {"walk_forward_tuning": "nested", "class_weight": "none", "threshold_mode": "prevalence",
          "tuning_metric": "log_loss"}
PIT = {"training_features": "point_in_time", "pit_estimation_window": 250, "pit_block": 21}
MARKETS = {"NS": {"costs": {"profile": "nse", "short_instrument": "futures", "futures_lot_sizes": {"AAA.NS": 25},
                            "futures_roll_every_bars": 5},
                  "metrics": {"risk_free_rate": 0.065}, "portfolio": {"cash_interest": True}}}


def make_config(**sections) -> ResearchConfig:
    overrides = {k: dict(v) for k, v in FAST.items()}
    for section, values in sections.items():
        overrides[section] = values if section == "markets" else {**overrides.get(section, {}), **values}
    return ResearchConfig().with_overrides(overrides)


def as_nse(assets):
    return [dataclasses.replace(a, ticker=t) for a, t in zip(assets, ("AAA.NS", "BBB.NS"))]


# --------------------------------------------------------------------------- half-life time stop
def test_rules_for_model_half_life_cap():
    base = ResearchConfig()
    assert rules_for_model(base, 7.3) == signal_rules_from_config(base)
    k2 = make_config(strategy={"max_holding_half_lives": 2.0})
    assert rules_for_model(k2, 7.3).max_holding_bars == 15
    assert rules_for_model(k2, 0.2).max_holding_bars == 1
    assert rules_for_model(k2, float("nan")).max_holding_bars is None
    capped = make_config(strategy={"max_holding_half_lives": 2.0, "max_holding_days": 10})
    assert rules_for_model(capped, 7.3).max_holding_bars == 10
    assert rules_for_model(capped, 3.0).max_holding_bars == 6


def test_static_half_life_stop_caps_every_trade(coint_pair):
    res = run_pair_research(coint_pair, make_config(strategy={"max_holding_half_lives": 1.0}),
                            include_walk_forward=False)
    cap = math.ceil(res.model.half_life)
    reasons = []
    for period in res.periods.values():
        assert period.rules.max_holding_bars == cap
        trades = period.backtest.trades
        closed = trades[trades["status"] == "closed"]
        assert (closed["holding_days"] <= cap).all()
        reasons += list(closed["exit_reason"])
    assert "max_holding" in reasons


def test_walk_forward_half_life_stop_is_per_window(coint_pair):
    wf = run_walk_forward(coint_pair.prices, make_config(strategy={"max_holding_half_lives": 1.0}))
    w = wf.windows
    np.testing.assert_array_equal(w["max_holding_bars"], np.ceil(w["half_life_bars"]).astype(int))
    closed = wf.trades[wf.trades["status"] == "closed"]
    caps = closed["window"].map(w.set_index("window")["max_holding_bars"])
    assert (closed["holding_days"] <= caps).all()


# --------------------------------------------------------------------------- static gate and pre-test refit
def test_static_gate_holds_periods_flat(indep_pair):
    gated = run_pair_research(indep_pair, make_config(evaluation={"static_require_cointegration": True}),
                              include_walk_forward=False)
    assert not gated.model.coint.reject_null and not static_tradable(gated, "validation")
    for name in ("train", "validation"):
        assert (gated.periods[name].backtest.daily["position"] == 0).all()
        assert gated.periods[name].metrics["completed_trades"] == 0
    assert any("gate" in w for w in gated.warnings)
    plain = run_pair_research(indep_pair, make_config(), include_walk_forward=False)
    assert static_tradable(plain, "validation") and plain.periods["train"].metrics["completed_trades"] > 0


def test_static_refit_before_test(coint_pair):
    plain = run_pair_research(coint_pair, make_config(), reveal_test=True, include_walk_forward=False)
    refit = run_pair_research(coint_pair, make_config(evaluation={"static_refit_before_test": True}),
                              reveal_test=True, include_walk_forward=False)
    val = refit.splits["validation"]
    assert refit.test_model.spread.n_obs == val.end_idx and refit.test_model.spread.fit_end == val.end_date
    assert refit.test_params["model"] == "refit on train + validation"
    for name in ("train", "validation"):  # earlier periods are untouched
        pd.testing.assert_frame_equal(refit.periods[name].backtest.daily, plain.periods[name].backtest.daily)
    sl = refit.splits["test"].slice
    logp = np.log(coint_pair.prices.iloc[sl])
    np.testing.assert_allclose(refit.zscore.iloc[sl], refit.test_model.spread.zscore(logp["A"], logp["B"]))
    assert refit.periods["test"].backtest.hedge_ratio == refit.test_model.spread.hedge_ratio
    assert refit.test_model.spread.hedge_ratio != plain.model.spread.hedge_ratio
    assert statistical_tests_payload(refit)["test_model_refit"]["n_obs"] == val.end_idx
    assert plain.test_model is None and statistical_tests_payload(plain)["test_model_refit"] is None


def test_training_suite_is_reported(coint_pair):
    res = run_pair_research(coint_pair, make_config(), include_walk_forward=False)
    payload = statistical_tests_payload(res)["cointegration_suite"]
    assert payload["summary"]["eg_ab_p_value"] == res.model.coint.p_value
    assert payload["eg_b_on_a"]["hedge_ratio"] != res.model.hedge.hedge_ratio
    assert payload["johansen"]["n_obs"] == res.splits["train"].n_obs


# --------------------------------------------------------------------------- walk-forward gates
@pytest.mark.parametrize("pair_name", ["coint_pair", "indep_pair"])
@pytest.mark.parametrize("gate", ["eg_both", "johansen", "eg_and_johansen"])
def test_walk_forward_gate_variants(request, pair_name, gate):
    pair = request.getfixturevalue(pair_name)
    w = run_walk_forward(pair.prices, make_config(cointegration={"gate": gate})).windows
    ab, ba, jo = (w[f"suite_{k}_pass"].astype(bool) for k in ("eg_ab", "eg_ba", "johansen"))
    expected = {"eg_both": ab & ba, "johansen": jo, "eg_and_johansen": ab & jo}[gate]
    assert (w["traded"].astype(bool) == expected).all()
    np.testing.assert_allclose(w["suite_eg_ab_p_value"], w["eg_p_value"])


def test_recording_all_tests_keeps_the_eg_gate(coint_pair):
    plain = run_walk_forward(coint_pair.prices, make_config())
    recorded = run_walk_forward(coint_pair.prices, make_config(cointegration={"record_all_tests": True}))
    assert "suite_johansen_trace" in recorded.windows and "suite_johansen_trace" not in plain.windows
    pd.testing.assert_frame_equal(recorded.daily, plain.daily)


# --------------------------------------------------------------------------- next-open execution
def test_next_open_execution_end_to_end(coint_pair):
    opened = run_pair_research(coint_pair, make_config(strategy={"execution_price": "open"}), include_walk_forward=False)
    closed = run_pair_research(coint_pair, make_config(), include_walk_forward=False)
    daily = opened.periods["validation"].backtest.daily
    np.testing.assert_allclose(daily["fill_price_a"], coint_pair.opens.loc[daily.index, "A"])
    np.testing.assert_allclose(daily["fill_price_b"], coint_pair.opens.loc[daily.index, "B"])
    # same signals, different fills
    pd.testing.assert_frame_equal(opened.periods["validation"].backtest.signals,
                                  closed.periods["validation"].backtest.signals)
    assert (opened.periods["validation"].metrics["cumulative_return"]
            != closed.periods["validation"].metrics["cumulative_return"])


# --------------------------------------------------------------------------- market costs, futures, interest
def test_market_overrides_reach_the_backtests(coint_assets):
    res = run_pair_research(prepare_pair(*as_nse(coint_assets)), make_config(markets=MARKETS), include_walk_forward=False)
    assert res.config.costs.profile == "nse" and res.config.metrics.risk_free_rate == 0.065
    bt = res.periods["validation"].backtest
    assert (bt.config.cost_profile, bt.config.short_instrument, bt.config.futures_lot_size_a) == ("nse", "futures", 25)
    assert bt.config.cash_interest_rate == 0.065 and bt.daily["interest"].sum() > 0
    assert bt.daily["borrow_cost"].sum() == 0.0  # futures shorts pay no borrow fee
    short_a = bt.daily["units_a"] < 0
    assert short_a.any()
    lots = -bt.daily.loc[short_a, "units_a"] / 25
    np.testing.assert_allclose(lots, np.round(lots), rtol=0, atol=1e-6)  # whole lots only
    assert bt.trades["rolls"].sum() > 0


def test_us_pair_is_untouched_by_nse_overrides(coint_pair):
    with_markets = run_pair_research(coint_pair, make_config(markets=MARKETS), include_walk_forward=False)
    plain = run_pair_research(coint_pair, make_config(), include_walk_forward=False)
    pd.testing.assert_frame_equal(with_markets.periods["validation"].backtest.daily,
                                  plain.periods["validation"].backtest.daily)


def test_cost_levels_for_market_profiles():
    generic = at_cost_level(ResearchConfig(), 20.0)
    assert (generic.costs.commission_bps, generic.costs.slippage_bps) == (20.0, 0.0)
    nse = at_cost_level(ResearchConfig().with_overrides({"costs": {"profile": "nse"}}), 20.0)
    assert (nse.costs.commission_bps, nse.costs.slippage_bps) == (ResearchConfig().costs.commission_bps, 20.0)


def test_nse_cost_sensitivity_varies_slippage(coint_assets):
    res = run_pair_research(prepare_pair(*as_nse(coint_assets)), make_config(markets=MARKETS), include_walk_forward=False)
    table = res.cost_sensitivity["validation"]
    values = table.sort_values(table.columns[0])["cumulative_return"].to_numpy()
    assert np.all(np.diff(values) <= 1e-12)


def test_pooled_excess_returns_subtract_each_markets_rate():
    idx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    d1 = {"traditional": pd.DataFrame({"net_return": [0.01, 0.02]}, index=idx[:2])}
    d2 = {"traditional": pd.DataFrame({"net_return": [0.04, -0.02]}, index=idx[1:])}
    excess = pooled_returns([d1, d2], ["traditional"], rf_daily=[0.001, 0.002])
    np.testing.assert_allclose(excess["traditional"], [(0.01 - 0.001) / 2, (0.02 - 0.001 + 0.04 - 0.002) / 2,
                                                       (-0.02 - 0.002) / 2])
    pd.testing.assert_frame_equal(pooled_returns([d1, d2], ["traditional"], rf_daily=[0.0, 0.0]),
                                  pooled_returns([d1, d2], ["traditional"]))


@pytest.fixture(scope="module")
def nse_run(coint_assets):
    cfg = make_config(markets=MARKETS, ml=NESTED, strategy={"execution_price": "open"})
    return run_pair(*as_nse(coint_assets), cfg)


def test_run_pair_resolves_market_settings(nse_run):
    assert nse_run.config.costs.profile == "nse" and nse_run.config.strategy.execution_price == "open"
    (rf,) = risk_free_daily([nse_run])
    assert rf.index.equals(nse_run.wf.daily["traditional"].index)
    assert rf.iloc[0] == 0.0  # the account starts at the first close
    np.testing.assert_allclose(rf.iloc[1:], daily_risk_free(0.065))
    assert nse_run.wf.daily["traditional"]["interest"].sum() > 0
    assert nse_run.wf.names == {"A": "AAA.NS", "B": "BBB.NS"}


def test_reprice_reproduces_open_execution_with_market_costs(nse_run):
    again = reprice_walk_forward(nse_run.pair.prices, nse_run.config, nse_run.wf, opens=nse_run.pair.opens)
    for variant, daily in nse_run.wf.daily.items():
        pd.testing.assert_frame_equal(again[variant], daily)


# --------------------------------------------------------------------------- point-in-time ML training rows
def test_point_in_time_rows_use_only_earlier_data(coint_pair):
    cfg = make_config(ml=PIT)
    spec, h = FeatureSpec.from_config(cfg), cfg.ml.horizon
    prices = coint_pair.prices
    pit = point_in_time_rows(prices, 0, 900, cfg, spec)
    assert len(pit) > 100 and pit.positions.min() >= 250
    assert (pit.positions + h <= 899).all()
    assert (pd.to_datetime(pit.blocks["estimation_end"]) < pd.to_datetime(pit.blocks["block_start"])).all()
    later = prices.copy()
    later.iloc[900:] *= 1.7  # nothing at or after fit_end is read
    pd.testing.assert_frame_equal(point_in_time_rows(later, 0, 900, cfg, spec).features, pit.features)
    t = 600  # a shock at bar t leaves every row before t - H unchanged and moves later blocks
    shocked = prices.copy()
    shocked.iloc[t, 0] *= 1.3
    moved = point_in_time_rows(shocked, 0, 900, cfg, spec)
    early, early_moved = pit.positions < t - h, moved.positions < t - h
    pd.testing.assert_frame_equal(moved.features[early_moved], pit.features[early])
    assert not moved.features[~early_moved].equals(pit.features[~early])


def test_point_in_time_training_in_static_ml(coint_pair):
    base = run_pair_research(coint_pair, make_config(ml=PIT), include_walk_forward=False)
    ml = run_ml_research(base, include_walk_forward=False)
    pit = point_in_time_rows(coint_pair.prices.iloc[:base.research_end_idx], 0, base.splits["train"].end_idx,
                             base.config, FeatureSpec.from_config(base.config), embargo=base.config.ml.embargo)
    assert ml.sample_table.loc["train", "rows_used"] == len(pit)
    assert ml.sample_table.loc["train", "features"].startswith("point in time")
    for name in MODEL_NAMES:
        report = ml.models[name]
        assert report.ok and report.fitted.n_train == len(pit) and report.metrics["train"]["n_samples"] == len(pit)
    assert len(ml.pit_blocks) == len(pit.blocks)
    plain = run_ml_research(run_pair_research(coint_pair, make_config(), include_walk_forward=False),
                            include_walk_forward=False)
    pd.testing.assert_series_equal(ml.dataset.z, plain.dataset.z)  # scored rows still use the frozen model
    assert ml.sample_table.loc["validation", "rows_used"] == plain.sample_table.loc["validation", "rows_used"]


def test_point_in_time_walk_forward(coint_pair):
    cfg = make_config(ml={**PIT, **NESTED})
    w = run_ml_walk_forward(coint_pair.prices, cfg, signal_rules_from_config(cfg), None, None).windows
    assert (w.loc[w["traded"].astype(bool), "pit_blocks"] > 0).all()
    first = point_in_time_rows(coint_pair.prices, 0, 500, cfg, FeatureSpec.from_config(cfg))
    assert w.loc[0, "fit_rows"] == int(first.purged(500, cfg.ml.embargo).sum())


# --------------------------------------------------------------------------- static ML consistency
def test_static_ml_follows_the_refit_and_the_gate(coint_pair, indep_pair):
    base = run_pair_research(coint_pair, make_config(evaluation={"static_refit_before_test": True}),
                             reveal_test=True, include_walk_forward=False)
    ml = run_ml_research(base, include_walk_forward=False)
    for name in MODEL_NAMES:
        report = ml.models[name]
        assert report.test_refit is not None and report.test_refit.n_train > report.fitted.n_train
        for kind in ("ml_only", "hybrid"):
            assert ml.comparisons["test"].results[f"{kind}_{name}"].hedge_ratio == base.test_model.spread.hedge_ratio
    sl = base.splits["test"].slice
    np.testing.assert_allclose(ml.dataset.z.iloc[sl], base.zscore.iloc[sl])
    gated = run_pair_research(indep_pair, make_config(evaluation={"static_require_cointegration": True}),
                              include_walk_forward=False)
    for comparison in run_ml_research(gated, include_walk_forward=False).comparisons.values():
        for result in comparison.results.values():
            assert (result.daily["position"] == 0).all()  # B and C held flat wherever A is
