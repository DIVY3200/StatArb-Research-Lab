"""Signal generation for the ML-only (B) and hybrid (C) variants, and their execution safeguards."""

import numpy as np
import pandas as pd
import pytest

from statarb.backtester import BacktestConfig, run_backtest
from statarb.ml_strategy import MLSignalRules, hybrid_signals, ml_only_signals, probability_gate
from statarb.signals import SignalRules, generate_signals


def _s(values):
    return pd.Series(values, index=pd.bdate_range("2023-01-02", periods=len(values)), dtype=float)


RULES = SignalRules(entry_z=2.0, exit_z=0.5)


# --------------------------------------------------------------------------- hybrid (C)
def test_hybrid_with_always_confirming_model_equals_traditional():
    rng = np.random.default_rng(3)
    z = _s(np.cumsum(rng.standard_normal(400)) * 0.3)
    traditional = generate_signals(z, RULES)
    hybrid = hybrid_signals(z, _s(np.ones(400)), RULES, threshold=0.6)
    pd.testing.assert_series_equal(hybrid["target_position"], traditional["target_position"])


def test_hybrid_never_enters_without_confirmation():
    z = _s([0.0, -2.5, -2.6, -1.0, 0.0])
    out = hybrid_signals(z, _s([0.9, 0.3, 0.4, 0.2, 0.1]), RULES, threshold=0.6)
    assert (out["target_position"] == 0).all()


def test_hybrid_can_enter_later_while_still_beyond_entry():
    z = _s([-2.5, -2.4, -2.2, -1.0, 0.0])
    out = hybrid_signals(z, _s([0.3, 0.7, 0.9, 0.0, 0.0]), RULES, threshold=0.6)
    assert out["target_position"].tolist() == [0, 1, 1, 1, 0]
    assert out["event"].iloc[1] == "entry_long"


def test_hybrid_exits_ignore_the_model():
    z = _s([-2.5, -1.5, -1.0, -0.3])
    out = hybrid_signals(z, _s([0.9, 0.0, np.nan, 0.0]), RULES, threshold=0.6)
    assert out["target_position"].tolist() == [1, 1, 1, 0]  # held despite p falling, exits on the z rule


def test_probability_gate_treats_missing_as_no():
    gate = probability_gate(_s([np.nan, 0.59, 0.6, 0.8]), 0.6)
    assert gate.tolist() == [False, False, True, True]


# --------------------------------------------------------------------------- ML-only (B)
def test_ml_only_direction_is_a_bet_on_reversion():
    rules = MLSignalRules(threshold=0.6, min_abs_z=1.0, exit_z=0.5, horizon=5)
    long_ = ml_only_signals(_s([-1.5, -1.2]), _s([0.8, 0.8]), rules)
    assert long_["target_position"].iloc[0] == 1 and long_["event"].iloc[0] == "entry_long"
    short = ml_only_signals(_s([1.5, 1.2]), _s([0.8, 0.8]), rules)
    assert short["target_position"].iloc[0] == -1 and short["event"].iloc[0] == "entry_short"


def test_ml_only_entry_requirements():
    rules = MLSignalRules(threshold=0.6, min_abs_z=1.0, exit_z=0.5, horizon=5, stop_loss_z=4.0)
    z = _s([-0.8, -1.5, -1.5, -4.5, -1.5])
    p = _s([0.9, 0.5, np.nan, 0.9, 0.9])
    # |z|<min_abs_z; p<threshold; missing p; beyond stop; then a valid entry
    assert ml_only_signals(z, p, rules)["target_position"].tolist() == [0, 0, 0, 0, 1]


def test_ml_only_horizon_exit_and_renewal():
    rules = MLSignalRules(threshold=0.6, min_abs_z=1.0, exit_z=0.5, horizon=3)
    z = _s([-1.5, -1.4, -1.3, -1.2, -1.2, -1.1, -1.1])
    expired = ml_only_signals(z, _s([0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), rules)
    assert expired["target_position"].tolist() == [1, 1, 1, 0, 0, 0, 0]
    assert expired["event"].iloc[3] == "horizon_exit"
    renewed = ml_only_signals(z, _s([0.8] * 7), rules)
    assert renewed["target_position"].tolist() == [1] * 7  # prediction still supports it: no round trip
    assert renewed["event"].iloc[3] == "renew"


def test_ml_only_exit_band_and_stop_loss():
    rules = MLSignalRules(threshold=0.6, min_abs_z=1.0, exit_z=0.5, horizon=10, stop_loss_z=3.0)
    reverted = ml_only_signals(_s([1.5, 1.0, 0.4]), _s([0.9, 0.9, 0.9]), rules)
    assert reverted["target_position"].tolist() == [-1, -1, 0] and reverted["event"].iloc[2] == "exit"
    stopped = ml_only_signals(_s([1.5, 2.5, 3.2]), _s([0.9, 0.9, 0.9]), rules)
    assert stopped["target_position"].tolist() == [-1, -1, 0] and stopped["event"].iloc[2] == "stop_loss"


@pytest.mark.parametrize("seed", range(10))
def test_ml_only_never_reverses_on_one_bar(seed):
    rng = np.random.default_rng(seed)
    z = _s(np.cumsum(rng.standard_normal(500)) * 0.3)
    p = _s(rng.uniform(size=500))
    out = ml_only_signals(z, p, MLSignalRules(threshold=0.5, min_abs_z=1.0, exit_z=0.25, horizon=5, stop_loss_z=3.5))
    target = out["target_position"].to_numpy()
    assert set(np.unique(target)).issubset({-1, 0, 1})
    assert not np.any(np.abs(np.diff(target)) == 2)


def test_rule_validation():
    with pytest.raises(ValueError):
        MLSignalRules(threshold=1.0)
    with pytest.raises(ValueError):
        MLSignalRules(horizon=0)
    with pytest.raises(ValueError):
        MLSignalRules(min_abs_z=2.0, stop_loss_z=1.5)
    with pytest.raises(ValueError, match="share an index"):
        ml_only_signals(_s([1.0, 2.0]), pd.Series([0.5, 0.5]), MLSignalRules())


# --------------------------------------------------------------------------- execution safeguards
def _prices(a, b):
    return pd.DataFrame({"A": a, "B": b}, index=pd.bdate_range("2023-01-02", periods=len(a)), dtype=float)


def test_ml_signal_executes_next_bar_and_cannot_earn_its_own_move():
    # The prediction on bar 2 sees A jump on bar 2; the trade fills at bar 3's close and earns nothing from it.
    prices = _prices([100, 100, 110, 110, 110, 110], [100] * 6)
    z = _s([0.0, 0.0, -1.5, -1.5, -1.5, -1.5])
    sig = ml_only_signals(z, _s([0.0, 0.0, 0.9, 0.9, 0.9, 0.9]), MLSignalRules(threshold=0.6, horizon=10))
    cfg = BacktestConfig(commission_bps=0, slippage_bps=0, borrow_bps_annual=0, rebalance_tolerance=None)
    result = run_backtest(prices, sig, 1.0, cfg)
    assert result.daily["position"].tolist() == [0, 0, 0, 1, 1, 0]
    assert result.daily["pnl"].abs().sum() == 0.0


def test_ml_variant_end_of_sample_and_costs_on_both_legs():
    prices = _prices(100 * np.ones(8), 50 * np.ones(8))
    sig = ml_only_signals(_s([-1.5] * 8), _s([0.9] * 8), MLSignalRules(threshold=0.6, horizon=3))
    cfg = BacktestConfig(commission_bps=5, slippage_bps=5, borrow_bps_annual=0, rebalance_tolerance=None)
    result = run_backtest(prices, sig, 1.0, cfg)
    trade = result.trades.iloc[0]
    assert trade["exit_reason"] == "end_of_period" and result.daily["position"].iloc[-1] == 0
    budget = 1_000_000 / 1.001
    np.testing.assert_allclose(trade["transaction_costs"], 2 * budget * 0.001)  # entry + exit, both legs
