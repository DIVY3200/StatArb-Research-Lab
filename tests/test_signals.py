import numpy as np
import pandas as pd
import pytest

from statarb.signals import SignalRules, generate_signals


def _run(z, **rules):
    series = pd.Series(z, index=pd.bdate_range("2021-01-01", periods=len(z)), dtype=float)
    return generate_signals(series, SignalRules(**rules))


def test_long_spread_entry_and_exit_negative_z():
    out = _run([0.0, -1.0, -2.1, -1.5, -0.8, -0.4, 0.3], entry_z=2.0, exit_z=0.5)
    assert out["target_position"].tolist() == [0, 0, 1, 1, 1, 0, 0]
    assert out["event"].tolist() == ["", "", "entry_long", "", "", "exit", ""]
    assert out["holding_bars"].tolist() == [0, 0, 0, 1, 2, 0, 0]


def test_short_spread_entry_and_exit_positive_z():
    out = _run([0.0, 2.0, 2.5, 1.0, 0.5, -0.1], entry_z=2.0, exit_z=0.5)
    assert out["target_position"].tolist() == [0, -1, -1, -1, 0, 0]  # >= entry triggers, <= exit exits
    assert out["event"].iloc[1] == "entry_short" and out["event"].iloc[4] == "exit"


def test_no_duplicate_entries_while_signal_persists():
    out = _run([-2.5] * 10 + [0.0], entry_z=2.0, exit_z=0.5)
    assert (out["event"] == "entry_long").sum() == 1
    assert out["target_position"].iloc[:10].eq(1).all()


def test_opposite_signal_exits_first_and_reenters_on_a_later_bar():
    out = _run([-2.5, -1.0, 2.5, 2.5, 0.0], entry_z=2.0, exit_z=0.5)
    assert out["target_position"].tolist() == [1, 1, 0, -1, 0]
    assert out["event"].tolist() == ["entry_long", "", "exit", "entry_short", "exit"]


def test_stop_loss_exits_and_blocks_reentry_until_back_inside_entry_band():
    z = [-2.2, -3.0, -4.1, -3.5, -2.5, -1.5, -2.3, 0.0]
    out = _run(z, entry_z=2.0, exit_z=0.5, stop_loss_z=4.0)
    assert out["event"].iloc[2] == "stop_loss"
    # Bars 3-4 are still beyond the entry level but re-entry is blocked; bar 5 resets; bar 6 re-enters.
    assert out["target_position"].tolist() == [1, 1, 0, 0, 0, 0, 1, 0]


def test_entry_refused_beyond_stop_level():
    out = _run([-4.5, -3.0, 0.0], entry_z=2.0, exit_z=0.5, stop_loss_z=4.0)
    assert out["target_position"].tolist() == [0, 1, 0]


def test_max_holding_exit():
    out = _run([-2.5, -2.0, -1.8, -1.9, -1.7, -2.6, 0.0], entry_z=2.0, exit_z=0.5, max_holding_bars=3)
    assert out["event"].iloc[3] == "max_holding"
    # Bar 4 (|z| = 1.7 < 2) lifts the re-entry block, so bar 5 (z = -2.6) opens a new trade.
    assert out["target_position"].tolist() == [1, 1, 1, 0, 0, 1, 0]


def test_max_holding_blocks_reentry_while_still_beyond_entry():
    out = _run([-2.5, -2.4, -2.3, -2.2, -2.1, 0.0], entry_z=2.0, exit_z=0.5, max_holding_bars=2)
    assert out["target_position"].tolist() == [1, 1, 0, 0, 0, 0]


def test_max_holding_reentry_after_reset():
    out = _run([-2.5, -1.8, -1.9, -1.0, -2.6, 0.0], entry_z=2.0, exit_z=0.5, max_holding_bars=2)
    # bar 2: max holding; bar 3 |z|<2 resets block; bar 4 re-enters.
    assert out["target_position"].tolist() == [1, 1, 0, 0, 1, 0]


def test_nan_zscore_holds_state_and_never_opens():
    out = _run([np.nan, -2.5, np.nan, -0.2, np.nan], entry_z=2.0, exit_z=0.5)
    assert out["target_position"].tolist() == [0, 1, 1, 0, 0]


def test_exit_at_zero_requires_crossing_mean():
    out = _run([-2.5, -0.1, 0.0], entry_z=2.0, exit_z=0.0)
    assert out["target_position"].tolist() == [1, 1, 0]


@pytest.mark.parametrize("seed", range(20))
def test_no_conflicting_positions_on_random_paths(seed):
    rng = np.random.default_rng(seed)
    z = np.cumsum(rng.standard_normal(500)) * 0.3
    out = _run(z, entry_z=1.5, exit_z=0.25, stop_loss_z=3.0, max_holding_bars=30)
    target = out["target_position"].to_numpy()
    assert set(np.unique(target)).issubset({-1, 0, 1})
    assert not np.any(np.abs(np.diff(target)) == 2)  # never a same-bar reversal
    changes = np.flatnonzero(np.diff(np.concatenate([[0], target])) != 0)
    assert all(out["event"].iloc[i] != "" for i in changes)  # every change is an explained event


def test_rule_validation():
    with pytest.raises(ValueError):
        SignalRules(entry_z=0)
    with pytest.raises(ValueError):
        SignalRules(entry_z=1.0, exit_z=1.0)
    with pytest.raises(ValueError):
        SignalRules(entry_z=2.0, stop_loss_z=1.5)
    with pytest.raises(ValueError):
        SignalRules(max_holding_bars=0)
