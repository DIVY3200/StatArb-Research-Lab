"""Finite-state mean-reversion signal generation.

Sign convention
---------------
``spread = log(P_A) - intercept - hedge_ratio * log(P_B)``

* ``z <= -entry_z``  -> A is cheap relative to B -> **LONG_SPREAD** (+1): long A, short B.
* ``z >= +entry_z``  -> A is expensive relative to B -> **SHORT_SPREAD** (-1): short A, long B.

State machine (evaluated once per bar on the z-score known at that bar's close)
-------------------------------------------------------------------------------
FLAT
    enter LONG_SPREAD if ``z <= -entry_z``; enter SHORT_SPREAD if ``z >= entry_z``.
    Entries are refused while ``|z| >= stop_loss_z`` (the trade would be stopped at once)
    and while the re-entry block is active (see below).
LONG_SPREAD / SHORT_SPREAD, checked in this order:
    1. stop loss: ``state * z <= -stop_loss_z`` (spread kept diverging) -> FLAT, block re-entry;
    2. exit: ``state * z >= -exit_z`` (spread reverted to within the exit band, or crossed it) -> FLAT;
    3. max holding: ``holding_bars >= max_holding_bars`` -> FLAT, block re-entry.

Rules for new signals while a position is open:

* A same-direction signal is ignored (no pyramiding, no duplicate entries).
* An opposite-direction signal always satisfies the exit rule first, so the position
  goes FLAT; a new position can open on a *later* bar only. There are no same-bar
  reversals, so conflicting positions cannot occur.
* After a stop-loss or max-holding exit, no new entry is allowed until ``|z|`` has
  returned strictly inside the entry band, preventing immediate re-entry into the same
  failed divergence.
* A missing z-score keeps the current state (positions are not opened on missing data).
* An optional ``entry_gate`` (used by the hybrid ML strategy) must also be True on the bar
  for an entry to happen. It never affects exits, stops or the re-entry block.

The output ``target_position`` is the *desired* state at each bar's close. The
backtester executes it ``execution_lag`` bars later; signals never trade on the bar
from which they were computed.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import pandas as pd


class PositionState(IntEnum):
    SHORT_SPREAD = -1
    FLAT = 0
    LONG_SPREAD = 1


EVENT_ENTRY_LONG = "entry_long"
EVENT_ENTRY_SHORT = "entry_short"
EVENT_EXIT = "exit"
EVENT_STOP = "stop_loss"
EVENT_MAX_HOLD = "max_holding"


@dataclass(frozen=True)
class SignalRules:
    entry_z: float = 2.0
    exit_z: float = 0.5
    stop_loss_z: float | None = None
    max_holding_bars: int | None = None

    def __post_init__(self) -> None:
        if self.entry_z <= 0:
            raise ValueError("entry_z must be positive")
        if not 0.0 <= self.exit_z < self.entry_z:
            raise ValueError("exit_z must satisfy 0 <= exit_z < entry_z")
        if self.stop_loss_z is not None and self.stop_loss_z <= self.entry_z:
            raise ValueError("stop_loss_z must be greater than entry_z")
        if self.max_holding_bars is not None and self.max_holding_bars < 1:
            raise ValueError("max_holding_bars must be at least 1")


def generate_signals(zscore: pd.Series, rules: SignalRules, entry_gate: pd.Series | None = None) -> pd.DataFrame:
    """Run the state machine over a z-score series.

    Returns a frame indexed like ``zscore`` with columns ``zscore``, ``target_position``
    (-1/0/+1), ``event`` (empty string when nothing happens) and ``holding_bars``
    (bars since the entry signal while a position is desired, else 0).
    """
    z_values = zscore.to_numpy(dtype=float)
    n = len(z_values)
    if entry_gate is None:
        gate = np.ones(n, dtype=bool)
    else:
        if not entry_gate.index.equals(zscore.index):
            raise ValueError("entry_gate must share the z-score index")
        gate = entry_gate.fillna(False).to_numpy(dtype=bool)
    target = np.zeros(n, dtype=int)
    holding = np.zeros(n, dtype=int)
    events = [""] * n

    state = PositionState.FLAT
    bars_held = 0
    reentry_blocked = False
    entry, exit_, stop, max_hold = rules.entry_z, rules.exit_z, rules.stop_loss_z, rules.max_holding_bars

    for t in range(n):
        z = z_values[t]
        if np.isnan(z):
            if state != PositionState.FLAT:
                bars_held += 1
        elif state == PositionState.FLAT:
            if reentry_blocked and abs(z) < entry:
                reentry_blocked = False
            stop_ok = stop is None or abs(z) < stop
            if not reentry_blocked and stop_ok and gate[t]:
                if z <= -entry:
                    state, bars_held, events[t] = PositionState.LONG_SPREAD, 0, EVENT_ENTRY_LONG
                elif z >= entry:
                    state, bars_held, events[t] = PositionState.SHORT_SPREAD, 0, EVENT_ENTRY_SHORT
        else:
            bars_held += 1
            directional_z = int(state) * z
            if stop is not None and directional_z <= -stop:
                state, events[t], reentry_blocked = PositionState.FLAT, EVENT_STOP, True
            elif directional_z >= -exit_:
                state, events[t] = PositionState.FLAT, EVENT_EXIT
            elif max_hold is not None and bars_held >= max_hold:
                state, events[t], reentry_blocked = PositionState.FLAT, EVENT_MAX_HOLD, True
            if state == PositionState.FLAT:
                bars_held = 0
        target[t] = int(state)
        holding[t] = bars_held if state != PositionState.FLAT else 0

    return pd.DataFrame(
        {"zscore": z_values, "target_position": target, "event": events, "holding_bars": holding},
        index=zscore.index,
    )


__all__ = [
    "EVENT_ENTRY_LONG",
    "EVENT_ENTRY_SHORT",
    "EVENT_EXIT",
    "EVENT_MAX_HOLD",
    "EVENT_STOP",
    "PositionState",
    "SignalRules",
    "generate_signals",
]
