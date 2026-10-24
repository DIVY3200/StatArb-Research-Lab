"""Signal rules for the ML-only (B) and hybrid (C) strategy variants.

All variants emit a ``target_position`` decided at the close of bar ``t`` from information
available at ``t`` (the z-score and the model's predicted probability ``p_t``, itself computed
from features up to ``t``). They are executed by the same backtester as the traditional
strategy: same execution lag, sizing, costs, short-selling assumption and end-of-period
liquidation. Each evaluation period starts flat.

A. Traditional
    The z-score state machine of :mod:`statarb.signals` (entry, exit, optional stop and
    maximum holding period).

B. ML-only research signal
    * Entry (when flat): ``p_t >= threshold`` and ``|z_t| >= min_abs_z`` (the model's training
      domain). Direction is a bet on reversion: long spread if ``z_t < 0``, short if ``z_t > 0``.
      No entry while ``|z_t|`` is at or beyond an enabled stop-loss level.
    * Exit, in priority order: stop-loss (``state * z_t <= -stop_loss_z``); reversion into the
      exit band (``state * z_t >= -exit_z``); horizon expiry after ``H`` bars. At expiry the
      position is *renewed* (kept, counter reset, no trade) only if the latest prediction still
      supports it (``p_t >= threshold``, same direction, ``|z_t| >= min_abs_z``); otherwise it exits.
    * Missing ``p_t`` or ``z_t`` never opens a position.

C. Hybrid
    The traditional state machine with one extra entry condition: ``p_t >= threshold`` on the
    entry bar. Bars beyond the entry threshold without ML confirmation are skipped; the trade can
    still open on a later bar if ``|z|`` remains beyond the entry level and ``p`` rises. Exits,
    stop-loss, maximum holding and the re-entry block are exactly those of strategy A.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .signals import EVENT_ENTRY_LONG, EVENT_ENTRY_SHORT, EVENT_EXIT, EVENT_STOP, SignalRules, generate_signals

EVENT_HORIZON = "horizon_exit"
EVENT_RENEW = "renew"

VARIANT_LABELS = {
    "traditional": "A. Traditional (z-score)",
    "ml_only_logistic_regression": "B. ML-only (LR)",
    "ml_only_random_forest": "B. ML-only (RF)",
    "hybrid_logistic_regression": "C. Hybrid (LR)",
    "hybrid_random_forest": "C. Hybrid (RF)",
    "ml_only_prevalence_baseline": "B0. B rules, prevalence forecast (no model)",
}


@dataclass(frozen=True)
class MLSignalRules:
    threshold: float = 0.5
    min_abs_z: float = 1.0
    exit_z: float = 0.5
    horizon: int = 5
    stop_loss_z: float | None = None

    def __post_init__(self) -> None:
        if not 0.0 < self.threshold < 1.0:
            raise ValueError("threshold must be in (0, 1)")
        if self.horizon < 1:
            raise ValueError("horizon must be at least 1")
        if self.exit_z < 0 or self.min_abs_z < 0:
            raise ValueError("exit_z and min_abs_z must be non-negative")
        if self.stop_loss_z is not None and self.stop_loss_z <= self.min_abs_z:
            raise ValueError("stop_loss_z must exceed min_abs_z")


def probability_gate(prob: pd.Series, threshold: float) -> pd.Series:
    """True where a prediction exists and ``p_t >= threshold``."""
    values = prob.to_numpy(dtype=float)
    return pd.Series(np.isfinite(values) & (np.nan_to_num(values, nan=-1.0) >= threshold), index=prob.index)


def hybrid_signals(zscore: pd.Series, prob: pd.Series, rules: SignalRules, threshold: float) -> pd.DataFrame:
    """Strategy C: traditional rules, with entries allowed only when the model agrees."""
    if not prob.index.equals(zscore.index):
        raise ValueError("probabilities and z-scores must share an index")
    signals = generate_signals(zscore, rules, entry_gate=probability_gate(prob, threshold))
    signals["probability"] = prob.to_numpy()
    return signals


def ml_only_signals(zscore: pd.Series, prob: pd.Series, rules: MLSignalRules) -> pd.DataFrame:
    """Strategy B state machine (see module docstring)."""
    if not prob.index.equals(zscore.index):
        raise ValueError("probabilities and z-scores must share an index")
    z_values = zscore.to_numpy(dtype=float)
    p_values = prob.to_numpy(dtype=float)
    n = len(z_values)
    target = np.zeros(n, dtype=int)
    holding = np.zeros(n, dtype=int)
    events = [""] * n
    state, held = 0, 0
    stop, thr = rules.stop_loss_z, rules.threshold

    def supports(z: float, p: float, direction: int) -> bool:
        return (np.isfinite(z) and np.isfinite(p) and p >= thr and abs(z) >= rules.min_abs_z and z != 0
                and -int(np.sign(z)) == direction and (stop is None or abs(z) < stop))

    for t in range(n):
        z, p = z_values[t], p_values[t]
        if state == 0:
            if np.isfinite(z) and z != 0:
                direction = -int(np.sign(z))
                if supports(z, p, direction):
                    state, held = direction, 0
                    events[t] = EVENT_ENTRY_LONG if direction > 0 else EVENT_ENTRY_SHORT
        elif np.isnan(z):
            held += 1
        else:
            held += 1
            directional = state * z
            if stop is not None and directional <= -stop:
                state, events[t] = 0, EVENT_STOP
            elif directional >= -rules.exit_z:
                state, events[t] = 0, EVENT_EXIT
            elif held >= rules.horizon:
                if supports(z, p, state):
                    held, events[t] = 0, EVENT_RENEW
                else:
                    state, events[t] = 0, EVENT_HORIZON
            if state == 0:
                held = 0
        target[t] = state
        holding[t] = held if state != 0 else 0
    return pd.DataFrame({"zscore": z_values, "probability": p_values, "target_position": target, "event": events,
                         "holding_bars": holding}, index=zscore.index)


__all__ = [
    "EVENT_HORIZON",
    "EVENT_RENEW",
    "MLSignalRules",
    "VARIANT_LABELS",
    "hybrid_signals",
    "ml_only_signals",
    "probability_gate",
]
