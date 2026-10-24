"""Static hedge-ratio estimation from training data.

Specification (default ``ols_log_price``)::

    log(P_A,t) = intercept + hedge_ratio * log(P_B,t) + residual_t

estimated by ordinary least squares on the *training window only*.

Interpretation and limitations
------------------------------
* ``hedge_ratio`` is an elasticity: a 1% move in B is associated with a
  ``hedge_ratio``% move in A in the long run. In dollar terms, a spread position
  that is long $1 of A is hedged by shorting about ``$hedge_ratio`` of B (see
  :mod:`statarb.portfolio`). It is *not* a share ratio.
* OLS treats B as exogenous; regressing B on A gives a different (not reciprocal)
  coefficient. The choice of dependent leg matters.
* With non-stationary regressors the usual OLS standard errors and t-statistics are
  invalid, so they are deliberately not reported. Under cointegration the estimate is
  super-consistent, but it can still be materially biased in finite samples.
* The relationship is assumed constant within each fitting window; it is re-estimated
  only on the documented walk-forward schedule.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm


@dataclass(frozen=True)
class HedgeRatioResult:
    intercept: float
    hedge_ratio: float
    r_squared: float
    residual_std: float
    n_obs: int
    fit_start: str
    fit_end: str
    method: str = "ols_log_price"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _validate_inputs(log_a: pd.Series, log_b: pd.Series, min_obs: int) -> None:
    if not log_a.index.equals(log_b.index):
        raise ValueError("log_a and log_b must share an identical index (align the pair first).")
    if len(log_a) < min_obs:
        raise ValueError(f"At least {min_obs} observations are required, got {len(log_a)}.")
    values = np.column_stack([log_a.to_numpy(float), log_b.to_numpy(float)])
    if not np.isfinite(values).all():
        raise ValueError("Log prices contain NaN or infinite values; clean the data first.")
    if np.std(values[:, 1]) < 1e-12:
        raise ValueError("Asset B's log price is constant in the fitting window; the hedge ratio is undefined.")


def estimate_hedge_ratio(
    log_a: pd.Series, log_b: pd.Series, method: str = "ols_log_price", min_obs: int = 30
) -> HedgeRatioResult:
    """Estimate intercept and hedge ratio of ``log_a`` on ``log_b`` by OLS.

    Callers are responsible for passing only training-window data.
    """
    if method != "ols_log_price":
        raise ValueError(f"Unsupported hedge-ratio method: {method!r}")
    _validate_inputs(log_a, log_b, min_obs)
    design = sm.add_constant(log_b.to_numpy(float), has_constant="add")
    model = sm.OLS(log_a.to_numpy(float), design).fit()
    intercept, beta = (float(v) for v in model.params)
    residuals = np.asarray(model.resid, dtype=float)
    return HedgeRatioResult(
        intercept=intercept,
        hedge_ratio=beta,
        r_squared=float(model.rsquared),
        residual_std=float(residuals.std(ddof=1)),
        n_obs=int(len(log_a)),
        fit_start=str(log_a.index[0].date()) if isinstance(log_a.index, pd.DatetimeIndex) else str(log_a.index[0]),
        fit_end=str(log_a.index[-1].date()) if isinstance(log_a.index, pd.DatetimeIndex) else str(log_a.index[-1]),
        method=method,
    )


__all__ = ["HedgeRatioResult", "estimate_hedge_ratio"]
