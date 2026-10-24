"""Engle-Granger two-step cointegration testing and multiple-testing correction.

Procedure (training window only)
--------------------------------
1. Regress ``log P_A`` on a constant and ``log P_B`` (step 1, the cointegrating regression).
2. Apply an augmented Dickey-Fuller regression to the residuals (step 2).
3. Compare the residual-ADF statistic with **Engle-Granger / MacKinnon (2010)** critical
   values for a two-variable system. Ordinary ADF critical values are *not* valid here
   because the residuals come from an estimated regression; ``statsmodels.tsa.stattools.coint``
   uses the correct residual-based distribution.

H0: no cointegration (the residual has a unit root). Rejecting H0 at the chosen
significance level is *statistical evidence* of a stationary long-run relation in the
sample. It is not proof that the relation holds in future, and the p-value is not the
probability that a trading strategy will be profitable.
"""

from __future__ import annotations

import inspect
import warnings
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint

from .hedge_ratio import estimate_hedge_ratio

ASSUMPTIONS: tuple[str, ...] = (
    "Each log-price series is integrated of order one, I(1), over the test window.",
    "The cointegrating relation (intercept and hedge ratio) is constant over the window (no structural breaks).",
    "The relation is linear in log prices with asset A as the dependent variable; reversing the order changes the test.",
    "Residual autocorrelation is absorbed by the ADF lag terms (lag order chosen by the configured criterion).",
    "P-values are asymptotic (MacKinnon surface) and approximate in finite samples.",
)

LIMITATIONS: tuple[str, ...] = (
    "Results depend on the sample period and specification; a different window can reverse the decision.",
    "Structural breaks (mergers, regulation, business-model changes) can invalidate a historical relation.",
    "Testing many pairs or windows inflates false discoveries; correct for multiple testing.",
    "A cointegrated pair is not necessarily profitable after transaction, borrow and financing costs.",
    "Statistical significance is not economic significance: the spread may revert too slowly or too little.",
)


@dataclass
class CointegrationResult:
    """Engle-Granger test output for one ordered pair on one sample window."""

    dependent: str
    independent: str
    test_statistic: float
    p_value: float
    critical_values: dict[str, float]
    significance: float
    reject_null: bool
    n_obs: int
    sample_start: str
    sample_end: str
    intercept: float
    hedge_ratio: float
    trend: str
    autolag: str | None
    adf_levels: dict[str, dict[str, float]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    test_name: str = "Engle-Granger two-step (residual ADF, MacKinnon critical values)"

    @property
    def decision(self) -> str:
        if self.reject_null:
            return (
                f"Reject H0 of no cointegration at the {self.significance:.0%} level "
                f"(p = {self.p_value:.4f})."
            )
        return (
            f"Fail to reject H0 of no cointegration at the {self.significance:.0%} level "
            f"(p = {self.p_value:.4f})."
        )

    def interpretation(self) -> str:
        period = f"training window {self.sample_start} to {self.sample_end}"
        if self.reject_null:
            return (
                f"In the {period}, the residual spread of {self.dependent} on {self.independent} shows "
                "statistically significant evidence of mean reversion. This is in-sample evidence only: it "
                "does not establish that the relation will persist or that trading it is profitable after costs."
            )
        return (
            f"In the {period}, there is insufficient evidence that {self.dependent} and {self.independent} are "
            "cointegrated. The pair should not be described as a cointegrated opportunity; any backtest of it "
            "is exploratory."
        )

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["decision"] = self.decision
        payload["interpretation"] = self.interpretation()
        payload["assumptions"] = list(ASSUMPTIONS)
        payload["limitations"] = list(LIMITATIONS)
        return payload


# statsmodels >= 0.15 can return a result object; request the stable tuple form when supported.
_ADF_TUPLE_KWARGS = {"result_object": False} if "result_object" in inspect.signature(adfuller).parameters else {}


def adf_summary(series: pd.Series, autolag: str | None = "AIC", regression: str = "c") -> dict[str, float]:
    """ADF unit-root test on one series (H0: unit root)."""
    values = series.to_numpy(float)
    stat, pvalue, used_lag, nobs, crit, _ = adfuller(
        values, regression=regression, autolag=autolag, **_ADF_TUPLE_KWARGS
    )
    return {
        "statistic": float(stat),
        "p_value": float(pvalue),
        "used_lag": int(used_lag),
        "n_obs": int(nobs),
        "crit_1%": float(crit["1%"]),
        "crit_5%": float(crit["5%"]),
        "crit_10%": float(crit["10%"]),
    }


def engle_granger_test(
    log_a: pd.Series,
    log_b: pd.Series,
    significance: float = 0.05,
    trend: str = "c",
    autolag: str | None = "AIC",
    min_obs: int = 100,
    name_a: str = "A",
    name_b: str = "B",
    check_integration: bool = True,
) -> CointegrationResult:
    """Run the Engle-Granger test of ``log_a`` on ``log_b`` (pass training-window data only).

    Raises ``ValueError`` for misaligned, too-short, non-finite, constant or perfectly
    collinear inputs, for which the test is undefined.
    """
    if not 0.0 < significance < 0.5:
        raise ValueError("significance must be in (0, 0.5)")
    if not log_a.index.equals(log_b.index):
        raise ValueError("log_a and log_b must share an identical index (align the pair first).")
    if len(log_a) < min_obs:
        raise ValueError(
            f"Engle-Granger test needs at least {min_obs} observations, got {len(log_a)}. "
            "Use a longer training window."
        )
    hedge = estimate_hedge_ratio(log_a, log_b, min_obs=min(min_obs, 30))
    if hedge.residual_std < 1e-10:
        raise ValueError("Residuals are numerically zero (perfectly collinear series); the test is undefined.")
    if np.std(log_a.to_numpy(float)) < 1e-12:
        raise ValueError("Asset A's log price is constant in the window; the test is undefined.")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, pvalue, crit = coint(
            log_a.to_numpy(float), log_b.to_numpy(float), trend=trend, method="aeg", autolag=autolag
        )
    critical_values = {"1%": float(crit[0]), "5%": float(crit[1]), "10%": float(crit[2])}
    result = CointegrationResult(
        dependent=name_a,
        independent=name_b,
        test_statistic=float(stat),
        p_value=float(pvalue),
        critical_values=critical_values,
        significance=float(significance),
        reject_null=bool(pvalue < significance),
        n_obs=int(len(log_a)),
        sample_start=hedge.fit_start,
        sample_end=hedge.fit_end,
        intercept=hedge.intercept,
        hedge_ratio=hedge.hedge_ratio,
        trend=trend,
        autolag=autolag,
    )

    if check_integration:
        for label, series in ((name_a, log_a), (name_b, log_b)):
            level = adf_summary(series, autolag=autolag)
            diff = adf_summary(series.diff().dropna(), autolag=autolag)
            result.adf_levels[label] = {"level_p_value": level["p_value"], "diff_p_value": diff["p_value"]}
            if level["p_value"] < significance:
                result.warnings.append(
                    f"{label}'s log price looks stationary in levels (ADF p = {level['p_value']:.3f}); the "
                    "Engle-Granger assumption that both series are I(1) may not hold."
                )
            if diff["p_value"] >= significance:
                result.warnings.append(
                    f"{label}'s log-price differences do not look stationary (ADF p = {diff['p_value']:.3f}); "
                    "the series may not be I(1)."
                )
    if hedge.hedge_ratio <= 0:
        result.warnings.append(
            "The estimated hedge ratio is non-positive; both legs would be traded in the same direction, "
            "which is unusual for a pairs trade."
        )
    if not result.reject_null:
        result.warnings.append(
            "The pair does not pass the training-period cointegration test at the chosen significance level."
        )
    return result


def benjamini_hochberg(p_values, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg false-discovery-rate procedure.

    Returns ``(rejected, adjusted_p_values)`` in the original order. This controls the
    expected share of false discoveries among the *tested* hypotheses under independence
    or positive dependence; it does not correct for pairs or parameters that were looked
    at informally and never formally tested, nor for survivorship bias.
    """
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1:
        raise ValueError("p_values must be one-dimensional")
    if np.any((p < 0) | (p > 1)) or not np.isfinite(p).all():
        raise ValueError("p_values must be finite and within [0, 1]")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    m = len(p)
    if m == 0:
        return np.array([], dtype=bool), np.array([], dtype=float)
    order = np.argsort(p)
    ranked = p[order] * m / np.arange(1, m + 1)
    adjusted_sorted = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted_sorted = np.clip(adjusted_sorted, 0.0, 1.0)
    adjusted = np.empty(m)
    adjusted[order] = adjusted_sorted
    return adjusted <= alpha, adjusted


__all__ = [
    "ASSUMPTIONS",
    "CointegrationResult",
    "LIMITATIONS",
    "adf_summary",
    "benjamini_hochberg",
    "engle_granger_test",
]
