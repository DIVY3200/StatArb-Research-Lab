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
from statsmodels.tsa.vector_ar.vecm import coint_johansen

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


_JOHANSEN_COLUMN = {0.10: 0, 0.05: 1, 0.01: 2}


@dataclass
class JohansenResult:
    """Johansen (1991) cointegration rank test for the two log prices (H0 for the trace test: rank 0)."""

    trace_statistic: float
    trace_critical_values: dict[str, float]
    max_eigen_statistic: float
    max_eigen_critical_values: dict[str, float]
    trace_statistic_r1: float
    trace_critical_value_r1: float
    significance: float
    reject_r0: bool
    rank: int
    hedge_ratio: float
    n_obs: int
    det_order: int
    k_ar_diff: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def johansen_test(log_a: pd.Series, log_b: pd.Series, significance: float = 0.05, det_order: int = 0,
                  k_ar_diff: int = 1, min_obs: int = 100) -> JohansenResult:
    """Johansen trace and maximum-eigenvalue tests on ``[log_a, log_b]``.

    Unlike Engle-Granger, the test is symmetric in the two series. ``rank`` follows the sequential trace
    procedure (0, 1 or 2; rank 2 would mean both series are stationary). ``hedge_ratio`` comes from the
    first cointegrating vector normalised on A, i.e. log A - h log B.

    Size caveat (measured, not assumed): ``det_order=0`` is statsmodels' unrestricted-constant case,
    whose critical values assume clearly trending prices. With the weak drift typical of daily equity
    prices it over-rejects: on independent random walks (756 bars, 0-0.05% drift per bar, 600 runs)
    the 5% trace test rejected 11-12% of the time, and the ``r <= 1`` step reported rank 2 for about a
    third of genuinely cointegrated synthetic pairs. Treat ``rank`` as a diagnostic, and prefer the
    ``eg_and_johansen`` gate (its size is bounded by Engle-Granger's) to Johansen alone.
    """
    column = next((c for level, c in _JOHANSEN_COLUMN.items() if abs(level - significance) < 1e-12), None)
    if column is None:
        raise ValueError("Johansen critical values exist only at the 10%, 5% and 1% levels")
    if not log_a.index.equals(log_b.index):
        raise ValueError("log_a and log_b must share an identical index (align the pair first).")
    if len(log_a) < min_obs:
        raise ValueError(f"Johansen test needs at least {min_obs} observations, got {len(log_a)}.")
    data = np.column_stack([log_a.to_numpy(float), log_b.to_numpy(float)])
    if not np.isfinite(data).all():
        raise ValueError("Log prices contain NaN or infinite values; clean the data first.")
    if np.std(data[:, 0]) < 1e-12 or np.std(data[:, 1]) < 1e-12:
        raise ValueError("A log-price series is constant in the window; the Johansen test is undefined.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = coint_johansen(data, det_order, k_ar_diff)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"Johansen test failed (singular system, e.g. perfectly collinear series): {exc}") from exc
    labels = ("90%", "95%", "99%")
    trace0, trace1 = float(np.real(res.lr1[0])), float(np.real(res.lr1[1]))
    crit0, crit1 = float(res.cvt[0, column]), float(res.cvt[1, column])
    rank = 0 if trace0 <= crit0 else (1 if trace1 <= crit1 else 2)
    vec = np.real(res.evec[:, 0])  # eig may return a complex dtype with zero imaginary parts
    hedge = float(-vec[1] / vec[0]) if abs(vec[0]) > 1e-12 else float("nan")
    return JohansenResult(
        trace_statistic=trace0, trace_critical_values=dict(zip(labels, map(float, res.cvt[0]))),
        max_eigen_statistic=float(np.real(res.lr2[0])),
        max_eigen_critical_values=dict(zip(labels, map(float, res.cvm[0]))),
        trace_statistic_r1=trace1, trace_critical_value_r1=crit1, significance=float(significance),
        reject_r0=bool(trace0 > crit0), rank=int(rank), hedge_ratio=hedge, n_obs=int(len(log_a)),
        det_order=int(det_order), k_ar_diff=int(k_ar_diff),
    )


GATES = ("eg", "eg_both", "johansen", "eg_and_johansen")


@dataclass
class CointegrationSuite:
    """Engle-Granger in both directions plus Johansen, on one fitting window."""

    eg_ab: CointegrationResult
    eg_ba: CointegrationResult
    johansen: JohansenResult | None
    johansen_error: str = ""

    def passes(self, gate: str = "eg") -> bool:
        """Whether the window passes a trading gate (see ``cointegration.gate`` in the config)."""
        johansen_ok = self.johansen is not None and self.johansen.reject_r0
        if gate == "eg":
            return self.eg_ab.reject_null
        if gate == "eg_both":
            return self.eg_ab.reject_null and self.eg_ba.reject_null
        if gate == "johansen":
            return johansen_ok
        if gate == "eg_and_johansen":
            return self.eg_ab.reject_null and johansen_ok
        raise ValueError(f"Unknown cointegration gate {gate!r}; expected one of {GATES}")

    def summary(self) -> dict[str, object]:
        j = self.johansen
        return {
            "eg_ab_p_value": self.eg_ab.p_value, "eg_ab_pass": self.eg_ab.reject_null,
            "eg_ba_p_value": self.eg_ba.p_value, "eg_ba_pass": self.eg_ba.reject_null,
            "johansen_trace": j.trace_statistic if j else float("nan"),
            "johansen_trace_crit": j.trace_critical_values[f"{(1 - j.significance):.0%}"] if j else float("nan"),
            "johansen_pass": bool(j.reject_r0) if j else False, "johansen_rank": j.rank if j else -1,
            "johansen_hedge_ratio": j.hedge_ratio if j else float("nan"),
            "eg_hedge_ratio": self.eg_ab.hedge_ratio,
            "johansen_error": self.johansen_error,
        }

    def to_dict(self) -> dict[str, object]:
        return {"eg_a_on_b": self.eg_ab.to_dict(), "eg_b_on_a": self.eg_ba.to_dict(),
                "johansen": self.johansen.to_dict() if self.johansen else None, "johansen_error": self.johansen_error,
                "summary": self.summary()}


def cointegration_suite(log_a: pd.Series, log_b: pd.Series, significance: float = 0.05, trend: str = "c",
                        autolag: str | None = "AIC", min_obs: int = 100, det_order: int = 0, k_ar_diff: int = 1,
                        name_a: str = "A", name_b: str = "B", eg_ab: CointegrationResult | None = None
                        ) -> CointegrationSuite:
    """Run EG (A on B), EG (B on A) and Johansen on the same window. Pass ``eg_ab`` to reuse a result."""
    if eg_ab is None:
        eg_ab = engle_granger_test(log_a, log_b, significance, trend, autolag, min_obs, name_a, name_b)
    eg_ba = engle_granger_test(log_b, log_a, significance, trend, autolag, min_obs, name_b, name_a,
                               check_integration=False)
    johansen, error = None, ""
    try:
        johansen = johansen_test(log_a, log_b, significance if significance in _JOHANSEN_COLUMN else 0.05,
                                 det_order, k_ar_diff, min_obs)
    except ValueError as exc:
        error = str(exc)
    return CointegrationSuite(eg_ab, eg_ba, johansen, error)


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
    "CointegrationSuite",
    "GATES",
    "JohansenResult",
    "cointegration_suite",
    "johansen_test",
    "LIMITATIONS",
    "adf_summary",
    "benjamini_hochberg",
    "engle_granger_test",
]
