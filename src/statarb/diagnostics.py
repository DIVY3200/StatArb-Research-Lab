"""Exploratory statistics for a price pair.

These diagnostics describe the sample; none of them establishes a tradable relation.
In particular, a high correlation of prices or returns is *not* evidence of
cointegration: two independent random walks routinely show large price correlations
(spurious correlation), and highly correlated returns say nothing about whether the
*level* spread is stationary.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data_validation import AlignedPair


def simple_returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Close-to-close simple returns; the first row is NaN (no prior observation)."""
    return prices / prices.shift(1) - 1.0


def log_returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    return np.log(prices).diff()


def normalized_prices(prices: pd.DataFrame, base: float = 100.0) -> pd.DataFrame:
    return base * prices / prices.iloc[0]


def rolling_volatility(returns: pd.DataFrame, window: int = 63, periods_per_year: int = 252) -> pd.DataFrame:
    return returns.rolling(window, min_periods=window).std(ddof=1) * np.sqrt(periods_per_year)


def rolling_correlation(returns: pd.DataFrame, window: int = 63) -> pd.Series:
    corr = returns["A"].rolling(window, min_periods=window).corr(returns["B"])
    corr.name = "rolling_correlation"
    return corr


def summary_statistics(prices: pd.DataFrame, names: dict[str, str], periods_per_year: int = 252) -> pd.DataFrame:
    """Per-asset descriptive statistics on the supplied (aligned) sample."""
    rets = simple_returns(prices).iloc[1:]
    rows = {}
    for col in prices.columns:
        series, r = prices[col], rets[col]
        years = len(r) / periods_per_year
        growth = float(series.iloc[-1] / series.iloc[0])
        rows[names.get(col, col)] = {
            "first_date": str(series.index[0].date()),
            "last_date": str(series.index[-1].date()),
            "observations": int(len(series)),
            "first_price": float(series.iloc[0]),
            "last_price": float(series.iloc[-1]),
            "total_return": growth - 1.0,
            "annualized_return": growth ** (1.0 / years) - 1.0 if years > 0 else np.nan,
            "annualized_volatility": float(r.std(ddof=1) * np.sqrt(periods_per_year)),
            "mean_daily_return": float(r.mean()),
            "min_daily_return": float(r.min()),
            "max_daily_return": float(r.max()),
            "skewness": float(r.skew()),
            "excess_kurtosis": float(r.kurt()),
        }
    return pd.DataFrame(rows)


def correlation_summary(prices: pd.DataFrame) -> dict[str, float]:
    rets = simple_returns(prices).dropna()
    logp = np.log(prices)
    return {
        "price_level_correlation": float(prices["A"].corr(prices["B"])),
        "log_price_correlation": float(logp["A"].corr(logp["B"])),
        "daily_return_correlation": float(rets["A"].corr(rets["B"])),
        "daily_return_spearman": float(rets["A"].corr(rets["B"], method="spearman")),
    }


def missing_data_summary(pair: AlignedPair) -> pd.DataFrame:
    """Rows removed during validation and alignment, per asset."""
    rows = []
    for ticker, report, dropped_in_alignment in (
        (pair.ticker_a, pair.validation_a, len(pair.report.dates_only_in_a)),
        (pair.ticker_b, pair.validation_b, len(pair.report.dates_only_in_b)),
    ):
        summary = report.summary() if report is not None else {}
        rows.append(
            {
                "ticker": ticker,
                "input_rows": summary.get("input_rows"),
                "removed_in_validation": summary.get("dropped_rows"),
                "missing_prices": summary.get("missing_prices_removed"),
                "non_positive_prices": summary.get("non_positive_prices_removed"),
                "conflicting_duplicates": summary.get("conflicting_duplicates_removed"),
                "exact_duplicates_collapsed": summary.get("exact_duplicates_collapsed"),
                "unmatched_dates_removed_in_alignment": dropped_in_alignment,
                "rows_used": pair.report.n_common,
            }
        )
    return pd.DataFrame(rows).set_index("ticker")


__all__ = [
    "correlation_summary",
    "log_returns",
    "missing_data_summary",
    "normalized_prices",
    "rolling_correlation",
    "rolling_volatility",
    "simple_returns",
    "summary_statistics",
]
