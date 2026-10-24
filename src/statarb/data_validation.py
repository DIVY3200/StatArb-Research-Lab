"""Data cleaning, validation and two-asset date alignment.

Principles
----------
* Rows are only ever *removed*, never invented: there is no forward filling or
  interpolation of prices. Every removed row is reported with a reason.
* Exact duplicate rows for a date are collapsed to one row; conflicting duplicates
  (same date, different values) are all removed because neither can be trusted.
* The two assets are aligned on the intersection of their valid dates. Each
  return therefore spans the same interval for both legs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data_loader import AssetData

_MAX_LISTED = 25  # cap on dates listed per issue in human-readable reports


@dataclass
class ValidationReport:
    """Outcome of validating one asset's price history."""

    ticker: str
    price_column: str
    n_input_rows: int
    n_valid_rows: int = 0
    was_unsorted: bool = False
    invalid_date_rows: int = 0
    exact_duplicate_dates: list[str] = field(default_factory=list)
    conflicting_duplicate_dates: list[str] = field(default_factory=list)
    missing_price_dates: list[str] = field(default_factory=list)
    nonpositive_price_dates: list[str] = field(default_factory=list)
    ohlc_inconsistent_dates: list[str] = field(default_factory=list)
    large_gaps: list[dict[str, object]] = field(default_factory=list)
    extreme_return_dates: list[dict[str, object]] = field(default_factory=list)
    first_date: str | None = None
    last_date: str | None = None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.errors

    @property
    def n_dropped(self) -> int:
        return self.n_input_rows - self.n_valid_rows

    def summary(self) -> dict[str, object]:
        return {
            "ticker": self.ticker,
            "price_column": self.price_column,
            "input_rows": self.n_input_rows,
            "valid_rows": self.n_valid_rows,
            "dropped_rows": self.n_dropped,
            "unparseable_dates": self.invalid_date_rows,
            "exact_duplicates_collapsed": len(self.exact_duplicate_dates),
            "conflicting_duplicates_removed": len(self.conflicting_duplicate_dates),
            "missing_prices_removed": len(self.missing_price_dates),
            "non_positive_prices_removed": len(self.nonpositive_price_dates),
            "ohlc_inconsistencies_flagged": len(self.ohlc_inconsistent_dates),
            "large_gaps_flagged": len(self.large_gaps),
            "extreme_returns_flagged": len(self.extreme_return_dates),
            "first_date": self.first_date,
            "last_date": self.last_date,
            "status": "PASS" if self.passed else "FAIL",
        }


@dataclass
class AlignmentReport:
    """How two cleaned price series were intersected."""

    ticker_a: str
    ticker_b: str
    n_a: int
    n_b: int
    n_common: int
    dates_only_in_a: list[str] = field(default_factory=list)
    dates_only_in_b: list[str] = field(default_factory=list)
    first_common_date: str | None = None
    last_common_date: str | None = None
    warnings: list[str] = field(default_factory=list)
    special_sessions: list[dict[str, str]] = field(default_factory=list)
    special_sessions_dropped: int = 0

    def summary(self) -> dict[str, object]:
        return {
            "ticker_a": self.ticker_a,
            "ticker_b": self.ticker_b,
            "valid_rows_a": self.n_a,
            "valid_rows_b": self.n_b,
            "common_dates": self.n_common,
            "dropped_only_in_a": len(self.dates_only_in_a),
            "dropped_only_in_b": len(self.dates_only_in_b),
            "first_common_date": self.first_common_date,
            "last_common_date": self.last_common_date,
            "special_sessions_flagged": len(self.special_sessions),
            "special_sessions_dropped": self.special_sessions_dropped,
        }


@dataclass
class AlignedPair:
    """Synchronised prices for the pair: columns ``A`` and ``B`` on common dates."""

    prices: pd.DataFrame
    ticker_a: str
    ticker_b: str
    price_column_a: str
    price_column_b: str
    report: AlignmentReport
    validation_a: ValidationReport | None = None
    validation_b: ValidationReport | None = None
    metadata_a: dict[str, object] = field(default_factory=dict)
    metadata_b: dict[str, object] = field(default_factory=dict)

    @property
    def log_prices(self) -> pd.DataFrame:
        return np.log(self.prices)

    @property
    def is_synthetic(self) -> bool:
        return bool(self.metadata_a.get("is_synthetic") or self.metadata_b.get("is_synthetic"))


def select_price_column(frame: pd.DataFrame, prefer_adjusted: bool = True) -> str:
    """Pick ``Adj Close`` when preferred and usable, otherwise ``Close``."""
    has_adj = "Adj Close" in frame.columns and frame["Adj Close"].notna().any()
    has_close = "Close" in frame.columns and frame["Close"].notna().any()
    if prefer_adjusted and has_adj:
        return "Adj Close"
    if has_close:
        return "Close"
    if has_adj:
        return "Adj Close"
    raise ValueError("Neither 'Adj Close' nor 'Close' contains any values.")


def _fmt(dates) -> list[str]:
    return [str(pd.Timestamp(d).date()) for d in dates]


def validate_price_frame(
    frame: pd.DataFrame,
    ticker: str,
    prefer_adjusted: bool = True,
    min_observations: int = 252,
    max_gap_days: int = 7,
    extreme_return_threshold: float = 0.25,
) -> tuple[pd.DataFrame, ValidationReport]:
    """Clean one asset's price frame and report every issue found.

    Returns the cleaned frame (chronological, unique dates, positive finite prices in the
    selected price column) and a :class:`ValidationReport`.
    """
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("frame must be indexed by a pandas DatetimeIndex (see standardize_price_frame)")
    price_col = select_price_column(frame, prefer_adjusted)
    report = ValidationReport(ticker=ticker, price_column=price_col, n_input_rows=int(len(frame)))
    if price_col == "Close" and prefer_adjusted:
        report.warnings.append(
            "Adjusted close unavailable; using 'Close'. Returns may exclude dividends and, depending "
            "on the source, may not be split-adjusted."
        )

    data = frame.copy()

    # 1. Unparseable dates.
    bad_dates = data.index.isna()
    report.invalid_date_rows = int(bad_dates.sum())
    if report.invalid_date_rows:
        report.warnings.append(f"{report.invalid_date_rows} row(s) with unparseable dates removed.")
        data = data.loc[~bad_dates]

    # 2. Chronological order.
    if not data.index.is_monotonic_increasing:
        report.was_unsorted = True
        report.warnings.append("Rows were not in chronological order; sorted by date.")
        data = data.sort_index(kind="mergesort")

    # 3. Duplicate dates.
    dup_mask = data.index.duplicated(keep=False)
    if dup_mask.any():
        dup_rows = data.loc[dup_mask]
        exact, conflicting = [], []
        for date, group in dup_rows.groupby(level=0):
            if len(group.drop_duplicates()) == 1:
                exact.append(date)
            else:
                conflicting.append(date)
        report.exact_duplicate_dates = _fmt(exact)
        report.conflicting_duplicate_dates = _fmt(conflicting)
        if conflicting:
            data = data.loc[~data.index.isin(conflicting)]
            report.warnings.append(
                f"{len(conflicting)} date(s) had conflicting duplicate rows; all copies removed."
            )
        data = data.loc[~data.index.duplicated(keep="first")]
        if exact:
            report.warnings.append(f"{len(exact)} exact duplicate date(s) collapsed to a single row.")

    # 4. Missing and non-positive prices in the price column used for research.
    price = data[price_col]
    missing = price.isna() | ~np.isfinite(price.to_numpy(dtype=float, na_value=np.nan))
    report.missing_price_dates = _fmt(data.index[missing])
    nonpositive = (~missing) & (price <= 0)
    report.nonpositive_price_dates = _fmt(data.index[nonpositive])
    if missing.any():
        report.warnings.append(
            f"{int(missing.sum())} row(s) with missing '{price_col}' removed (not filled)."
        )
    if nonpositive.any():
        report.warnings.append(
            f"{int(nonpositive.sum())} row(s) with non-positive '{price_col}' removed."
        )
    data = data.loc[~(missing | nonpositive)]

    # 5. OHLC consistency (flagged only; OHLC fields are not used in calculations).
    if {"High", "Low", "Close"}.issubset(data.columns):
        hi, lo, cl = data["High"], data["Low"], data["Close"]
        tol = 1e-6 * cl.abs()
        inconsistent = (hi < lo - tol) | (cl > hi + tol) | (cl < lo - tol)
        inconsistent &= hi.notna() & lo.notna() & cl.notna()
        report.ohlc_inconsistent_dates = _fmt(data.index[inconsistent])
        if inconsistent.any():
            report.warnings.append(
                f"{int(inconsistent.sum())} row(s) with inconsistent High/Low/Close values (flagged, kept)."
            )
    if "Volume" in data.columns and (data["Volume"] < 0).any():
        report.warnings.append("Negative volume values found (flagged, kept).")

    # 6. Calendar gaps and suspicious jumps.
    if len(data) > 1:
        gaps = data.index.to_series().diff().dt.days
        for end_date, days in gaps[gaps > max_gap_days].items():
            position = data.index.get_loc(end_date)
            report.large_gaps.append(
                {"from": str(data.index[position - 1].date()), "to": str(end_date.date()), "calendar_days": int(days)}
            )
        if report.large_gaps:
            report.warnings.append(
                f"{len(report.large_gaps)} gap(s) longer than {max_gap_days} calendar days; returns across "
                "these gaps span multiple days."
            )
        log_ret = np.log(data[price_col]).diff()
        jumps = log_ret[log_ret.abs() > extreme_return_threshold]
        report.extreme_return_dates = [
            {"date": str(d.date()), "log_return": round(float(r), 4)} for d, r in jumps.items()
        ]
        if len(jumps):
            report.warnings.append(
                f"{len(jumps)} daily move(s) larger than {extreme_return_threshold:.0%} in log terms; check "
                "for unadjusted splits or data errors."
            )

    report.n_valid_rows = int(len(data))
    if len(data):
        report.first_date = str(data.index[0].date())
        report.last_date = str(data.index[-1].date())
    if report.n_valid_rows < min_observations:
        report.errors.append(
            f"Insufficient history for {ticker}: {report.n_valid_rows} valid observations, at least "
            f"{min_observations} required. Extend the date range or choose another instrument."
        )
    return data, report


def align_pair(
    clean_a: pd.DataFrame,
    clean_b: pd.DataFrame,
    price_col_a: str,
    price_col_b: str,
    ticker_a: str = "A",
    ticker_b: str = "B",
    min_observations: int = 252,
) -> tuple[pd.DataFrame, AlignmentReport]:
    """Intersect two cleaned price series on common dates (inner join, no filling)."""
    series_a = clean_a[price_col_a]
    series_b = clean_b[price_col_b]
    common = series_a.index.intersection(series_b.index).sort_values()
    only_a = series_a.index.difference(series_b.index)
    only_b = series_b.index.difference(series_a.index)
    prices = pd.DataFrame({"A": series_a.reindex(common), "B": series_b.reindex(common)})
    prices.index.name = "Date"

    report = AlignmentReport(
        ticker_a=ticker_a,
        ticker_b=ticker_b,
        n_a=int(len(series_a)),
        n_b=int(len(series_b)),
        n_common=int(len(common)),
        dates_only_in_a=_fmt(only_a),
        dates_only_in_b=_fmt(only_b),
        first_common_date=str(common[0].date()) if len(common) else None,
        last_common_date=str(common[-1].date()) if len(common) else None,
    )
    for ticker, n_only, n_total in ((ticker_a, len(only_a), len(series_a)), (ticker_b, len(only_b), len(series_b))):
        if n_total and n_only / n_total > 0.05:
            report.warnings.append(
                f"{n_only} of {n_total} {ticker} dates ({n_only / n_total:.1%}) have no matching {ticker_b if ticker == ticker_a else ticker_a} "
                "observation and were excluded. Check for different trading calendars or data gaps."
            )
    if _exchange_suffix(ticker_a) != _exchange_suffix(ticker_b):
        report.warnings.append(
            "The assets appear to trade on different exchanges. Their closing prices are not "
            "synchronous, which can create spurious lead-lag effects and unrealistic fills."
        )
    if len(common) < min_observations:
        raise ValueError(
            f"Only {len(common)} overlapping observations between {ticker_a} and {ticker_b}; at least "
            f"{min_observations} are required. Choose assets with a longer common history."
        )
    return prices, report


def _exchange_suffix(ticker: str) -> str:
    return ticker.rsplit(".", 1)[1].upper() if "." in ticker else ""


def detect_special_sessions(
    dates: pd.DatetimeIndex,
    frames: tuple[pd.DataFrame, pd.DataFrame],
    listed_dates: tuple[str, ...] = (),
    volume_ratio: float | None = None,
    volume_window: int = 60,
) -> list[dict[str, str]]:
    """Flag non-regular sessions among ``dates`` (for example NSE Muhurat trading).

    A bar is flagged when it is dated on a weekend, appears in ``listed_dates``, or (if
    ``volume_ratio`` is set) either leg's volume is below ``volume_ratio`` times that leg's median
    volume over the *previous* ``volume_window`` bars. The volume rule uses only data up to the bar.
    """
    reasons: dict[pd.Timestamp, list[str]] = {}
    for date in dates[dates.dayofweek >= 5]:
        reasons.setdefault(date, []).append("weekend-dated bar")
    listed = {pd.Timestamp(d) for d in listed_dates}
    for date in dates[dates.isin(list(listed))]:
        reasons.setdefault(date, []).append("listed special session")
    if volume_ratio is not None:
        for leg, frame in zip(("A", "B"), frames):
            if "Volume" not in frame.columns:
                continue
            volume = frame["Volume"].astype(float)
            median = volume.shift(1).rolling(volume_window, min_periods=20).median()
            low = volume < volume_ratio * median
            for date in low.index[low.fillna(False).to_numpy(dtype=bool)]:
                if date in dates:
                    reasons.setdefault(date, []).append(f"leg {leg} volume < {volume_ratio:.0%} of trailing median")
    return [{"date": str(d.date()), "reasons": "; ".join(r)} for d, r in sorted(reasons.items())]


def prepare_pair(
    asset_a: AssetData,
    asset_b: AssetData,
    prefer_adjusted: bool = True,
    min_observations: int = 252,
    max_gap_days: int = 7,
    extreme_return_threshold: float = 0.25,
    special_sessions: str = "keep",
    special_session_dates: tuple[str, ...] = (),
    special_session_volume_ratio: float | None = None,
    special_session_dates_by_exchange: tuple[tuple[str, tuple[str, ...]], ...] = (),
) -> AlignedPair:
    """Validate both assets and align them; raises ``ValueError`` with actionable messages on failure.

    ``special_sessions="keep"`` (default, baseline behaviour) only reports flagged sessions;
    ``"drop"`` removes them from the aligned prices before any statistic, signal or fill is computed.
    """
    if special_sessions not in {"keep", "drop"}:
        raise ValueError("special_sessions must be keep or drop")
    if asset_a.ticker == asset_b.ticker:
        raise ValueError("Asset A and asset B must be different instruments.")
    clean_a, report_a = validate_price_frame(
        asset_a.frame, asset_a.ticker, prefer_adjusted, min_observations, max_gap_days, extreme_return_threshold
    )
    clean_b, report_b = validate_price_frame(
        asset_b.frame, asset_b.ticker, prefer_adjusted, min_observations, max_gap_days, extreme_return_threshold
    )
    errors = report_a.errors + report_b.errors
    if errors:
        raise ValueError(" ".join(errors))
    if report_a.price_column != report_b.price_column:
        report_a.warnings.append(
            f"Price conventions differ: {asset_a.ticker} uses '{report_a.price_column}', "
            f"{asset_b.ticker} uses '{report_b.price_column}'."
        )
    prices, alignment = align_pair(
        clean_a,
        clean_b,
        report_a.price_column,
        report_b.price_column,
        asset_a.ticker,
        asset_b.ticker,
        min_observations,
    )
    exchanges = {_exchange_suffix(asset_a.ticker), _exchange_suffix(asset_b.ticker)}
    listed = tuple(special_session_dates) + tuple(
        d for suffix, dates in special_session_dates_by_exchange if suffix in exchanges for d in dates)
    flagged = detect_special_sessions(prices.index, (clean_a, clean_b), listed, special_session_volume_ratio)
    alignment.special_sessions = flagged
    if flagged and special_sessions == "keep":
        alignment.warnings.append(
            f"{len(flagged)} special/non-regular session(s) kept as normal bars (e.g. {flagged[0]['date']}); "
            "set data.special_sessions: drop to exclude them."
        )
    elif flagged:
        drop = pd.DatetimeIndex([pd.Timestamp(item["date"]) for item in flagged])
        prices = prices.loc[~prices.index.isin(drop)]
        alignment.special_sessions_dropped = len(flagged)
        alignment.n_common = int(len(prices))
        alignment.warnings.append(f"{len(flagged)} special/non-regular session(s) dropped before analysis.")
        if len(prices) < min_observations:
            raise ValueError(f"Only {len(prices)} observations remain after dropping special sessions.")
    return AlignedPair(
        prices=prices,
        ticker_a=asset_a.ticker,
        ticker_b=asset_b.ticker,
        price_column_a=report_a.price_column,
        price_column_b=report_b.price_column,
        report=alignment,
        validation_a=report_a,
        validation_b=report_b,
        metadata_a=asset_a.metadata(),
        metadata_b=asset_b.metadata(),
    )


def prepare_pair_from_config(asset_a: AssetData, asset_b: AssetData, data_config) -> AlignedPair:
    """:func:`prepare_pair` with every option taken from a ``DataConfig``."""
    d = data_config
    return prepare_pair(asset_a, asset_b, d.prefer_adjusted, d.min_observations, d.max_gap_days,
                        d.extreme_return_threshold, d.special_sessions, d.special_session_dates,
                        d.special_session_volume_ratio, d.special_session_dates_by_exchange)


def truncate_list(items: list, limit: int = _MAX_LISTED) -> list:
    """Helper for displaying long issue lists."""
    return items if len(items) <= limit else items[:limit] + [f"... ({len(items) - limit} more)"]


__all__ = [
    "AlignedPair",
    "AlignmentReport",
    "ValidationReport",
    "align_pair",
    "detect_special_sessions",
    "prepare_pair",
    "prepare_pair_from_config",
    "select_price_column",
    "truncate_list",
    "validate_price_frame",
]
