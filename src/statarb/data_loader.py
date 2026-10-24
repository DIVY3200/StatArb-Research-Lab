"""Historical price loading: Yahoo Finance (optional), CSV files, and labelled synthetic data.

Price conventions
-----------------
* ``yfinance`` is called with ``auto_adjust=False`` so that both the raw ``Close``
  (split-adjusted by Yahoo) and ``Adj Close`` (split- and dividend-adjusted) are
  returned. The platform uses one of these columns *as delivered* and never
  applies further adjustments, so prices are not double-adjusted.
* CSV files are used exactly as supplied. If a CSV contains ``Adj Close`` it is
  assumed to already include split and dividend adjustments.
* Synthetic data has no corporate actions, so ``Adj Close == Close``. Synthetic
  frames are flagged ``is_synthetic=True`` and their tickers start with
  ``SYNTH_``; they exist for offline demonstrations and tests only and must never
  be presented as market history.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO

import numpy as np
import pandas as pd

STANDARD_COLUMNS: tuple[str, ...] = ("Open", "High", "Low", "Close", "Adj Close", "Volume")
SYNTHETIC_PREFIX = "SYNTH_"

_COLUMN_ALIASES: dict[str, str] = {
    "date": "Date",
    "datetime": "Date",
    "timestamp": "Date",
    "time": "Date",
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "adj close": "Adj Close",
    "adjclose": "Adj Close",
    "adjusted close": "Adj Close",
    "adj_close": "Adj Close",
    "adjusted_close": "Adj Close",
    "volume": "Volume",
    "ticker": "Ticker",
    "symbol": "Ticker",
}


class DataLoadError(RuntimeError):
    """Raised when price data cannot be obtained or parsed; messages are user-actionable."""


@dataclass
class AssetData:
    """Price history for one instrument plus provenance metadata."""

    ticker: str
    frame: pd.DataFrame
    source: str
    retrieved_at: str
    sha256: str | None = None
    is_synthetic: bool = False
    notes: list[str] = field(default_factory=list)

    def metadata(self) -> dict[str, object]:
        index = self.frame.index
        return {
            "ticker": self.ticker,
            "source": self.source,
            "retrieved_at": self.retrieved_at,
            "sha256": self.sha256,
            "is_synthetic": self.is_synthetic,
            "n_rows": int(len(self.frame)),
            "first_date": str(index.min().date()) if len(index) else None,
            "last_date": str(index.max().date()) if len(index) else None,
            "columns": list(self.frame.columns),
            "notes": list(self.notes),
        }


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def standardize_price_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalise column names, parse dates into a tz-naive ``DatetimeIndex`` and coerce numbers.

    Unparseable dates become ``NaT`` and unparseable numbers become ``NaN``; nothing is
    dropped or filled here so that :mod:`statarb.data_validation` can report it explicitly.
    """
    frame = raw.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = [col[0] for col in frame.columns]
    if "Date" not in frame.columns and frame.index.name is not None:
        frame = frame.reset_index()

    rename: dict[str, str] = {}
    for col in frame.columns:
        key = re.sub(r"\s+", " ", str(col).strip().lower())
        if key in _COLUMN_ALIASES:
            rename[col] = _COLUMN_ALIASES[key]
    frame = frame.rename(columns=rename)
    if frame.columns.duplicated().any():
        dupes = sorted(set(frame.columns[frame.columns.duplicated()]))
        raise DataLoadError(f"Ambiguous duplicate columns after normalisation: {dupes}")
    if "Date" not in frame.columns:
        raise DataLoadError("Price data must include a 'Date' column (or a date index).")
    if "Close" not in frame.columns and "Adj Close" not in frame.columns:
        raise DataLoadError("Price data must include a 'Close' or 'Adj Close' column.")

    dates = pd.to_datetime(frame["Date"], errors="coerce", utc=False)
    if getattr(dates.dt, "tz", None) is not None:
        dates = dates.dt.tz_localize(None)
    frame["Date"] = dates.dt.normalize()

    keep = ["Date"] + [c for c in STANDARD_COLUMNS if c in frame.columns]
    if "Ticker" in frame.columns:
        keep.append("Ticker")
    frame = frame[keep].copy()
    for col in STANDARD_COLUMNS:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce").astype(float)
    frame = frame.set_index("Date")
    frame.index.name = "Date"
    return frame


def load_csv(
    source: str | Path | bytes | BinaryIO,
    ticker: str | None = None,
    source_name: str | None = None,
) -> dict[str, AssetData]:
    """Load one CSV file containing a single asset, or several assets in long format.

    A long-format file has a ``Ticker`` (or ``Symbol``) column. Otherwise the asset is
    named ``ticker`` or, failing that, after the file stem. Returns ``{ticker: AssetData}``.
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise DataLoadError(f"CSV file not found: {path}")
        payload = path.read_bytes()
        name = source_name or path.name
    elif isinstance(source, bytes):
        payload = source
        name = source_name or "uploaded.csv"
    else:
        payload = source.read()
        name = source_name or getattr(source, "name", "uploaded.csv")

    try:
        raw = pd.read_csv(io.BytesIO(payload))
    except Exception as exc:  # pandas raises several parser exception types
        raise DataLoadError(f"Could not parse CSV '{name}': {exc}") from exc
    if raw.empty:
        raise DataLoadError(f"CSV '{name}' contains no rows.")

    frame = standardize_price_frame(raw)
    digest = sha256_bytes(payload)
    retrieved = utc_now_iso()
    file_flags_synthetic = "synthetic" in name.lower()

    assets: dict[str, AssetData] = {}
    if "Ticker" in frame.columns:
        tickers = frame["Ticker"].astype(str).str.strip()
        for symbol in sorted(tickers.dropna().unique()):
            sub = frame.loc[tickers == symbol].drop(columns="Ticker")
            assets[symbol] = _csv_asset(symbol, sub, name, digest, retrieved, file_flags_synthetic)
    else:
        symbol = ticker or Path(name).stem
        assets[symbol] = _csv_asset(symbol, frame, name, digest, retrieved, file_flags_synthetic)
    return assets


def _csv_asset(
    symbol: str, frame: pd.DataFrame, name: str, digest: str, retrieved: str, flagged: bool
) -> AssetData:
    synthetic = flagged or symbol.upper().startswith(SYNTHETIC_PREFIX)
    notes = [f"Loaded from CSV '{name}'. Prices are used as supplied (no additional adjustment)."]
    if synthetic:
        notes.append("SYNTHETIC DATA: generated for demonstration/testing, not market history.")
    if "Adj Close" not in frame.columns:
        notes.append("No 'Adj Close' column: 'Close' is used, which may omit dividend adjustments.")
    return AssetData(
        ticker=symbol,
        frame=frame,
        source=f"csv:{name}",
        retrieved_at=retrieved,
        sha256=digest,
        is_synthetic=synthetic,
        notes=notes,
    )


def _cache_paths(cache_dir: Path, ticker: str, start: str, end: str) -> tuple[Path, Path]:
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", ticker)
    stem = f"{safe}_{start}_{end}"
    return cache_dir / f"{stem}.csv", cache_dir / f"{stem}.meta.json"


def download_yfinance(
    ticker: str,
    start: str,
    end: str,
    cache_dir: str | Path | None = None,
    force_refresh: bool = False,
) -> AssetData:
    """Download daily OHLCV data from Yahoo Finance via ``yfinance``, with a local CSV cache.

    ``end`` is treated as inclusive (yfinance's own ``end`` is exclusive, so one day is added).
    The cache stores the raw download plus a metadata file with the retrieval timestamp and
    SHA-256 hash, so a later run can detect whether cached data has been modified.
    """
    ticker = ticker.strip().upper()
    if not ticker:
        raise DataLoadError("Ticker symbol is empty.")
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    if start_ts >= end_ts:
        raise DataLoadError(f"Start date {start} must be before end date {end}.")
    start_s, end_s = str(start_ts.date()), str(end_ts.date())

    cache_csv = cache_meta = None
    if cache_dir is not None:
        cache_root = Path(cache_dir)
        cache_root.mkdir(parents=True, exist_ok=True)
        cache_csv, cache_meta = _cache_paths(cache_root, ticker, start_s, end_s)
        if cache_csv.exists() and cache_meta.exists() and not force_refresh:
            return _load_cached(ticker, cache_csv, cache_meta)

    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise DataLoadError(
            "yfinance is not installed. Install it with `pip install yfinance`, or upload CSV data."
        ) from exc

    try:
        raw = yf.download(
            ticker,
            start=start_s,
            end=str((end_ts + pd.Timedelta(days=1)).date()),
            auto_adjust=False,
            actions=False,
            progress=False,
            threads=False,
        )
    except Exception as exc:  # network and provider errors vary widely
        raise DataLoadError(
            f"Download failed for '{ticker}': {exc}. Check the symbol and your network connection, "
            "or upload a CSV file instead."
        ) from exc
    if raw is None or raw.empty:
        raise DataLoadError(
            f"No data returned for '{ticker}' between {start_s} and {end_s}. The symbol may be "
            "invalid or delisted (Indian NSE symbols need the '.NS' suffix), or the provider may be "
            "unavailable. Try another symbol or date range, or upload a CSV file."
        )

    frame = standardize_price_frame(raw)
    retrieved = utc_now_iso()
    notes = [
        "Downloaded from Yahoo Finance via yfinance (auto_adjust=False). 'Adj Close' is split- and "
        "dividend-adjusted by the provider; it is used as delivered (no further adjustment).",
        "Free provider data may be revised, incomplete or removed; re-downloads can differ.",
    ]
    digest = None
    if cache_csv is not None and cache_meta is not None:
        frame.to_csv(cache_csv)
        digest = file_sha256(cache_csv)
        meta = {
            "ticker": ticker,
            "start": start_s,
            "end_inclusive": end_s,
            "source": "yfinance",
            "yfinance_version": getattr(yf, "__version__", "unknown"),
            "retrieved_at": retrieved,
            "sha256": digest,
            "n_rows": int(len(frame)),
        }
        cache_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    else:
        digest = sha256_bytes(frame.to_csv().encode("utf-8"))
    return AssetData(
        ticker=ticker,
        frame=frame,
        source="yfinance",
        retrieved_at=retrieved,
        sha256=digest,
        is_synthetic=False,
        notes=notes,
    )


def _load_cached(ticker: str, cache_csv: Path, cache_meta: Path) -> AssetData:
    meta = json.loads(cache_meta.read_text(encoding="utf-8"))
    frame = standardize_price_frame(pd.read_csv(cache_csv))
    digest = file_sha256(cache_csv)
    notes = [
        f"Loaded from local cache {cache_csv.name} (originally retrieved {meta.get('retrieved_at')}).",
        "Free provider data may be revised; use 'force refresh' to re-download.",
    ]
    if meta.get("sha256") and meta["sha256"] != digest:
        notes.append(
            "WARNING: cached file hash does not match the hash recorded at download time; "
            "the cached data has changed since retrieval."
        )
    return AssetData(
        ticker=ticker,
        frame=frame,
        source="yfinance (cached)",
        retrieved_at=str(meta.get("retrieved_at")),
        sha256=digest,
        is_synthetic=False,
        notes=notes,
    )


def generate_synthetic_pair(
    n_obs: int = 2520,
    seed: int = 42,
    cointegrated: bool = True,
    hedge_ratio: float = 0.8,
    intercept: float = 0.4,
    half_life: float = 15.0,
    residual_vol: float = 0.012,
    drift_b: float = 0.0002,
    vol_b: float = 0.015,
    start: str = "2015-01-02",
    prefix: str | None = None,
) -> tuple[AssetData, AssetData]:
    """Generate a deterministic, clearly labelled synthetic pair.

    ``log P_B`` is a random walk with drift. When ``cointegrated`` is true,
    ``log P_A = intercept + hedge_ratio * log P_B + u`` where ``u`` is a stationary
    AR(1) (discretised Ornstein-Uhlenbeck) process with the given half-life. Otherwise
    ``log P_A`` is an independent random walk, so no cointegrating relation exists.
    """
    if n_obs < 30:
        raise ValueError("n_obs must be at least 30")
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=n_obs, name="Date")

    log_b = np.log(50.0) + np.cumsum(drift_b + vol_b * rng.standard_normal(n_obs))
    if cointegrated:
        phi = float(np.exp(-np.log(2.0) / half_life))
        shocks = residual_vol * rng.standard_normal(n_obs)
        resid = np.empty(n_obs)
        resid[0] = shocks[0] / np.sqrt(1.0 - phi**2)
        for t in range(1, n_obs):
            resid[t] = phi * resid[t - 1] + shocks[t]
        log_a = intercept + hedge_ratio * log_b + resid
    else:
        log_a = np.log(60.0) + np.cumsum(drift_b + vol_b * rng.standard_normal(n_obs))

    label = prefix or (f"{SYNTHETIC_PREFIX}COINT" if cointegrated else f"{SYNTHETIC_PREFIX}INDEP")
    note = (
        "SYNTHETIC DATA: generated by statarb.data_loader.generate_synthetic_pair "
        f"(seed={seed}, cointegrated={cointegrated}). Not market history; for demos and tests only."
    )
    retrieved = utc_now_iso()
    assets = []
    for suffix, log_price in (("A", log_a), ("B", log_b)):
        frame = _ohlcv_from_close(np.exp(log_price), dates, rng)
        assets.append(
            AssetData(
                ticker=f"{label}_{suffix}",
                frame=frame,
                source="synthetic",
                retrieved_at=retrieved,
                sha256=sha256_bytes(frame.to_csv().encode("utf-8")),
                is_synthetic=True,
                notes=[note],
            )
        )
    return assets[0], assets[1]


def _ohlcv_from_close(close: np.ndarray, dates: pd.DatetimeIndex, rng: np.random.Generator) -> pd.DataFrame:
    n = len(close)
    prev = np.concatenate([[close[0]], close[:-1]])
    open_ = prev * np.exp(0.002 * rng.standard_normal(n))
    high = np.maximum(open_, close) * np.exp(np.abs(0.004 * rng.standard_normal(n)))
    low = np.minimum(open_, close) * np.exp(-np.abs(0.004 * rng.standard_normal(n)))
    volume = rng.integers(500_000, 5_000_000, size=n).astype(float)
    frame = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Adj Close": close, "Volume": volume},
        index=dates,
    )
    frame.index.name = "Date"
    return frame


def assets_to_long_csv(assets: list[AssetData], path: str | Path) -> Path:
    """Write several assets to one long-format CSV (``Date, Ticker, Open, ...``)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    for asset in assets:
        frame = asset.frame.copy()
        frame.insert(0, "Ticker", asset.ticker)
        frames.append(frame)
    pd.concat(frames).reset_index().to_csv(out, index=False, float_format="%.6f")
    return out


__all__ = [
    "AssetData",
    "DataLoadError",
    "STANDARD_COLUMNS",
    "SYNTHETIC_PREFIX",
    "assets_to_long_csv",
    "download_yfinance",
    "file_sha256",
    "generate_synthetic_pair",
    "load_csv",
    "sha256_bytes",
    "standardize_price_frame",
    "utc_now_iso",
]
