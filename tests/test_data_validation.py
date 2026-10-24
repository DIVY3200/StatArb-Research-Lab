import io
import json

import numpy as np
import pandas as pd
import pytest

from conftest import make_asset, make_frame
from statarb.data_loader import (
    DataLoadError,
    download_yfinance,
    file_sha256,
    generate_synthetic_pair,
    load_csv,
    standardize_price_frame,
)
from statarb.data_validation import prepare_pair, select_price_column, validate_price_frame


def test_sorts_unsorted_input_and_reports_it():
    frame = make_frame(np.linspace(10, 20, 50))
    shuffled = frame.iloc[np.random.default_rng(0).permutation(len(frame))]
    clean, report = validate_price_frame(shuffled, "X", min_observations=30)
    assert report.was_unsorted
    assert clean.index.is_monotonic_increasing
    pd.testing.assert_frame_equal(clean, frame, check_freq=False)


def test_missing_prices_are_removed_not_filled():
    close = np.linspace(10, 20, 60)
    frame = make_frame(close)
    frame.iloc[[5, 6, 30], frame.columns.get_loc("Adj Close")] = np.nan
    clean, report = validate_price_frame(frame, "X", min_observations=30)
    assert len(clean) == 57
    assert len(report.missing_price_dates) == 3
    assert str(frame.index[5].date()) in report.missing_price_dates
    # Every remaining price is an original observation: nothing was invented.
    assert clean["Adj Close"].isin(frame["Adj Close"].dropna()).all()
    assert frame.index[5] not in clean.index


def test_infinite_and_non_positive_prices_are_removed():
    frame = make_frame(np.linspace(10, 20, 60))
    frame.iloc[3, frame.columns.get_loc("Adj Close")] = 0.0
    frame.iloc[4, frame.columns.get_loc("Adj Close")] = -5.0
    frame.iloc[7, frame.columns.get_loc("Adj Close")] = np.inf
    clean, report = validate_price_frame(frame, "X", min_observations=30)
    assert len(report.nonpositive_price_dates) == 2
    assert len(report.missing_price_dates) == 1
    assert (clean["Adj Close"] > 0).all() and np.isfinite(clean["Adj Close"]).all()


def test_exact_duplicates_collapsed_conflicting_duplicates_removed():
    frame = make_frame(np.linspace(10, 20, 60))
    exact = frame.iloc[[10]]
    conflicting = frame.iloc[[20]].copy()
    conflicting["Adj Close"] *= 1.5
    dirty = pd.concat([frame, exact, conflicting])
    clean, report = validate_price_frame(dirty, "X", min_observations=30)
    assert report.exact_duplicate_dates == [str(frame.index[10].date())]
    assert report.conflicting_duplicate_dates == [str(frame.index[20].date())]
    assert not clean.index.has_duplicates
    assert frame.index[10] in clean.index
    assert frame.index[20] not in clean.index  # neither conflicting copy is trusted
    assert len(clean) == 59


def test_insufficient_history_is_an_error():
    frame = make_frame(np.linspace(10, 20, 40))
    _, report = validate_price_frame(frame, "X", min_observations=100)
    assert not report.passed
    assert "Insufficient history" in report.errors[0]
    with pytest.raises(ValueError, match="Insufficient history"):
        prepare_pair(make_asset("X", frame), make_asset("Y", frame.copy()), min_observations=100)


def test_large_gap_and_extreme_return_are_flagged():
    dates = list(pd.bdate_range("2020-01-01", periods=40)) + list(pd.bdate_range("2020-04-01", periods=40))
    close = np.linspace(10, 12, 80)
    close[60] = close[59] * 2.0  # a 100% jump, e.g. an unadjusted split
    frame = make_frame(close, dates=dates)
    _, report = validate_price_frame(frame, "X", min_observations=30, max_gap_days=7)
    assert len(report.large_gaps) == 1
    assert report.large_gaps[0]["calendar_days"] > 7
    assert any(abs(item["log_return"]) > 0.25 for item in report.extreme_return_dates)
    assert report.passed  # flagged, not fatal


def test_price_column_selection():
    frame = make_frame(np.linspace(10, 20, 50))
    assert select_price_column(frame) == "Adj Close"
    assert select_price_column(frame, prefer_adjusted=False) == "Close"
    no_adj = make_frame(np.linspace(10, 20, 50), adj=False)
    assert select_price_column(no_adj) == "Close"
    _, report = validate_price_frame(no_adj, "X", min_observations=30)
    assert any("Adjusted close unavailable" in w for w in report.warnings)


def test_standardize_handles_yfinance_multiindex_and_aliases():
    index = pd.DatetimeIndex(pd.bdate_range("2021-01-01", periods=3), name="Date")
    columns = pd.MultiIndex.from_product([["Adj Close", "Close", "High", "Low", "Open", "Volume"], ["KO"]])
    raw = pd.DataFrame(np.arange(18, dtype=float).reshape(3, 6) + 1, index=index, columns=columns)
    frame = standardize_price_frame(raw)
    assert list(frame.columns) == ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    assert frame.index.name == "Date"

    aliased = pd.DataFrame({"date": ["2021-01-04", "2021-01-05"], "adj_close": ["1.5", "bad"], "CLOSE": [1, 2]})
    frame = standardize_price_frame(aliased)
    assert np.isnan(frame["Adj Close"].iloc[1])  # unparseable number becomes NaN, reported by validation


def test_load_long_format_csv_and_synthetic_flag(tmp_path):
    a, b = generate_synthetic_pair(n_obs=60, seed=3)
    rows = []
    for asset in (a, b):
        f = asset.frame.reset_index()
        f.insert(1, "Ticker", asset.ticker)
        rows.append(f)
    path = tmp_path / "pair.csv"
    pd.concat(rows).to_csv(path, index=False)
    assets = load_csv(path)
    assert set(assets) == {a.ticker, b.ticker}
    loaded = assets[a.ticker]
    assert loaded.is_synthetic  # SYNTH_ prefix is detected
    assert loaded.sha256 == file_sha256(path)
    np.testing.assert_allclose(loaded.frame["Adj Close"].to_numpy(), a.frame["Adj Close"].to_numpy(), rtol=1e-12)


def test_load_single_asset_csv_from_bytes_uses_ticker_argument():
    csv = "Date,Close\n2021-01-04,10\n2021-01-05,11\n"
    assets = load_csv(io.BytesIO(csv.encode()), ticker="ABC", source_name="abc.csv")
    assert list(assets) == ["ABC"]
    assert not assets["ABC"].is_synthetic
    with pytest.raises(DataLoadError):
        load_csv(b"foo,bar\n1,2\n")


def test_synthetic_data_is_labelled():
    a, b = generate_synthetic_pair(n_obs=100, seed=1)
    assert a.is_synthetic and b.is_synthetic
    assert a.ticker.startswith("SYNTH_") and b.ticker.startswith("SYNTH_")
    assert "SYNTHETIC" in a.notes[0]
    a2, _ = generate_synthetic_pair(n_obs=100, seed=1)
    pd.testing.assert_frame_equal(a.frame, a2.frame)  # deterministic


def test_yfinance_cache_hit_is_offline_and_detects_modification(tmp_path):
    frame = make_frame(np.linspace(10, 20, 30))
    csv_path = tmp_path / "ABC_2020-01-01_2020-03-01.csv"
    frame.to_csv(csv_path)
    meta = {"retrieved_at": "2026-01-01T00:00:00+00:00", "sha256": file_sha256(csv_path)}
    (tmp_path / "ABC_2020-01-01_2020-03-01.meta.json").write_text(json.dumps(meta))
    asset = download_yfinance("ABC", "2020-01-01", "2020-03-01", cache_dir=tmp_path)
    assert asset.source == "yfinance (cached)"
    assert not any("does not match" in n for n in asset.notes)

    csv_path.write_text(csv_path.read_text() + "\n")  # tamper with the cached file
    asset = download_yfinance("ABC", "2020-01-01", "2020-03-01", cache_dir=tmp_path)
    assert any("does not match" in n for n in asset.notes)


def test_download_rejects_bad_inputs():
    with pytest.raises(DataLoadError):
        download_yfinance("", "2020-01-01", "2021-01-01")
    with pytest.raises(DataLoadError):
        download_yfinance("KO", "2021-01-01", "2020-01-01")
