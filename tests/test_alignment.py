import numpy as np
import pandas as pd
import pytest

from conftest import make_asset, make_frame
from statarb.data_validation import align_pair, prepare_pair


def _pair_with_gaps():
    dates = pd.bdate_range("2020-01-01", periods=300)
    a = make_frame(100 + np.arange(300, dtype=float), dates=dates)
    b = make_frame(50 + 0.5 * np.arange(300, dtype=float), dates=dates)
    a_drop = [dates[10], dates[11]]
    b_drop = [dates[100]]
    return a.drop(index=a_drop), b.drop(index=b_drop), a_drop, b_drop


def test_inner_join_reports_dropped_dates_without_filling():
    a, b, a_drop, b_drop = _pair_with_gaps()
    prices, report = align_pair(a, b, "Adj Close", "Adj Close", "AAA", "BBB", min_observations=100)
    assert report.n_common == 300 - 3
    assert report.dates_only_in_b == [str(d.date()) for d in a_drop]  # B had dates that A lacked
    assert report.dates_only_in_a == [str(d.date()) for d in b_drop]
    for missing in a_drop + b_drop:
        assert missing not in prices.index
    # Values are the original observations (no forward fill).
    pd.testing.assert_series_equal(prices["A"], a["Adj Close"].reindex(prices.index), check_names=False)
    pd.testing.assert_series_equal(prices["B"], b["Adj Close"].reindex(prices.index), check_names=False)
    assert prices.notna().all().all()


def test_aligned_returns_span_identical_intervals_for_both_legs():
    a, b, *_ = _pair_with_gaps()
    prices, _ = align_pair(a, b, "Adj Close", "Adj Close", min_observations=100)
    # Return on the first date after a gap is measured from the same previous date for both legs.
    returns = prices / prices.shift(1) - 1
    idx = prices.index.get_loc(pd.Timestamp(pd.bdate_range("2020-01-01", periods=300)[12]))
    prev_date = prices.index[idx - 1]
    expected_a = a.loc[prices.index[idx], "Adj Close"] / a.loc[prev_date, "Adj Close"] - 1
    expected_b = b.loc[prices.index[idx], "Adj Close"] / b.loc[prev_date, "Adj Close"] - 1
    assert returns["A"].iloc[idx] == pytest.approx(expected_a)
    assert returns["B"].iloc[idx] == pytest.approx(expected_b)


def test_insufficient_overlap_raises():
    a = make_frame(np.linspace(10, 20, 200), start="2020-01-01")
    b = make_frame(np.linspace(10, 20, 200), start="2020-08-01")
    with pytest.raises(ValueError, match="overlapping observations"):
        align_pair(a, b, "Adj Close", "Adj Close", min_observations=150)


def test_cross_exchange_warning_and_same_ticker_rejected():
    frame = make_frame(np.linspace(10, 20, 300))
    pair = prepare_pair(make_asset("TCS.NS", frame), make_asset("KO", frame.copy()), min_observations=100)
    assert any("different exchanges" in w for w in pair.report.warnings)
    with pytest.raises(ValueError, match="different instruments"):
        prepare_pair(make_asset("KO", frame), make_asset("KO", frame.copy()), min_observations=100)


def test_prepare_pair_end_to_end(coint_pair):
    assert list(coint_pair.prices.columns) == ["A", "B"]
    assert coint_pair.prices.index.is_monotonic_increasing
    assert coint_pair.is_synthetic
    assert coint_pair.price_column_a == "Adj Close"
