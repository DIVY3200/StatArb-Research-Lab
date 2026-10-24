"""Cointegration tests. Finite-sample tests are probabilistic, so fixed seeds are used and
rejection-rate checks use generous bounds."""

import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.multitest import multipletests

from statarb.cointegration import benjamini_hochberg, engle_granger_test
from statarb.data_loader import generate_synthetic_pair


def _logs(cointegrated: bool, seed: int, n: int = 1000):
    a, b = generate_synthetic_pair(n_obs=n, seed=seed, cointegrated=cointegrated)
    return np.log(a.frame["Adj Close"]), np.log(b.frame["Adj Close"])


def test_cointegrated_pair_rejects_null():
    log_a, log_b = _logs(True, seed=5)
    result = engle_granger_test(log_a, log_b, significance=0.05)
    assert result.reject_null
    assert result.p_value < 0.01
    assert result.hedge_ratio == pytest.approx(0.8, abs=0.1)
    assert "Reject H0" in result.decision


def test_independent_random_walks_fail_to_reject_for_fixed_seed():
    log_a, log_b = _logs(False, seed=42)
    result = engle_granger_test(log_a, log_b)
    assert not result.reject_null
    assert result.p_value > 0.2


def test_empirical_size_close_to_nominal_on_independent_walks():
    # Under H0 (no cointegration) a correctly sized 5% test rejects about 5% of the time.
    n_sims = 100
    rejections = sum(
        engle_granger_test(*_logs(False, seed=s, n=600), check_integration=False).reject_null for s in range(n_sims)
    )
    assert 0 <= rejections <= 12  # binomial(100, 0.05): P(X > 12) < 0.1%


def test_false_positives_occur_by_chance():
    # Seed 5 is a documented false positive: independent walks that pass at 5%. This is exactly
    # the multiple-testing risk of screening many pairs and is why rejection is not proof.
    result = engle_granger_test(*_logs(False, seed=5))
    assert result.reject_null and result.p_value < 0.05


def test_output_structure():
    log_a, log_b = _logs(True, seed=8)
    result = engle_granger_test(log_a, log_b, significance=0.10, name_a="AAA", name_b="BBB")
    assert set(result.critical_values) == {"1%", "5%", "10%"}
    assert result.critical_values["1%"] < result.critical_values["5%"] < result.critical_values["10%"]
    assert 0.0 <= result.p_value <= 1.0
    assert result.n_obs == len(log_a)
    assert result.sample_start == str(log_a.index[0].date())
    assert result.sample_end == str(log_a.index[-1].date())
    assert result.significance == 0.10
    assert set(result.adf_levels) == {"AAA", "BBB"}
    payload = result.to_dict()
    assert payload["assumptions"] and payload["limitations"] and payload["interpretation"]


def test_failed_pair_is_flagged_not_described_as_opportunity():
    result = engle_granger_test(*_logs(False, seed=42))
    assert "insufficient evidence" in result.interpretation()
    assert any("does not pass" in w for w in result.warnings)


@pytest.mark.parametrize("n", [10, 50])
def test_too_short_sample_raises(n):
    log_a, log_b = _logs(True, seed=1, n=200)
    with pytest.raises(ValueError, match="at least"):
        engle_granger_test(log_a.iloc[:n], log_b.iloc[:n], min_obs=100)


def test_invalid_inputs_raise():
    log_a, log_b = _logs(True, seed=1, n=300)
    with pytest.raises(ValueError, match="identical index"):
        engle_granger_test(log_a.iloc[1:], log_b.iloc[:-1])
    bad = log_a.copy()
    bad.iloc[5] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        engle_granger_test(bad, log_b)
    with pytest.raises(ValueError, match="collinear"):
        engle_granger_test(0.2 + 1.3 * log_b, log_b)
    with pytest.raises(ValueError, match="constant"):
        engle_granger_test(log_a, pd.Series(1.0, index=log_b.index))
    with pytest.raises(ValueError, match="significance"):
        engle_granger_test(log_a, log_b, significance=0.7)


def test_benjamini_hochberg_matches_statsmodels():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216])
    rejected, adjusted = benjamini_hochberg(p, alpha=0.05)
    ref_reject, ref_adjusted, *_ = multipletests(p, alpha=0.05, method="fdr_bh")
    np.testing.assert_allclose(adjusted, ref_adjusted)
    np.testing.assert_array_equal(rejected, ref_reject)
    shuffled = np.random.default_rng(0).permutation(len(p))
    r2, a2 = benjamini_hochberg(p[shuffled])
    np.testing.assert_allclose(a2, ref_adjusted[shuffled])  # original order preserved


def test_benjamini_hochberg_edge_cases():
    rejected, adjusted = benjamini_hochberg([])
    assert rejected.size == 0 and adjusted.size == 0
    with pytest.raises(ValueError):
        benjamini_hochberg([0.1, 1.2])
    rejected, adjusted = benjamini_hochberg([0.5, 0.9])
    assert not rejected.any() and (adjusted <= 1).all()
