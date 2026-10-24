import numpy as np
import pandas as pd
import pytest

from statarb.evaluation import chronological_split, run_pair_research
from statarb.data_validation import AlignedPair
from statarb.hedge_ratio import estimate_hedge_ratio


def _series(values, start="2020-01-01"):
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)))


def test_exact_linear_relation_recovered():
    log_b = _series(np.log(np.linspace(20, 40, 200)))
    log_a = 0.3 + 1.5 * log_b
    result = estimate_hedge_ratio(log_a, log_b)
    assert result.intercept == pytest.approx(0.3, abs=1e-10)
    assert result.hedge_ratio == pytest.approx(1.5, abs=1e-10)
    assert result.r_squared == pytest.approx(1.0)


def test_noisy_known_relation_recovered():
    rng = np.random.default_rng(0)
    log_b = _series(np.cumsum(0.01 * rng.standard_normal(2000)) + 3.0)
    log_a = 0.25 + 0.7 * log_b + 0.005 * rng.standard_normal(2000)
    result = estimate_hedge_ratio(log_a, log_b)
    assert result.hedge_ratio == pytest.approx(0.7, abs=0.01)
    assert result.intercept == pytest.approx(0.25, abs=0.05)
    assert result.n_obs == 2000
    assert result.fit_start == str(log_a.index[0].date())
    assert result.fit_end == str(log_a.index[-1].date())


def test_invalid_inputs():
    log_b = _series(np.log(np.linspace(20, 40, 100)))
    with pytest.raises(ValueError, match="identical index"):
        estimate_hedge_ratio(log_b.iloc[1:], log_b.iloc[:-1])
    with pytest.raises(ValueError, match="constant"):
        estimate_hedge_ratio(log_b, pd.Series(1.0, index=log_b.index))
    bad = log_b.copy()
    bad.iloc[3] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        estimate_hedge_ratio(bad, log_b)
    with pytest.raises(ValueError, match="Unsupported"):
        estimate_hedge_ratio(log_b, log_b, method="kalman")


def _tampered(pair: AlignedPair, start_idx: int, factor: float) -> AlignedPair:
    prices = pair.prices.copy()
    prices.iloc[start_idx:, 0] *= factor
    prices.iloc[start_idx:, 1] *= np.linspace(1.0, 1.0 / factor, len(prices) - start_idx)
    return AlignedPair(prices, pair.ticker_a, pair.ticker_b, pair.price_column_a, pair.price_column_b, pair.report,
                       pair.validation_a, pair.validation_b, pair.metadata_a, pair.metadata_b)


def test_parameters_estimated_on_training_data_only(coint_pair, config):
    """Changing every price after the training window must not change any fitted parameter."""
    splits = chronological_split(coint_pair.prices.index, 0.6, 0.2, 0.2)
    base = run_pair_research(coint_pair, config, include_walk_forward=False)
    tampered = run_pair_research(_tampered(coint_pair, splits["train"].end_idx, 1.7), config, include_walk_forward=False)
    assert base.model.spread == tampered.model.spread
    assert base.model.coint.p_value == tampered.model.coint.p_value
    assert base.model.hedge == tampered.model.hedge
    # And the fitted ratio differs from a (forbidden) full-sample estimate.
    logp = np.log(coint_pair.prices)
    full = estimate_hedge_ratio(logp["A"], logp["B"])
    assert full.hedge_ratio != base.model.hedge.hedge_ratio
