"""Engle-Granger in both directions, Johansen, the trading gates, and per-market config overrides."""

import dataclasses

import numpy as np
import pytest

import statarb.cointegration as ci
from statarb.cointegration import GATES, cointegration_suite, johansen_test
from statarb.config import ResearchConfig, market_key


def _logs(pair):
    logp = np.log(pair.prices)
    return logp["A"], logp["B"]


def test_cointegrated_pair_passes_every_gate(coint_pair):
    suite = cointegration_suite(*_logs(coint_pair))
    assert all(suite.passes(gate) for gate in GATES)
    assert suite.johansen.reject_r0 and suite.johansen.trace_statistic > suite.johansen.trace_critical_values["95%"]
    # Johansen's normalised vector and the EG regression estimate the same relation
    assert suite.johansen.hedge_ratio == pytest.approx(suite.eg_ab.hedge_ratio, rel=0.05)
    summary = suite.summary()
    assert summary["eg_ab_pass"] and summary["eg_ba_pass"] and summary["johansen_pass"]
    assert set(suite.to_dict()) == {"eg_a_on_b", "eg_b_on_a", "johansen", "johansen_error", "summary"}


def test_independent_pair_fails_every_gate(indep_pair):
    suite = cointegration_suite(*_logs(indep_pair))
    assert not any(suite.passes(gate) for gate in GATES)
    assert suite.johansen.rank == 0


def test_eg_direction_matters_and_eg_both_needs_both(coint_pair):
    a, b = _logs(coint_pair)
    suite = cointegration_suite(a, b)
    # the reverse regression is a different test: different residuals, statistic and hedge ratio
    assert suite.eg_ba.p_value != suite.eg_ab.p_value
    assert suite.eg_ba.hedge_ratio != pytest.approx(1.0 / suite.eg_ab.hedge_ratio, rel=1e-6)
    one_way = dataclasses.replace(suite, eg_ba=dataclasses.replace(suite.eg_ba, reject_null=False))
    assert one_way.passes("eg") and not one_way.passes("eg_both") and one_way.passes("eg_and_johansen")
    with pytest.raises(ValueError, match="Unknown cointegration gate"):
        suite.passes("nope")


def test_johansen_input_checks(coint_pair):
    a, b = _logs(coint_pair)
    with pytest.raises(ValueError, match="10%, 5% and 1%"):
        johansen_test(a, b, significance=0.02)
    with pytest.raises(ValueError, match="at least"):
        johansen_test(a.iloc[:50], b.iloc[:50])
    with pytest.raises(ValueError, match="identical index"):
        johansen_test(a, b.iloc[1:])
    with pytest.raises(ValueError, match="constant"):
        johansen_test(a * 0.0 + 1.0, b)


def test_johansen_failure_is_recorded_not_raised(coint_pair, monkeypatch):
    def singular(*args, **kwargs):
        raise ValueError("singular system")

    monkeypatch.setattr(ci, "johansen_test", singular)
    suite = ci.cointegration_suite(*_logs(coint_pair))
    assert suite.johansen is None and "singular" in suite.johansen_error
    assert suite.passes("eg") and not suite.passes("johansen") and not suite.passes("eg_and_johansen")
    assert np.isnan(suite.summary()["johansen_trace"]) and suite.to_dict()["johansen"] is None


def test_gate_configuration_is_validated():
    with pytest.raises(ValueError, match="gate"):
        ResearchConfig().with_overrides({"cointegration": {"gate": "nope"}})
    with pytest.raises(ValueError):
        ResearchConfig().with_overrides({"cointegration": {"gate": "johansen", "significance": 0.02}})


# --------------------------------------------------------------------------- market overrides
MARKETS = {"NS": {"costs": {"profile": "nse", "short_instrument": "futures", "futures_lot_sizes": {"TCS.NS": 175}},
                  "metrics": {"risk_free_rate": 0.065}},
           "US": {"metrics": {"risk_free_rate": 0.04}}}


def test_market_key():
    assert market_key("TCS.NS") == "NS" and market_key("tcs.ns") == "NS" and market_key("KO") == "US"


def test_for_pair_applies_the_market_of_leg_a():
    cfg = ResearchConfig().with_overrides({"markets": MARKETS})
    ns = cfg.for_pair("TCS.NS", "INFY.NS")
    assert (ns.costs.profile, ns.costs.short_instrument, ns.metrics.risk_free_rate) == ("nse", "futures", 0.065)
    assert ns.costs.lot_size("TCS.NS") == 175 and ns.costs.lot_size("INFY.NS") is None
    us = cfg.for_pair("KO", "PEP")
    assert us.costs.profile == "generic" and us.metrics.risk_free_rate == 0.04
    assert ns.for_pair("TCS.NS", "INFY.NS") == ns  # idempotent
    assert ResearchConfig().for_pair("TCS.NS", "INFY.NS") == ResearchConfig()  # no markets: unchanged


def test_market_overrides_are_validated():
    with pytest.raises(ValueError):
        ResearchConfig().with_overrides({"markets": {"NS": {"costs": {"profile": "lse"}}}})
    with pytest.raises(ValueError):
        ResearchConfig().with_overrides({"markets": {"NS": {"costs": {"no_such_key": 1.0}}}})
    with pytest.raises(ValueError):
        ResearchConfig().with_overrides({"markets": {"NS": {"no_such_section": {}}}})
