"""Headless smoke tests of the Streamlit dashboard (offline: synthetic data source)."""

from pathlib import Path

import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
APP = str(Path(__file__).resolve().parents[1] / "app.py")


def _run(at):
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    return at


def test_dashboard_renders_with_synthetic_data():
    at = _run(AppTest.from_file(APP, default_timeout=180))
    assert at.title[0].value == "StatArb Research Lab"
    assert any("SYNTHETIC DATA" in w.value for w in at.warning)
    assert len(at.tabs) == 11
    assert not any("does not pass" in e.value for e in at.error)  # the default synthetic pair is cointegrated
    assert any("Accuracy is not profitability" in i.value for i in at.info)  # ML tab rendered
    assert any("does not declare a winner" in w.value for w in at.warning)  # comparison tab rendered
    assert any("audit finding D1" in w.value for w in at.warning)  # baseline protocol discloses the tuning leak


def test_ml_layer_can_be_switched_off():
    at = _run(AppTest.from_file(APP, default_timeout=180))
    toggle = next(t for t in at.sidebar.toggle if t.label == "Run ML comparison (A vs B vs C)")
    toggle.set_value(False)
    _run(at)
    assert sum("ML comparison is switched off" in i.value for i in at.info) == 2


def test_dashboard_reveals_test_period_and_flags_failed_pair():
    at = _run(AppTest.from_file(APP, default_timeout=180))
    reveal = next(t for t in at.sidebar.toggle if t.label == "Reveal final test-period results")
    reveal.set_value(True)
    _run(at)
    relation = next(r for r in at.sidebar.radio if r.label == "Synthetic relation")
    relation.set_value("Independent random walks")
    _run(at)
    assert any("does not pass the training-period Engle-Granger test" in e.value for e in at.error)


def test_invalid_thresholds_show_actionable_error():
    at = _run(AppTest.from_file(APP, default_timeout=180))
    exit_input = next(n for n in at.sidebar.number_input if n.label == "Exit |z|")
    exit_input.set_value(3.0)  # exit above entry (2.0)
    _run(at)
    assert any("Invalid settings" in e.value for e in at.error)


def test_v2_protocol_runs_and_shows_prevalence_baseline():
    at = _run(AppTest.from_file(APP, default_timeout=300))
    protocol = next(r for r in at.sidebar.radio if r.label == "Protocol")
    protocol.set_value("v2 audited (config/research_v2.yaml)")
    _run(at)
    assert any("B0" in m.value for m in at.markdown)  # comparison tab describes the control
    assert not any("audit finding D1" in w.value for w in at.warning)  # nested tuning: no static-mode warning


def test_v3_protocol_shows_the_realism_options():
    at = _run(AppTest.from_file(APP, default_timeout=300))
    protocol = next(r for r in at.sidebar.radio if r.label == "Protocol")
    protocol.set_value("v3 realistic (config/research_v3.yaml)")
    _run(at)
    assert next(r for r in at.sidebar.radio if r.label == "Fill price").value == "open"
    assert next(r for r in at.sidebar.radio if r.label == "ML training rows").value == "point_in_time"
    assert next(s for s in at.sidebar.selectbox if s.label == "Cointegration gate").value == "eg_and_johansen"
    assert any("Both regression directions and Johansen" in h.value for h in at.subheader)
    text = " ".join(m.value for m in at.markdown)
    assert "Training rows are built point in time" in text
    assert "trades fill at the **open**" in text and "Cash earns the risk-free rate" in text
    assert not any("audit finding D1" in w.value for w in at.warning)


def test_nse_costs_with_futures_shorts_render():
    at = _run(AppTest.from_file(APP, default_timeout=300))
    next(r for r in at.sidebar.radio if r.label == "Cost model").set_value("nse")
    _run(at)
    next(r for r in at.sidebar.radio if r.label == "Short leg").set_value("futures")
    _run(at)
    text = " ".join(m.value for m in at.markdown)
    assert "NSE statutory schedule" in text and "single-stock futures" in text
    assert next(n for n in at.sidebar.number_input if n.label == "Short borrow fee (bps per year)").disabled


def test_johansen_gate_flags_independent_pair():
    at = _run(AppTest.from_file(APP, default_timeout=300))
    next(s for s in at.sidebar.selectbox if s.label == "Cointegration gate").set_value("johansen")
    _run(at)
    next(r for r in at.sidebar.radio if r.label == "Synthetic relation").set_value("Independent random walks")
    _run(at)
    assert any("does not pass the training-period cointegration gate (Johansen trace test)" in e.value for e in at.error)
