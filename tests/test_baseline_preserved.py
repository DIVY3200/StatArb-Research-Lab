"""The default config must reproduce the pre-audit baseline exactly.

``tests/data/golden_baseline_synthetic.json`` was captured from the original code before any
audit fix. Every fix since then is opt-in (config options whose defaults keep the old behaviour),
so these numbers must not move.

The golden values were captured on x86-64. The statistical layer is compared exactly on every
platform. The ML outputs are compared exactly on x86-64 only: Random Forest fits are bit-sensitive
to floating-point differences on other CPUs (Apple Silicon), where they differ in the last digits.
"""

import json
import math
import platform
from pathlib import Path

import pytest

GOLDEN = Path(__file__).with_name("data") / "golden_baseline_synthetic.json"
STATISTICAL_KEYS = ("base_periods", "base_walk_forward", "base_walk_forward_windows")
X86_64 = platform.machine().lower() in {"x86_64", "amd64"}


def _compare(expected, actual, path=""):
    if isinstance(expected, dict):
        assert set(expected) <= set(actual), f"missing keys at {path}: {set(expected) - set(actual)}"
        for key in expected:
            _compare(expected[key], actual[key], f"{path}/{key}")
    elif expected is None:
        assert actual is None, f"{path}: expected undefined, got {actual}"
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        assert actual is not None and math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12), \
            f"{path}: {actual} != {expected}"
    else:
        assert actual == expected, f"{path}: {actual} != {expected}"


@pytest.fixture(scope="module")
def golden_and_actual():
    import importlib.util

    spec = importlib.util.spec_from_file_location("capture", GOLDEN.with_name("capture_golden_baseline.py"))
    capture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(capture)
    return json.loads(GOLDEN.read_text(encoding="utf-8")), capture.capture()


@pytest.mark.slow
def test_default_config_reproduces_pre_audit_statistical_baseline(golden_and_actual):
    expected, actual = golden_and_actual
    for key in STATISTICAL_KEYS:
        _compare(expected[key], actual[key], key)


@pytest.mark.slow
@pytest.mark.skipif(not X86_64, reason="golden ML outputs were captured on x86-64; Random Forest fits differ in the "
                                       "last digits on other CPUs (macos-26-arm64: test-period RF Brier score "
                                       "0.2510946 vs 0.2510814). The statistical layer is still compared exactly.")
def test_default_config_reproduces_pre_audit_ml_baseline(golden_and_actual):
    expected, actual = golden_and_actual
    ml_keys = sorted(set(expected) - set(STATISTICAL_KEYS))
    assert ml_keys == ["classification", "comparisons", "ml_walk_forward", "model_params"]
    for key in ml_keys:
        _compare(expected[key], actual[key], key)
