"""The default configuration must reproduce the pre-audit baseline exactly.

``tests/data/golden_baseline_synthetic.json`` was captured from the original code before any
audit fix. Every fix since then is opt-in (config options whose defaults keep the old behaviour),
so these numbers must not move.
"""

import json
import math
from pathlib import Path

import pytest

GOLDEN = Path(__file__).with_name("data") / "golden_baseline_synthetic.json"


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


@pytest.mark.slow
def test_default_config_reproduces_pre_audit_baseline():
    import importlib.util

    spec = importlib.util.spec_from_file_location("capture", GOLDEN.with_name("capture_golden_baseline.py"))
    capture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(capture)
    _compare(json.loads(GOLDEN.read_text(encoding="utf-8")), capture.capture())
