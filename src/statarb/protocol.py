"""Research-protocol guards: keep the pre-registered holdout untouched until the final evaluation.

With ``holdout.enforce: true`` (as in ``config/research_v2.yaml``), any request for data after
``holdout.development_end`` raises :class:`HoldoutAccessError` unless the caller explicitly declares
a final evaluation. The final evaluation is meant to happen exactly once, after the specification
freeze recorded in ``docs/research/PREREGISTRATION.md``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from .config import PROJECT_ROOT, ResearchConfig


class HoldoutAccessError(PermissionError):
    """Raised when holdout data is requested outside a declared final evaluation."""


def allowed_end_date(config: ResearchConfig, requested_end: str, final_evaluation: bool = False) -> str:
    """Return the end date that may be loaded, or raise if the request reaches into the holdout."""
    h = config.holdout
    if not h.enforce:
        return requested_end
    if pd.Timestamp(requested_end) <= pd.Timestamp(h.development_end):
        return requested_end
    if not final_evaluation:
        raise HoldoutAccessError(
            f"Requested data up to {requested_end}, but the holdout ({h.holdout_start} to {h.holdout_end}) is locked. "
            f"Use an end date on or before {h.development_end}, or run the declared final evaluation."
        )
    if pd.Timestamp(requested_end) > pd.Timestamp(h.holdout_end):
        raise HoldoutAccessError(f"Final evaluation may not go beyond holdout_end {h.holdout_end}.")
    return requested_end


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_fingerprint(paths: list[Path] | None = None) -> str:
    """SHA-256 over the research engine source (and any extra files), in a stable order."""
    files = sorted((PROJECT_ROOT / "src" / "statarb").glob("*.py")) + list(paths or [])
    digest = hashlib.sha256()
    for path in files:
        digest.update(Path(path).name.encode("utf-8"))
        digest.update(Path(path).read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


__all__ = ["HoldoutAccessError", "allowed_end_date", "code_fingerprint", "file_sha256"]
