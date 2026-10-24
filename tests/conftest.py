"""Shared deterministic fixtures. No test touches the network."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statarb.config import ResearchConfig
from statarb.data_loader import AssetData, generate_synthetic_pair
from statarb.data_validation import prepare_pair


def make_frame(close, start: str = "2020-01-01", dates=None, adj=True) -> pd.DataFrame:
    """Minimal OHLCV frame from a close series (business-day dates unless given)."""
    close = np.asarray(close, dtype=float)
    index = pd.DatetimeIndex(dates) if dates is not None else pd.bdate_range(start, periods=len(close))
    index.name = "Date"
    frame = pd.DataFrame(
        {"Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close, "Volume": 1_000.0}, index=index
    )
    if adj:
        frame["Adj Close"] = close
    return frame


def make_asset(ticker: str, frame: pd.DataFrame) -> AssetData:
    return AssetData(ticker=ticker, frame=frame, source="test", retrieved_at="2026-01-01T00:00:00+00:00")


@pytest.fixture
def config() -> ResearchConfig:
    return ResearchConfig()


@pytest.fixture(scope="session")
def coint_assets():
    return generate_synthetic_pair(n_obs=1500, seed=11, cointegrated=True)


@pytest.fixture(scope="session")
def indep_assets():
    return generate_synthetic_pair(n_obs=1500, seed=11, cointegrated=False)


@pytest.fixture(scope="session")
def coint_pair(coint_assets):
    return prepare_pair(*coint_assets)


@pytest.fixture(scope="session")
def indep_pair(indep_assets):
    return prepare_pair(*indep_assets)


def signal_frame(index, targets, events=None, z=None) -> pd.DataFrame:
    targets = np.asarray(targets, dtype=int)
    n = len(targets)
    return pd.DataFrame(
        {
            "zscore": np.zeros(n) if z is None else np.asarray(z, dtype=float),
            "target_position": targets,
            "event": [""] * n if events is None else list(events),
            "holding_bars": np.zeros(n, dtype=int),
        },
        index=index,
    )
