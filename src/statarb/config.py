"""Typed research configuration: loading, validation, overrides and snapshots.

The YAML file in ``config/default_config.yaml`` is parsed into nested frozen
dataclasses. Unknown keys raise an error so that typos (for example
``entry_zz``) cannot silently fall back to defaults.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default_config.yaml"


@dataclass(frozen=True)
class DataConfig:
    source: str = "synthetic"
    asset_a: str = "KO"
    asset_b: str = "PEP"
    candidate_pairs: tuple[tuple[str, str], ...] = (
        ("HDFCBANK.NS", "ICICIBANK.NS"),
        ("TCS.NS", "INFY.NS"),
        ("KO", "PEP"),
    )
    start_date: str = "2015-01-01"
    end_date: str = "2024-12-31"
    prefer_adjusted: bool = True
    cache_dir: str = "data/raw"
    min_observations: int = 252
    max_gap_days: int = 7
    extreme_return_threshold: float = 0.25
    # Special (non-regular) sessions such as NSE Muhurat trading: "keep" (baseline) or "drop".
    special_sessions: str = "keep"
    special_session_dates: tuple[str, ...] = ()
    # Exchange-specific listed sessions, keyed by ticker suffix (e.g. NS for NSE); YAML mapping or list of pairs.
    special_session_dates_by_exchange: tuple[tuple[str, tuple[str, ...]], ...] = ()
    special_session_volume_ratio: float | None = None

    def __post_init__(self) -> None:
        if self.source not in {"yfinance", "csv", "synthetic"}:
            raise ValueError(f"data.source must be yfinance, csv or synthetic, got {self.source!r}")
        pairs = tuple(tuple(str(t) for t in pair) for pair in self.candidate_pairs)
        if any(len(pair) != 2 for pair in pairs):
            raise ValueError("data.candidate_pairs must contain pairs of two tickers")
        object.__setattr__(self, "candidate_pairs", pairs)
        object.__setattr__(self, "start_date", str(self.start_date))
        object.__setattr__(self, "end_date", str(self.end_date))
        if self.start_date >= self.end_date:
            raise ValueError("data.start_date must be earlier than data.end_date")
        if self.min_observations < 30:
            raise ValueError("data.min_observations must be at least 30")
        if self.max_gap_days < 1:
            raise ValueError("data.max_gap_days must be positive")
        if self.extreme_return_threshold <= 0:
            raise ValueError("data.extreme_return_threshold must be positive")
        if self.special_sessions not in {"keep", "drop"}:
            raise ValueError("data.special_sessions must be keep or drop")
        object.__setattr__(self, "special_session_dates", tuple(str(d) for d in self.special_session_dates))
        by_exchange = self.special_session_dates_by_exchange
        items = by_exchange.items() if isinstance(by_exchange, dict) else by_exchange
        object.__setattr__(self, "special_session_dates_by_exchange",
                           tuple((str(k).upper().lstrip("."), tuple(str(d) for d in v)) for k, v in items))
        ratio = self.special_session_volume_ratio
        if ratio is not None and not 0.0 < ratio < 1.0:
            raise ValueError("data.special_session_volume_ratio must be in (0, 1) or null")


@dataclass(frozen=True)
class SplitConfig:
    train: float = 0.6
    validation: float = 0.2
    test: float = 0.2

    def __post_init__(self) -> None:
        for name in ("train", "validation", "test"):
            value = getattr(self, name)
            if not 0.0 < value < 1.0:
                raise ValueError(f"splits.{name} must be strictly between 0 and 1, got {value}")
        total = self.train + self.validation + self.test
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"split proportions must sum to 1.0, got {total:.6f}")


@dataclass(frozen=True)
class CointegrationConfig:
    significance: float = 0.05
    trend: str = "c"
    autolag: str | None = "AIC"
    min_observations: int = 100

    def __post_init__(self) -> None:
        if not 0.0 < self.significance < 0.5:
            raise ValueError("cointegration.significance must be in (0, 0.5)")
        if self.trend not in {"c", "ct", "ctt", "n"}:
            raise ValueError("cointegration.trend must be one of c, ct, ctt, n")
        if self.autolag is not None and self.autolag not in {"AIC", "BIC", "t-stat"}:
            raise ValueError("cointegration.autolag must be AIC, BIC, t-stat or null")
        if self.min_observations < 30:
            raise ValueError("cointegration.min_observations must be at least 30")


@dataclass(frozen=True)
class HedgeRatioConfig:
    method: str = "ols_log_price"

    def __post_init__(self) -> None:
        if self.method != "ols_log_price":
            raise ValueError("hedge_ratio.method currently supports only 'ols_log_price'")


@dataclass(frozen=True)
class SpreadConfig:
    min_std: float = 1e-8

    def __post_init__(self) -> None:
        if self.min_std <= 0:
            raise ValueError("spread.min_std must be positive")


@dataclass(frozen=True)
class StrategyConfig:
    entry_z: float = 2.0
    exit_z: float = 0.5
    stop_loss_z: float | None = None
    max_holding_days: int | None = None
    execution_lag: int = 1

    def __post_init__(self) -> None:
        if self.entry_z <= 0:
            raise ValueError("strategy.entry_z must be positive")
        if not 0.0 <= self.exit_z < self.entry_z:
            raise ValueError("strategy.exit_z must satisfy 0 <= exit_z < entry_z")
        if self.stop_loss_z is not None and self.stop_loss_z <= self.entry_z:
            raise ValueError("strategy.stop_loss_z must exceed entry_z when enabled")
        if self.max_holding_days is not None and self.max_holding_days < 1:
            raise ValueError("strategy.max_holding_days must be >= 1 when enabled")
        if self.execution_lag < 1:
            raise ValueError(
                "strategy.execution_lag must be >= 1: a signal computed from the close of day t "
                "cannot be traded at that same close without look-ahead bias"
            )


@dataclass(frozen=True)
class PortfolioConfig:
    initial_capital: float = 1_000_000.0
    max_gross_exposure: float = 1.0
    rebalance_tolerance: float | None = 0.10
    close_at_period_end: bool = True

    def __post_init__(self) -> None:
        if self.initial_capital <= 0:
            raise ValueError("portfolio.initial_capital must be positive")
        if not 0.0 < self.max_gross_exposure <= 10.0:
            raise ValueError("portfolio.max_gross_exposure must be in (0, 10]")
        if self.rebalance_tolerance is not None and self.rebalance_tolerance < 0:
            raise ValueError("portfolio.rebalance_tolerance must be non-negative or null")


@dataclass(frozen=True)
class CostConfig:
    commission_bps: float = 5.0
    slippage_bps: float = 5.0
    borrow_bps_annual: float = 50.0
    short_selling_allowed: bool = True

    def __post_init__(self) -> None:
        for name in ("commission_bps", "slippage_bps", "borrow_bps_annual"):
            if getattr(self, name) < 0:
                raise ValueError(f"costs.{name} must be non-negative")


@dataclass(frozen=True)
class MetricsConfig:
    periods_per_year: int = 252
    risk_free_rate: float = 0.0
    min_years_for_cagr: float = 1.0

    def __post_init__(self) -> None:
        if self.periods_per_year <= 0:
            raise ValueError("metrics.periods_per_year must be positive")
        if not -0.05 <= self.risk_free_rate <= 0.5:
            raise ValueError("metrics.risk_free_rate must be an annual decimal rate, e.g. 0.03")
        if self.min_years_for_cagr < 0:
            raise ValueError("metrics.min_years_for_cagr must be non-negative")


@dataclass(frozen=True)
class EvaluationConfig:
    test_parameter_source: str = "config"
    selection_metric: str = "sharpe"
    min_trades_for_selection: int = 3

    def __post_init__(self) -> None:
        if self.test_parameter_source not in {"config", "validation_selected"}:
            raise ValueError("evaluation.test_parameter_source must be config or validation_selected")
        if self.selection_metric not in {"sharpe", "cumulative_return", "calmar"}:
            raise ValueError("evaluation.selection_metric must be sharpe, cumulative_return or calmar")
        if self.min_trades_for_selection < 0:
            raise ValueError("evaluation.min_trades_for_selection must be non-negative")


@dataclass(frozen=True)
class WalkForwardConfig:
    train_window: int = 504
    test_window: int = 126
    expanding: bool = False
    require_cointegration: bool = True
    min_eval_observations: int = 20

    def __post_init__(self) -> None:
        if self.train_window < 60:
            raise ValueError("walk_forward.train_window must be at least 60 bars")
        if self.test_window < 5:
            raise ValueError("walk_forward.test_window must be at least 5 bars")
        if self.min_eval_observations < 2:
            raise ValueError("walk_forward.min_eval_observations must be at least 2")


@dataclass(frozen=True)
class RobustnessConfig:
    entry_grid: tuple[float, ...] = (1.5, 2.0, 2.5, 3.0)
    exit_grid: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0)
    cost_grid_bps: tuple[float, ...] = (0.0, 5.0, 10.0, 20.0, 40.0)

    def __post_init__(self) -> None:
        for name in ("entry_grid", "exit_grid", "cost_grid_bps"):
            values = tuple(float(v) for v in getattr(self, name))
            if not values:
                raise ValueError(f"robustness.{name} must not be empty")
            if any(v < 0 for v in values):
                raise ValueError(f"robustness.{name} values must be non-negative")
            object.__setattr__(self, name, values)


@dataclass(frozen=True)
class SyntheticConfig:
    n_observations: int = 2520
    hedge_ratio: float = 0.8
    intercept: float = 0.4
    half_life: float = 15.0
    residual_volatility: float = 0.012
    seed: int = 42

    def __post_init__(self) -> None:
        if self.n_observations < 100:
            raise ValueError("synthetic.n_observations must be at least 100")
        if self.half_life <= 0 or self.residual_volatility <= 0:
            raise ValueError("synthetic.half_life and residual_volatility must be positive")


@dataclass(frozen=True)
class ReproducibilityConfig:
    random_seed: int = 42
    reports_dir: str = "reports"


@dataclass(frozen=True)
class MLConfig:
    """Machine-learning layer: reversion target, features, models and the B/C strategy variants."""

    enabled: bool = True
    horizon: int = 5
    tolerance_z: float = 0.25
    min_abs_z: float = 1.0
    embargo: int = 0
    z_lags: tuple[int, ...] = (1, 2, 5)
    momentum_window: int = 10
    volatility_window: int = 20
    correlation_window: int = 20
    return_windows: tuple[int, ...] = (1, 5)
    probability_threshold: float = 0.5
    threshold_source: str = "config"
    threshold_grid: tuple[float, ...] = (0.5, 0.55, 0.6, 0.65, 0.7)
    tuning_metric: str = "roc_auc"
    logistic_c_grid: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)
    rf_n_estimators: int = 200
    rf_max_depth_grid: tuple[int | None, ...] = (3, 6)
    rf_min_samples_leaf_grid: tuple[int, ...] = (10, 30)
    min_train_samples: int = 60
    min_class_count: int = 10
    min_eval_samples: int = 30
    calibration_bins: int = 5
    # Audit-driven protocol options; defaults reproduce the baseline.
    walk_forward_tuning: str = "static"      # static (baseline: reuse static-validation choice) | nested
    inner_validation_fraction: float = 0.3   # nested: last share of each fit window used for tuning
    class_weight: str = "balanced"           # balanced (baseline) | none
    threshold_mode: str = "absolute"         # absolute (probability_threshold) | prevalence (fit-window base rate)
    prevalence_multiplier: float = 1.0
    include_prevalence_baseline: bool = False  # adds diagnostic variant B0 (B rules, constant prevalence forecast)

    def __post_init__(self) -> None:
        for name in ("z_lags", "return_windows", "rf_min_samples_leaf_grid"):
            values = tuple(int(v) for v in getattr(self, name))
            if not values or any(v < 1 for v in values):
                raise ValueError(f"ml.{name} must contain positive integers")
            object.__setattr__(self, name, values)
        for name in ("threshold_grid", "logistic_c_grid"):
            object.__setattr__(self, name, tuple(float(v) for v in getattr(self, name)))
        depths = tuple(None if v is None else int(v) for v in self.rf_max_depth_grid)
        object.__setattr__(self, "rf_max_depth_grid", depths)
        if self.horizon < 1:
            raise ValueError("ml.horizon must be at least 1 bar")
        if self.tolerance_z < 0:
            raise ValueError("ml.tolerance_z must be non-negative")
        if self.min_abs_z < 0:
            raise ValueError("ml.min_abs_z must be non-negative")
        if self.embargo < 0:
            raise ValueError("ml.embargo must be non-negative")
        for name in ("momentum_window", "volatility_window", "correlation_window"):
            if getattr(self, name) < 2:
                raise ValueError(f"ml.{name} must be at least 2")
        if not 0.0 < self.probability_threshold < 1.0:
            raise ValueError("ml.probability_threshold must be in (0, 1)")
        if any(not 0.0 < t < 1.0 for t in self.threshold_grid) or not self.threshold_grid:
            raise ValueError("ml.threshold_grid values must be in (0, 1)")
        if self.threshold_source not in {"config", "validation_selected"}:
            raise ValueError("ml.threshold_source must be config or validation_selected")
        if self.tuning_metric not in {"roc_auc", "log_loss"}:
            raise ValueError("ml.tuning_metric must be roc_auc or log_loss")
        if not self.logistic_c_grid or any(c <= 0 for c in self.logistic_c_grid):
            raise ValueError("ml.logistic_c_grid values must be positive")
        if not self.rf_max_depth_grid or any(d is not None and d < 1 for d in self.rf_max_depth_grid):
            raise ValueError("ml.rf_max_depth_grid values must be positive integers or null")
        if self.rf_n_estimators < 10:
            raise ValueError("ml.rf_n_estimators must be at least 10")
        if self.min_class_count < 1 or self.min_train_samples < 10 or self.min_eval_samples < 2:
            raise ValueError("ml.min_class_count >= 1, min_train_samples >= 10 and min_eval_samples >= 2 are required")
        if self.calibration_bins < 2:
            raise ValueError("ml.calibration_bins must be at least 2")
        if self.walk_forward_tuning not in {"static", "nested"}:
            raise ValueError("ml.walk_forward_tuning must be static or nested")
        if not 0.1 <= self.inner_validation_fraction <= 0.5:
            raise ValueError("ml.inner_validation_fraction must be in [0.1, 0.5]")
        if self.class_weight not in {"balanced", "none"}:
            raise ValueError("ml.class_weight must be balanced or none")
        if self.threshold_mode not in {"absolute", "prevalence"}:
            raise ValueError("ml.threshold_mode must be absolute or prevalence")
        if self.prevalence_multiplier <= 0:
            raise ValueError("ml.prevalence_multiplier must be positive")
        if (self.walk_forward_tuning == "nested" and self.threshold_source == "validation_selected"
                and self.threshold_mode == "absolute"):
            raise ValueError("ml.walk_forward_tuning: nested cannot reuse validation-selected thresholds (that would "
                             "reintroduce look-ahead); use threshold_mode: prevalence or threshold_source: config")


@dataclass(frozen=True)
class HoldoutConfig:
    """Untouched final-evaluation period. When ``enforce`` is true, data after ``development_end``
    may only be loaded through an explicit final evaluation (see :mod:`statarb.protocol`)."""

    enforce: bool = False
    development_end: str | None = None
    holdout_start: str | None = None
    holdout_end: str | None = None

    def __post_init__(self) -> None:
        for name in ("development_end", "holdout_start", "holdout_end"):
            value = getattr(self, name)
            object.__setattr__(self, name, None if value is None else str(value))
        if self.enforce:
            if None in (self.development_end, self.holdout_start, self.holdout_end):
                raise ValueError("holdout.enforce requires development_end, holdout_start and holdout_end")
            if not self.development_end < self.holdout_start <= self.holdout_end:
                raise ValueError("holdout dates must satisfy development_end < holdout_start <= holdout_end")


@dataclass(frozen=True)
class InferenceConfig:
    """Stationary-bootstrap settings for confidence intervals."""

    bootstrap_samples: int = 5000
    mean_block_length: float = 20.0
    confidence: float = 0.95
    seed: int = 20261021

    def __post_init__(self) -> None:
        if self.bootstrap_samples < 100:
            raise ValueError("inference.bootstrap_samples must be at least 100")
        if self.mean_block_length < 1:
            raise ValueError("inference.mean_block_length must be at least 1")
        if not 0.5 < self.confidence < 1.0:
            raise ValueError("inference.confidence must be in (0.5, 1)")


_SECTIONS: dict[str, type] = {
    "data": DataConfig,
    "splits": SplitConfig,
    "cointegration": CointegrationConfig,
    "hedge_ratio": HedgeRatioConfig,
    "spread": SpreadConfig,
    "strategy": StrategyConfig,
    "portfolio": PortfolioConfig,
    "costs": CostConfig,
    "metrics": MetricsConfig,
    "evaluation": EvaluationConfig,
    "walk_forward": WalkForwardConfig,
    "robustness": RobustnessConfig,
    "synthetic": SyntheticConfig,
    "reproducibility": ReproducibilityConfig,
    "ml": MLConfig,
    "holdout": HoldoutConfig,
    "inference": InferenceConfig,
}


@dataclass(frozen=True)
class ResearchConfig:
    data: DataConfig = field(default_factory=DataConfig)
    splits: SplitConfig = field(default_factory=SplitConfig)
    cointegration: CointegrationConfig = field(default_factory=CointegrationConfig)
    hedge_ratio: HedgeRatioConfig = field(default_factory=HedgeRatioConfig)
    spread: SpreadConfig = field(default_factory=SpreadConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    costs: CostConfig = field(default_factory=CostConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    walk_forward: WalkForwardConfig = field(default_factory=WalkForwardConfig)
    robustness: RobustnessConfig = field(default_factory=RobustnessConfig)
    synthetic: SyntheticConfig = field(default_factory=SyntheticConfig)
    reproducibility: ReproducibilityConfig = field(default_factory=ReproducibilityConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    holdout: HoldoutConfig = field(default_factory=HoldoutConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "ResearchConfig":
        """Build a validated config from a nested dictionary (missing keys use defaults)."""
        raw = raw or {}
        unknown_sections = set(raw) - set(_SECTIONS)
        if unknown_sections:
            raise ValueError(f"Unknown configuration sections: {sorted(unknown_sections)}")
        sections: dict[str, Any] = {}
        for name, section_cls in _SECTIONS.items():
            values = raw.get(name) or {}
            allowed = {f.name for f in fields(section_cls)}
            unknown = set(values) - allowed
            if unknown:
                raise ValueError(f"Unknown keys in '{name}': {sorted(unknown)}")
            sections[name] = section_cls(**values)
        return cls(**sections)

    def to_dict(self) -> dict[str, Any]:
        """Plain nested dictionary (tuples converted to lists) suitable for YAML/JSON."""
        return _to_builtin(asdict(self))

    def with_overrides(self, overrides: dict[str, dict[str, Any]]) -> "ResearchConfig":
        """Return a new validated config with selected section keys replaced."""
        merged = self.to_dict()
        for section, values in overrides.items():
            if section not in merged:
                raise ValueError(f"Unknown configuration section: {section!r}")
            merged[section] = {**merged[section], **values}
        return ResearchConfig.from_dict(merged)

    def fingerprint(self) -> str:
        """Short, stable hash of the configuration used to label research runs."""
        payload = json.dumps(self.to_dict(), sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:12]


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    return value


def load_config(path: str | Path | None = None) -> ResearchConfig:
    """Load a YAML configuration file; defaults to ``config/default_config.yaml``."""
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Configuration file {config_path} must contain a mapping at top level")
    return ResearchConfig.from_dict(raw)


def save_config_snapshot(config: ResearchConfig, path: str | Path) -> Path:
    """Write the exact configuration used by a run to YAML."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.to_dict(), handle, sort_keys=False)
    return out


def resolve_path(path: str | Path) -> Path:
    """Resolve a project-relative path (as stored in the config) against the project root."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


__all__ = [
    "CointegrationConfig",
    "CostConfig",
    "DataConfig",
    "DEFAULT_CONFIG_PATH",
    "EvaluationConfig",
    "HedgeRatioConfig",
    "HoldoutConfig",
    "InferenceConfig",
    "MetricsConfig",
    "MLConfig",
    "PortfolioConfig",
    "PROJECT_ROOT",
    "ReproducibilityConfig",
    "ResearchConfig",
    "RobustnessConfig",
    "SplitConfig",
    "SpreadConfig",
    "StrategyConfig",
    "SyntheticConfig",
    "WalkForwardConfig",
    "load_config",
    "resolve_path",
    "save_config_snapshot",
]
