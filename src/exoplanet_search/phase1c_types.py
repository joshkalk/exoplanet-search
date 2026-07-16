"""Configuration and typed records for Phase 1C posterior sampling."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


PARAMETER_ORDER = (
    "log_rp",
    "log_a",
    "z_b",
    "q1",
    "q2",
    "log_jitter",
    "period_offset",
    "mid_epoch_offset",
)


@dataclass(frozen=True)
class Phase1CConfig:
    """Primary Phase 1C settings for frozen Phase 1B posterior sampling."""

    phase1b_output_dir: Path = Path("data/interim/kepler5_phase1b_fit")
    output_dir: Path = Path("data/interim/kepler5_phase1c_posterior")
    run_id: str | None = None
    random_seed: int = 20260715
    n_ensembles: int = 4
    n_walkers: int = 24
    pilot_steps: int = 24
    synthetic_steps: int = 80
    synthetic_recovery_steps: int = 2000
    production_steps: int = 2000
    chunk_steps: int = 12
    warmup_steps: int = 8
    supersample_factor: int = 11
    rp_bounds: tuple[float, float] = (0.001, 0.35)
    a_bounds: tuple[float, float] = (1.01, 100.0)
    q_bounds: tuple[float, float] = (0.0, 1.0)
    limb_darkening_sigma_floor: float = 0.08
    jitter_lower: float = 1.0e-8
    jitter_upper: float = 0.02
    jitter_prior_median_uncertainty_multiple: float = 5.0
    baseline_intercept_sigma: float = 0.05
    baseline_slope_sigma: float = 0.05
    mid_epoch_half_width_duration_scale: float = 0.5
    convergence_rhat_threshold: float = 1.01
    convergence_ess_minimum: float = 1000.0
    convergence_tau_multiple: float = 50.0
    max_pilot_seconds: float = 600.0
    minimum_meaningful_summary_draws: int = 1000
    local_tight_scales: tuple[float, ...] = (0.015, 0.02, 0.015, 0.015, 0.015, 0.05, 1.0e-5, 2.0e-4)
    local_moderate_scales: tuple[float, ...] = (0.04, 0.05, 0.04, 0.04, 0.04, 0.12, 5.0e-5, 1.0e-3)
    local_broad_scales: tuple[float, ...] = (0.08, 0.10, 0.08, 0.08, 0.08, 0.25, 2.0e-4, 5.0e-3)

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        for key, value in list(values.items()):
            if isinstance(value, Path):
                values[key] = str(value)
            elif isinstance(value, tuple):
                values[key] = list(value)
        values["parameter_order"] = list(PARAMETER_ORDER)
        values["notes"] = {
            "phase1b_snapshot": "Frozen accepted Phase 1B cadences are consumed as-is.",
            "q_prior_width": "max(recorded q sigma, limb_darkening_sigma_floor)",
            "jitter_prior": "Half-normal in physical jitter; sampled in log jitter with Jacobian.",
            "baseline_marginalization": "Exact Gaussian marginalization per event.",
            "pilot_label": "PILOT - NONPRODUCTION - NONCONVERGED",
        }
        return values


@dataclass(frozen=True)
class FrozenPhase1BData:
    """Frozen unbinned Phase 1B accepted cadences and fit metadata."""

    time: np.ndarray
    flux: np.ndarray
    flux_uncertainty: np.ndarray
    event_number: np.ndarray
    predicted_center: np.ndarray
    product_id: np.ndarray
    quarter: np.ndarray
    exposure_days: np.ndarray
    deterministic_parameters: dict[str, float]
    limb_darkening: dict[str, Any]
    phase1b_configuration: dict[str, Any]
    phase1b_summary: dict[str, Any]
    provenance: dict[str, Any]
    input_manifest: dict[str, Any]

    @property
    def event_count(self) -> int:
        return int(len(np.unique(self.event_number)))

    @property
    def cadence_count(self) -> int:
        return int(len(self.time))


@dataclass(frozen=True)
class TimingReference:
    """Internal mid-mission timing reference used by Phase 1C."""

    period_reference: float
    original_epoch_reference: float
    mid_epoch_reference: float
    mid_epoch_cycle: int
    period_half_width: float
    mid_epoch_half_width: float


@dataclass(frozen=True)
class PhysicalSample:
    """Physical transit parameters derived from transformed coordinates."""

    rp: float
    a: float
    b: float
    q1: float
    q2: float
    jitter: float
    period: float
    mid_epoch: float
    original_epoch: float
