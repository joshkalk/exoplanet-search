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

DIAGNOSTIC_METHODOLOGY_VERSION = "phase1c_emcee_ensemble_state_v2"


@dataclass(frozen=True)
class Phase1CConfig:
    """Primary Phase 1C settings for frozen Phase 1B posterior sampling."""

    phase1b_output_dir: Path = Path("data/interim/kepler5_phase1b_fit")
    output_dir: Path = Path("data/interim/kepler5_phase1c_posterior")
    run_id: str | None = None
    random_seed: int = 20260715
    n_ensembles: int = 4
    ensemble_processes: int = 1
    n_walkers: int = 24
    pilot_steps: int = 24
    synthetic_steps: int = 80
    synthetic_recovery_steps: int = 2000
    production_steps: int = 2000
    target_total_steps: int | None = None
    additional_steps: int | None = None
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
    convergence_stability_chunks: int = 3
    convergence_stability_sigma_threshold: float = 0.25
    convergence_ensemble_shift_threshold: float = 3.0
    convergence_interval_overlap_minimum: float = 0.25
    convergence_tail_interval_overlap_minimum: float = 0.25
    convergence_ensemble_scale_ratio_max: float = 3.0
    autocorrelation_min_usable_walkers: int = 2
    max_pilot_seconds: float = 600.0
    minimum_meaningful_summary_draws: int = 1000
    diagnostic_methodology_version: str = DIAGNOSTIC_METHODOLOGY_VERSION
    prior_informed_pool_size: int = 1024
    prior_informed_max_pool_size: int = 8192
    prior_informed_pool_growth_factor: int = 2
    prior_informed_pool_scale_multiplier: float = 0.8
    prior_informed_elite_size: int = 16
    prior_informed_min_finite_candidates: int = 8
    prior_informed_max_logp_deficit: float = 30.0
    local_tight_scales: tuple[float, ...] = (0.015, 0.02, 0.015, 0.015, 0.015, 0.05, 1.0e-5, 2.0e-4)
    local_moderate_scales: tuple[float, ...] = (0.04, 0.05, 0.04, 0.04, 0.04, 0.12, 5.0e-5, 1.0e-3)
    local_broad_scales: tuple[float, ...] = (0.08, 0.10, 0.08, 0.08, 0.08, 0.25, 2.0e-4, 5.0e-3)
    prior_informed_cloud_scales: tuple[float, ...] = (
        0.06,
        0.075,
        0.06,
        0.06,
        0.06,
        0.18,
        1.0e-4,
        2.0e-3,
    )
    severe_walker_acceptance_max: float = 0.05
    severe_walker_repeated_fraction_min: float = 0.95
    severe_walker_logp_deficit_min: float = 100.0
    severe_walker_final_distance_min: float = 25.0

    def __post_init__(self) -> None:
        if self.diagnostic_methodology_version != DIAGNOSTIC_METHODOLOGY_VERSION:
            raise ValueError(
                "diagnostic_methodology_version must equal "
                f"{DIAGNOSTIC_METHODOLOGY_VERSION!r}."
            )
        if int(self.autocorrelation_min_usable_walkers) < 2:
            raise ValueError("autocorrelation_min_usable_walkers must be at least 2.")
        ensemble_processes = _require_integer(self.ensemble_processes, "ensemble_processes")
        if ensemble_processes <= 0:
            raise ValueError("ensemble_processes must be positive.")
        if ensemble_processes > int(self.n_ensembles):
            raise ValueError("ensemble_processes cannot exceed n_ensembles.")
        initial_pool_size = _require_integer(self.prior_informed_pool_size, "prior_informed_pool_size")
        max_pool_size = _require_integer(
            self.prior_informed_max_pool_size,
            "prior_informed_max_pool_size",
        )
        growth_factor = _require_integer(
            self.prior_informed_pool_growth_factor,
            "prior_informed_pool_growth_factor",
        )
        required_candidates = _require_integer(
            self.prior_informed_min_finite_candidates,
            "prior_informed_min_finite_candidates",
        )
        elite_size = _require_integer(self.prior_informed_elite_size, "prior_informed_elite_size")
        if initial_pool_size <= 0:
            raise ValueError("prior_informed_pool_size must be positive.")
        if max_pool_size < initial_pool_size:
            raise ValueError("prior_informed_max_pool_size must be at least prior_informed_pool_size.")
        if growth_factor < 2:
            raise ValueError("prior_informed_pool_growth_factor must be an integer at least 2.")
        if required_candidates <= 0:
            raise ValueError("prior_informed_min_finite_candidates must be positive.")
        if elite_size <= 0:
            raise ValueError("prior_informed_elite_size must be positive.")
        if float(self.prior_informed_pool_scale_multiplier) <= 0.0:
            raise ValueError("prior_informed_pool_scale_multiplier must be positive.")
        if (
            not np.isfinite(float(self.prior_informed_max_logp_deficit))
            or float(self.prior_informed_max_logp_deficit) <= 0.0
        ):
            raise ValueError("prior_informed_max_logp_deficit must be finite and positive.")

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
            "diagnostic_methodology": DIAGNOSTIC_METHODOLOGY_VERSION,
            "prior_informed_initialization": (
                "Broad prior pool is posterior-screened to select one remote anchor, then walkers "
                "start as a coherent local cloud around that anchor."
            ),
        }
        return values


def _require_integer(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer.")
    try:
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer.") from exc
    if integer != value:
        raise ValueError(f"{field_name} must be an integer.")
    return integer


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
