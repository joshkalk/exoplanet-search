"""Typed configuration and data containers for Phase 1B transit fitting."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Phase1BConfig:
    """Deterministic physical-fit settings independent of published planet values."""

    phase1a_summary_path: Path = Path("data/interim/kepler5_phase1a_search/search_summary.json")
    phase1a_provenance_path: Path = Path(
        "data/interim/kepler5_phase1a_search/provenance_manifest.json"
    )
    stellar_inputs_path: Path = Path("data/interim/kepler5_phase1b_stellar_inputs.json")
    output_dir: Path = Path("data/interim/kepler5_phase1b_fit")
    cadence: str = "long"
    random_seed: int = 481516
    n_starts: int = 14
    test_n_starts: int = 5
    window_half_width_duration_scale: float = 3.0
    window_period_cap_fraction: float = 0.49
    baseline_inner_duration_scale: float = 0.75
    minimum_in_transit_points: int = 3
    minimum_points_each_side_in_transit: int = 1
    minimum_pre_baseline_points: int = 8
    minimum_post_baseline_points: int = 8
    supersample_factor: int = 11
    high_supersample_factor: int = 21
    limb_darkening_prior_sigma_floor: float = 0.08
    timing_refinement_t0_half_width_duration_scale: float = 0.5
    jitter_upper_bound: float = 0.02
    near_equal_objective_delta: float = 2.0

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        for key, value in list(values.items()):
            if isinstance(value, Path):
                values[key] = str(value)
        return values


@dataclass(frozen=True)
class ObservationSet:
    """Accepted unbinned cadences with source-product metadata."""

    time: np.ndarray
    flux: np.ndarray
    flux_err: np.ndarray
    product_id: np.ndarray
    quarter: np.ndarray
    cadence: np.ndarray
    exposure_days: np.ndarray
    mission: np.ndarray
    flux_product: np.ndarray

    def __len__(self) -> int:
        return int(len(self.time))


@dataclass(frozen=True)
class TransitWindows:
    """Window audit table plus accepted cadence mapping."""

    audit_rows: list[dict[str, Any]]
    accepted_mask: np.ndarray
    event_number: np.ndarray
    predicted_center: np.ndarray


@dataclass(frozen=True)
class LimbDarkeningInputs:
    """Reproducible quadratic limb-darkening input package."""

    q1: float
    q2: float
    q1_sigma: float
    q2_sigma: float
    u1: float
    u2: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FitData:
    """The cadences passed to deterministic fitting."""

    observations: ObservationSet
    time: np.ndarray
    flux: np.ndarray
    flux_err: np.ndarray
    event_number: np.ndarray
    predicted_center: np.ndarray
    exposure_days: np.ndarray
    product_id: np.ndarray
    quarter: np.ndarray
    phase1a_period_days: float
    phase1a_transit_time: float
    phase1a_duration_days: float

