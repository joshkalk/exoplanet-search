"""BATMAN model evaluation, physical constraints, and local-baseline solving."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import batman
import numpy as np


@dataclass(frozen=True)
class PhysicalParameters:
    """Circular-transit parameters optimized by Phase 1B."""

    rp: float
    a: float
    b: float
    q1: float
    q2: float
    jitter: float = 0.0
    period: float | None = None
    t0: float | None = None


def q_to_u(q1: float, q2: float) -> tuple[float, float]:
    """Convert Kipping q1/q2 to quadratic limb-darkening coefficients."""
    root = np.sqrt(float(q1))
    return float(2.0 * root * q2), float(root * (1.0 - 2.0 * q2))


def u_to_q(u1: float, u2: float) -> tuple[float, float]:
    """Convert quadratic coefficients to Kipping q1/q2."""
    total = float(u1) + float(u2)
    if total <= 0.0:
        raise ValueError("Quadratic coefficients cannot be converted to physical q1/q2.")
    return float(total**2), float(u1 / (2.0 * total))


def inclination_degrees(a: float, b: float) -> float:
    """Return circular-orbit inclination from a/Rstar and impact parameter."""
    ratio = float(b) / float(a)
    if ratio < 0.0 or ratio > 1.0:
        raise ValueError("Impact parameter and a/Rstar imply invalid inclination.")
    return float(np.degrees(np.arccos(ratio)))


def validate_physical(params: PhysicalParameters) -> list[str]:
    """Return constraint violations for the physical transit model."""
    failures: list[str] = []
    if not np.isfinite(params.rp) or params.rp <= 0.0:
        failures.append("rp_must_be_positive")
    if not np.isfinite(params.a) or params.a <= 1.0 + max(params.rp, 0.0):
        failures.append("a_over_rstar_must_exceed_1_plus_rp")
    if not np.isfinite(params.b) or params.b < 0.0 or params.b >= 1.0 + max(params.rp, 0.0):
        failures.append("impact_parameter_outside_transiting_range")
    if np.isfinite(params.a) and np.isfinite(params.b) and params.a > 0 and params.b / params.a > 1.0:
        failures.append("inclination_invalid")
    if not np.isfinite(params.q1) or not 0.0 <= params.q1 <= 1.0:
        failures.append("q1_outside_unit_interval")
    if not np.isfinite(params.q2) or not 0.0 <= params.q2 <= 1.0:
        failures.append("q2_outside_unit_interval")
    if not np.isfinite(params.jitter) or params.jitter < 0.0:
        failures.append("jitter_must_be_nonnegative")
    return failures


def batman_flux(
    time: np.ndarray,
    exposure_days: np.ndarray,
    *,
    rp: float,
    a: float,
    b: float,
    q1: float,
    q2: float,
    period: float,
    t0: float,
    supersample_factor: int,
) -> np.ndarray:
    """Evaluate an exposure-integrated circular BATMAN transit model."""
    params = PhysicalParameters(rp=rp, a=a, b=b, q1=q1, q2=q2)
    failures = validate_physical(params)
    if failures:
        raise ValueError(";".join(failures))
    u1, u2 = q_to_u(q1, q2)
    model_flux = np.empty_like(time, dtype=float)
    for exposure in np.unique(np.asarray(exposure_days, dtype=float)):
        mask = np.isclose(exposure_days, exposure, rtol=0.0, atol=max(abs(exposure) * 1.0e-10, 1.0e-12))
        transit_params = batman.TransitParams()
        transit_params.t0 = float(t0)
        transit_params.per = float(period)
        transit_params.rp = float(rp)
        transit_params.a = float(a)
        transit_params.inc = inclination_degrees(a, b)
        transit_params.ecc = 0.0
        transit_params.w = 90.0
        transit_params.u = [u1, u2]
        transit_params.limb_dark = "quadratic"
        model = batman.TransitModel(
            transit_params,
            np.asarray(time[mask], dtype=float),
            supersample_factor=int(supersample_factor),
            exp_time=float(exposure),
        )
        model_flux[mask] = model.light_curve(transit_params)
    return model_flux


def solve_local_baselines(
    time: np.ndarray,
    flux: np.ndarray,
    sigma: np.ndarray,
    event_number: np.ndarray,
    centers: np.ndarray,
    transit_model: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Solve one multiplicative linear baseline per transit window."""
    baseline = np.empty_like(flux, dtype=float)
    combined = np.empty_like(flux, dtype=float)
    rows: list[dict[str, Any]] = []
    for event in np.unique(event_number):
        mask = event_number == event
        center = float(np.nanmedian(centers[mask]))
        x = time[mask] - center
        design = np.column_stack([transit_model[mask], transit_model[mask] * x])
        weights = 1.0 / np.square(np.maximum(sigma[mask], 1.0e-12))
        lhs = design.T @ (weights[:, None] * design)
        rhs = design.T @ (weights * flux[mask])
        try:
            coeff = np.linalg.solve(lhs, rhs)
            status = "solved"
        except np.linalg.LinAlgError:
            coeff = np.linalg.lstsq(weights[:, None] ** 0.5 * design, weights**0.5 * flux[mask], rcond=None)[0]
            status = "least_squares_fallback"
        local_baseline = coeff[0] + coeff[1] * x
        baseline[mask] = local_baseline
        combined[mask] = local_baseline * transit_model[mask]
        rows.append(
            {
                "event_number": int(event),
                "predicted_center": center,
                "baseline_intercept": float(coeff[0]),
                "baseline_slope_per_day": float(coeff[1]),
                "cadence_count": int(np.count_nonzero(mask)),
                "solve_status": status,
            }
        )
    return baseline, combined, rows

