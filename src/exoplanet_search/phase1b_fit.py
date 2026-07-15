"""Deterministic multi-start optimization for Phase 1B."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from .phase1b_model import (
    PhysicalParameters,
    batman_flux,
    inclination_degrees,
    q_to_u,
    solve_local_baselines,
    validate_physical,
)
from .phase1b_types import FitData, LimbDarkeningInputs, Phase1BConfig


@dataclass(frozen=True)
class FitResult:
    """Result of one deterministic fit stage."""

    stage: str
    parameters: dict[str, float]
    objective_value: float
    optimizer_success: bool
    optimizer_message: str
    baseline_rows: list[dict[str, Any]]
    residuals: np.ndarray
    transit_model: np.ndarray
    local_baseline: np.ndarray
    combined_model: np.ndarray
    multistart_rows: list[dict[str, Any]]
    warnings: list[str]


def run_fit_stage(
    data: FitData,
    limb: LimbDarkeningInputs,
    config: Phase1BConfig,
    *,
    stage: str,
    fit_timing: bool,
    fixed_limb_darkening: bool = False,
    n_starts: int | None = None,
) -> FitResult:
    """Run bounded deterministic optimization from multiple reproducible starts."""
    n_starts = n_starts or config.n_starts
    starts = initial_parameter_vectors(data, limb, config, fit_timing, fixed_limb_darkening, n_starts)
    bounds = parameter_bounds(data, config, fit_timing, fixed_limb_darkening)
    rows: list[dict[str, Any]] = []
    best = None
    for start_index, start in enumerate(starts):
        lower = np.asarray([bound[0] for bound in bounds], dtype=float)
        upper = np.asarray([bound[1] for bound in bounds], dtype=float)
        result = least_squares(
            residual_vector,
            start,
            args=(data, limb, config, fit_timing, fixed_limb_darkening),
            bounds=(lower, upper),
            method="trf",
            x_scale="jac",
            ftol=1.0e-8,
            xtol=1.0e-8,
            gtol=1.0e-8,
            max_nfev=250,
        )
        params = vector_to_params(result.x, data, fit_timing, fixed_limb_darkening, limb)
        failures = validate_physical(params)
        objective_value = objective(result.x, data, limb, config, fit_timing, fixed_limb_darkening)
        row = {
            "stage": stage,
            "start_index": start_index,
            "optimizer_success": bool(result.success),
            "termination_status": int(result.status),
            "message": str(result.message),
            "objective_value": float(objective_value),
            **parameter_output(params),
            "constraint_failures": ";".join(failures),
            "boundary_hits": ";".join(boundary_hits(result.x, bounds)),
        }
        rows.append(row)
        if result.success and not failures and np.isfinite(objective_value):
            if best is None or objective_value < best["objective_value"]:
                best = {"result": result, "objective_value": objective_value}
    if best is None:
        raise RuntimeError(f"All optimizer starts failed for {stage}.")
    best_result = best["result"]
    params = vector_to_params(best_result.x, data, fit_timing, fixed_limb_darkening, limb)
    model = evaluate_solution(data, params, config)
    warnings = _solution_warnings(rows, float(best["objective_value"]), config)
    return FitResult(
        stage=stage,
        parameters=parameter_output(params),
        objective_value=float(best["objective_value"]),
        optimizer_success=bool(best_result.success),
        optimizer_message=str(best_result.message),
        baseline_rows=model["baseline_rows"],
        residuals=model["residuals"],
        transit_model=model["transit_model"],
        local_baseline=model["local_baseline"],
        combined_model=model["combined_model"],
        multistart_rows=rows,
        warnings=warnings,
    )


def objective(
    vector: np.ndarray,
    data: FitData,
    limb: LimbDarkeningInputs,
    config: Phase1BConfig,
    fit_timing: bool,
    fixed_limb_darkening: bool,
) -> float:
    params = vector_to_params(vector, data, fit_timing, fixed_limb_darkening, limb)
    failures = validate_physical(params)
    if failures:
        return 1.0e30 + 1.0e26 * len(failures)
    try:
        model = evaluate_solution(data, params, config)
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        return 1.0e30
    sigma = np.sqrt(np.square(data.flux_err) + params.jitter**2)
    chi2 = float(np.sum(np.square(model["residuals"] / sigma) + 2.0 * np.log(sigma)))
    penalty = 0.0
    if not fixed_limb_darkening:
        penalty += ((params.q1 - limb.q1) / max(limb.q1_sigma, config.limb_darkening_prior_sigma_floor)) ** 2
        penalty += ((params.q2 - limb.q2) / max(limb.q2_sigma, config.limb_darkening_prior_sigma_floor)) ** 2
    return chi2 + float(penalty)


def residual_vector(
    vector: np.ndarray,
    data: FitData,
    limb: LimbDarkeningInputs,
    config: Phase1BConfig,
    fit_timing: bool,
    fixed_limb_darkening: bool,
) -> np.ndarray:
    """Return least-squares residuals equivalent to the deterministic objective."""
    params = vector_to_params(vector, data, fit_timing, fixed_limb_darkening, limb)
    failures = validate_physical(params)
    if failures:
        return np.full(len(data.time) + 4, 1.0e15, dtype=float)
    try:
        model = evaluate_solution(data, params, config)
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        return np.full(len(data.time) + 4, 1.0e15, dtype=float)
    sigma0 = np.maximum(data.flux_err, 1.0e-12)
    sigma = np.sqrt(np.square(sigma0) + params.jitter**2)
    residuals = [model["residuals"] / sigma]
    jitter_penalty = np.sqrt(np.maximum(0.0, 2.0 * np.log(sigma / sigma0)))
    residuals.append(jitter_penalty)
    if not fixed_limb_darkening:
        residuals.append(
            np.asarray(
                [
                    (params.q1 - limb.q1)
                    / max(limb.q1_sigma, config.limb_darkening_prior_sigma_floor),
                    (params.q2 - limb.q2)
                    / max(limb.q2_sigma, config.limb_darkening_prior_sigma_floor),
                ],
                dtype=float,
            )
        )
    return np.concatenate(residuals)


def evaluate_solution(data: FitData, params: PhysicalParameters, config: Phase1BConfig) -> dict[str, Any]:
    period = float(params.period if params.period is not None else data.phase1a_period_days)
    t0 = float(params.t0 if params.t0 is not None else data.phase1a_transit_time)
    transit_model = batman_flux(
        data.time,
        data.exposure_days,
        rp=params.rp,
        a=params.a,
        b=params.b,
        q1=params.q1,
        q2=params.q2,
        period=period,
        t0=t0,
        supersample_factor=config.supersample_factor,
    )
    sigma = np.sqrt(np.square(data.flux_err) + params.jitter**2)
    baseline, combined, baseline_rows = solve_local_baselines(
        data.time,
        data.flux,
        sigma,
        data.event_number,
        data.predicted_center,
        transit_model,
    )
    residuals = data.flux - combined
    return {
        "transit_model": transit_model,
        "local_baseline": baseline,
        "combined_model": combined,
        "residuals": residuals,
        "baseline_rows": baseline_rows,
    }


def initial_parameter_vectors(
    data: FitData,
    limb: LimbDarkeningInputs,
    config: Phase1BConfig,
    fit_timing: bool,
    fixed_limb_darkening: bool,
    n_starts: int,
) -> list[np.ndarray]:
    depth = max(estimate_transit_depth(data), 1.0e-5)
    rp0 = float(np.clip(np.sqrt(abs(depth)), 0.015, 0.2))
    starts: list[np.ndarray] = []
    rng = np.random.default_rng(config.random_seed + (11 if fit_timing else 0) + (23 if fixed_limb_darkening else 0))
    b_values = [0.2, 0.55, 0.8]
    for index in range(n_starts):
        if index == 0:
            rp = rp0
            b = 0.35
        else:
            rp = float(rng.uniform(max(0.01, 0.5 * rp0), min(0.25, 2.5 * rp0 + 0.02)))
            b = float(b_values[index % len(b_values)] + rng.normal(0.0, 0.05))
            b = float(np.clip(b, 0.02, 0.95))
        a = duration_informed_a(data.phase1a_period_days, data.phase1a_duration_days, rp, b)
        q1 = limb.q1 if index == 0 else float(np.clip(rng.normal(limb.q1, 0.16), 0.02, 0.98))
        q2 = limb.q2 if index == 0 else float(np.clip(rng.normal(limb.q2, 0.16), 0.02, 0.98))
        jitter = 0.0 if index == 0 else float(rng.uniform(0.0, min(0.004, config.jitter_upper_bound)))
        values = [rp, a, b]
        if not fixed_limb_darkening:
            values.extend([q1, q2])
        values.append(jitter)
        if fit_timing:
            values.extend([data.phase1a_period_days, data.phase1a_transit_time])
        starts.append(np.asarray(values, dtype=float))
    return starts


def duration_informed_a(period: float, duration: float, rp: float, b: float) -> float:
    sine = np.sin(np.pi * max(duration, 1.0e-4) / max(period, duration + 1.0e-4))
    chord = max((1.0 + rp) ** 2 - b**2, 1.0e-4) ** 0.5
    return float(np.clip(chord / max(sine, 1.0e-3), 1.0 + rp + 1.0e-3, 80.0))


def estimate_transit_depth(data: FitData) -> float:
    """Estimate a generic starting depth from recovered Phase 1A timing."""
    phase = ((data.time - data.phase1a_transit_time + 0.5 * data.phase1a_period_days) % data.phase1a_period_days) - (
        0.5 * data.phase1a_period_days
    )
    in_mask = np.abs(phase) <= 0.5 * data.phase1a_duration_days
    baseline_mask = np.abs(phase) >= 0.75 * data.phase1a_duration_days
    if np.count_nonzero(in_mask) == 0 or np.count_nonzero(baseline_mask) == 0:
        return float(np.nanmedian(1.0 - data.flux))
    return float(np.nanmedian(data.flux[baseline_mask]) - np.nanmedian(data.flux[in_mask]))


def parameter_bounds(
    data: FitData,
    config: Phase1BConfig,
    fit_timing: bool,
    fixed_limb_darkening: bool,
) -> list[tuple[float, float]]:
    bounds: list[tuple[float, float]] = [(0.001, 0.35), (1.01, 100.0), (0.0, 1.3)]
    if not fixed_limb_darkening:
        bounds.extend([(0.0, 1.0), (0.0, 1.0)])
    bounds.append((0.0, config.jitter_upper_bound))
    if fit_timing:
        period_half_width = _phase1a_period_half_width(data)
        t0_half_width = config.timing_refinement_t0_half_width_duration_scale * data.phase1a_duration_days
        bounds.extend(
            [
                (data.phase1a_period_days - period_half_width, data.phase1a_period_days + period_half_width),
                (data.phase1a_transit_time - t0_half_width, data.phase1a_transit_time + t0_half_width),
            ]
        )
    return bounds


def vector_to_params(
    vector: np.ndarray,
    data: FitData,
    fit_timing: bool,
    fixed_limb_darkening: bool,
    limb: LimbDarkeningInputs,
) -> PhysicalParameters:
    index = 0
    rp, a, b = (float(vector[0]), float(vector[1]), float(vector[2]))
    index = 3
    if fixed_limb_darkening:
        q1, q2 = limb.q1, limb.q2
    else:
        q1, q2 = float(vector[index]), float(vector[index + 1])
        index += 2
    jitter = float(vector[index])
    index += 1
    period = data.phase1a_period_days
    t0 = data.phase1a_transit_time
    if fit_timing:
        period, t0 = float(vector[index]), float(vector[index + 1])
    return PhysicalParameters(rp=rp, a=a, b=b, q1=q1, q2=q2, jitter=jitter, period=period, t0=t0)


def parameter_output(params: PhysicalParameters) -> dict[str, float]:
    u1, u2 = q_to_u(params.q1, params.q2)
    return {
        "rp_over_rstar": float(params.rp),
        "a_over_rstar": float(params.a),
        "impact_parameter": float(params.b),
        "inclination_degrees": inclination_degrees(params.a, params.b),
        "q1": float(params.q1),
        "q2": float(params.q2),
        "u1": float(u1),
        "u2": float(u2),
        "white_noise_jitter": float(params.jitter),
        "period_days": float(params.period) if params.period is not None else float("nan"),
        "transit_time": float(params.t0) if params.t0 is not None else float("nan"),
    }


def boundary_hits(vector: np.ndarray, bounds: list[tuple[float, float]]) -> list[str]:
    names = ["rp", "a", "b", "q1", "q2", "jitter", "period", "t0"]
    hits: list[str] = []
    offset = len(bounds) - len(vector)
    del offset
    for index, (value, bound) in enumerate(zip(vector, bounds, strict=False)):
        name = names[index] if index < len(names) else f"param_{index}"
        if np.isclose(value, bound[0], rtol=0.0, atol=1.0e-6):
            hits.append(f"{name}_lower")
        if np.isclose(value, bound[1], rtol=0.0, atol=1.0e-6):
            hits.append(f"{name}_upper")
    return hits


def _phase1a_period_half_width(data: FitData) -> float:
    return max(0.02 * data.phase1a_period_days, data.phase1a_duration_days)


def _solution_warnings(
    rows: list[dict[str, Any]],
    best_objective: float,
    config: Phase1BConfig,
) -> list[str]:
    warnings: list[str] = []
    failed = [row for row in rows if not row["optimizer_success"]]
    if failed:
        warnings.append(f"{len(failed)}_optimizer_starts_failed")
    competitors = [
        row
        for row in rows
        if row["optimizer_success"]
        and row["objective_value"] <= best_objective + config.near_equal_objective_delta
    ]
    if len(competitors) > 1:
        rp_values = np.asarray([row["rp_over_rstar"] for row in competitors], dtype=float)
        if np.nanmax(rp_values) - np.nanmin(rp_values) > 0.01:
            warnings.append("materially_different_near_equal_multistart_solutions")
    if any(row["boundary_hits"] for row in rows):
        warnings.append("one_or_more_optimizer_solutions_hit_bounds")
    return warnings
