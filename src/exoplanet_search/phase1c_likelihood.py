"""Gaussian Phase 1C likelihood with exact local-baseline marginalization."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import batman
import numpy as np

from .phase1b_model import PhysicalParameters, batman_flux, inclination_degrees, q_to_u, validate_physical
from .phase1c_parameters import log_prior, vector_to_physical
from .phase1c_types import FrozenPhase1BData, PARAMETER_ORDER, Phase1CConfig, PhysicalSample, TimingReference


@dataclass(frozen=True)
class EventLikelihoodResult:
    """Marginalized likelihood and conditional baseline posterior for one event."""

    log_likelihood: float
    baseline_mean: np.ndarray
    baseline_covariance: np.ndarray


@dataclass
class PosteriorProfiler:
    """Mutable counters for direct log-posterior instrumentation."""

    posterior_calls: int = 0
    prior_seconds: float = 0.0
    batman_seconds: float = 0.0
    baseline_likelihood_seconds: float = 0.0
    total_seconds: float = 0.0
    invalid_prior_count: int = 0
    invalid_likelihood_count: int = 0

    def summary(self) -> dict[str, float | int]:
        return {
            "posterior_calls": int(self.posterior_calls),
            "prior_transform_seconds": float(self.prior_seconds),
            "batman_model_seconds": float(self.batman_seconds),
            "marginalized_baseline_likelihood_seconds": float(self.baseline_likelihood_seconds),
            "total_log_posterior_seconds": float(self.total_seconds),
            "invalid_prior_count": int(self.invalid_prior_count),
            "invalid_likelihood_count": int(self.invalid_likelihood_count),
        }

    def add(self, other: "PosteriorProfiler") -> None:
        self.posterior_calls += other.posterior_calls
        self.prior_seconds += other.prior_seconds
        self.batman_seconds += other.batman_seconds
        self.baseline_likelihood_seconds += other.baseline_likelihood_seconds
        self.total_seconds += other.total_seconds
        self.invalid_prior_count += other.invalid_prior_count
        self.invalid_likelihood_count += other.invalid_likelihood_count


@dataclass(frozen=True)
class EventLikelihoodContext:
    """Immutable, event-local data used by every likelihood evaluation."""

    event_number: int
    data_slice: slice | None
    data_indices: np.ndarray
    time: np.ndarray
    flux: np.ndarray
    flux_uncertainty_squared: np.ndarray
    local_coordinate: np.ndarray
    frozen_center: float
    cadence_count: int
    is_contiguous: bool

    def select_model(self, transit_model: np.ndarray) -> np.ndarray:
        """Return this event's model values without rebuilding an event mask."""
        if self.data_slice is not None:
            return transit_model[self.data_slice]
        return transit_model[self.data_indices]


@dataclass(frozen=True)
class ExposureGroupContext:
    """Fixed exposure-duration group for BATMAN exposure integration."""

    exposure_days: float
    data_slice: slice | None
    data_indices: np.ndarray
    time: np.ndarray
    cadence_count: int


@dataclass(frozen=True)
class Phase1CLikelihoodContext:
    """Immutable precomputed data context for the Phase 1C posterior path."""

    data: FrozenPhase1BData
    config: Phase1CConfig
    timing: TimingReference
    time: np.ndarray
    exposure_days: np.ndarray
    event_number: np.ndarray
    event_numbers: np.ndarray
    events: tuple[EventLikelihoodContext, ...]
    event_index: tuple[tuple[int, int], ...]
    exposure_groups: tuple[ExposureGroupContext, ...]
    exposure_groups_safe: bool
    cadence_count: int
    event_count: int
    parameter_index: tuple[tuple[str, int], ...]
    baseline_mean: np.ndarray
    baseline_variance: np.ndarray
    baseline_precision: np.ndarray
    baseline_logdet: float
    baseline_intercept_sigma: float
    baseline_slope_sigma: float

    @classmethod
    def from_data(
        cls,
        data: FrozenPhase1BData,
        config: Phase1CConfig,
        timing: TimingReference,
    ) -> "Phase1CLikelihoodContext":
        """Build a read-only likelihood context from frozen Phase 1B data."""
        return build_likelihood_context(data, config, timing)


def build_likelihood_context(
    data: FrozenPhase1BData,
    config: Phase1CConfig,
    timing: TimingReference,
) -> Phase1CLikelihoodContext:
    """Precompute static Phase 1C likelihood quantities once per sampler run."""
    time_array = _readonly_array(data.time, dtype=float)
    flux_array = _readonly_array(data.flux, dtype=float)
    flux_uncertainty_array = _readonly_array(data.flux_uncertainty, dtype=float)
    exposure_days = _readonly_array(data.exposure_days, dtype=float)
    event_number = _readonly_array(data.event_number, dtype=int)
    predicted_center = _readonly_array(data.predicted_center, dtype=float)
    _validate_data_arrays(time_array, flux_array, flux_uncertainty_array, exposure_days, event_number, predicted_center)
    if config.baseline_intercept_sigma <= 0.0 or config.baseline_slope_sigma <= 0.0:
        raise ValueError("Baseline prior widths must be positive.")

    event_numbers = _readonly_array(np.unique(event_number).astype(int), dtype=int)
    events: list[EventLikelihoodContext] = []
    covered_indices: list[np.ndarray] = []
    for event in event_numbers:
        indices = np.flatnonzero(event_number == int(event))
        if indices.size == 0:
            raise ValueError(f"Phase 1C event {int(event)} has no cadences.")
        data_slice, is_contiguous = _slice_if_contiguous(indices, event_number, int(event))
        event_time = _readonly_array(_select(time_array, data_slice, indices), dtype=float)
        event_flux = _readonly_array(_select(flux_array, data_slice, indices), dtype=float)
        event_sigma_sq = _readonly_array(np.square(_select(flux_uncertainty_array, data_slice, indices)), dtype=float)
        center = float(np.median(_select(predicted_center, data_slice, indices)))
        if not np.isfinite(center):
            raise ValueError(f"Phase 1C event {int(event)} has a nonfinite frozen center.")
        local_coordinate = _readonly_array(event_local_coordinate(event_time, center), dtype=float)
        events.append(
            EventLikelihoodContext(
                event_number=int(event),
                data_slice=data_slice,
                data_indices=_readonly_array(indices, dtype=int),
                time=event_time,
                flux=event_flux,
                flux_uncertainty_squared=event_sigma_sq,
                local_coordinate=local_coordinate,
                frozen_center=center,
                cadence_count=int(indices.size),
                is_contiguous=bool(is_contiguous),
            )
        )
        covered_indices.append(indices)
    _validate_event_partition(covered_indices, time_array.size)

    exposure_groups, exposure_groups_safe = _build_exposure_groups(time_array, exposure_days)
    baseline_variance = _readonly_array(
        np.asarray([config.baseline_intercept_sigma**2, config.baseline_slope_sigma**2], dtype=float)
    )
    baseline_precision = _readonly_array(1.0 / baseline_variance, dtype=float)
    baseline_mean = _readonly_array(np.asarray([1.0, 0.0], dtype=float))
    return Phase1CLikelihoodContext(
        data=data,
        config=config,
        timing=timing,
        time=time_array,
        exposure_days=exposure_days,
        event_number=event_number,
        event_numbers=event_numbers,
        events=tuple(events),
        event_index=tuple((event.event_number, index) for index, event in enumerate(events)),
        exposure_groups=tuple(exposure_groups),
        exposure_groups_safe=bool(exposure_groups_safe),
        cadence_count=int(time_array.size),
        event_count=int(len(events)),
        parameter_index=tuple((name, index) for index, name in enumerate(PARAMETER_ORDER)),
        baseline_mean=baseline_mean,
        baseline_variance=baseline_variance,
        baseline_precision=baseline_precision,
        baseline_logdet=float(np.sum(np.log(baseline_variance))),
        baseline_intercept_sigma=float(config.baseline_intercept_sigma),
        baseline_slope_sigma=float(config.baseline_slope_sigma),
    )


def _readonly_array(values: Any, *, dtype: Any | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=dtype).copy()
    array.setflags(write=False)
    return array


def _validate_data_arrays(
    time_array: np.ndarray,
    flux_array: np.ndarray,
    flux_uncertainty_array: np.ndarray,
    exposure_days: np.ndarray,
    event_number: np.ndarray,
    predicted_center: np.ndarray,
) -> None:
    shapes = {
        "time": time_array.shape,
        "flux": flux_array.shape,
        "flux_uncertainty": flux_uncertainty_array.shape,
        "exposure_days": exposure_days.shape,
        "event_number": event_number.shape,
        "predicted_center": predicted_center.shape,
    }
    if len(set(shapes.values())) != 1:
        raise ValueError(f"Phase 1C likelihood arrays must have matching shapes: {shapes}")
    if time_array.ndim != 1:
        raise ValueError("Phase 1C likelihood arrays must be one-dimensional.")
    if time_array.size == 0:
        raise ValueError("Cannot build a Phase 1C likelihood context for empty data.")
    for name, values in (
        ("time", time_array),
        ("flux", flux_array),
        ("flux_uncertainty", flux_uncertainty_array),
        ("exposure_days", exposure_days),
        ("predicted_center", predicted_center),
    ):
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Phase 1C likelihood array {name!r} contains nonfinite values.")
    if np.any(flux_uncertainty_array <= 0.0):
        raise ValueError("Phase 1C cadence uncertainties must be positive.")
    if np.any(exposure_days <= 0.0):
        raise ValueError("Phase 1C exposure durations must be positive.")


def _slice_if_contiguous(indices: np.ndarray, event_number: np.ndarray, event: int) -> tuple[slice | None, bool]:
    if indices.size == 0:
        return None, False
    is_contiguous = bool(np.all(np.diff(indices) == 1)) if indices.size > 1 else True
    if not is_contiguous:
        return None, False
    candidate = slice(int(indices[0]), int(indices[-1]) + 1)
    if not np.all(event_number[candidate] == event):
        raise ValueError(f"Phase 1C event {event} failed contiguous-slice validation.")
    return candidate, True


def _select(values: np.ndarray, data_slice: slice | None, indices: np.ndarray) -> np.ndarray:
    if data_slice is not None:
        return values[data_slice]
    return values[indices]


def _validate_event_partition(index_blocks: list[np.ndarray], cadence_count: int) -> None:
    combined = np.concatenate(index_blocks) if index_blocks else np.asarray([], dtype=int)
    if combined.size != cadence_count:
        raise ValueError("Phase 1C event partition does not cover every cadence exactly once.")
    if not np.array_equal(np.sort(combined), np.arange(cadence_count, dtype=int)):
        raise ValueError("Phase 1C event partition has missing or duplicate cadence indices.")


def _build_exposure_groups(
    time_array: np.ndarray,
    exposure_days: np.ndarray,
) -> tuple[list[ExposureGroupContext], bool]:
    groups = []
    coverage = np.zeros(exposure_days.size, dtype=int)
    for exposure in np.unique(exposure_days):
        tolerance = max(abs(float(exposure)) * 1.0e-10, 1.0e-12)
        indices = np.flatnonzero(np.isclose(exposure_days, exposure, rtol=0.0, atol=tolerance))
        if indices.size == 0:
            continue
        coverage[indices] += 1
        group_mask = np.zeros(exposure_days.size, dtype=bool)
        group_mask[indices] = True
        data_slice, _ = _slice_if_contiguous(indices, group_mask, True)
        groups.append(
            ExposureGroupContext(
                exposure_days=float(exposure),
                data_slice=data_slice,
                data_indices=_readonly_array(indices, dtype=int),
                time=_readonly_array(_select(time_array, data_slice, indices), dtype=float),
                cadence_count=int(indices.size),
            )
        )
    safe = bool(np.all(coverage == 1))
    return groups, safe


def event_local_coordinate(time: np.ndarray, frozen_center: float) -> np.ndarray:
    """Return deterministic local coordinates scaled to approximately [-1, 1]."""
    x = np.asarray(time, dtype=float) - float(frozen_center)
    scale = float(np.max(np.abs(x))) if x.size else 0.0
    if not np.isfinite(scale) or scale <= 0.0:
        return np.zeros_like(x, dtype=float)
    return x / scale


def marginalized_event_log_likelihood(
    *,
    time: np.ndarray,
    flux: np.ndarray,
    flux_uncertainty: np.ndarray,
    transit_model: np.ndarray,
    frozen_center: float,
    jitter: float,
    baseline_intercept_sigma: float,
    baseline_slope_sigma: float,
) -> EventLikelihoodResult:
    """Evaluate the exact marginalized event likelihood using Woodbury identities."""
    y = np.asarray(flux, dtype=float)
    sigma = np.asarray(flux_uncertainty, dtype=float)
    m = np.asarray(transit_model, dtype=float)
    if y.shape != sigma.shape or y.shape != m.shape:
        raise ValueError("Event likelihood arrays must have matching shapes.")
    if y.size == 0:
        raise ValueError("Cannot evaluate event likelihood for an empty event.")
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(sigma)) or not np.all(np.isfinite(m)):
        raise ValueError("Event likelihood arrays contain nonfinite values.")
    variance = np.square(sigma) + float(jitter) ** 2
    if not np.all(np.isfinite(variance)) or np.any(variance <= 0.0):
        raise ValueError("Event likelihood variance must be finite and positive.")
    if baseline_intercept_sigma <= 0.0 or baseline_slope_sigma <= 0.0:
        raise ValueError("Baseline prior widths must be positive.")

    x = event_local_coordinate(np.asarray(time, dtype=float), frozen_center)
    design = np.column_stack([m, m * x])
    beta_mean = np.asarray([1.0, 0.0], dtype=float)
    lambda_diag = np.asarray(
        [baseline_intercept_sigma**2, baseline_slope_sigma**2],
        dtype=float,
    )
    lambda_inv_diag = 1.0 / lambda_diag
    residual = y - design @ beta_mean
    cinv = 1.0 / variance
    weighted_design = cinv[:, None] * design
    a_matrix = np.diag(lambda_inv_diag) + design.T @ weighted_design
    try:
        chol = np.linalg.cholesky(a_matrix)
        solved = np.linalg.solve(chol.T, np.linalg.solve(chol, design.T @ (cinv * residual)))
        cov = np.linalg.solve(chol.T, np.linalg.solve(chol, np.eye(2)))
    except np.linalg.LinAlgError as exc:
        raise ValueError("Baseline marginalization system is not positive definite.") from exc

    v = design.T @ (cinv * residual)
    quadratic = float(residual @ (cinv * residual) - v @ solved)
    logdet_c = float(np.sum(np.log(variance)))
    logdet_lambda = float(np.sum(np.log(lambda_diag)))
    logdet_a = float(2.0 * np.sum(np.log(np.diag(chol))))
    log_likelihood = -0.5 * (quadratic + logdet_c + logdet_lambda + logdet_a + y.size * math.log(2.0 * math.pi))
    return EventLikelihoodResult(
        log_likelihood=float(log_likelihood),
        baseline_mean=beta_mean + cov @ v,
        baseline_covariance=cov,
    )


def marginalized_event_log_likelihood_from_context(
    event: EventLikelihoodContext,
    transit_model: np.ndarray,
    jitter: float,
    context: Phase1CLikelihoodContext,
) -> EventLikelihoodResult:
    """Evaluate an event likelihood from precomputed static event arrays."""
    m = np.asarray(event.select_model(transit_model), dtype=float)
    if m.shape != event.flux.shape:
        raise ValueError("Event likelihood model has the wrong shape for the precomputed context.")
    if not np.all(np.isfinite(m)):
        raise ValueError("Event likelihood model contains nonfinite values.")
    variance = event.flux_uncertainty_squared + float(jitter) ** 2
    if not np.all(np.isfinite(variance)) or np.any(variance <= 0.0):
        raise ValueError("Event likelihood variance must be finite and positive.")

    design = np.empty((event.cadence_count, 2), dtype=float)
    design[:, 0] = m
    design[:, 1] = m * event.local_coordinate
    residual = event.flux - design @ context.baseline_mean
    cinv = 1.0 / variance
    weighted_design = cinv[:, None] * design
    a_matrix = np.diag(context.baseline_precision) + design.T @ weighted_design
    try:
        chol = np.linalg.cholesky(a_matrix)
        solved = np.linalg.solve(chol.T, np.linalg.solve(chol, design.T @ (cinv * residual)))
        cov = np.linalg.solve(chol.T, np.linalg.solve(chol, np.eye(2)))
    except np.linalg.LinAlgError as exc:
        raise ValueError("Baseline marginalization system is not positive definite.") from exc

    v = design.T @ (cinv * residual)
    quadratic = float(residual @ (cinv * residual) - v @ solved)
    logdet_c = float(np.sum(np.log(variance)))
    logdet_a = float(2.0 * np.sum(np.log(np.diag(chol))))
    log_likelihood = -0.5 * (
        quadratic
        + logdet_c
        + context.baseline_logdet
        + logdet_a
        + event.cadence_count * math.log(2.0 * math.pi)
    )
    return EventLikelihoodResult(
        log_likelihood=float(log_likelihood),
        baseline_mean=context.baseline_mean + cov @ v,
        baseline_covariance=cov,
    )


def transit_model_for_vector(
    vector: np.ndarray,
    data: FrozenPhase1BData,
    config: Phase1CConfig,
    timing: TimingReference,
) -> np.ndarray:
    """Evaluate the reused Phase 1B BATMAN model for a transformed vector."""
    sample = vector_to_physical(vector, timing)
    return batman_flux(
        data.time,
        data.exposure_days,
        rp=sample.rp,
        a=sample.a,
        b=sample.b,
        q1=sample.q1,
        q2=sample.q2,
        period=sample.period,
        t0=sample.original_epoch,
        supersample_factor=config.supersample_factor,
    )


def transit_model_for_sample(
    sample: PhysicalSample,
    context: Phase1CLikelihoodContext,
) -> np.ndarray:
    """Evaluate the Phase 1B BATMAN model for a precomputed likelihood context."""
    if not context.exposure_groups_safe:
        return batman_flux(
            context.time,
            context.exposure_days,
            rp=sample.rp,
            a=sample.a,
            b=sample.b,
            q1=sample.q1,
            q2=sample.q2,
            period=sample.period,
            t0=sample.original_epoch,
            supersample_factor=context.config.supersample_factor,
        )
    params = PhysicalParameters(rp=sample.rp, a=sample.a, b=sample.b, q1=sample.q1, q2=sample.q2)
    failures = validate_physical(params)
    if failures:
        raise ValueError(";".join(failures))
    u1, u2 = q_to_u(sample.q1, sample.q2)
    inclination = inclination_degrees(sample.a, sample.b)
    model_flux = np.empty(context.cadence_count, dtype=float)
    for group in context.exposure_groups:
        transit_params = _batman_params(sample, u1, u2, inclination)
        model = batman.TransitModel(
            transit_params,
            group.time,
            supersample_factor=int(context.config.supersample_factor),
            exp_time=float(group.exposure_days),
        )
        group_flux = model.light_curve(transit_params)
        if group.data_slice is not None:
            model_flux[group.data_slice] = group_flux
        else:
            model_flux[group.data_indices] = group_flux
    return model_flux


def _batman_params(sample: PhysicalSample, u1: float, u2: float, inclination: float) -> batman.TransitParams:
    transit_params = batman.TransitParams()
    transit_params.t0 = float(sample.original_epoch)
    transit_params.per = float(sample.period)
    transit_params.rp = float(sample.rp)
    transit_params.a = float(sample.a)
    transit_params.inc = float(inclination)
    transit_params.ecc = 0.0
    transit_params.w = 90.0
    transit_params.u = [float(u1), float(u2)]
    transit_params.limb_dark = "quadratic"
    return transit_params


def reference_log_likelihood(
    vector: np.ndarray,
    data: FrozenPhase1BData,
    config: Phase1CConfig,
    timing: TimingReference,
) -> float:
    """Reference likelihood retaining the original mask-based event loop."""
    sample = vector_to_physical(vector, timing)
    try:
        transit_model = transit_model_for_vector(vector, data, config, timing)
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        return -math.inf
    total = 0.0
    for event in np.unique(data.event_number):
        mask = data.event_number == event
        result = marginalized_event_log_likelihood(
            time=data.time[mask],
            flux=data.flux[mask],
            flux_uncertainty=data.flux_uncertainty[mask],
            transit_model=transit_model[mask],
            frozen_center=float(np.median(data.predicted_center[mask])),
            jitter=sample.jitter,
            baseline_intercept_sigma=config.baseline_intercept_sigma,
            baseline_slope_sigma=config.baseline_slope_sigma,
        )
        total += result.log_likelihood
    return float(total)


def log_likelihood_with_context(
    vector: np.ndarray,
    context: Phase1CLikelihoodContext,
    *,
    sample: PhysicalSample | None = None,
) -> float:
    """Return the context-backed marginalized Gaussian log likelihood."""
    physical = vector_to_physical(vector, context.timing) if sample is None else sample
    try:
        transit_model = transit_model_for_sample(physical, context)
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        return -math.inf
    total = 0.0
    try:
        for event in context.events:
            total += marginalized_event_log_likelihood_from_context(
                event,
                transit_model,
                physical.jitter,
                context,
            ).log_likelihood
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        return -math.inf
    return float(total)


def log_likelihood(
    vector: np.ndarray,
    data: FrozenPhase1BData,
    config: Phase1CConfig,
    timing: TimingReference,
) -> float:
    """Return the full marginalized Gaussian log likelihood summed over events."""
    return log_likelihood_with_context(vector, Phase1CLikelihoodContext.from_data(data, config, timing))


def reference_log_probability(
    vector: np.ndarray,
    data: FrozenPhase1BData,
    config: Phase1CConfig,
    timing: TimingReference,
) -> float:
    """Reference log posterior retaining the original mask-based event loop."""
    prior = log_prior(vector, data, config, timing)
    if not np.isfinite(prior):
        return -math.inf
    likelihood = reference_log_likelihood(vector, data, config, timing)
    if not np.isfinite(likelihood):
        return -math.inf
    return float(prior + likelihood)


def log_probability_with_context(vector: np.ndarray, context: Phase1CLikelihoodContext) -> float:
    """Return normalized log posterior density using a precomputed likelihood context."""
    prior = log_prior(vector, context.data, context.config, context.timing)
    if not np.isfinite(prior):
        return -math.inf
    likelihood = log_likelihood_with_context(vector, context)
    if not np.isfinite(likelihood):
        return -math.inf
    return float(prior + likelihood)


def log_probability(
    vector: np.ndarray,
    data: FrozenPhase1BData,
    config: Phase1CConfig,
    timing: TimingReference,
) -> float:
    """Return normalized log posterior density in transformed coordinates."""
    return log_probability_with_context(vector, Phase1CLikelihoodContext.from_data(data, config, timing))


def reference_profiled_log_probability(
    vector: np.ndarray,
    data: FrozenPhase1BData,
    config: Phase1CConfig,
    timing: TimingReference,
    profiler: PosteriorProfiler,
) -> float:
    """Reference profiled log posterior retaining the original mask-based event loop."""
    profiler.posterior_calls += 1
    total_start = time.perf_counter()
    prior_start = time.perf_counter()
    prior = log_prior(vector, data, config, timing)
    profiler.prior_seconds += time.perf_counter() - prior_start
    if not np.isfinite(prior):
        profiler.invalid_prior_count += 1
        profiler.total_seconds += time.perf_counter() - total_start
        return -math.inf

    batman_start = time.perf_counter()
    try:
        sample = vector_to_physical(vector, timing)
        transit_model = transit_model_for_vector(vector, data, config, timing)
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        profiler.invalid_likelihood_count += 1
        profiler.batman_seconds += time.perf_counter() - batman_start
        profiler.total_seconds += time.perf_counter() - total_start
        return -math.inf
    profiler.batman_seconds += time.perf_counter() - batman_start

    likelihood_start = time.perf_counter()
    total = 0.0
    try:
        for event in np.unique(data.event_number):
            mask = data.event_number == event
            result = marginalized_event_log_likelihood(
                time=data.time[mask],
                flux=data.flux[mask],
                flux_uncertainty=data.flux_uncertainty[mask],
                transit_model=transit_model[mask],
                frozen_center=float(np.median(data.predicted_center[mask])),
                jitter=sample.jitter,
                baseline_intercept_sigma=config.baseline_intercept_sigma,
                baseline_slope_sigma=config.baseline_slope_sigma,
            )
            total += result.log_likelihood
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        profiler.invalid_likelihood_count += 1
        profiler.baseline_likelihood_seconds += time.perf_counter() - likelihood_start
        profiler.total_seconds += time.perf_counter() - total_start
        return -math.inf
    profiler.baseline_likelihood_seconds += time.perf_counter() - likelihood_start
    likelihood = float(total)
    if not np.isfinite(likelihood):
        profiler.invalid_likelihood_count += 1
        profiler.total_seconds += time.perf_counter() - total_start
        return -math.inf
    profiler.total_seconds += time.perf_counter() - total_start
    return float(prior + likelihood)


def profiled_log_probability_with_context(
    vector: np.ndarray,
    context: Phase1CLikelihoodContext,
    profiler: PosteriorProfiler,
) -> float:
    """Return log posterior using precomputed context while recording timings."""
    profiler.posterior_calls += 1
    total_start = time.perf_counter()
    prior_start = time.perf_counter()
    prior = log_prior(vector, context.data, context.config, context.timing)
    profiler.prior_seconds += time.perf_counter() - prior_start
    if not np.isfinite(prior):
        profiler.invalid_prior_count += 1
        profiler.total_seconds += time.perf_counter() - total_start
        return -math.inf

    batman_start = time.perf_counter()
    try:
        sample = vector_to_physical(vector, context.timing)
        transit_model = transit_model_for_sample(sample, context)
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        profiler.invalid_likelihood_count += 1
        profiler.batman_seconds += time.perf_counter() - batman_start
        profiler.total_seconds += time.perf_counter() - total_start
        return -math.inf
    profiler.batman_seconds += time.perf_counter() - batman_start

    likelihood_start = time.perf_counter()
    total = 0.0
    try:
        for event in context.events:
            total += marginalized_event_log_likelihood_from_context(
                event,
                transit_model,
                sample.jitter,
                context,
            ).log_likelihood
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        profiler.invalid_likelihood_count += 1
        profiler.baseline_likelihood_seconds += time.perf_counter() - likelihood_start
        profiler.total_seconds += time.perf_counter() - total_start
        return -math.inf
    profiler.baseline_likelihood_seconds += time.perf_counter() - likelihood_start
    likelihood = float(total)
    if not np.isfinite(likelihood):
        profiler.invalid_likelihood_count += 1
        profiler.total_seconds += time.perf_counter() - total_start
        return -math.inf
    profiler.total_seconds += time.perf_counter() - total_start
    return float(prior + likelihood)


def profiled_log_probability(
    vector: np.ndarray,
    data: FrozenPhase1BData,
    config: Phase1CConfig,
    timing: TimingReference,
    profiler: PosteriorProfiler,
) -> float:
    """Return log posterior while recording direct call counts and section timings."""
    context = Phase1CLikelihoodContext.from_data(data, config, timing)
    return profiled_log_probability_with_context(vector, context, profiler)


def baseline_conditionals_for_vector(
    vector: np.ndarray,
    data: FrozenPhase1BData,
    config: Phase1CConfig,
    timing: TimingReference,
) -> list[dict[str, Any]]:
    """Return conditional baseline posterior summaries for each frozen event."""
    context = Phase1CLikelihoodContext.from_data(data, config, timing)
    sample = vector_to_physical(vector, timing)
    transit_model = transit_model_for_sample(sample, context)
    rows: list[dict[str, Any]] = []
    for event in context.events:
        result = marginalized_event_log_likelihood_from_context(event, transit_model, sample.jitter, context)
        rows.append(
            {
                "event_number": int(event.event_number),
                "baseline_intercept_mean": float(result.baseline_mean[0]),
                "baseline_slope_mean": float(result.baseline_mean[1]),
                "baseline_intercept_sd": float(np.sqrt(result.baseline_covariance[0, 0])),
                "baseline_slope_sd": float(np.sqrt(result.baseline_covariance[1, 1])),
                "baseline_covariance_01": float(result.baseline_covariance[0, 1]),
            }
        )
    return rows
