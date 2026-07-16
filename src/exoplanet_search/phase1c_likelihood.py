"""Gaussian Phase 1C likelihood with exact local-baseline marginalization."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .phase1b_model import batman_flux
from .phase1c_parameters import log_prior, vector_to_physical
from .phase1c_types import FrozenPhase1BData, Phase1CConfig, TimingReference


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


def log_likelihood(
    vector: np.ndarray,
    data: FrozenPhase1BData,
    config: Phase1CConfig,
    timing: TimingReference,
) -> float:
    """Return the full marginalized Gaussian log likelihood summed over events."""
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


def log_probability(
    vector: np.ndarray,
    data: FrozenPhase1BData,
    config: Phase1CConfig,
    timing: TimingReference,
) -> float:
    """Return normalized log posterior density in transformed coordinates."""
    prior = log_prior(vector, data, config, timing)
    if not np.isfinite(prior):
        return -math.inf
    likelihood = log_likelihood(vector, data, config, timing)
    if not np.isfinite(likelihood):
        return -math.inf
    return float(prior + likelihood)


def profiled_log_probability(
    vector: np.ndarray,
    data: FrozenPhase1BData,
    config: Phase1CConfig,
    timing: TimingReference,
    profiler: PosteriorProfiler,
) -> float:
    """Return log posterior while recording direct call counts and section timings."""
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


def baseline_conditionals_for_vector(
    vector: np.ndarray,
    data: FrozenPhase1BData,
    config: Phase1CConfig,
    timing: TimingReference,
) -> list[dict[str, Any]]:
    """Return conditional baseline posterior summaries for each frozen event."""
    sample = vector_to_physical(vector, timing)
    transit_model = transit_model_for_vector(vector, data, config, timing)
    rows: list[dict[str, Any]] = []
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
        rows.append(
            {
                "event_number": int(event),
                "baseline_intercept_mean": float(result.baseline_mean[0]),
                "baseline_slope_mean": float(result.baseline_mean[1]),
                "baseline_intercept_sd": float(np.sqrt(result.baseline_covariance[0, 0])),
                "baseline_slope_sd": float(np.sqrt(result.baseline_covariance[1, 1])),
                "baseline_covariance_01": float(result.baseline_covariance[0, 1]),
            }
        )
    return rows
