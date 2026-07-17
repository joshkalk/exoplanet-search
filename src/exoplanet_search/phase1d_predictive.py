"""Phase 1D conditional-baseline draws and posterior-predictive flux generation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .phase1c_likelihood import event_local_coordinate, marginalized_event_log_likelihood, transit_model_for_vector
from .phase1c_parameters import vector_to_physical
from .phase1c_types import FrozenPhase1BData, Phase1CConfig, TimingReference
from .phase1d_draws import SelectedPosteriorDraw


@dataclass(frozen=True)
class EventBaselineDraw:
    """One conditional Gaussian draw of event-local baseline coefficients."""

    event_number: int
    coefficients: np.ndarray
    conditional_mean: np.ndarray
    conditional_covariance: np.ndarray
    design_matrix: np.ndarray
    transit_model: np.ndarray
    local_coordinate: np.ndarray
    covariance_eigenvalues: np.ndarray
    cholesky_diagonal: np.ndarray

    def audit_record(self, draw: SelectedPosteriorDraw, *, replication_index: int) -> dict[str, Any]:
        return {
            "source_run_id": draw.run_id,
            "source_mode": draw.mode,
            "selection_position": draw.selection_position,
            "predictive_replication_index": replication_index,
            "ensemble": draw.ensemble,
            "walker": draw.walker,
            "step": draw.step,
            "event_number": self.event_number,
            "baseline_intercept": float(self.coefficients[0]),
            "baseline_slope": float(self.coefficients[1]),
            "conditional_mean": [float(value) for value in self.conditional_mean],
            "conditional_covariance": self.conditional_covariance.astype(float).tolist(),
            "covariance_eigenvalues": [float(value) for value in self.covariance_eigenvalues],
            "cholesky_diagonal": [float(value) for value in self.cholesky_diagonal],
        }


def draw_conditional_event_baseline(
    draw: SelectedPosteriorDraw,
    event_number: int,
    data: FrozenPhase1BData,
    config: Phase1CConfig,
    timing: TimingReference,
    rng: np.random.Generator,
    *,
    transit_model: np.ndarray | None = None,
    physical=None,
) -> EventBaselineDraw:
    """Draw beta | y, theta for one event using the exact Phase 1C Gaussian conditional.

    Phase 1C analytically marginalizes multiplicative local event baselines with
    design matrix X = [m, m*x]. This function reuses that same likelihood path,
    then samples beta from the conditional Normal distribution returned by the
    marginalized calculation. It does not refit or approximate the Gaussian path.
    """
    sample = physical if physical is not None else vector_to_physical(draw.vector, timing)
    if transit_model is None:
        transit_model = transit_model_for_vector(draw.vector, data, config, timing)
    mask = data.event_number == int(event_number)
    if not np.any(mask):
        raise ValueError(f"Unknown event_number {event_number}.")
    event_model = transit_model[mask]
    x = event_local_coordinate(data.time[mask], float(np.median(data.predicted_center[mask])))
    result = marginalized_event_log_likelihood(
        time=data.time[mask],
        flux=data.flux[mask],
        flux_uncertainty=data.flux_uncertainty[mask],
        transit_model=event_model,
        frozen_center=float(np.median(data.predicted_center[mask])),
        jitter=sample.jitter,
        baseline_intercept_sigma=config.baseline_intercept_sigma,
        baseline_slope_sigma=config.baseline_slope_sigma,
    )
    covariance = np.asarray(result.baseline_covariance, dtype=float)
    eigenvalues, cholesky = _validate_covariance(covariance)
    coefficients = np.asarray(result.baseline_mean, dtype=float) + cholesky @ rng.standard_normal(2)
    return EventBaselineDraw(
        event_number=int(event_number),
        coefficients=np.asarray(coefficients, dtype=float),
        conditional_mean=np.asarray(result.baseline_mean, dtype=float),
        conditional_covariance=covariance,
        design_matrix=np.column_stack([event_model, event_model * x]),
        transit_model=np.asarray(event_model, dtype=float),
        local_coordinate=np.asarray(x, dtype=float),
        covariance_eigenvalues=eigenvalues,
        cholesky_diagonal=np.diag(cholesky),
    )


def generate_replicated_flux(
    draw: SelectedPosteriorDraw,
    data: FrozenPhase1BData,
    config: Phase1CConfig,
    timing: TimingReference,
    rng: np.random.Generator,
    *,
    replication_index: int = 0,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    """Generate one posterior-predictive flux replicate aligned to frozen cadences.

    For each frozen event this draws beta from the exact conditional Gaussian
    baseline posterior, computes predictive_mean = X beta, and adds newly
    generated independent Gaussian noise with variance sigma_i^2 + jitter^2.
    Observed residuals are never resampled.
    """
    sample = vector_to_physical(draw.vector, timing)
    full_transit_model = transit_model_for_vector(draw.vector, data, config, timing)
    n_cadences = data.cadence_count
    model_flux = np.empty(n_cadences, dtype=float)
    predictive_mean = np.empty(n_cadences, dtype=float)
    predictive_noise = np.empty(n_cadences, dtype=float)
    replicated_flux = np.empty(n_cadences, dtype=float)
    baseline_intercept = np.empty(n_cadences, dtype=float)
    baseline_slope = np.empty(n_cadences, dtype=float)
    baseline_audit = []
    for event in np.unique(data.event_number):
        mask = data.event_number == event
        baseline = draw_conditional_event_baseline(
            draw,
            int(event),
            data,
            config,
            timing,
            rng,
            transit_model=full_transit_model,
            physical=sample,
        )
        mean = baseline.design_matrix @ baseline.coefficients
        variance = np.square(data.flux_uncertainty[mask]) + sample.jitter**2
        noise = rng.normal(0.0, np.sqrt(variance))
        model_flux[mask] = baseline.transit_model
        predictive_mean[mask] = mean
        predictive_noise[mask] = noise
        replicated_flux[mask] = mean + noise
        baseline_intercept[mask] = baseline.coefficients[0]
        baseline_slope[mask] = baseline.coefficients[1]
        baseline_audit.append(baseline.audit_record(draw, replication_index=replication_index))
    rows = {
        "cadence_index": np.arange(n_cadences, dtype=int),
        "time": np.asarray(data.time, dtype=float),
        "event_number": np.asarray(data.event_number, dtype=int),
        "product_id": np.asarray(data.product_id, dtype=str),
        "quarter": np.asarray(data.quarter, dtype=str),
        "exposure_days": np.asarray(data.exposure_days, dtype=float),
        "flux_uncertainty": np.asarray(data.flux_uncertainty, dtype=float),
        "source_ensemble": np.full(n_cadences, draw.ensemble, dtype=int),
        "source_walker": np.full(n_cadences, draw.walker, dtype=int),
        "source_step": np.full(n_cadences, draw.step, dtype=int),
        "source_run_id": np.full(n_cadences, draw.run_id, dtype=str),
        "source_mode": np.full(n_cadences, draw.mode, dtype=str),
        "selection_position": np.full(n_cadences, draw.selection_position, dtype=int),
        "predictive_replication_index": np.full(n_cadences, replication_index, dtype=int),
        "model_flux": model_flux,
        "baseline_intercept": baseline_intercept,
        "baseline_slope": baseline_slope,
        "predictive_mean": predictive_mean,
        "predictive_noise": predictive_noise,
        "replicated_flux": replicated_flux,
        "observed_flux": np.asarray(data.flux, dtype=float),
        "resampled_observed_residual": np.zeros(n_cadences, dtype=bool),
    }
    return rows, baseline_audit


def write_development_predictive_output(
    output_dir: Path,
    *,
    predictive_rows: list[dict[str, np.ndarray]],
    baseline_audit: list[dict[str, Any]],
    config_payload: dict[str, Any],
) -> None:
    """Write compact nonauthoritative development predictive outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stacked: dict[str, np.ndarray] = {}
    for key in predictive_rows[0]:
        stacked[key] = np.stack([rows[key] for rows in predictive_rows], axis=0)
    np.savez_compressed(output_dir / "development_predictive_flux.npz", **stacked)
    with (output_dir / "event_baseline_draw_audit.jsonl").open("w", encoding="utf-8") as output_file:
        for row in baseline_audit:
            output_file.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    with (output_dir / "phase1d_predictive_configuration.json").open("w", encoding="utf-8") as output_file:
        json.dump(config_payload, output_file, indent=2)


def _validate_covariance(covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if covariance.shape != (2, 2):
        raise ValueError("Baseline covariance must be 2x2.")
    if not np.all(np.isfinite(covariance)):
        raise ValueError("Baseline covariance contains nonfinite values.")
    if not np.allclose(covariance, covariance.T, rtol=0.0, atol=1.0e-12):
        raise ValueError("Baseline covariance is not symmetric.")
    eigenvalues = np.linalg.eigvalsh(covariance)
    if np.min(eigenvalues) <= 0.0:
        raise ValueError("Baseline covariance is not positive definite.")
    try:
        cholesky = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Baseline covariance Cholesky factorization failed.") from exc
    return eigenvalues, cholesky
