"""Phase 1C posterior-sampling orchestration for frozen Phase 1B transit data."""

from __future__ import annotations

import re
import json
import subprocess
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .phase1b_model import batman_flux
from .phase1c_diagnostics import (
    convergence_diagnostics,
    ensemble_summary_frame,
    post_warmup_flat_chain,
    post_warmup_chain,
    posterior_summary_frame,
)
from .phase1c_inputs import load_frozen_phase1b, write_phase1b_input_manifest
from .phase1c_likelihood import log_probability
from .phase1c_outputs import write_correlation_plot, write_json, write_marginal_plot, write_trace_plot
from .phase1c_parameters import (
    build_timing_reference,
    physical_to_vector,
    prior_description,
    timing_support_audit,
)
from .phase1c_sampler import (
    checkpoint_metadata,
    dependency_versions,
    estimate_autocorrelation_time,
    load_backend_chains,
    load_backend_chains_by_ensemble,
    run_ensembles,
)
from .phase1c_types import FrozenPhase1BData, Phase1CConfig, PhysicalSample, TimingReference

PILOT_LABEL = "PILOT - NONPRODUCTION - NONCONVERGED"


def validate_phase1c_inputs(config: Phase1CConfig) -> dict[str, Any]:
    """Validate frozen Phase 1B inputs and write Phase 1C input records."""
    run_config = prepare_run_config(config, "validate", resume=False)
    data = load_frozen_phase1b(run_config)
    timing = build_timing_reference(data, run_config)
    audit = timing_support_audit(data, timing)
    _write_common_inputs(data, run_config, timing, mode="validate-inputs", timing_audit=audit)
    manifest = write_phase1b_input_manifest(run_config.phase1b_output_dir, run_config.output_dir)
    result = {
        "status": "valid",
        "run_id": run_config.run_id,
        "run_directory": str(run_config.output_dir),
        "phase1b_output_dir": str(run_config.phase1b_output_dir),
        "phase1b_manifest_sha256": manifest["manifest_sha256"],
        "cadence_count": data.cadence_count,
        "accepted_event_count": data.event_count,
        "required_file_count": len(manifest["files"]),
        "residuals_csv_used_as_input": False,
        "timing_reference": timing.__dict__,
        "timing_support_audit": _audit_summary(audit),
    }
    write_json(run_config.output_dir / "input_validation_summary.json", result)
    update_run_index(config.output_dir, run_config, "validate", result["status"])
    return result


def run_phase1c_pilot(config: Phase1CConfig, *, resume: bool = False) -> dict[str, Any]:
    """Run a short nonproduction real-data pilot from frozen Phase 1B cadences."""
    run_config = prepare_run_config(config, "pilot", resume=resume)
    data = load_frozen_phase1b(run_config)
    timing = build_timing_reference(data, run_config)
    return _run_sampling_mode(data, run_config, timing, mode="pilot", steps=run_config.pilot_steps, resume=resume)


def run_phase1c_production(config: Phase1CConfig, *, resume: bool = False) -> dict[str, Any]:
    """Run production Phase 1C sampling configuration."""
    run_config = prepare_run_config(config, "production", resume=resume)
    data = load_frozen_phase1b(run_config)
    timing = build_timing_reference(data, run_config)
    return _run_sampling_mode(
        data,
        run_config,
        timing,
        mode="production",
        steps=run_config.production_steps,
        resume=resume,
    )


def run_phase1c_synthetic_validation(
    config: Phase1CConfig,
    *,
    resume: bool = False,
    recovery: bool = False,
) -> dict[str, Any]:
    """Run a reproducible synthetic Phase 1C smoke or recovery validation."""
    mode = "synthetic_recovery" if recovery else "synthetic"
    run_config = prepare_run_config(config, mode, resume=resume)
    data, timing, injected = synthetic_dataset(run_config)
    steps = run_config.synthetic_recovery_steps if recovery else run_config.synthetic_steps
    result = _run_sampling_mode(data, run_config, timing, mode=mode, steps=steps, resume=resume)
    payload = _synthetic_summary(run_config, timing, injected, result, recovery=recovery)
    write_json(run_config.output_dir / "synthetic_recovery_summary.json", payload)
    update_run_index(config.output_dir, run_config, mode, payload["status"])
    return payload


def summarize_phase1c_checkpoints(config: Phase1CConfig, *, mode: str = "pilot") -> dict[str, Any]:
    """Summarize existing checkpoints without running additional MCMC steps."""
    run_config = prepare_run_config(config, mode, resume=True)
    data = load_frozen_phase1b(run_config)
    timing = build_timing_reference(data, run_config)
    results = []
    from .phase1c_sampler import EnsembleRunResult

    for ensemble_index in range(run_config.n_ensembles):
        path = run_config.output_dir / f"ensemble_{ensemble_index:02d}.h5"
        if path.exists():
            results.append(
                EnsembleRunResult(
                    ensemble_index=ensemble_index,
                    seed=run_config.random_seed + 1000 * ensemble_index,
                    strategy="checkpoint",
                    backend_path=path,
                    iterations=0,
                    runtime_seconds=0.0,
                    acceptance_fraction=np.asarray([np.nan]),
                    initialization_summary={"resume": True},
                    profiler_summary={},
                )
            )
    if not results:
        raise FileNotFoundError(f"No Phase 1C {mode!r} checkpoints found in {run_config.output_dir}.")
    return _write_sampling_summaries(data, run_config, timing, results, mode=mode, elapsed_seconds=0.0)


def prepare_run_config(config: Phase1CConfig, mode: str, *, resume: bool) -> Phase1CConfig:
    """Return a config whose output directory is an isolated immutable run directory."""
    if resume and not config.run_id:
        raise ValueError("Phase 1C resume requires --phase1c-run-id to identify an existing run directory.")
    run_id = sanitize_run_id(config.run_id or timestamp_run_id())
    run_dir = config.output_dir / f"{mode}_{run_id}"
    if resume:
        if not run_dir.exists():
            raise FileNotFoundError(f"Cannot resume missing Phase 1C run directory: {run_dir}")
    elif run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite existing Phase 1C run directory: {run_dir}. "
            "Use a new --phase1c-run-id or resume the existing run."
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    return replace(config, output_dir=run_dir, run_id=run_id)


def sanitize_run_id(run_id: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_id.strip())
    if not cleaned:
        raise ValueError("Phase 1C run ID cannot be empty.")
    return cleaned


def timestamp_run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _run_sampling_mode(
    data: FrozenPhase1BData,
    config: Phase1CConfig,
    timing: TimingReference,
    *,
    mode: str,
    steps: int,
    resume: bool,
) -> dict[str, Any]:
    start = time.perf_counter()
    audit = timing_support_audit(data, timing)
    if not audit["center_remains_inside_every_frozen_window"]:
        raise RuntimeError("Phase 1C timing support audit failed: centers leave frozen windows.")
    _write_common_inputs(data, config, timing, mode=mode, timing_audit=audit)
    if mode not in {"synthetic", "synthetic_recovery"}:
        write_phase1b_input_manifest(config.phase1b_output_dir, config.output_dir)
    else:
        write_json(config.output_dir / "synthetic_input_record.json", _synthetic_input_record(data, timing))

    history_state: dict[str, Any] = {"previous_medians": None, "rows": _read_existing_history(config.output_dir)}

    def on_chunk(results, elapsed_seconds, profiler_summary):
        _append_convergence_history(
            config,
            timing,
            results,
            mode=mode,
            elapsed_seconds=elapsed_seconds,
            profiler_summary=profiler_summary,
            history_state=history_state,
        )

    results = run_ensembles(
        data,
        config,
        timing,
        steps=steps,
        mode=mode,
        resume=resume,
        chunk_callback=on_chunk,
    )
    elapsed = time.perf_counter() - start
    summary = _write_sampling_summaries(data, config, timing, results, mode=mode, elapsed_seconds=elapsed)
    update_run_index(config.output_dir.parent, config, mode, summary["diagnostic_status"])
    return summary


def _write_common_inputs(
    data: FrozenPhase1BData,
    config: Phase1CConfig,
    timing: TimingReference,
    *,
    mode: str,
    timing_audit: dict[str, Any],
) -> None:
    write_json(config.output_dir / "phase1c_configuration.json", config.to_dict())
    write_json(config.output_dir / "parameter_transformations.json", prior_description(data, config, timing))
    write_json(config.output_dir / "timing_support_audit.json", timing_audit)
    write_json(config.output_dir / "provenance_manifest.json", build_phase1c_provenance(data, config, timing, mode=mode))


def _write_sampling_summaries(
    data: FrozenPhase1BData,
    config: Phase1CConfig,
    timing: TimingReference,
    results,
    *,
    mode: str,
    elapsed_seconds: float,
) -> dict[str, Any]:
    diagnostic_start = time.perf_counter()
    chain, log_prob = load_backend_chains(results)
    chains_by_ensemble = load_backend_chains_by_ensemble(results)
    acceptance = np.concatenate([result.acceptance_fraction for result in results])
    autocorr = estimate_autocorrelation_time(results, config.warmup_steps)
    diagnostics = convergence_diagnostics(
        chain,
        log_prob,
        acceptance,
        autocorr,
        config,
        warmup_steps=config.warmup_steps,
    )
    posterior = posterior_summary_frame(chain, timing, warmup_steps=config.warmup_steps)
    ensemble = ensemble_summary_frame(chains_by_ensemble, config.warmup_steps)
    posterior.to_csv(config.output_dir / "posterior_parameter_summary.csv", index=False)
    ensemble.to_csv(config.output_dir / "ensemble_summary.csv", index=False)
    diagnostics["mode"] = mode
    diagnostics["run_id"] = config.run_id
    diagnostics["nonproduction"] = mode in {"pilot", "synthetic", "synthetic_recovery"}
    diagnostics["convergence_claim"] = (
        "no convergence claim; fixed short nonproduction run"
        if mode in {"pilot", "synthetic"}
        else diagnostics["status"]
    )
    write_json(config.output_dir / "sampler_diagnostics.json", diagnostics)
    profiler_summary = _aggregate_result_profilers(results)
    output_overhead = max(elapsed_seconds - float(profiler_summary.get("total_log_posterior_seconds", 0.0)), 0.0)
    runtime = {
        "mode": mode,
        "run_id": config.run_id,
        "elapsed_seconds": elapsed_seconds,
        "total_iterations_per_ensemble": [result.iterations for result in results],
        "stored_log_probability_entries": int(np.size(log_prob)),
        "finite_log_probability_fraction": float(np.mean(np.isfinite(log_prob))),
        "actual_log_posterior_calls": int(profiler_summary.get("posterior_calls", 0)),
        "posterior_calls_per_second": (
            float(profiler_summary.get("posterior_calls", 0) / elapsed_seconds) if elapsed_seconds > 0.0 else None
        ),
        "stored_draws_per_second": float(np.size(log_prob) / elapsed_seconds) if elapsed_seconds > 0.0 else None,
        "timing_seconds": {
            **profiler_summary,
            "output_and_diagnostic_overhead_seconds": output_overhead + (time.perf_counter() - diagnostic_start),
        },
        "ensemble_results": [_ensemble_runtime_row(result) for result in results],
    }
    write_json(config.output_dir / "sampler_runtime.json", runtime)
    label = PILOT_LABEL if mode == "pilot" else None
    if chain.shape[1] >= 2:
        write_trace_plot(config.output_dir / "trace_plot.png", chain, label=label)
        flat = post_warmup_flat_chain(chain, config.warmup_steps)
        if flat.shape[0] > 10:
            write_marginal_plot(config.output_dir / "marginal_plot.png", flat, label=label)
            write_correlation_plot(config.output_dir / "correlation_plot.png", flat, label=label)
    if mode == "pilot":
        retained_draws = int(post_warmup_flat_chain(chain, config.warmup_steps).shape[0])
        pilot = {
            "label": PILOT_LABEL,
            "status": "complete_nonproduction_nonconverged"
            if diagnostics["status"] != "converged"
            else "complete_nonproduction_review",
            "warning": (
                "Posterior summaries from this pilot are not scientific uncertainty estimates."
                if retained_draws < config.minimum_meaningful_summary_draws
                else ""
            ),
            "cadence_count": data.cadence_count,
            "accepted_event_count": data.event_count,
            "retained_draws": retained_draws,
            "runtime": runtime,
            "diagnostics": diagnostics,
            "checkpoint_status": "written",
            "residuals_csv_used_as_input": False,
        }
        write_json(config.output_dir / "pilot_summary.json", pilot)
    return {
        "mode": mode,
        "run_id": config.run_id,
        "output_dir": str(config.output_dir),
        "diagnostic_status": diagnostics["status"],
        "nonproduction": mode in {"pilot", "synthetic", "synthetic_recovery"},
        "elapsed_seconds": elapsed_seconds,
        "actual_log_posterior_calls": int(profiler_summary.get("posterior_calls", 0)),
        "posterior_calls_per_second": runtime["posterior_calls_per_second"],
        "finite_log_probability_fraction": diagnostics["finite_log_probability_fraction"],
        "acceptance_fraction": diagnostics["acceptance_fraction"],
    }


def _append_convergence_history(
    config: Phase1CConfig,
    timing: TimingReference,
    results,
    *,
    mode: str,
    elapsed_seconds: float,
    profiler_summary: dict[str, Any],
    history_state: dict[str, Any],
) -> None:
    chain, log_prob = load_backend_chains(results)
    acceptance = np.concatenate([result.acceptance_fraction for result in results])
    autocorr = estimate_autocorrelation_time(results, config.warmup_steps)
    diagnostics = convergence_diagnostics(
        chain,
        log_prob,
        acceptance,
        autocorr,
        config,
        warmup_steps=config.warmup_steps,
    )
    kept = post_warmup_chain(chain, config.warmup_steps)
    completed_steps = int(min(result.iterations for result in results))
    retained_steps = int(kept.shape[1])
    stability = _posterior_stability_metric(chain, timing, config, history_state)
    row = {
        "mode": mode,
        "run_id": config.run_id,
        "completed_steps": completed_steps,
        "retained_post_warmup_steps": retained_steps,
        "elapsed_seconds": float(elapsed_seconds),
        "actual_log_posterior_calls": int(profiler_summary.get("posterior_calls", 0)),
        "acceptance_fraction_min": diagnostics["acceptance_fraction"]["min"],
        "acceptance_fraction_median": diagnostics["acceptance_fraction"]["median"],
        "acceptance_fraction_max": diagnostics["acceptance_fraction"]["max"],
        "finite_log_probability_fraction": diagnostics["finite_log_probability_fraction"],
        "rhat_max": _max_numeric(diagnostics["split_rhat"]),
        "bulk_ess_min": _min_numeric(diagnostics["bulk_ess"]),
        "tail_ess_min": _min_numeric(diagnostics["tail_ess"]),
        "autocorrelation_time_max": None
        if diagnostics["emcee_autocorrelation_time"] is None
        else _max_list(diagnostics["emcee_autocorrelation_time"]),
        "diagnostic_backend": diagnostics["standard_diagnostic_backend"],
        "diagnostic_availability": diagnostics["standard_diagnostic_backend"],
        "posterior_median_max_abs_change_from_previous": stability,
        "convergence_status": diagnostics["status"],
    }
    history_state["rows"].append(row)
    pd.DataFrame(history_state["rows"]).to_csv(config.output_dir / "convergence_history.csv", index=False)


def synthetic_dataset(config: Phase1CConfig) -> tuple[FrozenPhase1BData, TimingReference, dict[str, float]]:
    """Build a small synthetic frozen Phase 1B-like dataset for sampler validation."""
    rng = np.random.default_rng(config.random_seed + 77)
    period = 3.0
    original_epoch = 10.0
    event_ids = np.arange(8)
    offsets = np.linspace(-0.35, 0.35, 28)
    time = np.concatenate([original_epoch + event * period + offsets for event in event_ids])
    event_number = np.concatenate([np.full(offsets.size, event, dtype=int) for event in event_ids])
    predicted_center = original_epoch + event_number * period
    exposure_days = np.full(time.size, 0.02)
    true = PhysicalSample(
        rp=0.08,
        a=8.5,
        b=0.35,
        q1=0.30,
        q2=0.40,
        jitter=0.00008,
        period=period,
        mid_epoch=original_epoch + 4 * period,
        original_epoch=original_epoch,
    )
    model = batman_flux(
        time,
        exposure_days,
        rp=true.rp,
        a=true.a,
        b=true.b,
        q1=true.q1,
        q2=true.q2,
        period=true.period,
        t0=true.original_epoch,
        supersample_factor=config.supersample_factor,
    )
    flux_uncertainty = rng.uniform(0.00008, 0.00013, size=time.size)
    flux = np.empty_like(time)
    for event in event_ids:
        mask = event_number == event
        x = (time[mask] - predicted_center[mask]) / np.max(np.abs(offsets))
        c0 = rng.normal(1.0, config.baseline_intercept_sigma)
        c1 = rng.normal(0.0, config.baseline_slope_sigma)
        noise = rng.normal(0.0, np.sqrt(flux_uncertainty[mask] ** 2 + true.jitter**2))
        flux[mask] = model[mask] * (c0 + c1 * x) + noise
    deterministic_parameters = {
        "rp_over_rstar": true.rp,
        "a_over_rstar": true.a,
        "impact_parameter": true.b,
        "q1": true.q1,
        "q2": true.q2,
        "white_noise_jitter": true.jitter,
        "period_days": true.period,
        "transit_time": true.original_epoch,
    }
    data = FrozenPhase1BData(
        time=time,
        flux=flux,
        flux_uncertainty=flux_uncertainty,
        event_number=event_number,
        predicted_center=predicted_center,
        product_id=np.full(time.size, "synthetic"),
        quarter=np.full(time.size, "synthetic"),
        exposure_days=exposure_days,
        deterministic_parameters=deterministic_parameters,
        limb_darkening={"q1": true.q1, "q2": true.q2, "q1_sigma": 0.04, "q2_sigma": 0.04},
        phase1b_configuration={
            "supersample_factor": config.supersample_factor,
            "timing_refinement_t0_half_width_duration_scale": config.mid_epoch_half_width_duration_scale,
        },
        phase1b_summary=_synthetic_phase1b_summary(period),
        provenance={"cadence_counts": {"phase1b_fit_cadence_count": int(time.size)}},
        input_manifest={"manifest_sha256": "synthetic", "residuals_csv_used_as_input": False, "files": []},
    )
    timing = build_timing_reference(data, config)
    injected = {
        "rp_over_rstar": true.rp,
        "a_over_rstar": true.a,
        "impact_parameter": true.b,
        "q1": true.q1,
        "q2": true.q2,
        "white_noise_jitter": true.jitter,
        "period_days": true.period,
        "transit_time_original_reference": true.original_epoch,
        "transit_time_mid_mission_reference": true.mid_epoch,
    }
    vector = physical_to_vector(true, timing)
    if not np.isfinite(log_probability(vector, data, config, timing)):
        raise RuntimeError("Synthetic injected solution has nonfinite posterior density.")
    return data, timing, injected


def build_phase1c_provenance(
    data: FrozenPhase1BData,
    config: Phase1CConfig,
    timing: TimingReference,
    *,
    mode: str,
) -> dict[str, Any]:
    """Build Phase 1C provenance independent of Phase 0/1A/1B rebuilding."""
    return {
        "run_timestamp_utc": datetime.now(UTC).isoformat(),
        "mode": mode,
        "run_id": config.run_id,
        "phase1": "1C",
        "git": {
            "commit": _git_output("rev-parse", "HEAD"),
            "status": _git_output("status", "--short"),
            "is_dirty": bool(_git_output("status", "--short")),
        },
        "dependencies": dependency_versions(),
        "phase1b_input_manifest": data.input_manifest,
        "phase1c_configuration": config.to_dict(),
        "checkpoint_metadata": checkpoint_metadata(data, config, mode=mode),
        "timing_reference": timing.__dict__,
        "cadence_count": data.cadence_count,
        "accepted_event_count": data.event_count,
        "scientific_safeguards": {
            "phase0_phase1a_phase1b_rebuilt": False,
            "kepler_data_redownloaded": False,
            "residuals_csv_used_as_fitting_input": False,
            "published_planet_parameters_used": False,
            "accepted_cadences_altered": False,
            "posterior_predictive_suite_implemented": False,
        },
    }


def update_run_index(base_output_dir: Path, config: Phase1CConfig, mode: str, status: str) -> None:
    """Write a small top-level index pointing to immutable run directories."""
    base_output_dir.mkdir(parents=True, exist_ok=True)
    index_path = base_output_dir / "run_index.json"
    if index_path.exists():
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8")).get("runs", [])
        except (ValueError, OSError, json.JSONDecodeError):
            payload = []
    else:
        payload = []
    payload.append(
        {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "mode": mode,
            "run_id": config.run_id,
            "run_directory": str(config.output_dir),
            "status": status,
        }
    )
    write_json(index_path, {"runs": payload})


def _synthetic_summary(
    config: Phase1CConfig,
    timing: TimingReference,
    injected: dict[str, float],
    sampler_result: dict[str, Any],
    *,
    recovery: bool,
) -> dict[str, Any]:
    posterior = pd.read_csv(config.output_dir / "posterior_parameter_summary.csv")
    diagnostics = _read_json(config.output_dir / "sampler_diagnostics.json")
    ensemble = pd.read_csv(config.output_dir / "ensemble_summary.csv")
    rows = []
    for _, row in posterior.iterrows():
        parameter = row["parameter"]
        if parameter not in injected:
            continue
        rows.append(
            {
                "parameter": parameter,
                "injected_value": injected[parameter],
                "posterior_median": float(row["median"]),
                "q16": float(row["q16"]),
                "q84": float(row["q84"]),
                "q02_5": float(row["q02_5"]),
                "q97_5": float(row["q97_5"]),
                "interval_68_width": float(row["q84"] - row["q16"]),
                "interval_95_width": float(row["q97_5"] - row["q02_5"]),
                "injected_inside_68_percent_interval": bool(row["q16"] <= injected[parameter] <= row["q84"]),
                "injected_inside_95_percent_interval": bool(row["q02_5"] <= injected[parameter] <= row["q97_5"]),
            }
        )
    all_inside_95 = all(row["injected_inside_95_percent_interval"] for row in rows)
    if diagnostics["status"] != "converged":
        status = "smoke_test_completed_nonconverged"
    elif recovery and all_inside_95:
        status = "converged_recovery_passed"
    elif recovery:
        status = "converged_recovery_failed"
    else:
        status = "smoke_test_completed_converged"
    return {
        "status": status,
        "run_id": config.run_id,
        "run_directory": str(config.output_dir),
        "configured_steps": config.synthetic_recovery_steps if recovery else config.synthetic_steps,
        "configured_walkers": config.n_walkers,
        "configured_ensembles": config.n_ensembles,
        "configured_warmup_steps": config.warmup_steps,
        "convergence_diagnostics": diagnostics,
        "injected_parameters": injected,
        "parameter_recovery_rows": rows,
        "per_ensemble_summary": ensemble.to_dict(orient="records"),
        "retained_log_posterior_neighborhood_fractions": _log_posterior_neighborhood_fractions(config),
        "coverage_statement": (
            "Interval coverage from this nonconverged smoke chain is not validation."
            if diagnostics["status"] != "converged"
            else "Converged synthetic recovery status is determined by configured convergence criteria."
        ),
        "sampler_result": sampler_result,
        "nonproduction": not recovery,
        "timing_reference": timing.__dict__,
    }


def _log_posterior_neighborhood_fractions(config: Phase1CConfig) -> dict[str, float]:
    import emcee

    values = []
    for path in sorted(config.output_dir.glob("ensemble_*.h5")):
        backend = emcee.backends.HDFBackend(str(path), read_only=True)
        log_prob = backend.get_log_prob()
        discard = min(config.warmup_steps, max(log_prob.shape[0] - 1, 0))
        values.extend(np.ravel(log_prob[discard:, :]).tolist())
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {str(distance): 0.0 for distance in (1.0, 2.0, 5.0, 10.0)}
    best = float(np.max(arr))
    return {str(distance): float(np.mean(arr >= best - distance)) for distance in (1.0, 2.0, 5.0, 10.0)}


def _ensemble_runtime_row(result) -> dict[str, Any]:
    import emcee

    backend = emcee.backends.HDFBackend(str(result.backend_path), read_only=True)
    log_prob = backend.get_log_prob()
    finite = log_prob[np.isfinite(log_prob)]
    final_log_posterior = {}
    if finite.size:
        final_log_posterior = {
            "min": float(np.min(finite)),
            "median": float(np.median(finite)),
            "max": float(np.max(finite)),
        }
    return {
        "ensemble": result.ensemble_index,
        "seed": result.seed,
        "strategy": result.strategy,
        "backend_path": str(result.backend_path),
        "iterations": result.iterations,
        "runtime_seconds": result.runtime_seconds,
        "acceptance_fraction_min": float(np.nanmin(result.acceptance_fraction)),
        "acceptance_fraction_median": float(np.nanmedian(result.acceptance_fraction)),
        "acceptance_fraction_max": float(np.nanmax(result.acceptance_fraction)),
        "initialization": result.initialization_summary,
        "final_log_posterior": final_log_posterior,
        "profiler": result.profiler_summary,
    }


def _aggregate_result_profilers(results) -> dict[str, Any]:
    aggregate = {}
    if not results:
        return aggregate
    keys = set().union(*(result.profiler_summary.keys() for result in results))
    for key in keys:
        values = [result.profiler_summary.get(key, 0.0) for result in results]
        aggregate[key] = float(np.sum(values)) if any(isinstance(value, float) for value in values) else int(np.sum(values))
    return aggregate


def _posterior_stability_metric(
    chain: np.ndarray,
    timing: TimingReference,
    config: Phase1CConfig,
    history_state: dict[str, Any],
) -> float | None:
    try:
        summary = posterior_summary_frame(chain, timing, warmup_steps=config.warmup_steps)
    except (ValueError, IndexError):
        return None
    medians = {row["parameter"]: float(row["median"]) for _, row in summary.iterrows()}
    previous = history_state.get("previous_medians")
    history_state["previous_medians"] = medians
    if previous is None:
        return None
    common = set(previous) & set(medians)
    if not common:
        return None
    return float(max(abs(medians[key] - previous[key]) for key in common))


def _synthetic_input_record(data: FrozenPhase1BData, timing: TimingReference) -> dict[str, Any]:
    return {
        "type": "synthetic_phase1c_dataset",
        "cadence_count": data.cadence_count,
        "event_count": data.event_count,
        "timing_reference": timing.__dict__,
        "residuals_csv_used_as_input": False,
    }


def _synthetic_phase1b_summary(period: float) -> dict[str, Any]:
    return {
        "established_inputs": {
            "full_mission_local_refinement": {
                "refined_period_days": period,
                "refined_transit_time": 10.0,
                "refined_duration_days": 0.12,
            }
        },
        "transit_windows": {"included_count": 8, "predicted_count": 8},
        "acceptance_checks": {"published_physical_planet_parameters_used_or_compared": False},
    }


def _audit_summary(audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "earliest_frozen_event_number": audit["earliest_frozen_event_number"],
        "latest_frozen_event_number": audit["latest_frozen_event_number"],
        "maximum_center_displacement_days": audit["maximum_center_displacement_days"],
        "minimum_center_to_window_edge_margin_days": audit["minimum_center_to_window_edge_margin_days"],
        "minimum_complete_transit_margin_days": audit["minimum_complete_transit_margin_days"],
        "center_remains_inside_every_frozen_window": audit["center_remains_inside_every_frozen_window"],
        "complete_nominal_transit_remains_inside_every_frozen_window": audit[
            "complete_nominal_transit_remains_inside_every_frozen_window"
        ],
        "period_support_rule": audit["period_support_rule"],
    }


def _read_existing_history(output_dir: Path) -> list[dict[str, Any]]:
    path = output_dir / "convergence_history.csv"
    if not path.exists():
        return []
    return pd.read_csv(path).to_dict(orient="records")


def _read_json(path: Path) -> dict[str, Any]:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def _max_numeric(values: dict[str, Any]) -> float | None:
    numeric = [float(value) for value in values.values() if value is not None and np.isfinite(value)]
    return max(numeric) if numeric else None


def _min_numeric(values: dict[str, Any]) -> float | None:
    numeric = [float(value) for value in values.values() if value is not None and np.isfinite(value)]
    return min(numeric) if numeric else None


def _max_list(values: list[Any]) -> float | None:
    numeric = [float(value) for value in values if value is not None and np.isfinite(value)]
    return max(numeric) if numeric else None


def _git_output(*args: str) -> str:
    try:
        completed = subprocess.run(["git", *args], check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return ""
    return completed.stdout.strip()
