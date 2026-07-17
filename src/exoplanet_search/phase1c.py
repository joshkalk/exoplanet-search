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
    independent_ensemble_agreement,
    post_warmup_flat_chain,
    post_warmup_chain,
    posterior_stability_check,
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
    return _run_sampling_mode(
        data,
        run_config,
        timing,
        mode="pilot",
        steps=_target_steps(run_config, run_config.pilot_steps, resume=resume),
        resume=resume,
    )


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
        steps=_target_steps(run_config, run_config.production_steps, resume=resume),
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
    default_steps = run_config.synthetic_recovery_steps if recovery else run_config.synthetic_steps
    steps = _target_steps(run_config, default_steps, resume=resume)
    result = _run_sampling_mode(data, run_config, timing, mode=mode, steps=steps, resume=resume)
    payload = _synthetic_summary(run_config, timing, injected, result, recovery=recovery)
    write_json(run_config.output_dir / "synthetic_recovery_summary.json", payload)
    return payload


def summarize_phase1c_checkpoints(config: Phase1CConfig, *, mode: str = "pilot") -> dict[str, Any]:
    """Summarize existing checkpoints without running additional MCMC steps."""
    run_config = prepare_run_config(config, mode, resume=True)
    run_config = _stored_phase1c_config(run_config)
    iterations = _stored_iterations(run_config.output_dir)
    invocation = _begin_invocation(
        run_config,
        f"{mode}_summarize",
        starting_iterations=iterations,
        target_steps=min(iterations.values(), default=0),
    )
    start = time.perf_counter()
    try:
        data, timing = _load_summarize_data(run_config, mode)
        _validate_synthetic_input_record_if_needed(data, timing, run_config, mode)
        result = _summarize_phase1c_checkpoints(run_config, data, timing, mode=mode)
    except KeyboardInterrupt:
        _finish_invocation(
            run_config,
            invocation,
            status="interrupted",
            ending_iterations=_stored_iterations(run_config.output_dir),
            elapsed_seconds=time.perf_counter() - start,
            posterior_calls=0,
            stop_reason="interrupted",
        )
        raise
    except Exception:
        _finish_invocation(
            run_config,
            invocation,
            status="failed",
            ending_iterations=_stored_iterations(run_config.output_dir),
            elapsed_seconds=time.perf_counter() - start,
            posterior_calls=0,
            stop_reason="failed",
        )
        raise
    _finish_invocation(
        run_config,
        invocation,
        status="completed",
        ending_iterations=_stored_iterations(run_config.output_dir),
        elapsed_seconds=time.perf_counter() - start,
        posterior_calls=0,
        stop_reason="summarized_existing_checkpoints",
    )
    return result


def _summarize_phase1c_checkpoints(
    run_config: Phase1CConfig,
    data: FrozenPhase1BData,
    timing: TimingReference,
    *,
    mode: str,
) -> dict[str, Any]:
    from .phase1c_sampler import validate_checkpoint_metadata
    import emcee

    metadata = checkpoint_metadata(data, run_config, mode=mode)
    rows = []
    for ensemble_index in range(run_config.n_ensembles):
        path = run_config.output_dir / f"ensemble_{ensemble_index:02d}.h5"
        if not path.exists():
            continue
        validate_checkpoint_metadata(path, metadata, run_config.random_seed + 1000 * ensemble_index)
        backend = emcee.backends.HDFBackend(str(path), read_only=True)
        accepted = np.asarray(getattr(backend, "accepted", np.asarray([], dtype=float)), dtype=float)
        rows.append(
            {
                "ensemble": ensemble_index,
                "path": str(path),
                "iterations": int(backend.iteration),
                "accepted_min": float(np.min(accepted)) if accepted.size else None,
                "accepted_median": float(np.median(accepted)) if accepted.size else None,
                "accepted_max": float(np.max(accepted)) if accepted.size else None,
            }
        )
    if not rows:
        raise FileNotFoundError(f"No Phase 1C {mode!r} checkpoints found in {run_config.output_dir}.")
    chain_parts = []
    log_prob_parts = []
    for row in rows:
        backend = emcee.backends.HDFBackend(row["path"], read_only=True)
        chain_parts.append(np.transpose(backend.get_chain(), (1, 0, 2)))
        log_prob_parts.append(np.transpose(backend.get_log_prob(), (1, 0)))
    chain = np.concatenate(chain_parts, axis=0)
    log_prob = np.concatenate(log_prob_parts, axis=0)
    summary = posterior_summary_frame(chain, timing, warmup_steps=run_config.warmup_steps)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    summary_path = run_config.output_dir / f"checkpoint_summary_{timestamp}.csv"
    summary.to_csv(summary_path, index=False)
    payload = {
        "mode": mode,
        "run_id": run_config.run_id,
        "run_directory": str(run_config.output_dir),
        "checkpoint_metadata_validated": True,
        "stored_log_probability_entries": int(np.size(log_prob)),
        "finite_log_probability_fraction": float(np.mean(np.isfinite(log_prob))),
        "ensembles": rows,
        "summary_path": str(summary_path),
        "authoritative_runtime_modified": False,
        "authoritative_provenance_modified": False,
    }
    write_json(run_config.output_dir / f"checkpoint_summary_{timestamp}.json", payload)
    return payload


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


def _stored_phase1c_config(config: Phase1CConfig) -> Phase1CConfig:
    path = config.output_dir / "phase1c_configuration.json"
    if not path.exists():
        return config
    payload = _read_json(path)
    payload.pop("parameter_order", None)
    payload.pop("notes", None)
    path_fields = {"phase1b_output_dir", "output_dir"}
    tuple_fields = {
        "rp_bounds",
        "a_bounds",
        "q_bounds",
        "local_tight_scales",
        "local_moderate_scales",
        "local_broad_scales",
        "prior_informed_cloud_scales",
    }
    for key in path_fields & set(payload):
        payload[key] = Path(payload[key])
    for key in tuple_fields & set(payload):
        payload[key] = tuple(payload[key])
    if "prior_informed_cloud_logp_drop" in payload:
        payload.setdefault("prior_informed_max_logp_deficit", payload["prior_informed_cloud_logp_drop"])
        payload.pop("prior_informed_cloud_logp_drop", None)
    stored = Phase1CConfig(**payload)
    return replace(stored, output_dir=config.output_dir, run_id=config.run_id)


def _target_steps(config: Phase1CConfig, default_steps: int, *, resume: bool) -> int:
    """Resolve target total steps, supporting extension by additional steps."""
    if config.additional_steps is not None and config.target_total_steps is not None:
        raise ValueError("Use either --phase1c-additional-steps or --phase1c-target-total-steps, not both.")
    current = min(_stored_iterations(config.output_dir).values(), default=0)
    if config.additional_steps is not None:
        if not resume:
            raise ValueError("--phase1c-additional-steps requires --phase1c-resume.")
        if config.additional_steps <= 0:
            raise ValueError("--phase1c-additional-steps must be positive.")
        return int(current + config.additional_steps)
    if config.target_total_steps is not None:
        if config.target_total_steps < current:
            raise ValueError("Requested target total steps is below the stored checkpoint length.")
        return int(config.target_total_steps)
    return int(default_steps)


def sanitize_run_id(run_id: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_id.strip())
    if not cleaned:
        raise ValueError("Phase 1C run ID cannot be empty.")
    return cleaned


def timestamp_run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _stored_iterations(output_dir: Path) -> dict[str, int]:
    import emcee

    iterations = {}
    for path in sorted(output_dir.glob("ensemble_*.h5")):
        try:
            backend = emcee.backends.HDFBackend(str(path), read_only=True)
            iterations[path.name] = int(backend.iteration)
        except (OSError, AttributeError):
            continue
    return iterations


def _begin_invocation(
    config: Phase1CConfig,
    mode: str,
    *,
    starting_iterations: dict[str, int],
    target_steps: int,
) -> dict[str, Any]:
    history = _read_invocation_history(config.output_dir)
    record = {
        "invocation_sequence_number": len(history) + 1,
        "timestamp_start_utc": datetime.now(UTC).isoformat(),
        "status": "started",
        "mode": mode,
        "run_id": config.run_id,
        "starting_iterations": starting_iterations,
        "target_total_steps": int(target_steps),
        "mutable_execution_controls": _mutable_execution_controls(config),
        "git": {"commit": _git_output("rev-parse", "HEAD"), "status": _git_output("status", "--short")},
    }
    history.append(record)
    write_json(config.output_dir / "invocation_history.json", {"invocations": history})
    return record


def _finish_invocation(
    config: Phase1CConfig,
    record: dict[str, Any],
    *,
    status: str,
    ending_iterations: dict[str, int],
    elapsed_seconds: float,
    posterior_calls: int,
    stop_reason: str,
) -> None:
    history = _read_invocation_history(config.output_dir)
    sequence = int(record["invocation_sequence_number"])
    prior = [item for item in history if int(item.get("invocation_sequence_number", -1)) != sequence]
    cumulative_calls = posterior_calls + sum(int(item.get("new_posterior_calls", 0)) for item in prior)
    cumulative_runtime = elapsed_seconds + sum(float(item.get("invocation_runtime_seconds", 0.0)) for item in prior)
    record.update(
        {
            "timestamp_end_utc": datetime.now(UTC).isoformat(),
            "status": status,
            "ending_iterations": ending_iterations,
            "new_posterior_calls": int(posterior_calls),
            "invocation_runtime_seconds": float(elapsed_seconds),
            "cumulative_posterior_calls": int(cumulative_calls),
            "cumulative_sampling_runtime_seconds": float(cumulative_runtime),
            "stop_reason": stop_reason,
        }
    )
    history = [record if int(item.get("invocation_sequence_number", -1)) == sequence else item for item in history]
    write_json(config.output_dir / "invocation_history.json", {"invocations": history})


def _mutable_execution_controls(config: Phase1CConfig) -> dict[str, Any]:
    return {
        "pilot_steps": config.pilot_steps,
        "synthetic_steps": config.synthetic_steps,
        "synthetic_recovery_steps": config.synthetic_recovery_steps,
        "production_steps": config.production_steps,
        "target_total_steps": config.target_total_steps,
        "additional_steps": config.additional_steps,
        "chunk_steps": config.chunk_steps,
        "max_pilot_seconds": config.max_pilot_seconds,
        "minimum_meaningful_summary_draws": config.minimum_meaningful_summary_draws,
    }


def _read_invocation_history(output_dir: Path) -> list[dict[str, Any]]:
    path = output_dir / "invocation_history.json"
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("invocations", [])
    except (OSError, json.JSONDecodeError):
        return []


def _write_json_if_absent(path: Path, payload: dict[str, Any]) -> None:
    if not path.exists():
        write_json(path, payload)


def _load_summarize_data(config: Phase1CConfig, mode: str) -> tuple[FrozenPhase1BData, TimingReference]:
    if mode in {"synthetic", "synthetic_recovery"}:
        data, timing, _ = synthetic_dataset(config)
        return data, timing
    data = load_frozen_phase1b(config)
    return data, build_timing_reference(data, config)


def _validate_synthetic_input_record_if_needed(
    data: FrozenPhase1BData,
    timing: TimingReference,
    config: Phase1CConfig,
    mode: str,
) -> None:
    if mode not in {"synthetic", "synthetic_recovery"}:
        return
    path = config.output_dir / "synthetic_input_record.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing synthetic input record: {path}")
    recorded = _read_json(path)
    expected = _synthetic_input_record(data, timing)
    for key in ("type", "cadence_count", "event_count", "residuals_csv_used_as_input"):
        if recorded.get(key) != expected[key]:
            raise ValueError(f"synthetic_input_record.json mismatch for {key}.")
    for key, value in expected["timing_reference"].items():
        recorded_value = recorded.get("timing_reference", {}).get(key)
        if isinstance(value, float):
            if not np.isclose(float(recorded_value), value, rtol=0.0, atol=1.0e-12):
                raise ValueError(f"synthetic_input_record.json timing mismatch for {key}.")
        elif recorded_value != value:
            raise ValueError(f"synthetic_input_record.json timing mismatch for {key}.")


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
    starting_iterations = _stored_iterations(config.output_dir)
    invocation = _begin_invocation(config, mode, starting_iterations=starting_iterations, target_steps=steps)
    try:
        audit = timing_support_audit(data, timing)
        if not audit["center_remains_inside_every_frozen_window"]:
            raise RuntimeError("Phase 1C timing support audit failed: centers leave frozen windows.")
        _write_common_inputs(data, config, timing, mode=mode, timing_audit=audit, immutable=resume)
        if mode not in {"synthetic", "synthetic_recovery"}:
            if not resume or not (config.output_dir / "phase1b_input_manifest.json").exists():
                write_phase1b_input_manifest(config.phase1b_output_dir, config.output_dir)
        else:
            writer = _write_json_if_absent if resume else write_json
            writer(config.output_dir / "synthetic_input_record.json", _synthetic_input_record(data, timing))

        history_state: dict[str, Any] = {
            "summary_history": _read_posterior_summary_history(config.output_dir),
            "rows": _read_existing_history(config.output_dir),
        }

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
        summary = _write_sampling_summaries(
            data,
            config,
            timing,
            results,
            mode=mode,
            elapsed_seconds=elapsed,
            summary_history=history_state["summary_history"],
            latest_invocation=invocation,
        )
    except KeyboardInterrupt:
        _finish_invocation(
            config,
            invocation,
            status="interrupted",
            ending_iterations=_stored_iterations(config.output_dir),
            elapsed_seconds=time.perf_counter() - start,
            posterior_calls=0,
            stop_reason="interrupted",
        )
        raise
    except Exception:
        _finish_invocation(
            config,
            invocation,
            status="failed",
            ending_iterations=_stored_iterations(config.output_dir),
            elapsed_seconds=time.perf_counter() - start,
            posterior_calls=0,
            stop_reason="failed",
        )
        raise
    _finish_invocation(
        config,
        invocation,
        status="completed",
        ending_iterations=_stored_iterations(config.output_dir),
        elapsed_seconds=summary["elapsed_seconds"],
        posterior_calls=summary["actual_log_posterior_calls"],
        stop_reason="target_steps_reached",
    )
    _rewrite_runtime_with_finished_invocation(config)
    update_run_index(config.output_dir.parent, config, mode, summary["diagnostic_status"])
    return summary


def _write_common_inputs(
    data: FrozenPhase1BData,
    config: Phase1CConfig,
    timing: TimingReference,
    *,
    mode: str,
    timing_audit: dict[str, Any],
    immutable: bool = False,
) -> None:
    writer = _write_json_if_absent if immutable else write_json
    writer(config.output_dir / "phase1c_configuration.json", config.to_dict())
    writer(config.output_dir / "parameter_transformations.json", prior_description(data, config, timing))
    writer(config.output_dir / "timing_support_audit.json", timing_audit)
    writer(config.output_dir / "provenance_manifest.json", build_phase1c_provenance(data, config, timing, mode=mode))


def _write_sampling_summaries(
    data: FrozenPhase1BData,
    config: Phase1CConfig,
    timing: TimingReference,
    results,
    *,
    mode: str,
    elapsed_seconds: float,
    summary_history: list[pd.DataFrame],
    latest_invocation: dict[str, Any],
) -> dict[str, Any]:
    diagnostic_start = time.perf_counter()
    chain, log_prob = load_backend_chains(results)
    chains_by_ensemble = load_backend_chains_by_ensemble(results)
    acceptance = np.concatenate([result.acceptance_fraction for result in results])
    autocorr = estimate_autocorrelation_time(results, config.warmup_steps)
    posterior = posterior_summary_frame(chain, timing, warmup_steps=config.warmup_steps)
    ensemble = ensemble_summary_frame(chains_by_ensemble, config.warmup_steps)
    history_for_diagnostics = summary_history if summary_history else [posterior]
    stability = posterior_stability_check(history_for_diagnostics, config)
    agreement = independent_ensemble_agreement(chains_by_ensemble, timing, config, warmup_steps=config.warmup_steps)
    diagnostics = convergence_diagnostics(
        chain,
        log_prob,
        acceptance,
        autocorr,
        config,
        warmup_steps=config.warmup_steps,
        posterior_stability=stability,
        ensemble_agreement=agreement,
    )
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
    invocation_calls = int(profiler_summary.get("posterior_calls", 0))
    runtime = {
        "mode": mode,
        "run_id": config.run_id,
        "cumulative_totals": _cumulative_runtime_totals(
            config.output_dir,
            pending_calls=invocation_calls,
            pending_elapsed_seconds=elapsed_seconds,
        ),
        "latest_invocation": {
            **latest_invocation,
            "elapsed_seconds_so_far": float(elapsed_seconds),
            "posterior_calls_this_invocation": invocation_calls,
            "posterior_calls_per_second": float(invocation_calls / elapsed_seconds) if elapsed_seconds > 0.0 else None,
            "stored_draws_per_second": float(np.size(log_prob) / elapsed_seconds) if elapsed_seconds > 0.0 else None,
            "timing_seconds": {
                **profiler_summary,
                "output_and_diagnostic_overhead_seconds": output_overhead + (time.perf_counter() - diagnostic_start),
            },
        },
        "invocation_history_path": str(config.output_dir / "invocation_history.json"),
        "invocation_history": _read_invocation_history(config.output_dir),
        "total_iterations_per_ensemble": [result.iterations for result in results],
        "stored_log_probability_entries": int(np.size(log_prob)),
        "finite_log_probability_fraction": float(np.mean(np.isfinite(log_prob))),
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
        "actual_log_posterior_calls": invocation_calls,
        "posterior_calls_per_second": runtime["latest_invocation"]["posterior_calls_per_second"],
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
    summary = None
    try:
        summary = posterior_summary_frame(chain, timing, warmup_steps=config.warmup_steps)
        history_state["summary_history"].append(summary)
        _append_posterior_summary_history(
            config.output_dir,
            {
                "mode": mode,
                "run_id": config.run_id,
                "completed_steps": int(min(result.iterations for result in results)),
                "timestamp_utc": datetime.now(UTC).isoformat(),
                "summary": summary.to_dict(orient="records"),
            },
        )
    except (ValueError, IndexError):
        pass
    chains_by_ensemble = load_backend_chains_by_ensemble(results)
    stability = posterior_stability_check(history_state["summary_history"], config)
    agreement = independent_ensemble_agreement(chains_by_ensemble, timing, config, warmup_steps=config.warmup_steps)
    diagnostics = convergence_diagnostics(
        chain,
        log_prob,
        acceptance,
        autocorr,
        config,
        warmup_steps=config.warmup_steps,
        posterior_stability=stability,
        ensemble_agreement=agreement,
    )
    kept = post_warmup_chain(chain, config.warmup_steps)
    completed_steps = int(min(result.iterations for result in results))
    retained_steps = int(kept.shape[1])
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
        "autocorrelation_time_max": diagnostics["autocorrelation_worst_tau"],
        "complete_valid_autocorrelation": diagnostics["criteria"]["complete_valid_autocorrelation"],
        "chain_length_exceeds_tau_multiple": diagnostics["criteria"]["chain_length_exceeds_tau_multiple"],
        "finite_log_probability_fraction_is_one": diagnostics["criteria"]["finite_log_probability_fraction_is_one"],
        "no_severe_walker_pathology": diagnostics["criteria"]["no_severe_walker_pathology"],
        "severe_walker_count": diagnostics["walker_health"]["severe_walker_count"],
        "diagnostic_backend": diagnostics["standard_diagnostic_backend"],
        "diagnostic_availability": diagnostics["standard_diagnostic_backend"],
        "diagnostic_methodology_version": diagnostics["diagnostic_methodology_version"],
        "posterior_stability_passed": stability["passed"],
        "independent_ensemble_agreement_passed": agreement["passed"],
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
        "diagnostic_methodology_version": config.diagnostic_methodology_version,
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
    entry = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "mode": mode,
        "run_id": config.run_id,
        "run_directory": str(config.output_dir),
        "status": status,
    }
    payload = [
        item
        for item in payload
        if not (item.get("mode") == mode and item.get("run_id") == config.run_id)
    ]
    payload.append(entry)
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
        "nonproduction": True,
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


def _posterior_summary_history_path(output_dir: Path) -> Path:
    return output_dir / "posterior_summary_history.jsonl"


def _append_posterior_summary_history(output_dir: Path, record: dict[str, Any]) -> None:
    path = _posterior_summary_history_path(output_dir)
    with path.open("a", encoding="utf-8") as output_file:
        output_file.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")


def _read_posterior_summary_history(output_dir: Path) -> list[pd.DataFrame]:
    path = _posterior_summary_history_path(output_dir)
    if not path.exists():
        return []
    frames = []
    with path.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            if not line.strip():
                continue
            payload = json.loads(line)
            frames.append(pd.DataFrame(payload["summary"]))
    return frames


def _cumulative_runtime_totals(
    output_dir: Path,
    *,
    pending_calls: int = 0,
    pending_elapsed_seconds: float = 0.0,
) -> dict[str, Any]:
    history = _read_invocation_history(output_dir)
    completed = [item for item in history if item.get("status") == "completed"]
    return {
        "posterior_calls": int(pending_calls + sum(int(item.get("new_posterior_calls", 0)) for item in completed)),
        "sampling_runtime_seconds": float(
            pending_elapsed_seconds + sum(float(item.get("invocation_runtime_seconds", 0.0)) for item in completed)
        ),
        "completed_invocation_count": len(completed),
        "invocation_count": len(history),
    }


def _rewrite_runtime_with_finished_invocation(config: Phase1CConfig) -> None:
    path = config.output_dir / "sampler_runtime.json"
    if not path.exists():
        return
    runtime = _read_json(path)
    history = _read_invocation_history(config.output_dir)
    latest = next(
        (
            item
            for item in history
            if int(item.get("invocation_sequence_number", -1))
            == int(runtime.get("latest_invocation", {}).get("invocation_sequence_number", -2))
        ),
        runtime.get("latest_invocation", {}),
    )
    runtime["latest_invocation"] = latest
    runtime["invocation_history"] = history
    runtime["cumulative_totals"] = _cumulative_runtime_totals(config.output_dir)
    write_json(path, runtime)


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
