"""Phase 1B deterministic physical transit fitting pipeline."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .config import DEFAULT_TIME_SYSTEM
from .phase1b_fit import FitResult, evaluate_solution, run_fit_stage
from .phase1b_model import PhysicalParameters, q_to_u, u_to_q
from .phase1b_observations import build_transit_windows, observations_from_bundle
from .phase1b_outputs import (
    acf_rows,
    residual_rows,
    residual_summary,
    rms_binning_rows,
    save_diagnostic_plots,
    write_csv,
)
from .phase1b_types import FitData, LimbDarkeningInputs, Phase1BConfig
from .provenance import build_provenance_manifest, write_json

ANTI_LEAKAGE_STATEMENT = (
    "Published physical planet parameters and published ephemeris constants are not imported "
    "or used by generic Phase 1B fitting code, initialization, bounds, diagnostics, or tests."
)


def run_phase1b_fit(
    *,
    bundle,
    output_dir: Path,
    config: Phase1BConfig,
    target: str,
    mission: str,
    author: str,
    cadence: str,
    flux_product: str,
    quality_bitmask: str | int,
    preprocessing: dict[str, Any],
) -> dict[str, Any]:
    """Run Phase 1B and write auditable deterministic fit products."""
    if not str(cadence).lower().startswith("long"):
        raise ValueError("Phase 1B currently accepts Kepler long cadence only.")
    output_dir.mkdir(parents=True, exist_ok=True)
    phase1a_summary = _load_phase1a_summary(config.phase1a_summary_path)
    phase1a_provenance = _read_json_if_exists(config.phase1a_provenance_path)
    refinement = _full_mission_refinement(phase1a_summary)
    limb = load_limb_darkening_inputs(config.stellar_inputs_path)

    observations, metadata_rows = observations_from_bundle(bundle, flux_product)
    _reject_mixed_cadence(observations.cadence)
    windows = build_transit_windows(
        observations,
        period_days=refinement["refined_period_days"],
        transit_time=refinement["refined_transit_time"],
        duration_days=refinement["refined_duration_days"],
        config=config,
    )
    if not np.any(windows.accepted_mask):
        write_csv(output_dir / "transit_window_audit.csv", windows.audit_rows)
        raise RuntimeError("No transit windows passed objective Phase 1B coverage criteria.")

    data = FitData(
        observations=observations,
        time=observations.time[windows.accepted_mask],
        flux=observations.flux[windows.accepted_mask],
        flux_err=observations.flux_err[windows.accepted_mask],
        event_number=windows.event_number[windows.accepted_mask],
        predicted_center=windows.predicted_center[windows.accepted_mask],
        exposure_days=observations.exposure_days[windows.accepted_mask],
        product_id=observations.product_id[windows.accepted_mask],
        quarter=observations.quarter[windows.accepted_mask],
        phase1a_period_days=refinement["refined_period_days"],
        phase1a_transit_time=refinement["refined_transit_time"],
        phase1a_duration_days=refinement["refined_duration_days"],
    )

    fixed = run_fit_stage(data, limb, config, stage="fixed_phase1a_timing", fit_timing=False)
    refined = run_fit_stage(data, limb, config, stage="global_timing_refinement", fit_timing=True)
    fixed_ld = run_fit_stage(
        data,
        limb,
        config,
        stage="fixed_theoretical_limb_darkening",
        fit_timing=False,
        fixed_limb_darkening=True,
        n_starts=max(3, min(config.n_starts, 6)),
    )

    exposure_rows = exposure_convergence_rows(data, refined, config)
    stability_rows = stability_diagnostics(data, limb, config, refined)
    residuals = residual_rows(data, refined, "global_timing_refinement")
    acf = acf_rows(refined.residuals)
    rms = rms_binning_rows(data, refined.residuals)
    residual_stats = residual_summary(data, refined)
    input_record = phase1a_input_record(
        config.phase1a_summary_path,
        config.phase1a_provenance_path,
        phase1a_summary,
        phase1a_provenance,
    )
    provenance = build_provenance_manifest(
        target=target,
        mission=mission,
        author=author,
        cadence=cadence,
        flux_product=flux_product,
        time_system=DEFAULT_TIME_SYSTEM,
        quality_bitmask=quality_bitmask,
        preprocessing=preprocessing,
        stitching_policy=bundle.stitching_policy,
        downloaded_paths=bundle.downloaded_paths,
        cadence_counts={"phase1b_fit_cadence_count": len(data.time)},
    )
    provenance["phase1b"] = {
        "configuration": config.to_dict(),
        "anti_leakage_statement": ANTI_LEAKAGE_STATEMENT,
        "phase1a_input_record": input_record,
        "random_seed": config.random_seed,
        "python_executable": sys.executable,
        "raw_input_match_to_phase1a": compare_raw_inputs(provenance, phase1a_provenance),
    }

    write_json(output_dir / "phase1b_configuration.json", config.to_dict())
    write_json(output_dir / "phase1a_input_record.json", input_record)
    write_csv(output_dir / "observation_product_metadata.csv", metadata_rows)
    write_csv(output_dir / "transit_window_audit.csv", windows.audit_rows)
    write_csv(output_dir / "accepted_fit_cadences.csv", accepted_cadence_rows(data))
    write_csv(output_dir / "event_baseline_parameters.csv", refined.baseline_rows)
    write_csv(output_dir / "multistart_diagnostics.csv", [*fixed.multistart_rows, *refined.multistart_rows, *fixed_ld.multistart_rows])
    write_csv(
        output_dir / "deterministic_fit_parameters.csv",
        [
            {"stage": fixed.stage, "objective_value": fixed.objective_value, **fixed.parameters},
            {"stage": refined.stage, "objective_value": refined.objective_value, **refined.parameters},
            {"stage": fixed_ld.stage, "objective_value": fixed_ld.objective_value, **fixed_ld.parameters},
        ],
    )
    write_json(output_dir / "timing_refinement_comparison.json", timing_comparison(fixed, refined))
    write_json(output_dir / "limb_darkening_inputs.json", limb.metadata)
    write_json(output_dir / "limb_darkening_comparison.json", limb_darkening_comparison(refined, fixed_ld))
    write_csv(output_dir / "exposure_integration_convergence.csv", exposure_rows)
    write_csv(output_dir / "stability_diagnostics.csv", stability_rows)
    write_json(output_dir / "residual_summary.json", residual_stats)
    write_csv(output_dir / "residuals.csv", residuals)
    write_csv(output_dir / "residual_acf.csv", acf)
    write_csv(output_dir / "residual_rms_binning.csv", rms)
    write_json(output_dir / "synthetic_recovery_summary.json", {"see_tests": "tests/test_phase1b.py"})
    write_json(output_dir / "provenance_manifest.json", provenance)
    save_diagnostic_plots(output_dir, data, refined, acf, rms)

    summary = build_summary(
        config,
        input_record,
        limb,
        windows.audit_rows,
        fixed,
        refined,
        fixed_ld,
        exposure_rows,
        stability_rows,
        residual_stats,
        provenance,
    )
    write_json(output_dir / "phase1b_summary.json", summary)
    return summary


def load_limb_darkening_inputs(path: Path) -> LimbDarkeningInputs:
    """Load reproducible stellar/limb-darkening inputs from JSON."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing limb-darkening input file: {path}. Provide independently sourced stellar "
            "inputs and reproducible Kepler-band quadratic coefficients before running real Phase 1B."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    coefficients = payload.get("quadratic_coefficients", {})
    if "q1" in coefficients and "q2" in coefficients:
        q1, q2 = float(coefficients["q1"]), float(coefficients["q2"])
        u1, u2 = q_to_u(q1, q2)
    else:
        u1, u2 = float(coefficients["u1"]), float(coefficients["u2"])
        q1, q2 = u_to_q(u1, u2)
    uncertainties = payload.get("coefficient_uncertainties", {})
    q1_sigma = float(uncertainties.get("q1_sigma", payload.get("q1_sigma", 0.1)))
    q2_sigma = float(uncertainties.get("q2_sigma", payload.get("q2_sigma", 0.1)))
    if not (0.0 <= q1 <= 1.0 and 0.0 <= q2 <= 1.0):
        raise ValueError("Limb-darkening q1/q2 inputs are outside the physical unit square.")
    metadata = dict(payload)
    metadata["loaded_path"] = str(path)
    metadata["loaded_sha256"] = _sha256(path)
    metadata["converted_q1"] = q1
    metadata["converted_q2"] = q2
    metadata["converted_u1"] = u1
    metadata["converted_u2"] = u2
    metadata["prior_treatment"] = "Gaussian penalty on q1/q2 for deterministic MAP fit"
    return LimbDarkeningInputs(q1=q1, q2=q2, q1_sigma=q1_sigma, q2_sigma=q2_sigma, u1=u1, u2=u2, metadata=metadata)


def exposure_convergence_rows(data: FitData, fit: FitResult, config: Phase1BConfig) -> list[dict[str, Any]]:
    params = PhysicalParameters(
        rp=fit.parameters["rp_over_rstar"],
        a=fit.parameters["a_over_rstar"],
        b=fit.parameters["impact_parameter"],
        q1=fit.parameters["q1"],
        q2=fit.parameters["q2"],
        jitter=fit.parameters["white_noise_jitter"],
        period=fit.parameters["period_days"],
        t0=fit.parameters["transit_time"],
    )
    default_model = fit.transit_model
    high_config = replace(config, supersample_factor=config.high_supersample_factor)
    high_model = evaluate_solution(data, params, high_config)["transit_model"]
    diff = high_model - default_model
    scatter = float(np.nanstd(fit.residuals, ddof=1))
    return [
        {
            "default_supersample_factor": config.supersample_factor,
            "comparison_supersample_factor": config.high_supersample_factor,
            "max_abs_transit_model_difference": float(np.nanmax(np.abs(diff))),
            "rms_transit_model_difference": float(np.sqrt(np.nanmean(diff**2))),
            "residual_scatter": scatter,
            "max_difference_fraction_of_residual_scatter": float(np.nanmax(np.abs(diff)) / scatter) if scatter else float("nan"),
            "parameter_refit_performed": False,
            "acceptance_note": "Model-grid convergence checked; high-supersample parameter refit deferred if runtime is prohibitive.",
        }
    ]


def stability_diagnostics(
    data: FitData,
    limb: LimbDarkeningInputs,
    config: Phase1BConfig,
    reference: FitResult,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    midpoint = float(np.nanmedian(data.time))
    groups = [("early_mission", data.time <= midpoint), ("late_mission", data.time > midpoint)]
    for label, mask in groups:
        rows.append(_fit_subset(label, mask, data, limb, config, reference))
    for quarter in sorted(set(map(str, data.quarter))):
        if not quarter:
            continue
        rows.append(_fit_subset(f"quarter_{quarter}", data.quarter.astype(str) == quarter, data, limb, config, reference))
    rows.append(
        {
            "diagnostic": "fixed_vs_fitted_limb_darkening",
            "status": "see_limb_darkening_comparison_json",
            "sample_cadence_count": int(len(data.time)),
        }
    )
    rows.append(
        {
            "diagnostic": "fixed_timing_vs_global_timing_refinement",
            "status": "see_timing_refinement_comparison_json",
            "sample_cadence_count": int(len(data.time)),
        }
    )
    return rows


def _fit_subset(
    label: str,
    mask: np.ndarray,
    data: FitData,
    limb: LimbDarkeningInputs,
    config: Phase1BConfig,
    reference: FitResult,
) -> dict[str, Any]:
    if np.count_nonzero(mask) < 40 or len(np.unique(data.event_number[mask])) < 2:
        return {"diagnostic": label, "status": "skipped_insufficient_coverage", "sample_cadence_count": int(np.count_nonzero(mask))}
    subset = FitData(
        observations=data.observations,
        time=data.time[mask],
        flux=data.flux[mask],
        flux_err=data.flux_err[mask],
        event_number=data.event_number[mask],
        predicted_center=data.predicted_center[mask],
        exposure_days=data.exposure_days[mask],
        product_id=data.product_id[mask],
        quarter=data.quarter[mask],
        phase1a_period_days=data.phase1a_period_days,
        phase1a_transit_time=data.phase1a_transit_time,
        phase1a_duration_days=data.phase1a_duration_days,
    )
    try:
        fit = run_fit_stage(subset, limb, config, stage=label, fit_timing=False, n_starts=max(3, min(5, config.n_starts)))
    except RuntimeError as exc:
        return {"diagnostic": label, "status": "fit_failed", "message": str(exc), "sample_cadence_count": int(np.count_nonzero(mask))}
    return {
        "diagnostic": label,
        "status": "fit_completed",
        "sample_cadence_count": int(np.count_nonzero(mask)),
        "event_count": int(len(np.unique(subset.event_number))),
        "objective_value": fit.objective_value,
        "rp_over_rstar": fit.parameters["rp_over_rstar"],
        "a_over_rstar": fit.parameters["a_over_rstar"],
        "impact_parameter": fit.parameters["impact_parameter"],
        "delta_rp_over_rstar_from_reference": fit.parameters["rp_over_rstar"] - reference.parameters["rp_over_rstar"],
        "delta_a_over_rstar_from_reference": fit.parameters["a_over_rstar"] - reference.parameters["a_over_rstar"],
        "delta_impact_parameter_from_reference": fit.parameters["impact_parameter"] - reference.parameters["impact_parameter"],
        "warnings": ";".join(fit.warnings),
    }


def accepted_cadence_rows(data: FitData) -> list[dict[str, Any]]:
    return [
        {
            "time": float(data.time[index]),
            "flux": float(data.flux[index]),
            "flux_uncertainty": float(data.flux_err[index]),
            "event_number": int(data.event_number[index]),
            "predicted_center": float(data.predicted_center[index]),
            "product_id": str(data.product_id[index]),
            "quarter": str(data.quarter[index]),
            "exposure_days": float(data.exposure_days[index]),
        }
        for index in range(len(data.time))
    ]


def timing_comparison(fixed: FitResult, refined: FitResult) -> dict[str, Any]:
    return {
        "fixed_timing": {"objective_value": fixed.objective_value, "parameters": fixed.parameters},
        "timing_refined": {"objective_value": refined.objective_value, "parameters": refined.parameters},
        "shifts": {
            key: refined.parameters[key] - fixed.parameters[key]
            for key in ("period_days", "transit_time", "rp_over_rstar", "a_over_rstar", "impact_parameter")
        },
    }


def limb_darkening_comparison(fitted: FitResult, fixed: FitResult) -> dict[str, Any]:
    keys = ("rp_over_rstar", "a_over_rstar", "impact_parameter", "objective_value")
    return {
        "atmosphere_informed_fitted": {"objective_value": fitted.objective_value, "parameters": fitted.parameters},
        "fixed_theoretical_coefficients": {"objective_value": fixed.objective_value, "parameters": fixed.parameters},
        "shifts_fixed_minus_fitted": {
            key: (fixed.objective_value if key == "objective_value" else fixed.parameters[key])
            - (fitted.objective_value if key == "objective_value" else fitted.parameters[key])
            for key in keys
        },
    }


def build_summary(
    config: Phase1BConfig,
    input_record: dict[str, Any],
    limb: LimbDarkeningInputs,
    audit_rows: list[dict[str, Any]],
    fixed: FitResult,
    refined: FitResult,
    fixed_ld: FitResult,
    exposure_rows: list[dict[str, Any]],
    stability_rows: list[dict[str, Any]],
    residual_stats: dict[str, Any],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    included = [row for row in audit_rows if row["included"]]
    excluded = [row for row in audit_rows if not row["included"]]
    reason_counts: dict[str, int] = {}
    for row in excluded:
        for reason in str(row["exclusion_reasons"]).split(";"):
            if reason:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
    warnings = [*fixed.warnings, *refined.warnings]
    if any(not row.get("parameter_refit_performed", False) for row in exposure_rows):
        warnings.append("high_supersample_parameter_refit_not_run")
    return {
        "established_inputs": input_record,
        "modeling_assumptions": {
            "transit_model": "spherical star, spherical opaque planet, circular orbit, no dilution, quadratic limb darkening",
            "fitting_data": "unbinned accepted cadences in objective transit windows",
            "local_baseline": "one multiplicative linear baseline solved analytically per accepted transit",
            "exposure_integration": "BATMAN supersample exposure integration grouped by exposure duration",
            "posterior_uncertainty": "deferred to Phase 1B-B",
            "anti_leakage_statement": ANTI_LEAKAGE_STATEMENT,
        },
        "configuration": config.to_dict(),
        "limb_darkening": limb.metadata,
        "transit_windows": {
            "predicted_count": len(audit_rows),
            "included_count": len(included),
            "excluded_count": len(excluded),
            "exclusion_reason_counts": reason_counts,
        },
        "fitted_results": {
            "fixed_phase1a_timing": {"objective_value": fixed.objective_value, **fixed.parameters},
            "global_timing_refinement": {"objective_value": refined.objective_value, **refined.parameters},
            "fixed_theoretical_limb_darkening": {"objective_value": fixed_ld.objective_value, **fixed_ld.parameters},
        },
        "diagnostic_results": {
            "exposure_integration_convergence": exposure_rows,
            "stability_diagnostics": stability_rows,
            "residual_summary": residual_stats,
            "raw_input_match_to_phase1a": provenance["phase1b"]["raw_input_match_to_phase1a"],
        },
        "warnings": sorted(set(warnings)),
        "unresolved_limitations": [
            "Deterministic optimizer diagnostics are not final parameter uncertainties.",
            "Posterior sampling and convergence analysis are deferred to Phase 1B-B.",
        ],
        "acceptance_checks": {
            "physical_batman_model_fit": True,
            "multiple_initializations_recorded": True,
            "local_baselines_solved_simultaneously": True,
            "published_physical_planet_parameters_used_or_compared": False,
        },
    }


def phase1a_input_record(summary_path: Path, provenance_path: Path, summary: dict[str, Any], provenance: dict[str, Any]) -> dict[str, Any]:
    return {
        "phase1a_summary_path": str(summary_path),
        "phase1a_summary_sha256": _sha256(summary_path),
        "phase1a_provenance_path": str(provenance_path),
        "phase1a_provenance_sha256": _sha256(provenance_path) if provenance_path.exists() else None,
        "locked_training_ephemeris": summary.get("locked_refined_training_candidate", {}),
        "full_mission_local_refinement": summary.get("full_mission_local_refinement", {}),
        "phase1a_provenance_available": bool(provenance),
    }


def compare_raw_inputs(current: dict[str, Any], previous: dict[str, Any]) -> dict[str, Any]:
    current_raw = {row.get("path"): row.get("sha256") for row in current.get("raw_inputs", []) if row.get("path")}
    previous_raw = {row.get("path"): row.get("sha256") for row in previous.get("raw_inputs", []) if row.get("path")}
    if not current_raw or not previous_raw:
        return {
            "status": "not_verifiable",
            "reason": "Current or Phase 1A provenance lacks exact raw FITS paths/checksums.",
        }
    mismatches = [
        {"path": path, "phase1a_sha256": checksum, "phase1b_sha256": current_raw.get(path)}
        for path, checksum in previous_raw.items()
        if current_raw.get(path) != checksum
    ]
    if mismatches:
        raise RuntimeError(f"Phase 1B raw FITS checksums differ from Phase 1A provenance: {mismatches}")
    return {"status": "verified_match", "matched_file_count": len(previous_raw)}


def _load_phase1a_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing Phase 1A summary: {path}")
    summary = json.loads(path.read_text(encoding="utf-8"))
    refinement = _full_mission_refinement(summary)
    required = ("refined_period_days", "refined_transit_time", "refined_duration_days", "refined_depth")
    missing = [key for key in required if key not in refinement]
    if missing:
        raise ValueError(f"Phase 1A full_mission_local_refinement missing required fields: {missing}")
    return summary


def _full_mission_refinement(summary: dict[str, Any]) -> dict[str, Any]:
    try:
        return summary["full_mission_local_refinement"]
    except KeyError as exc:
        raise ValueError("Phase 1A summary lacks full_mission_local_refinement.") from exc


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _reject_mixed_cadence(cadence_values: np.ndarray) -> None:
    cadences = {str(value).lower() for value in cadence_values}
    if len(cadences) != 1 or not next(iter(cadences)).startswith("long"):
        raise ValueError(f"Phase 1B currently rejects mixed or non-long cadence fitting: {sorted(cadences)}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
