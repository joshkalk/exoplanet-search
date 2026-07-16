"""Frozen Phase 1B input loading and integrity checks for Phase 1C."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .phase1c_types import FrozenPhase1BData, Phase1CConfig

REQUIRED_PHASE1B_FILES = (
    "accepted_fit_cadences.csv",
    "transit_window_audit.csv",
    "observation_product_metadata.csv",
    "deterministic_fit_parameters.csv",
    "limb_darkening_inputs.json",
    "phase1b_configuration.json",
    "phase1b_summary.json",
    "provenance_manifest.json",
)


def load_frozen_phase1b(config: Phase1CConfig) -> FrozenPhase1BData:
    """Load and validate the frozen Phase 1B accepted-cadence snapshot."""
    input_dir = config.phase1b_output_dir
    manifest = build_phase1b_input_manifest(input_dir)
    accepted = pd.read_csv(input_dir / "accepted_fit_cadences.csv")
    audit = pd.read_csv(input_dir / "transit_window_audit.csv")
    products = pd.read_csv(input_dir / "observation_product_metadata.csv")
    parameters = pd.read_csv(input_dir / "deterministic_fit_parameters.csv")
    limb_darkening = _normalize_limb_darkening(_read_json(input_dir / "limb_darkening_inputs.json"))
    phase1b_configuration = _read_json(input_dir / "phase1b_configuration.json")
    phase1b_summary = _read_json(input_dir / "phase1b_summary.json")
    provenance = _read_json(input_dir / "provenance_manifest.json")

    _validate_accepted_cadences(accepted)
    _validate_phase1b_consistency(
        accepted=accepted,
        audit=audit,
        products=products,
        parameters=parameters,
        phase1b_configuration=phase1b_configuration,
        phase1b_summary=phase1b_summary,
        provenance=provenance,
    )
    timing_refined = _timing_refined_parameters(parameters)
    return FrozenPhase1BData(
        time=accepted["time"].to_numpy(dtype=float),
        flux=accepted["flux"].to_numpy(dtype=float),
        flux_uncertainty=accepted["flux_uncertainty"].to_numpy(dtype=float),
        event_number=accepted["event_number"].to_numpy(dtype=int),
        predicted_center=accepted["predicted_center"].to_numpy(dtype=float),
        product_id=accepted["product_id"].astype(str).to_numpy(),
        quarter=accepted["quarter"].astype(str).to_numpy(),
        exposure_days=accepted["exposure_days"].to_numpy(dtype=float),
        deterministic_parameters=timing_refined,
        limb_darkening=limb_darkening,
        phase1b_configuration=phase1b_configuration,
        phase1b_summary=phase1b_summary,
        provenance=provenance,
        input_manifest=manifest,
    )


def build_phase1b_input_manifest(input_dir: Path) -> dict[str, Any]:
    """Return required Phase 1B input paths and SHA-256 checksums."""
    files = []
    for name in REQUIRED_PHASE1B_FILES:
        path = input_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Missing required Phase 1B input: {path}")
        files.append({"name": name, "path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size})
    optional = input_dir / "phase1a_input_record.json"
    if optional.exists():
        files.append(
            {
                "name": optional.name,
                "path": str(optional),
                "sha256": sha256(optional),
                "bytes": optional.stat().st_size,
                "optional": True,
            }
        )
    digest = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "phase1b_output_dir": str(input_dir),
        "files": files,
        "manifest_sha256": digest,
        "residuals_csv_used_as_input": False,
    }


def write_phase1b_input_manifest(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    manifest = build_phase1b_input_manifest(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "phase1b_input_manifest.json").open("w", encoding="utf-8") as output_file:
        json.dump(manifest, output_file, indent=2)
    return manifest


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_accepted_cadences(accepted: pd.DataFrame) -> None:
    required = {
        "time",
        "flux",
        "flux_uncertainty",
        "event_number",
        "predicted_center",
        "product_id",
        "quarter",
        "exposure_days",
    }
    missing = sorted(required - set(accepted.columns))
    if missing:
        raise ValueError(f"accepted_fit_cadences.csv missing columns: {missing}")
    numeric_columns = ["time", "flux", "flux_uncertainty", "event_number", "predicted_center", "exposure_days"]
    for column in numeric_columns:
        values = pd.to_numeric(accepted[column], errors="coerce").to_numpy(dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"accepted_fit_cadences.csv has nonfinite {column} values.")
    if np.any(accepted["flux_uncertainty"].to_numpy(dtype=float) <= 0.0):
        raise ValueError("accepted_fit_cadences.csv has nonpositive flux uncertainties.")
    if np.any(accepted["exposure_days"].to_numpy(dtype=float) <= 0.0):
        raise ValueError("accepted_fit_cadences.csv has nonpositive exposure durations.")
    if accepted.duplicated(subset=["time", "event_number", "product_id"]).any():
        raise ValueError("accepted_fit_cadences.csv contains duplicate accepted observations.")


def _validate_phase1b_consistency(
    *,
    accepted: pd.DataFrame,
    audit: pd.DataFrame,
    products: pd.DataFrame,
    parameters: pd.DataFrame,
    phase1b_configuration: dict[str, Any],
    phase1b_summary: dict[str, Any],
    provenance: dict[str, Any],
) -> None:
    included = audit[audit["included"].astype(str).str.lower().isin({"true", "1"})]
    if len(included) != int(phase1b_summary["transit_windows"]["included_count"]):
        raise ValueError("Phase 1B included transit-window count mismatch.")
    if len(audit) != int(phase1b_summary["transit_windows"]["predicted_count"]):
        raise ValueError("Phase 1B predicted transit-window count mismatch.")
    event_ids = set(accepted["event_number"].astype(int))
    included_ids = set(included["event_number"].astype(int))
    if event_ids != included_ids:
        raise ValueError("Accepted cadence event identifiers do not match included audit rows.")
    if int(provenance.get("cadence_counts", {}).get("phase1b_fit_cadence_count", -1)) != len(accepted):
        raise ValueError("Phase 1B provenance cadence count does not match accepted cadences.")
    if set(accepted["product_id"].astype(str)) - set(products["product_id"].astype(str)):
        raise ValueError("Accepted cadence product identifiers are absent from product metadata.")
    if "global_timing_refinement" not in set(parameters["stage"].astype(str)):
        raise ValueError("deterministic_fit_parameters.csv lacks global_timing_refinement row.")
    if int(phase1b_configuration.get("supersample_factor", -1)) <= 0:
        raise ValueError("Phase 1B configuration has invalid supersample factor.")
    if phase1b_summary["acceptance_checks"].get("published_physical_planet_parameters_used_or_compared") is not False:
        raise ValueError("Phase 1B summary does not preserve anti-leakage acceptance state.")


def _timing_refined_parameters(parameters: pd.DataFrame) -> dict[str, float]:
    row = parameters[parameters["stage"].astype(str) == "global_timing_refinement"].iloc[0]
    keys = (
        "rp_over_rstar",
        "a_over_rstar",
        "impact_parameter",
        "q1",
        "q2",
        "white_noise_jitter",
        "period_days",
        "transit_time",
    )
    return {key: float(row[key]) for key in keys}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as input_file:
        return json.load(input_file)


def _normalize_limb_darkening(payload: dict[str, Any]) -> dict[str, Any]:
    if {"q1", "q2", "q1_sigma", "q2_sigma"} <= set(payload):
        return payload
    coefficients = payload.get("quadratic_coefficients", {})
    uncertainties = payload.get("coefficient_uncertainties", {})
    required = ("q1", "q2")
    missing = [key for key in required if key not in coefficients]
    missing.extend(key for key in ("q1_sigma", "q2_sigma") if key not in uncertainties)
    if missing:
        raise ValueError(f"limb_darkening_inputs.json missing coefficient fields: {sorted(missing)}")
    normalized = dict(payload)
    normalized.update(
        {
            "q1": float(coefficients["q1"]),
            "q2": float(coefficients["q2"]),
            "q1_sigma": float(uncertainties["q1_sigma"]),
            "q2_sigma": float(uncertainties["q2_sigma"]),
        }
    )
    return normalized
