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
    phase1a_input_record = None
    phase1a_path = input_dir / "phase1a_input_record.json"
    if phase1a_path.exists():
        phase1a_input_record = _read_json(phase1a_path)

    _validate_accepted_cadences(accepted)
    _validate_phase1b_consistency(
        accepted=accepted,
        audit=audit,
        products=products,
        parameters=parameters,
        phase1b_configuration=phase1b_configuration,
        phase1b_summary=phase1b_summary,
        provenance=provenance,
        phase1a_input_record=phase1a_input_record,
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
    _require_integral_values(accepted, "event_number", "accepted_fit_cadences.csv")
    for column in ("product_id", "quarter"):
        if accepted[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"accepted_fit_cadences.csv has blank {column} values.")


def _validate_phase1b_consistency(
    *,
    accepted: pd.DataFrame,
    audit: pd.DataFrame,
    products: pd.DataFrame,
    parameters: pd.DataFrame,
    phase1b_configuration: dict[str, Any],
    phase1b_summary: dict[str, Any],
    provenance: dict[str, Any],
    phase1a_input_record: dict[str, Any] | None,
) -> None:
    _require_columns(audit, {"event_number", "predicted_midpoint", "included"}, "transit_window_audit.csv")
    _require_columns(products, {"product_id", "quarter", "exposure_duration_days"}, "observation_product_metadata.csv")
    _require_columns(parameters, {"stage"}, "deterministic_fit_parameters.csv")
    _validate_embedded_phase1b_configuration(phase1b_configuration, provenance)
    _validate_embedded_phase1a_record(phase1a_input_record, provenance)
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
    _validate_integer_columns(accepted, audit, products)
    _validate_event_centers(accepted, included, phase1b_summary)
    _validate_product_metadata(accepted, products, phase1b_summary)
    if set(accepted["product_id"].astype(str)) - set(products["product_id"].astype(str)):
        raise ValueError("Accepted cadence product identifiers are absent from product metadata.")
    if "global_timing_refinement" not in set(parameters["stage"].astype(str)):
        raise ValueError("deterministic_fit_parameters.csv lacks global_timing_refinement row.")
    _validate_timing_refinement_parameters(parameters, phase1b_summary)
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


def _require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")


def _validate_embedded_phase1b_configuration(
    phase1b_configuration: dict[str, Any],
    provenance: dict[str, Any],
) -> None:
    embedded = provenance.get("phase1b", {}).get("configuration")
    if not isinstance(embedded, dict):
        raise ValueError("provenance_manifest.json lacks embedded Phase 1B configuration.")
    if _canonical_payload(embedded) != _canonical_payload(phase1b_configuration):
        raise ValueError("phase1b_configuration.json does not match provenance embedded configuration.")


def _validate_embedded_phase1a_record(
    phase1a_input_record: dict[str, Any] | None,
    provenance: dict[str, Any],
) -> None:
    embedded = provenance.get("phase1b", {}).get("phase1a_input_record")
    if phase1a_input_record is None:
        if embedded is not None:
            raise ValueError("provenance_manifest.json embeds a Phase 1A record but phase1a_input_record.json is absent.")
        return
    if not isinstance(embedded, dict):
        raise ValueError("phase1a_input_record.json exists but provenance lacks embedded Phase 1A record.")
    if _canonical_payload(embedded) != _canonical_payload(phase1a_input_record):
        raise ValueError("phase1a_input_record.json does not match provenance embedded Phase 1A record.")


def _validate_integer_columns(
    accepted: pd.DataFrame,
    audit: pd.DataFrame,
    products: pd.DataFrame,
) -> None:
    _require_integral_values(audit, "event_number", "transit_window_audit.csv")
    for column in (
        "total_point_count",
        "expected_in_transit_point_count",
        "left_in_transit_point_count",
        "right_in_transit_point_count",
        "pre_transit_baseline_count",
        "post_transit_baseline_count",
    ):
        if column in audit:
            _require_integral_values(audit, column, "transit_window_audit.csv")
    for column in ("product_index", "input_cadence_count", "finite_cadence_count"):
        if column in products:
            _require_integral_values(products, column, "observation_product_metadata.csv")
    if "quarter" in products:
        _require_integral_values(products, "quarter", "observation_product_metadata.csv")
    if accepted["quarter"].astype(str).str.fullmatch(r"[+-]?\d+").all():
        _require_integral_values(accepted, "quarter", "accepted_fit_cadences.csv")


def _validate_event_centers(
    accepted: pd.DataFrame,
    included: pd.DataFrame,
    phase1b_summary: dict[str, Any],
) -> None:
    ephemeris = phase1b_summary["established_inputs"]["full_mission_local_refinement"]
    frozen_period = float(ephemeris["refined_period_days"])
    frozen_t0 = float(ephemeris["refined_transit_time"])
    audit_midpoints = {
        int(row["event_number"]): float(row["predicted_midpoint"])
        for _, row in included.iterrows()
    }
    for event, group in accepted.groupby(accepted["event_number"].astype(int)):
        centers = pd.to_numeric(group["predicted_center"], errors="coerce").to_numpy(dtype=float)
        if float(np.max(centers) - np.min(centers)) > 1.0e-9:
            raise ValueError(f"Accepted cadences have nonconstant predicted_center for event {event}.")
        expected = frozen_t0 + event * frozen_period
        center = float(np.median(centers))
        if event not in audit_midpoints:
            raise ValueError(f"Accepted event {event} is absent from included transit audit rows.")
        if not np.isclose(center, audit_midpoints[event], rtol=0.0, atol=1.0e-9):
            raise ValueError(f"Accepted predicted_center disagrees with audit midpoint for event {event}.")
        if not np.isclose(center, expected, rtol=0.0, atol=1.0e-8):
            raise ValueError(f"Accepted predicted_center disagrees with frozen ephemeris for event {event}.")


def _validate_product_metadata(
    accepted: pd.DataFrame,
    products: pd.DataFrame,
    phase1b_summary: dict[str, Any],
) -> None:
    product_table = products.set_index(products["product_id"].astype(str), drop=False)
    product_counts = accepted.groupby(accepted["product_id"].astype(str)).size().to_dict()
    diagnostic_results = phase1b_summary.get("diagnostic_results", {})
    residual_counts = {
        str(row["product_id"]): int(row["cadence_count"])
        for row in diagnostic_results.get("residual_summary", {}).get("by_product", [])
        if "product_id" in row and "cadence_count" in row
    }
    for product_id, group in accepted.groupby(accepted["product_id"].astype(str)):
        if product_id not in product_table.index:
            raise ValueError(f"Accepted product {product_id} is absent from product metadata.")
        product_row = product_table.loc[product_id]
        if isinstance(product_row, pd.DataFrame):
            raise ValueError(f"Product metadata has duplicate product_id {product_id}.")
        quarters = {str(value) for value in group["quarter"]}
        if quarters != {str(product_row["quarter"])}:
            raise ValueError(f"Accepted quarter disagrees with product metadata for {product_id}.")
        accepted_exposure = pd.to_numeric(group["exposure_days"], errors="coerce").to_numpy(dtype=float)
        metadata_exposure = float(product_row["exposure_duration_days"])
        if not np.isfinite(metadata_exposure) or metadata_exposure <= 0.0:
            raise ValueError(f"Product metadata has invalid exposure duration for {product_id}.")
        if not np.allclose(accepted_exposure, metadata_exposure, rtol=0.0, atol=1.0e-12):
            raise ValueError(f"Accepted exposure duration disagrees with product metadata for {product_id}.")
    if residual_counts and residual_counts != {key: int(value) for key, value in product_counts.items()}:
        raise ValueError("Phase 1B residual by-product cadence counts do not match accepted cadences.")


def _validate_timing_refinement_parameters(
    parameters: pd.DataFrame,
    phase1b_summary: dict[str, Any],
) -> None:
    row = parameters[parameters["stage"].astype(str) == "global_timing_refinement"].iloc[0]
    summary = phase1b_summary.get("fitted_results", {}).get("global_timing_refinement")
    if not isinstance(summary, dict):
        raise ValueError("phase1b_summary.json lacks fitted_results.global_timing_refinement.")
    required = (
        "objective_value",
        "rp_over_rstar",
        "a_over_rstar",
        "impact_parameter",
        "q1",
        "q2",
        "white_noise_jitter",
        "period_days",
        "transit_time",
    )
    for key in required:
        if key not in row or key not in summary:
            raise ValueError(f"Global timing refinement is missing {key}.")
        tolerance = 1.0e-9 if key == "objective_value" else 1.0e-12
        if not np.isclose(float(row[key]), float(summary[key]), rtol=0.0, atol=tolerance):
            raise ValueError(f"Global timing refinement {key} disagrees between CSV and summary JSON.")


def _require_integral_values(frame: pd.DataFrame, column: str, label: str) -> None:
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    if not np.all(np.isfinite(values)) or not np.all(np.equal(values, np.rint(values))):
        raise ValueError(f"{label} has nonintegral {column} values.")


def _canonical_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _canonical_payload(item) for key, item in sorted(value.items())}
    if isinstance(value, list):
        return [_canonical_payload(item) for item in value]
    if isinstance(value, str):
        return value.replace("\\", "/")
    return value


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
