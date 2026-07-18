"""Synthetic Phase 1C datasets and authoritative recovery-gate identity."""

from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import math
import platform
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Mapping

import numpy as np

from .phase1b_model import batman_flux
from .phase1c_inputs import load_frozen_phase1b
from .phase1c_likelihood import event_local_coordinate, log_probability
from .phase1c_parameters import build_timing_reference, physical_to_vector
from .phase1c_types import FrozenPhase1BData, Phase1CConfig, PhysicalSample, TimingReference

TOY_SMOKE_DATASET_DESIGN = "toy_smoke_v1"
LEGACY_TOY_DATASET_DESIGN = "legacy_toy_synthetic"
REALISTIC_DATASET_DESIGN = "realistic_frozen_cadence_v1"
TOY_GENERATOR_VERSION = "phase1c_toy_smoke_generator_v1"
REALISTIC_GENERATOR_VERSION = "phase1c_realistic_frozen_cadence_generator_v1"
SYNTHETIC_RECORD_TYPE = "synthetic_phase1c_dataset"
SYNTHETIC_RECORD_SCHEMA_VERSION = "phase1c_synthetic_input_record_v2"

EXPECTED_REALISTIC_SOURCE_MANIFEST_SHA256 = "bed35b602e925d5da93773ee72037dbf5019498e3bb1308975f7aa9f9671082f"
EXPECTED_REALISTIC_CADENCE_COUNT = 18_041
EXPECTED_REALISTIC_EVENT_COUNT = 373
REALISTIC_SYNTHETIC_FLUX_SEED = 20_260_718
REALISTIC_SUPERSAMPLE_FACTOR = 11
REALISTIC_MID_MISSION_CYCLE = 206

RECOVERY_PARAMETER_REGISTRY = (
    "rp_over_rstar",
    "a_over_rstar",
    "impact_parameter",
    "q1",
    "q2",
    "white_noise_jitter",
    "period_days",
    "transit_time_mid_mission_reference",
)

REALISTIC_INJECTED_PARAMETERS = {
    "rp_over_rstar": 0.07875,
    "a_over_rstar": 6.45,
    "impact_parameter": 0.30,
    "q1": 0.45,
    "q2": 0.19,
    "white_noise_jitter": 0.00012,
    "period_days": 3.548715437667639,
    "transit_time_mid_mission_reference": 853.9003172957171,
    "transit_time_original_reference": 122.86493713618336,
}


@dataclass(frozen=True)
class RealisticSyntheticRecoverySpec:
    """Immutable specification for the authoritative realistic recovery dataset."""

    dataset_design: str = REALISTIC_DATASET_DESIGN
    generator_version: str = REALISTIC_GENERATOR_VERSION
    synthetic_flux_seed: int = REALISTIC_SYNTHETIC_FLUX_SEED
    supersample_factor: int = REALISTIC_SUPERSAMPLE_FACTOR
    mid_mission_cycle: int = REALISTIC_MID_MISSION_CYCLE
    injected_parameters: Mapping[str, float] = field(default_factory=lambda: dict(REALISTIC_INJECTED_PARAMETERS))
    expected_source_manifest_sha256: str | None = EXPECTED_REALISTIC_SOURCE_MANIFEST_SHA256
    expected_cadence_count: int | None = EXPECTED_REALISTIC_CADENCE_COUNT
    expected_event_count: int | None = EXPECTED_REALISTIC_EVENT_COUNT


@dataclass(frozen=True)
class SyntheticDatasetResult:
    """Generated synthetic data plus auditable identity."""

    data: FrozenPhase1BData
    timing: TimingReference
    injected_parameters: dict[str, float]
    derived_timing: dict[str, Any]
    identity: dict[str, Any]
    baseline_coefficients: tuple[dict[str, float | int], ...] = ()
    baseline_coefficients_csv: str | None = None

    @property
    def dataset_design(self) -> str:
        return str(self.identity.get("dataset_design", LEGACY_TOY_DATASET_DESIGN))

    def legacy_tuple(self) -> tuple[FrozenPhase1BData, TimingReference, dict[str, float]]:
        return self.data, self.timing, self.injected_parameters


def build_synthetic_dataset_for_mode(
    config: Phase1CConfig,
    mode: str,
    *,
    recorded_input: dict[str, Any] | None = None,
) -> SyntheticDatasetResult:
    """Return the synthetic dataset required by a Phase 1C mode or existing record."""
    if mode == "synthetic":
        if recorded_input is not None and dataset_design_from_record(recorded_input) == LEGACY_TOY_DATASET_DESIGN:
            return build_toy_synthetic_dataset(config, legacy_identity=True)
        return build_toy_synthetic_dataset(config)
    if mode != "synthetic_recovery":
        raise ValueError(f"Mode {mode!r} does not use a synthetic dataset.")
    if recorded_input is not None:
        design = dataset_design_from_record(recorded_input)
        if design == LEGACY_TOY_DATASET_DESIGN:
            return build_toy_synthetic_dataset(config, legacy_identity=True)
        if design != REALISTIC_DATASET_DESIGN:
            raise ValueError(f"Unsupported synthetic dataset design: {design!r}")
    return build_realistic_synthetic_recovery_dataset(config)


def dataset_design_from_record(record: Mapping[str, Any]) -> str:
    """Return the explicit or legacy synthetic dataset design from a stored record."""
    design = record.get("dataset_design")
    if design is None:
        return LEGACY_TOY_DATASET_DESIGN
    return str(design)


def build_toy_synthetic_dataset(
    config: Phase1CConfig,
    *,
    legacy_identity: bool = False,
) -> SyntheticDatasetResult:
    """Build the fast 224-cadence synthetic dataset retained for validation tests."""
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
    baseline_rows = []
    for event in event_ids:
        mask = event_number == event
        x = event_local_coordinate(time[mask], float(np.median(predicted_center[mask])))
        c0 = float(rng.normal(1.0, config.baseline_intercept_sigma))
        c1 = float(rng.normal(0.0, config.baseline_slope_sigma))
        noise = rng.normal(0.0, np.sqrt(flux_uncertainty[mask] ** 2 + true.jitter**2))
        flux[mask] = model[mask] * (c0 + c1 * x) + noise
        baseline_rows.append({"event_number": int(event), "c0": c0, "c1": c1})
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
    identity = _toy_identity(config, time, flux_uncertainty, flux, baseline_rows, legacy_identity=legacy_identity)
    manifest_sha = "synthetic" if legacy_identity else identity["overall_canonical_identity_sha256"]
    input_manifest = {"manifest_sha256": manifest_sha, "residuals_csv_used_as_input": False, "files": []}
    if not legacy_identity:
        input_manifest["synthetic_dataset_identity"] = identity
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
        input_manifest=input_manifest,
    )
    timing = build_timing_reference(data, config)
    vector = physical_to_vector(true, timing)
    if not np.isfinite(log_probability(vector, data, config, timing)):
        raise RuntimeError("Synthetic injected solution has nonfinite posterior density.")
    derived_timing = {
        "mid_mission_cycle": int(timing.mid_epoch_cycle),
        "transit_time_original_reference": float(true.original_epoch),
        "transit_time_mid_mission_reference": float(true.mid_epoch),
        "original_epoch_is_derived_audit_quantity": True,
    }
    return SyntheticDatasetResult(
        data=data,
        timing=timing,
        injected_parameters=injected,
        derived_timing=derived_timing,
        identity=identity,
        baseline_coefficients=tuple(baseline_rows),
        baseline_coefficients_csv=baseline_coefficients_csv(baseline_rows),
    )


def build_realistic_synthetic_recovery_dataset(
    config: Phase1CConfig,
    *,
    spec: RealisticSyntheticRecoverySpec | None = None,
    source_data: FrozenPhase1BData | None = None,
) -> SyntheticDatasetResult:
    """Build the authoritative frozen-cadence synthetic recovery dataset."""
    spec = spec or RealisticSyntheticRecoverySpec()
    if spec.dataset_design != REALISTIC_DATASET_DESIGN:
        raise ValueError(f"Realistic recovery requires dataset_design={REALISTIC_DATASET_DESIGN!r}.")
    if spec.supersample_factor != REALISTIC_SUPERSAMPLE_FACTOR:
        raise ValueError("The realistic frozen-cadence generator v1 requires supersampling factor 11.")
    if config.supersample_factor != REALISTIC_SUPERSAMPLE_FACTOR:
        raise ValueError("Realistic recovery requires Phase1CConfig.supersample_factor == 11.")
    source = source_data if source_data is not None else load_frozen_phase1b(config)
    _validate_realistic_source_preflight(source, spec)
    timing = build_timing_reference(source, config)
    true = _realistic_physical_sample(spec)
    _validate_realistic_timing(true, spec)
    vector = physical_to_vector(true, timing)
    if abs(vector[6]) > timing.period_half_width or abs(vector[7]) > timing.mid_epoch_half_width:
        raise ValueError("Realistic injected timing vector is outside the frozen Phase 1C support.")

    transit_model = batman_flux(
        source.time,
        source.exposure_days,
        rp=true.rp,
        a=true.a,
        b=true.b,
        q1=true.q1,
        q2=true.q2,
        period=true.period,
        t0=true.original_epoch,
        supersample_factor=spec.supersample_factor,
    )
    seed_sequence = np.random.SeedSequence(spec.synthetic_flux_seed)
    baseline_sequence, noise_sequence = seed_sequence.spawn(2)
    baseline_rng = np.random.default_rng(baseline_sequence)
    noise_rng = np.random.default_rng(noise_sequence)
    synthetic_flux = np.empty_like(source.time, dtype=float)
    baseline_rows = []
    for event in sorted(int(value) for value in np.unique(source.event_number)):
        mask = source.event_number == event
        center = float(np.median(source.predicted_center[mask]))
        local_coordinate = event_local_coordinate(source.time[mask], center)
        c0 = float(baseline_rng.normal(1.0, config.baseline_intercept_sigma))
        c1 = float(baseline_rng.normal(0.0, config.baseline_slope_sigma))
        baseline = c0 + c1 * local_coordinate
        synthetic_flux[mask] = transit_model[mask] * baseline
        baseline_rows.append({"event_number": event, "c0": c0, "c1": c1})
    sigma_total = np.sqrt(np.square(source.flux_uncertainty) + true.jitter**2)
    synthetic_flux += noise_rng.normal(0.0, sigma_total)

    injected = _realistic_injected_registry(spec)
    baseline_csv = baseline_coefficients_csv(baseline_rows)
    identity = _realistic_identity(
        source,
        synthetic_flux,
        baseline_rows,
        baseline_csv,
        spec,
        config,
        seed_sequence,
        (baseline_sequence, noise_sequence),
    )
    data = FrozenPhase1BData(
        time=np.array(source.time, copy=True),
        flux=synthetic_flux,
        flux_uncertainty=np.array(source.flux_uncertainty, copy=True),
        event_number=np.array(source.event_number, copy=True),
        predicted_center=np.array(source.predicted_center, copy=True),
        product_id=np.array(source.product_id, copy=True),
        quarter=np.array(source.quarter, copy=True),
        exposure_days=np.array(source.exposure_days, copy=True),
        deterministic_parameters=copy.deepcopy(source.deterministic_parameters),
        limb_darkening=copy.deepcopy(source.limb_darkening),
        phase1b_configuration=copy.deepcopy(source.phase1b_configuration),
        phase1b_summary=copy.deepcopy(source.phase1b_summary),
        provenance={
            "source_phase1b_provenance": copy.deepcopy(source.provenance),
            "synthetic_dataset_identity_sha256": identity["overall_canonical_identity_sha256"],
        },
        input_manifest={
            "manifest_sha256": identity["overall_canonical_identity_sha256"],
            "residuals_csv_used_as_input": False,
            "observed_flux_used": False,
            "synthetic_dataset_identity": identity,
            "source_phase1b_input_manifest": copy.deepcopy(source.input_manifest),
            "files": [],
        },
    )
    if not np.isfinite(log_probability(vector, data, config, timing)):
        raise RuntimeError("Realistic synthetic injected solution has nonfinite posterior density.")
    derived_timing = {
        "mid_mission_cycle": int(spec.mid_mission_cycle),
        "transit_time_original_reference": injected["transit_time_original_reference"],
        "transit_time_mid_mission_reference": injected["transit_time_mid_mission_reference"],
        "original_epoch_formula": "mid_mission_epoch - mid_mission_cycle * period",
        "original_epoch_is_derived_audit_quantity": True,
        "independent_coverage_parameters": list(RECOVERY_PARAMETER_REGISTRY),
    }
    return SyntheticDatasetResult(
        data=data,
        timing=timing,
        injected_parameters=injected,
        derived_timing=derived_timing,
        identity=identity,
        baseline_coefficients=tuple(baseline_rows),
        baseline_coefficients_csv=baseline_csv,
    )


def synthetic_input_record(result: SyntheticDatasetResult) -> dict[str, Any]:
    """Return the stored synthetic input record for a generated dataset."""
    record = {
        "type": SYNTHETIC_RECORD_TYPE,
        "schema_version": SYNTHETIC_RECORD_SCHEMA_VERSION,
        "dataset_design": result.dataset_design,
        "generator_version": result.identity.get("generator_version"),
        "cadence_count": result.data.cadence_count,
        "event_count": result.data.event_count,
        "timing_reference": result.timing.__dict__,
        "residuals_csv_used_as_input": False,
        "observed_flux_used": bool(result.identity.get("observed_flux_used", False)),
        "residuals_used": bool(result.identity.get("residuals_used", False)),
        "injected_parameters": result.injected_parameters,
        "derived_timing": result.derived_timing,
        "dataset_identity": result.identity,
    }
    if result.baseline_coefficients_csv is not None:
        record["baseline_coefficients_audit_csv_sha256"] = hashlib.sha256(
            result.baseline_coefficients_csv.encode("utf-8")
        ).hexdigest()
    return record


def synthetic_input_record_from_data(data: FrozenPhase1BData, timing: TimingReference) -> dict[str, Any]:
    """Return a synthetic input record for data when only the frozen data object is available."""
    identity = data.input_manifest.get("synthetic_dataset_identity")
    if identity is None:
        return legacy_synthetic_input_record(data, timing)
    return {
        "type": SYNTHETIC_RECORD_TYPE,
        "schema_version": SYNTHETIC_RECORD_SCHEMA_VERSION,
        "dataset_design": identity.get("dataset_design"),
        "generator_version": identity.get("generator_version"),
        "cadence_count": data.cadence_count,
        "event_count": data.event_count,
        "timing_reference": timing.__dict__,
        "residuals_csv_used_as_input": False,
        "observed_flux_used": bool(identity.get("observed_flux_used", False)),
        "residuals_used": bool(identity.get("residuals_used", False)),
        "injected_parameters": identity.get("injected_parameters", {}),
        "derived_timing": identity.get("derived_timing", {}),
        "dataset_identity": identity,
    }


def legacy_synthetic_input_record(data: FrozenPhase1BData, timing: TimingReference) -> dict[str, Any]:
    """Return the pre-design synthetic record shape for legacy toy checkpoints."""
    return {
        "type": SYNTHETIC_RECORD_TYPE,
        "cadence_count": data.cadence_count,
        "event_count": data.event_count,
        "timing_reference": timing.__dict__,
        "residuals_csv_used_as_input": False,
    }


def validate_synthetic_input_record(
    recorded: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    """Validate a stored synthetic input record against regenerated data."""
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
    design = dataset_design_from_record(recorded)
    if design == LEGACY_TOY_DATASET_DESIGN:
        return
    expected_design = expected.get("dataset_design")
    if design != expected_design:
        raise ValueError("synthetic_input_record.json mismatch for dataset_design.")
    expected_identity = expected.get("dataset_identity", {})
    recorded_identity = recorded.get("dataset_identity", {})
    for key in (
        "overall_canonical_identity_sha256",
        "generated_synthetic_flux_sha256",
        "baseline_coefficients_hash_sha256",
    ):
        if recorded_identity.get(key) != expected_identity.get(key):
            raise ValueError(f"synthetic_input_record.json identity mismatch for {key}.")


def baseline_coefficients_csv(rows: list[dict[str, float | int]] | tuple[dict[str, float | int], ...]) -> str:
    """Return the stable audit CSV text for baseline coefficients."""
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["event_number", "c0", "c1"], lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                "event_number": int(row["event_number"]),
                "c0": format(float(row["c0"]), ".17g"),
                "c1": format(float(row["c1"]), ".17g"),
            }
        )
    return output.getvalue()


def canonical_array_hash(array: np.ndarray) -> str:
    """Hash dtype, shape, and stable C-order contents for numeric or string arrays."""
    arr = np.asarray(array)
    digest = hashlib.sha256()
    digest.update(str(arr.dtype).encode("utf-8"))
    digest.update(json.dumps(list(arr.shape), separators=(",", ":")).encode("utf-8"))
    if np.issubdtype(arr.dtype, np.number) or np.issubdtype(arr.dtype, np.bool_):
        contiguous = np.ascontiguousarray(arr)
        digest.update(contiguous.view(np.uint8).tobytes())
    else:
        for value in arr.ravel(order="C"):
            digest.update(str(value).encode("utf-8"))
            digest.update(b"\0")
    return digest.hexdigest()


def canonical_payload_hash(payload: Mapping[str, Any]) -> str:
    """Hash a JSON-canonical payload."""
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _realistic_physical_sample(spec: RealisticSyntheticRecoverySpec) -> PhysicalSample:
    injected = spec.injected_parameters
    return PhysicalSample(
        rp=float(injected["rp_over_rstar"]),
        a=float(injected["a_over_rstar"]),
        b=float(injected["impact_parameter"]),
        q1=float(injected["q1"]),
        q2=float(injected["q2"]),
        jitter=float(injected["white_noise_jitter"]),
        period=float(injected["period_days"]),
        mid_epoch=float(injected["transit_time_mid_mission_reference"]),
        original_epoch=float(injected["transit_time_original_reference"]),
    )


def _realistic_injected_registry(spec: RealisticSyntheticRecoverySpec) -> dict[str, float]:
    injected = {key: float(spec.injected_parameters[key]) for key in RECOVERY_PARAMETER_REGISTRY}
    injected["transit_time_original_reference"] = float(spec.injected_parameters["transit_time_original_reference"])
    return injected


def _validate_realistic_timing(sample: PhysicalSample, spec: RealisticSyntheticRecoverySpec) -> None:
    derived = sample.mid_epoch - int(spec.mid_mission_cycle) * sample.period
    if not math.isclose(derived, sample.original_epoch, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("Realistic injected original epoch is inconsistent with mid-mission timing.")


def _validate_realistic_source_preflight(
    source: FrozenPhase1BData,
    spec: RealisticSyntheticRecoverySpec,
) -> None:
    if spec.expected_source_manifest_sha256 is not None:
        actual = str(source.input_manifest.get("manifest_sha256"))
        if actual != spec.expected_source_manifest_sha256:
            raise ValueError(
                "Realistic recovery source manifest mismatch: "
                f"expected {spec.expected_source_manifest_sha256}, found {actual}."
            )
    if spec.expected_cadence_count is not None and source.cadence_count != spec.expected_cadence_count:
        raise ValueError(
            f"Realistic recovery source cadence count mismatch: "
            f"expected {spec.expected_cadence_count}, found {source.cadence_count}."
        )
    if spec.expected_event_count is not None and source.event_count != spec.expected_event_count:
        raise ValueError(
            f"Realistic recovery source event count mismatch: "
            f"expected {spec.expected_event_count}, found {source.event_count}."
        )
    if source.input_manifest.get("residuals_csv_used_as_input") is not False:
        raise ValueError("Realistic recovery requires residuals_csv_used_as_input=false.")


def _toy_identity(
    config: Phase1CConfig,
    time: np.ndarray,
    flux_uncertainty: np.ndarray,
    flux: np.ndarray,
    baseline_rows: list[dict[str, float | int]],
    *,
    legacy_identity: bool,
) -> dict[str, Any]:
    if legacy_identity:
        return {
            "dataset_design": LEGACY_TOY_DATASET_DESIGN,
            "generator_version": "legacy_pre_dataset_design",
            "overall_canonical_identity_sha256": "synthetic",
            "observed_flux_used": False,
            "residuals_used": False,
        }
    core = {
        "dataset_design": TOY_SMOKE_DATASET_DESIGN,
        "generator_version": TOY_GENERATOR_VERSION,
        "root_rng_seed": int(config.random_seed + 77),
        "cadence_count": int(time.size),
        "event_count": 8,
        "structural_field_hashes": {
            "time": canonical_array_hash(time),
            "flux_uncertainty": canonical_array_hash(flux_uncertainty),
        },
        "generated_synthetic_flux_sha256": canonical_array_hash(flux),
        "baseline_coefficients_hash_sha256": canonical_payload_hash({"rows": baseline_rows}),
        "observed_flux_used": False,
        "residuals_used": False,
    }
    return _with_identity_sha(core)


def _realistic_identity(
    source: FrozenPhase1BData,
    synthetic_flux: np.ndarray,
    baseline_rows: list[dict[str, float | int]],
    baseline_csv: str,
    spec: RealisticSyntheticRecoverySpec,
    config: Phase1CConfig,
    seed_sequence: np.random.SeedSequence,
    child_sequences: tuple[np.random.SeedSequence, np.random.SeedSequence],
) -> dict[str, Any]:
    injected = _realistic_injected_registry(spec)
    derived_timing = {
        "mid_mission_cycle": int(spec.mid_mission_cycle),
        "transit_time_original_reference": injected["transit_time_original_reference"],
        "transit_time_mid_mission_reference": injected["transit_time_mid_mission_reference"],
        "original_epoch_formula": "mid_mission_epoch - mid_mission_cycle * period",
        "original_epoch_is_derived_audit_quantity": True,
        "independent_coverage_parameters": list(RECOVERY_PARAMETER_REGISTRY),
    }
    baseline_hash = canonical_payload_hash({"rows": baseline_rows})
    core = {
        "dataset_design": spec.dataset_design,
        "generator_version": spec.generator_version,
        "source_phase1b_manifest_sha256": source.input_manifest.get("manifest_sha256"),
        "source_cadence_count": source.cadence_count,
        "source_event_count": source.event_count,
        "source_residuals_csv_used_as_input": bool(source.input_manifest.get("residuals_csv_used_as_input")),
        "preserved_structural_field_hashes": {
            "time": canonical_array_hash(source.time),
            "event_number": canonical_array_hash(source.event_number),
            "predicted_center": canonical_array_hash(source.predicted_center),
            "exposure_days": canonical_array_hash(source.exposure_days),
            "flux_uncertainty": canonical_array_hash(source.flux_uncertainty),
            "product_id": canonical_array_hash(source.product_id),
            "quarter": canonical_array_hash(source.quarter),
        },
        "injected_parameters": injected,
        "derived_timing": derived_timing,
        "root_rng_seed": int(spec.synthetic_flux_seed),
        "rng": {
            "bit_generator": "PCG64",
            "root_seed_sequence": _seed_sequence_record(seed_sequence, "root", None),
            "child_streams": [
                _seed_sequence_record(child_sequences[0], "event_baseline_coefficients", 0),
                _seed_sequence_record(child_sequences[1], "cadence_noise", 1),
            ],
        },
        "baseline_generation_rule": (
            "For each sorted accepted event label, draw c0~Normal(1, baseline_intercept_sigma) "
            "and c1~Normal(0, baseline_slope_sigma); multiply transit model by c0+c1*x, "
            "where x is phase1c_likelihood.event_local_coordinate."
        ),
        "noise_generation_rule": (
            "Independent Gaussian cadence noise with sigma_total=sqrt(frozen_flux_uncertainty^2 + injected_jitter^2)."
        ),
        "baseline_coefficients_hash_sha256": baseline_hash,
        "baseline_coefficients_audit_csv_sha256": hashlib.sha256(baseline_csv.encode("utf-8")).hexdigest(),
        "generated_synthetic_flux_sha256": canonical_array_hash(synthetic_flux),
        "supersampling_factor": int(spec.supersample_factor),
        "baseline_prior_widths": {
            "baseline_intercept_sigma": float(config.baseline_intercept_sigma),
            "baseline_slope_sigma": float(config.baseline_slope_sigma),
        },
        "observed_flux_used": False,
        "residuals_used": False,
        "dependency_versions": _dependency_versions(),
    }
    return _with_identity_sha(core)


def _with_identity_sha(core: dict[str, Any]) -> dict[str, Any]:
    payload = dict(core)
    payload["overall_canonical_identity_sha256"] = canonical_payload_hash(core)
    return payload


def _seed_sequence_record(
    seed_sequence: np.random.SeedSequence,
    purpose: str,
    child_index: int | None,
) -> dict[str, Any]:
    entropy = seed_sequence.entropy
    if isinstance(entropy, np.ndarray):
        entropy = entropy.tolist()
    return {
        "purpose": purpose,
        "child_index": child_index,
        "entropy": entropy,
        "spawn_key": list(seed_sequence.spawn_key),
        "pool_size": int(seed_sequence.pool_size),
        "state_preview_uint32": [int(value) for value in seed_sequence.generate_state(4)],
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


def _dependency_versions() -> dict[str, str]:
    packages = {"numpy": "numpy", "batman": "batman-package"}
    versions = {"python": platform.python_version()}
    for label, package in packages.items():
        try:
            versions[label] = version(package)
        except PackageNotFoundError:
            versions[label] = "not-installed"
    return versions


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        _canonical_json_safe(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _canonical_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _canonical_json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Cannot canonicalize nonfinite float.")
        return value
    return value
