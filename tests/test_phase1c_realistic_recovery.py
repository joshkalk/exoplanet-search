import json
from dataclasses import replace
from pathlib import Path

import emcee
import h5py
import numpy as np
import pandas as pd
import pytest

from exoplanet_search import phase1c as phase1c_module
from exoplanet_search.phase1c import _boundary_audit, _recovery_gate_record, _strict_recovery_rows
from exoplanet_search.phase1c_inputs import load_frozen_phase1b
from exoplanet_search.phase1c_likelihood import log_probability
from exoplanet_search.phase1c_parameters import physical_to_vector
from exoplanet_search.phase1c_sampler import checkpoint_metadata, immutable_checkpoint_identity, run_ensembles
from exoplanet_search.phase1c_synthetic import (
    EXPECTED_REALISTIC_CADENCE_COUNT,
    EXPECTED_REALISTIC_EVENT_COUNT,
    EXPECTED_REALISTIC_SOURCE_MANIFEST_SHA256,
    REALISTIC_DATASET_DESIGN,
    RECOVERY_PARAMETER_REGISTRY,
    RealisticSyntheticRecoverySpec,
    build_realistic_synthetic_recovery_dataset,
    build_synthetic_dataset_for_mode,
    build_toy_synthetic_dataset,
    canonical_array_hash,
    legacy_synthetic_input_record,
    synthetic_input_record,
    validate_synthetic_input_record,
)
from exoplanet_search.phase1c_types import FrozenPhase1BData, Phase1CConfig, PhysicalSample


def test_generator_separation_keeps_toy_validation_and_realistic_recovery(tmp_path):
    config = _realistic_config(tmp_path)
    toy = build_synthetic_dataset_for_mode(config, "synthetic")
    realistic = build_synthetic_dataset_for_mode(config, "synthetic_recovery")

    assert toy.dataset_design == "toy_smoke_v1"
    assert toy.data.cadence_count == 224
    assert realistic.dataset_design == REALISTIC_DATASET_DESIGN
    assert realistic.data.cadence_count == EXPECTED_REALISTIC_CADENCE_COUNT
    assert realistic.data.event_count == EXPECTED_REALISTIC_EVENT_COUNT
    assert realistic.data.input_manifest["manifest_sha256"] != "synthetic"


def test_realistic_generator_preserves_frozen_structure_and_centers(tmp_path):
    config = _realistic_config(tmp_path)
    source = load_frozen_phase1b(config)
    result = build_realistic_synthetic_recovery_dataset(config)
    injected = result.injected_parameters

    for field in (
        "time",
        "event_number",
        "predicted_center",
        "exposure_days",
        "flux_uncertainty",
        "product_id",
        "quarter",
    ):
        assert np.array_equal(getattr(result.data, field), getattr(source, field))
    assert result.data.deterministic_parameters == source.deterministic_parameters
    assert result.data.limb_darkening == source.limb_darkening
    assert not np.array_equal(result.data.flux, source.flux)
    assert result.injected_parameters["rp_over_rstar"] != pytest.approx(
        source.deterministic_parameters["rp_over_rstar"]
    )
    assert result.injected_parameters["a_over_rstar"] != pytest.approx(
        source.deterministic_parameters["a_over_rstar"]
    )
    assert result.injected_parameters["period_days"] != pytest.approx(
        source.deterministic_parameters["period_days"]
    )
    assert injected["q1"] == 0.45
    assert injected["q2"] == 0.19
    assert injected["q1"] != source.limb_darkening["q1"]
    assert injected["q2"] != source.limb_darkening["q2"]
    assert injected["q1"] != source.deterministic_parameters["q1"]
    assert injected["q2"] != source.deterministic_parameters["q2"]
    assert 0.0 < injected["q1"] < 1.0
    assert 0.0 < injected["q2"] < 1.0
    sample = _physical_sample_from_injected(injected)
    vector = physical_to_vector(sample, result.timing)
    assert np.isfinite(log_probability(vector, result.data, config, result.timing))
    rows = _strict_recovery_rows(_posterior_frame(injected), injected)
    assert [row["parameter"] for row in rows] == list(RECOVERY_PARAMETER_REGISTRY)
    assert len({row["parameter"] for row in rows}) == len(RECOVERY_PARAMETER_REGISTRY)
    assert _boundary_audit(rows, injected, config, result.timing)["passed"] is True


def test_realistic_generator_does_not_use_observed_flux_or_residuals(monkeypatch, tmp_path):
    config = _realistic_config(tmp_path)
    source = load_frozen_phase1b(config)
    sentinel_a = _with_flux(source, np.full(source.cadence_count, 0.123))
    sentinel_b = _with_flux(source, np.full(source.cadence_count, 2.0))

    flux_a = build_realistic_synthetic_recovery_dataset(config, source_data=sentinel_a).data.flux
    flux_b = build_realistic_synthetic_recovery_dataset(config, source_data=sentinel_b).data.flux
    assert np.array_equal(flux_a, flux_b)

    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        if Path(path).name == "residuals.csv":
            raise AssertionError("residuals.csv must not be opened by realistic synthetic generation")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    build_realistic_synthetic_recovery_dataset(config)


def test_realistic_generation_is_deterministic_and_identity_changes(tmp_path):
    config = _realistic_config(tmp_path)
    first = build_realistic_synthetic_recovery_dataset(config)
    second = build_realistic_synthetic_recovery_dataset(config)

    assert np.array_equal(first.data.flux, second.data.flux)
    assert first.baseline_coefficients == second.baseline_coefficients
    assert first.identity == second.identity

    changed_seed = build_realistic_synthetic_recovery_dataset(
        config,
        spec=RealisticSyntheticRecoverySpec(synthetic_flux_seed=123, expected_source_manifest_sha256=None),
    )
    assert not np.array_equal(first.data.flux, changed_seed.data.flux)
    assert first.identity["overall_canonical_identity_sha256"] != changed_seed.identity[
        "overall_canonical_identity_sha256"
    ]

    source = load_frozen_phase1b(config)
    changed_manifest = replace(
        source,
        input_manifest={**source.input_manifest, "manifest_sha256": "changed"},
    )
    changed_source = build_realistic_synthetic_recovery_dataset(
        config,
        source_data=changed_manifest,
        spec=RealisticSyntheticRecoverySpec(expected_source_manifest_sha256=None),
    )
    assert first.identity["overall_canonical_identity_sha256"] != changed_source.identity[
        "overall_canonical_identity_sha256"
    ]

    changed_time = replace(source, time=source.time.copy())
    changed_time.time[0] += 1.0e-8
    structural = build_realistic_synthetic_recovery_dataset(
        config,
        source_data=changed_time,
        spec=RealisticSyntheticRecoverySpec(expected_source_manifest_sha256=None),
    )
    assert first.identity["preserved_structural_field_hashes"]["time"] != structural.identity[
        "preserved_structural_field_hashes"
    ]["time"]


def test_identity_record_validation_and_legacy_toy_behavior(tmp_path):
    config = _realistic_config(tmp_path)
    realistic = build_realistic_synthetic_recovery_dataset(config)
    record = synthetic_input_record(realistic)
    validate_synthetic_input_record(record, record)

    tampered = json.loads(json.dumps(record))
    tampered["dataset_identity"]["generated_synthetic_flux_sha256"] = "bad"
    with pytest.raises(ValueError, match="generated_synthetic_flux_sha256"):
        validate_synthetic_input_record(tampered, record)

    toy = build_toy_synthetic_dataset(config)
    legacy_toy = build_toy_synthetic_dataset(config, legacy_identity=True)
    toy_identity = immutable_checkpoint_identity(toy.data, config, mode="synthetic")
    realistic_identity = immutable_checkpoint_identity(realistic.data, config, mode="synthetic_recovery")
    assert toy_identity != realistic_identity
    assert checkpoint_metadata(legacy_toy.data, config, mode="synthetic")["immutable_scientific_identity"][
        "phase1b_input_manifest_sha256"
    ] == "synthetic"
    legacy_record = legacy_synthetic_input_record(legacy_toy.data, legacy_toy.timing)
    validate_synthetic_input_record(legacy_record, legacy_record)


def test_strict_eight_parameter_registry_rejects_missing_duplicate_and_bad_intervals(tmp_path):
    result = build_realistic_synthetic_recovery_dataset(_realistic_config(tmp_path))
    posterior = _posterior_frame(result.injected_parameters)
    rows = _strict_recovery_rows(posterior, result.injected_parameters)
    assert [row["parameter"] for row in rows] == list(RECOVERY_PARAMETER_REGISTRY)

    with pytest.raises(ValueError, match="missing required"):
        _strict_recovery_rows(posterior, {"rp_over_rstar": 0.1})
    with pytest.raises(ValueError, match="exactly once"):
        _strict_recovery_rows(pd.concat([posterior, posterior.iloc[[0]]], ignore_index=True), result.injected_parameters)
    bad = posterior.copy()
    bad.loc[0, "q02_5"] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        _strict_recovery_rows(bad, result.injected_parameters)
    bad = posterior.copy()
    bad.loc[0, "q97_5"] = bad.loc[0, "q02_5"] - 1.0
    with pytest.raises(ValueError, match="malformed"):
        _strict_recovery_rows(bad, result.injected_parameters)
    assert "transit_time_original_reference" not in RECOVERY_PARAMETER_REGISTRY


def test_authoritative_gate_statuses_and_boundary_audit(monkeypatch, tmp_path):
    config = _realistic_config(tmp_path, n_ensembles=4, n_walkers=32)
    result = build_realistic_synthetic_recovery_dataset(config)
    rows = _strict_recovery_rows(_posterior_frame(result.injected_parameters), result.injected_parameters)
    diagnostics = _passing_diagnostics()
    ensemble = pd.DataFrame({"ensemble": [0, 1, 2, 3]})

    monkeypatch.setattr(
        phase1c_module,
        "_hdf_iteration_counts",
        lambda _config: {f"ensemble_{index:02d}.h5": 4 for index in range(4)},
    )
    monkeypatch.setattr(
        phase1c_module,
        "_read_json_if_exists",
        lambda _path: {
            "ensemble_results": [
                {"strategy": "local_tight"},
                {"strategy": "local_moderate"},
                {"strategy": "local_broad"},
                {"strategy": "prior_informed"},
            ]
        },
    )
    gate = _recovery_gate_record(config, result, diagnostics, ensemble, rows, None)
    assert gate["status"] == "realistic_recovery_gate_passed"
    assert gate["boundary_audit"]["passed"] is True

    toy = build_toy_synthetic_dataset(config)
    toy_gate = _recovery_gate_record(config, toy, diagnostics, ensemble, rows, None)
    assert toy_gate["status"] == "nonauthoritative_toy_recovery"

    wrong_ensembles = _recovery_gate_record(
        replace(config, n_ensembles=3),
        result,
        diagnostics,
        pd.DataFrame({"ensemble": [0, 1, 2]}),
        rows,
        None,
    )
    assert wrong_ensembles["criteria"]["four_ensembles_configured_and_represented"] is False

    boundary_rows = [dict(row) for row in rows]
    boundary_rows[0]["q02_5"] = config.rp_bounds[0]
    boundary_gate = _recovery_gate_record(config, result, diagnostics, ensemble, boundary_rows, None)
    assert boundary_gate["criteria"]["hard_boundary_audit_passes"] is False

    nonconverged = _recovery_gate_record(config, result, {**diagnostics, "status": "nonconverged"}, ensemble, rows, None)
    assert nonconverged["status"] == "realistic_recovery_nonconverged"


def test_real_artifact_integration_preflight_and_finite_injected_posterior(tmp_path):
    config = _realistic_config(tmp_path)
    result = build_realistic_synthetic_recovery_dataset(config)

    assert result.identity["source_phase1b_manifest_sha256"] == EXPECTED_REALISTIC_SOURCE_MANIFEST_SHA256
    assert result.data.cadence_count == EXPECTED_REALISTIC_CADENCE_COUNT
    assert result.data.event_count == EXPECTED_REALISTIC_EVENT_COUNT
    assert result.identity["observed_flux_used"] is False
    assert result.identity["residuals_used"] is False
    assert result.data.input_manifest["residuals_csv_used_as_input"] is False
    assert result.identity["generated_synthetic_flux_sha256"] == canonical_array_hash(result.data.flux)


def test_realistic_short_sequential_parallel_equivalence(tmp_path):
    config_seq = _realistic_config(
        tmp_path / "seq",
        n_ensembles=2,
        n_walkers=16,
        chunk_steps=1,
        warmup_steps=0,
        ensemble_processes=1,
    )
    config_par = replace(config_seq, output_dir=tmp_path / "par", ensemble_processes=2)
    seq_data = build_realistic_synthetic_recovery_dataset(config_seq)
    par_data = build_realistic_synthetic_recovery_dataset(config_par)

    seq = run_ensembles(seq_data.data, config_seq, seq_data.timing, steps=2, mode="synthetic_recovery")
    par = run_ensembles(par_data.data, config_par, par_data.timing, steps=2, mode="synthetic_recovery")

    for left, right in zip(seq, par, strict=True):
        assert left.initialization_summary == right.initialization_summary
        _assert_backend_equivalent(left.backend_path, right.backend_path)


def _realistic_config(path, **overrides):
    base = Phase1CConfig(
        output_dir=Path(path),
        n_ensembles=overrides.pop("n_ensembles", 4),
        n_walkers=overrides.pop("n_walkers", 32),
        chunk_steps=overrides.pop("chunk_steps", 2),
        warmup_steps=overrides.pop("warmup_steps", 1),
        synthetic_recovery_steps=overrides.pop("synthetic_recovery_steps", 4),
        ensemble_processes=overrides.pop("ensemble_processes", 1),
    )
    return replace(base, **overrides) if overrides else base


def _with_flux(source: FrozenPhase1BData, flux: np.ndarray) -> FrozenPhase1BData:
    return replace(source, flux=np.asarray(flux, dtype=float))


def _physical_sample_from_injected(injected: dict[str, float]) -> PhysicalSample:
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


def _posterior_frame(injected: dict[str, float]) -> pd.DataFrame:
    rows = []
    for parameter in RECOVERY_PARAMETER_REGISTRY:
        value = float(injected[parameter])
        delta = max(abs(value) * 1.0e-6, 1.0e-7)
        rows.append(
            {
                "parameter": parameter,
                "median": value,
                "q02_5": value - delta,
                "q16": value - 0.5 * delta,
                "q84": value + 0.5 * delta,
                "q97_5": value + delta,
                "mean": value,
                "sd": delta,
            }
        )
    return pd.DataFrame(rows)


def _passing_diagnostics():
    return {
        "status": "converged",
        "finite_log_probability_fraction": 1.0,
        "criteria": {
            "complete_valid_autocorrelation": True,
            "chain_length_exceeds_tau_multiple": True,
            "ess_all_above_minimum": True,
            "tail_ess_all_above_minimum": True,
            "posterior_summary_stability": True,
            "independent_ensemble_agreement": True,
            "no_severe_walker_pathology": True,
        },
    }


def _assert_backend_equivalent(left_path: Path, right_path: Path) -> None:
    left = emcee.backends.HDFBackend(str(left_path), read_only=True)
    right = emcee.backends.HDFBackend(str(right_path), read_only=True)
    assert left.iteration == right.iteration
    assert np.array_equal(left.get_chain(), right.get_chain())
    assert np.array_equal(left.get_log_prob(), right.get_log_prob())
    assert np.array_equal(left.accepted, right.accepted)
    assert np.array_equal(left.get_last_sample().coords, right.get_last_sample().coords)
    assert np.array_equal(left.get_last_sample().log_prob, right.get_last_sample().log_prob)
    _assert_random_states_equal(left.random_state, right.random_state)

    with h5py.File(left_path, "r") as left_hdf, h5py.File(right_path, "r") as right_hdf:
        for key in (
            "phase1c_mode",
            "phase1c_immutable_scientific_identity_sha256",
            "phase1c_immutable_scientific_identity",
            "phase1c_ensemble_seed",
        ):
            assert left_hdf.attrs[key] == right_hdf.attrs[key]


def _assert_random_states_equal(left, right) -> None:
    assert left[0] == right[0]
    assert np.array_equal(left[1], right[1])
    assert left[2:] == right[2:]
