import json
from dataclasses import replace
from pathlib import Path

import emcee
import h5py
import numpy as np
import pandas as pd
import pytest

from exoplanet_search import phase1c as phase1c_module
from exoplanet_search import phase1d_draws as draws_module
from exoplanet_search.phase1c import (
    _boundary_audit,
    _ensemble_target_audit,
    _identity_audit,
    _recovery_gate_record,
    _strict_recovery_rows,
    run_phase1c_synthetic_validation,
)
from exoplanet_search.phase1c_inputs import load_frozen_phase1b
from exoplanet_search.phase1c_likelihood import log_probability
from exoplanet_search.phase1c_parameters import physical_to_vector
from exoplanet_search.phase1c_sampler import checkpoint_metadata, immutable_checkpoint_identity, run_ensembles
from exoplanet_search.phase1c_synthetic import (
    EXPECTED_REALISTIC_CADENCE_COUNT,
    EXPECTED_REALISTIC_EVENT_COUNT,
    EXPECTED_REALISTIC_SOURCE_MANIFEST_SHA256,
    REALISTIC_GENERATOR_VERSION,
    REALISTIC_DATASET_DESIGN,
    RECOVERY_PARAMETER_REGISTRY,
    RealisticSyntheticRecoverySpec,
    authoritative_realistic_spec_record,
    authoritative_realistic_spec_sha256,
    build_realistic_synthetic_recovery_dataset,
    build_synthetic_dataset_for_mode,
    build_toy_synthetic_dataset,
    canonical_array_hash,
    canonical_payload_hash,
    legacy_synthetic_input_record,
    realistic_spec_record,
    synthetic_input_record,
    validate_synthetic_input_record,
)
from exoplanet_search.phase1c_outputs import write_json
from exoplanet_search.phase1c_types import FrozenPhase1BData, PARAMETER_ORDER, Phase1CConfig, PhysicalSample


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


def test_frozen_authoritative_v1_specification_and_sha(tmp_path):
    config = _realistic_config(tmp_path)
    default_spec = RealisticSyntheticRecoverySpec()
    default_record = realistic_spec_record(default_spec)

    assert default_record == authoritative_realistic_spec_record()
    assert canonical_payload_hash(default_record) == authoritative_realistic_spec_sha256()
    default = build_realistic_synthetic_recovery_dataset(config)
    assert default.identity["authoritative_v1_specification"] == authoritative_realistic_spec_record()
    assert default.identity["authoritative_v1_specification_sha256"] == authoritative_realistic_spec_sha256()
    assert default.identity["authoritative_v1_specification_matches_exactly"] is True

    changed_seed = build_realistic_synthetic_recovery_dataset(
        config,
        spec=RealisticSyntheticRecoverySpec(synthetic_flux_seed=123),
    )
    assert changed_seed.identity["authoritative_v1_specification_matches_exactly"] is False

    injected = dict(default_spec.injected_parameters)
    injected["q1"] = 0.44
    changed_q = build_realistic_synthetic_recovery_dataset(
        config,
        spec=RealisticSyntheticRecoverySpec(injected_parameters=injected),
    )
    assert changed_q.identity["authoritative_v1_specification_matches_exactly"] is False

    changed_generator = build_realistic_synthetic_recovery_dataset(
        config,
        spec=RealisticSyntheticRecoverySpec(generator_version=f"{REALISTIC_GENERATOR_VERSION}_test"),
    )
    assert changed_generator.identity["authoritative_v1_specification_matches_exactly"] is False

    cycle_injected = dict(default_spec.injected_parameters)
    cycle_injected["transit_time_original_reference"] = (
        cycle_injected["transit_time_mid_mission_reference"]
        - 205 * cycle_injected["period_days"]
    )
    changed_cycle = build_realistic_synthetic_recovery_dataset(
        config,
        spec=RealisticSyntheticRecoverySpec(mid_mission_cycle=205, injected_parameters=cycle_injected),
    )
    assert changed_cycle.identity["authoritative_v1_specification_matches_exactly"] is False

    changed_source_expectation = build_realistic_synthetic_recovery_dataset(
        config,
        spec=RealisticSyntheticRecoverySpec(expected_cadence_count=1),
    )
    assert changed_source_expectation.identity["authoritative_v1_specification_matches_exactly"] is False

    changed_source_manifest = build_realistic_synthetic_recovery_dataset(
        config,
        spec=RealisticSyntheticRecoverySpec(expected_source_manifest_sha256="changed"),
    )
    assert changed_source_manifest.identity["authoritative_v1_specification_matches_exactly"] is False

    changed_supersample = build_realistic_synthetic_recovery_dataset(
        replace(config, supersample_factor=7),
        spec=RealisticSyntheticRecoverySpec(supersample_factor=7),
    )
    assert changed_supersample.identity["authoritative_v1_specification_matches_exactly"] is False


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

    def assert_invalid(mutator) -> None:
        tampered = json.loads(json.dumps(record))
        mutator(tampered)
        validate_synthetic_input_record(tampered, record)

    with pytest.raises(ValueError, match="injected_parameters"):
        assert_invalid(lambda item: item["injected_parameters"].__setitem__("q1", 0.451))
    with pytest.raises(ValueError, match="SHA does not recompute"):
        assert_invalid(
            lambda item: item["dataset_identity"]["rng"]["child_streams"][0].__setitem__("purpose", "changed"),
        )
    with pytest.raises(ValueError, match="derived_timing"):
        assert_invalid(lambda item: item["derived_timing"].__setitem__("mid_mission_cycle", 207))
    with pytest.raises(ValueError, match="generator_version"):
        assert_invalid(lambda item: item.__setitem__("generator_version", "changed"))
    with pytest.raises(ValueError, match="dataset_design"):
        assert_invalid(lambda item: item.pop("dataset_design"))
    with pytest.raises(ValueError, match="observed_flux_used"):
        assert_invalid(lambda item: item.__setitem__("observed_flux_used", True))
    with pytest.raises(ValueError, match="residuals_used"):
        assert_invalid(lambda item: item.__setitem__("residuals_used", True))
    with pytest.raises(ValueError, match="SHA does not recompute"):
        assert_invalid(
            lambda item: item["dataset_identity"].__setitem__("overall_canonical_identity_sha256", "0" * 64),
        )
    with pytest.raises(ValueError, match="SHA does not recompute"):
        assert_invalid(
            lambda item: item["dataset_identity"].__setitem__("source_event_count", 999),
        )

    toy = build_toy_synthetic_dataset(config)
    legacy_toy = build_toy_synthetic_dataset(config, legacy_identity=True)
    toy_identity = immutable_checkpoint_identity(toy.data, config, mode="synthetic")
    realistic_identity = immutable_checkpoint_identity(realistic.data, config, mode="synthetic_recovery")
    assert toy_identity != realistic_identity
    assert checkpoint_metadata(legacy_toy.data, config, mode="synthetic")["immutable_scientific_identity"][
        "phase1b_input_manifest_sha256"
    ] == "synthetic"
    legacy_record = legacy_synthetic_input_record(legacy_toy.data, legacy_toy.timing)
    assert "dataset_design" not in legacy_record
    validate_synthetic_input_record(legacy_record, legacy_record)


def test_realistic_baseline_audit_artifact_validates_exact_content(tmp_path):
    config = _realistic_config(tmp_path)
    result = build_realistic_synthetic_recovery_dataset(config)
    _write_realistic_artifacts(config, result)

    phase1c_module._validate_synthetic_baseline_audit_artifact(config, result)

    (config.output_dir / "synthetic_baseline_coefficients.csv").write_text(
        _tampered_baseline_csv(result.baseline_coefficients_csv),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="baseline audit artifact does not match"):
        phase1c_module._validate_synthetic_baseline_audit_artifact(config, result)


def test_realistic_baseline_audit_resume_fails_missing_or_altered_before_sampling(monkeypatch, tmp_path):
    missing_config = _realistic_config(tmp_path / "missing")
    missing_result = build_realistic_synthetic_recovery_dataset(missing_config)
    missing_config.output_dir.mkdir(parents=True)
    write_json(missing_config.output_dir / "synthetic_input_record.json", synthetic_input_record(missing_result))
    with pytest.raises(FileNotFoundError, match="Missing realistic synthetic baseline audit artifact"):
        phase1c_module._validate_synthetic_baseline_audit_artifact(missing_config, missing_result)

    altered_config = _realistic_config(tmp_path / "altered")
    altered_result = build_realistic_synthetic_recovery_dataset(altered_config)
    _write_realistic_artifacts(altered_config, altered_result)
    (altered_config.output_dir / "synthetic_baseline_coefficients.csv").write_text(
        _tampered_baseline_csv(altered_result.baseline_coefficients_csv),
        encoding="utf-8",
    )
    called = {"run_ensembles": False}

    def forbidden_run_ensembles(*args, **kwargs):
        called["run_ensembles"] = True
        raise AssertionError("run_ensembles must not be reached for a malformed synthetic resume")

    monkeypatch.setattr(phase1c_module, "run_ensembles", forbidden_run_ensembles)
    with pytest.raises(ValueError, match="baseline audit artifact does not match"):
        phase1c_module._run_sampling_mode(
            altered_result.data,
            altered_config,
            altered_result.timing,
            mode="synthetic_recovery",
            steps=4,
            resume=True,
            synthetic_result=altered_result,
        )
    assert called["run_ensembles"] is False


def test_realistic_baseline_audit_valid_resume_reaches_sampler(monkeypatch, tmp_path):
    config = _realistic_config(tmp_path)
    result = build_realistic_synthetic_recovery_dataset(config)
    _write_realistic_artifacts(config, result)
    called = {"run_ensembles": False}

    def fake_run_ensembles(*args, **kwargs):
        called["run_ensembles"] = True
        return []

    def fake_write_sampling_summaries(*args, **kwargs):
        return {
            "elapsed_seconds": 0.25,
            "actual_log_posterior_calls": 0,
            "diagnostic_status": "nonconverged",
        }

    monkeypatch.setattr(phase1c_module, "run_ensembles", fake_run_ensembles)
    monkeypatch.setattr(phase1c_module, "_write_sampling_summaries", fake_write_sampling_summaries)
    summary = phase1c_module._run_sampling_mode(
        result.data,
        config,
        result.timing,
        mode="synthetic_recovery",
        steps=4,
        resume=True,
        synthetic_result=result,
    )
    assert called["run_ensembles"] is True
    assert summary["diagnostic_status"] == "nonconverged"


def test_summarize_and_phase1d_source_loading_reject_tampered_baseline_audit(tmp_path):
    config = _realistic_config(tmp_path)
    result = build_realistic_synthetic_recovery_dataset(config)
    _write_realistic_artifacts(config, result)
    (config.output_dir / "synthetic_baseline_coefficients.csv").write_text(
        _tampered_baseline_csv(result.baseline_coefficients_csv),
        encoding="utf-8",
    )

    data, timing, regenerated = phase1c_module._load_summarize_data(config, "synthetic_recovery")
    with pytest.raises(ValueError, match="baseline audit artifact does not match"):
        phase1c_module._validate_synthetic_input_record_if_needed(
            data,
            timing,
            config,
            "synthetic_recovery",
            synthetic_result=regenerated,
        )
    with pytest.raises(ValueError, match="baseline audit artifact does not match"):
        draws_module._load_bound_data_and_timing(config, "synthetic_recovery")


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


def test_authoritative_gate_statuses_and_boundary_audit(tmp_path):
    config = _realistic_config(tmp_path, n_ensembles=4, n_walkers=32)
    result = build_realistic_synthetic_recovery_dataset(config)
    rows = _strict_recovery_rows(_posterior_frame(result.injected_parameters), result.injected_parameters)
    diagnostics = _passing_diagnostics()
    ensemble = _write_authoritative_gate_artifacts(config)

    gate = _recovery_gate_record(config, result, diagnostics, ensemble, rows, None)
    assert gate["status"] == "realistic_recovery_gate_passed"
    assert gate["boundary_audit"]["passed"] is True
    assert gate["criteria"]["authoritative_v1_specification_matches_exactly"] is True
    assert gate["criteria"]["generator_and_identity_validation_pass"] is True
    assert gate["ensemble_target_audit"]["passed"] is True

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


def test_recovery_gate_identity_audit_rejects_altered_spec_and_hashes(tmp_path):
    config = _realistic_config(tmp_path, n_ensembles=4, n_walkers=32)
    result = build_realistic_synthetic_recovery_dataset(config)
    rows = _strict_recovery_rows(_posterior_frame(result.injected_parameters), result.injected_parameters)
    diagnostics = _passing_diagnostics()
    ensemble = _write_authoritative_gate_artifacts(config)

    passing_gate = _recovery_gate_record(config, result, diagnostics, ensemble, rows, None)
    assert passing_gate["status"] == "realistic_recovery_gate_passed"
    assert _identity_audit(result)["passed"] is True

    changed_seed = build_realistic_synthetic_recovery_dataset(
        config,
        spec=RealisticSyntheticRecoverySpec(synthetic_flux_seed=123),
    )
    changed_seed_gate = _recovery_gate_record(
        config,
        changed_seed,
        diagnostics,
        ensemble,
        _strict_recovery_rows(_posterior_frame(changed_seed.injected_parameters), changed_seed.injected_parameters),
        None,
    )
    assert changed_seed_gate["status"] == "realistic_recovery_gate_failed"
    assert changed_seed_gate["criteria"]["authoritative_v1_specification_matches_exactly"] is False

    for key, criterion in (
        ("preserved_structural_field_hashes", "preserved_structural_hashes_match_generated_data"),
        ("generated_synthetic_flux_sha256", "generated_flux_hash_matches_generated_flux"),
        ("baseline_coefficients_hash_sha256", "baseline_coefficient_hash_matches_rows"),
    ):
        tampered = _with_identity(result, key, "bad")
        audit = _identity_audit(tampered)
        assert audit["passed"] is False
        assert audit["criteria"][criterion] is False
        gate = _recovery_gate_record(config, tampered, diagnostics, ensemble, rows, None)
        assert gate["status"] == "realistic_recovery_gate_failed"
        assert gate["criteria"]["generator_and_identity_validation_pass"] is False


def test_ensemble_target_audit_requires_exact_ids_hdfs_shapes_and_target(tmp_path):
    config = _realistic_config(tmp_path, n_ensembles=4, n_walkers=32)
    ensemble = _write_authoritative_gate_artifacts(config, iterations=4)
    passing = _ensemble_target_audit(config, ensemble)
    assert passing["passed"] is True
    assert passing["hdf_iterations"] == {f"ensemble_{index:02d}.h5": 4 for index in range(4)}
    assert passing["completed_invocation_target_total_steps"] == 4

    duplicate = _ensemble_target_audit(config, pd.DataFrame({"ensemble": [0, 1, 1, 3]}))
    assert duplicate["criteria"]["ensemble_summary_ids_exact_once"] is False

    missing_id = _ensemble_target_audit(config, pd.DataFrame({"ensemble": [0, 1, 3]}))
    assert missing_id["criteria"]["ensemble_summary_ids_exact_once"] is False

    wrong_walkers_config = _realistic_config(tmp_path / "wrong_walkers", n_ensembles=4, n_walkers=32)
    wrong_walkers_ensemble = _write_authoritative_gate_artifacts(wrong_walkers_config, walkers=31)
    wrong_walkers = _ensemble_target_audit(wrong_walkers_config, wrong_walkers_ensemble)
    assert wrong_walkers["criteria"]["hdf_shapes_match_authoritative_requirements"] is False

    unequal_config = _realistic_config(tmp_path / "unequal", n_ensembles=4, n_walkers=32)
    unequal_ensemble = _write_authoritative_gate_artifacts(
        unequal_config,
        iterations=4,
        iterations_by_id={3: 3},
    )
    unequal = _ensemble_target_audit(unequal_config, unequal_ensemble)
    assert unequal["criteria"]["every_hdf_has_same_completed_target_length"] is False

    below_target_config = _realistic_config(tmp_path / "below_target", n_ensembles=4, n_walkers=32)
    below_target_ensemble = _write_authoritative_gate_artifacts(
        below_target_config,
        iterations=4,
        completed_target=5,
    )
    below_target = _ensemble_target_audit(below_target_config, below_target_ensemble)
    assert below_target["criteria"]["hdf_iterations_equal_completed_invocation_target"] is False

    missing_file_config = _realistic_config(tmp_path / "missing_file", n_ensembles=4, n_walkers=32)
    missing_file_ensemble = _write_authoritative_gate_artifacts(missing_file_config, missing_hdfs={2})
    missing_file = _ensemble_target_audit(missing_file_config, missing_file_ensemble)
    assert missing_file["criteria"]["expected_hdf_files_exist"] is False


def test_realistic_recovery_summary_updates_run_index_with_gate_status(monkeypatch, tmp_path):
    config = _realistic_config(tmp_path, run_id="gate_status", n_ensembles=4, n_walkers=32)
    run_config = phase1c_module.prepare_run_config(config, "synthetic_recovery", resume=False)

    def fake_run_sampling_mode(*args, **kwargs):
        run_config.output_dir.mkdir(parents=True, exist_ok=True)
        result = build_realistic_synthetic_recovery_dataset(run_config)
        write_json(run_config.output_dir / "synthetic_input_record.json", synthetic_input_record(result))
        (run_config.output_dir / "synthetic_baseline_coefficients.csv").write_text(
            result.baseline_coefficients_csv,
            encoding="utf-8",
        )
        return {"diagnostic_status": "converged"}

    def fake_synthetic_summary(*args, **kwargs):
        return {
            "mode": "synthetic_recovery",
            "run_id": run_config.run_id,
            "status": "realistic_recovery_gate_failed",
        }

    monkeypatch.setattr(phase1c_module, "_run_sampling_mode", fake_run_sampling_mode)
    monkeypatch.setattr(phase1c_module, "_synthetic_summary", fake_synthetic_summary)
    payload = run_phase1c_synthetic_validation(config, recovery=True, resume=False)
    run_index = json.loads((config.output_dir / "run_index.json").read_text(encoding="utf-8"))

    assert payload["status"] == "realistic_recovery_gate_failed"
    assert run_index["runs"][-1]["status"] == "realistic_recovery_gate_failed"


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


def _write_realistic_artifacts(config: Phase1CConfig, result) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(config.output_dir / "synthetic_input_record.json", synthetic_input_record(result))
    (config.output_dir / "synthetic_baseline_coefficients.csv").write_text(
        result.baseline_coefficients_csv,
        encoding="utf-8",
    )


def _tampered_baseline_csv(text: str) -> str:
    lines = text.splitlines()
    first = lines[1].split(",")
    first[1] = format(float(first[1]) + 1.0e-12, ".17g")
    lines[1] = ",".join(first)
    return "\n".join(lines) + "\n"


def _with_identity(result, key: str, value):
    identity = json.loads(json.dumps(result.identity))
    identity[key] = value
    return replace(result, identity=identity)


def _write_authoritative_gate_artifacts(
    config: Phase1CConfig,
    *,
    iterations: int = 4,
    completed_target: int | None = None,
    iterations_by_id: dict[int, int] | None = None,
    walkers: int = 32,
    ndim: int = len(PARAMETER_ORDER),
    missing_hdfs: set[int] | None = None,
    ensemble_ids: tuple[int, ...] = (0, 1, 2, 3),
) -> pd.DataFrame:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    target = iterations if completed_target is None else completed_target
    write_json(
        config.output_dir / "invocation_history.json",
        {
            "invocations": [
                {
                    "invocation_sequence_number": 1,
                    "status": "completed",
                    "target_total_steps": target,
                }
            ]
        },
    )
    write_json(
        config.output_dir / "sampler_runtime.json",
        {
            "ensemble_results": [
                {"strategy": "local_tight"},
                {"strategy": "local_moderate"},
                {"strategy": "local_broad"},
                {"strategy": "prior_informed"},
            ]
        },
    )
    _write_gate_hdfs(
        config.output_dir,
        iterations=iterations,
        iterations_by_id=iterations_by_id,
        walkers=walkers,
        ndim=ndim,
        missing_hdfs=missing_hdfs or set(),
    )
    return pd.DataFrame({"ensemble": list(ensemble_ids)})


def _write_gate_hdfs(
    output_dir: Path,
    *,
    iterations: int,
    iterations_by_id: dict[int, int] | None,
    walkers: int,
    ndim: int,
    missing_hdfs: set[int],
) -> None:
    for index in range(4):
        if index in missing_hdfs:
            continue
        path = output_dir / f"ensemble_{index:02d}.h5"
        count = iterations_by_id.get(index, iterations) if iterations_by_id else iterations
        with h5py.File(path, "w") as hdf:
            group = hdf.create_group("mcmc")
            group.create_dataset("chain", shape=(count, walkers, ndim), dtype="float64")


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
