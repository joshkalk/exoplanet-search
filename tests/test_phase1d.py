import json
import shutil
import time
from dataclasses import replace
from inspect import signature
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from exoplanet_search import phase1d_predictive as predictive_module
from exoplanet_search import phase1d_draws as draws_module
from exoplanet_search.phase1c import _synthetic_input_record, checkpoint_metadata, synthetic_dataset
from exoplanet_search.phase1c_parameters import vector_to_physical
from exoplanet_search.phase1c_sampler import write_checkpoint_metadata
from exoplanet_search.phase1c_types import PARAMETER_ORDER, Phase1CConfig
from exoplanet_search.phase1d import Phase1DDevelopmentConfig, run_phase1d_development_predictive
from exoplanet_search.phase1d_draws import (
    Phase1DSourceRequirements,
    Phase1DSourcePolicy,
    load_phase1d_source,
    select_posterior_draws,
)
from exoplanet_search.phase1d_predictive import draw_conditional_event_baseline, generate_replicated_flux


def test_invalid_authoritative_policy_combinations_are_rejected():
    with pytest.raises(ValueError, match="requires require_converged"):
        Phase1DSourcePolicy(authoritative=True, require_converged=False)
    with pytest.raises(ValueError, match="cannot allow nonproduction"):
        Phase1DSourcePolicy(authoritative=True, allow_nonproduction=True)
    with pytest.raises(ValueError, match="cannot include an override"):
        Phase1DSourcePolicy(authoritative=True, override_reason="because")
    with pytest.raises(ValueError, match="nonempty reason"):
        Phase1DSourcePolicy(authoritative=False, allow_nonproduction=True)
    with pytest.raises(ValueError, match="nonempty reason"):
        Phase1DSourcePolicy(authoritative=False, require_converged=False, override_reason="")


def test_hdf_shape_parameter_order_and_accepted_validation(tmp_path):
    run_dir, _, _ = _write_phase1c_hdf_fixture(tmp_path)
    load_phase1d_source(run_dir, _development_policy())

    bad = _clone_fixture(tmp_path, run_dir, "bad_order")
    with h5py.File(bad / "ensemble_00.h5", "a") as hdf:
        hdf.attrs["phase1c_parameter_order"] = json.dumps(["wrong"] + list(PARAMETER_ORDER[1:]))
    with pytest.raises(ValueError, match="Parameter order"):
        load_phase1d_source(bad, _development_policy())

    bad = _clone_fixture(tmp_path, run_dir, "bad_accepted_shape")
    with h5py.File(bad / "ensemble_00.h5", "a") as hdf:
        del hdf["mcmc/accepted"]
        hdf["mcmc"].create_dataset("accepted", data=np.ones(2))
    with pytest.raises(ValueError, match="accepted shape"):
        load_phase1d_source(bad, _development_policy())

    bad = _clone_fixture(tmp_path, run_dir, "bad_accepted_value")
    with h5py.File(bad / "ensemble_00.h5", "a") as hdf:
        hdf["mcmc/accepted"][0] = -1.0
    with pytest.raises(ValueError, match="accepted contains"):
        load_phase1d_source(bad, _development_policy())


def test_missing_mixed_and_nonfinite_ensemble_rejection(tmp_path):
    run_dir, _, _ = _write_phase1c_hdf_fixture(tmp_path, n_ensembles=4)
    (run_dir / "ensemble_03.h5").unlink()
    with pytest.raises(FileNotFoundError, match="ensemble_03"):
        load_phase1d_source(run_dir, _development_policy())

    run_dir, _, _ = _write_phase1c_hdf_fixture(tmp_path / "mixed", n_ensembles=4)
    with h5py.File(run_dir / "ensemble_02.h5", "a") as hdf:
        hdf.attrs["phase1c_run_id"] = json.dumps("other")
    with pytest.raises(ValueError, match="Checkpoint metadata mismatch|Run ID mismatch"):
        load_phase1d_source(run_dir, _development_policy())

    run_dir, _, _ = _write_phase1c_hdf_fixture(tmp_path / "nonfinite")
    with h5py.File(run_dir / "ensemble_01.h5", "a") as hdf:
        hdf["mcmc/chain"][3, 0, 0] = np.nan
    with pytest.raises(ValueError, match="Nonfinite retained vectors"):
        load_phase1d_source(run_dir, _development_policy())


def test_source_diagnostics_identity_and_checkpoint_validation(monkeypatch, tmp_path):
    run_dir, config, _ = _write_phase1c_hdf_fixture(tmp_path, mode="production", diagnostics_status="converged")
    data, _, _ = synthetic_dataset(config)
    monkeypatch.setattr(draws_module, "load_frozen_phase1b", lambda _: data)
    requirements = Phase1DSourceRequirements(
        name="fixture_requirements",
        expected_run_id="fixture",
        n_walkers=config.n_walkers,
        warmup_steps=config.warmup_steps,
        convergence_ess_minimum=config.convergence_ess_minimum,
    )

    bad = _clone_fixture(tmp_path, run_dir, "bad_diag_run")
    _write_json(bad / "sampler_diagnostics.json", {"status": "converged", "mode": "production", "run_id": "other"})
    with pytest.raises(ValueError, match="diagnostics.*run ID"):
        load_phase1d_source(bad, Phase1DSourcePolicy.authoritative_production(), requirements=requirements)

    bad = _clone_fixture(tmp_path, run_dir, "bad_diag_mode")
    _write_json(bad / "sampler_diagnostics.json", {"status": "converged", "mode": "synthetic", "run_id": config.run_id})
    with pytest.raises((ValueError, FileNotFoundError), match="synthetic input record|mode"):
        load_phase1d_source(bad, Phase1DSourcePolicy.authoritative_production(), requirements=requirements)

    bad = _clone_fixture(tmp_path, run_dir, "stale_history")
    _write_convergence_history(bad, mode="production", run_id=config.run_id, steps=6, status="converged")
    with pytest.raises(ValueError, match="stale"):
        load_phase1d_source(bad, Phase1DSourcePolicy.authoritative_production(), requirements=requirements)

    bad = _clone_fixture(tmp_path, run_dir, "unequal_iterations")
    with h5py.File(bad / "ensemble_01.h5", "a") as hdf:
        hdf["mcmc/chain"].resize((8, config.n_walkers, len(PARAMETER_ORDER)))
        hdf["mcmc/log_prob"].resize((8, config.n_walkers))
    with pytest.raises(ValueError, match="Checkpoint metadata mismatch|unequal"):
        load_phase1d_source(bad, Phase1DSourcePolicy.authoritative_production(), requirements=requirements)

    bad = _clone_fixture(tmp_path, run_dir, "metadata_mismatch")
    with h5py.File(bad / "ensemble_00.h5", "a") as hdf:
        hdf.attrs["phase1c_immutable_scientific_identity_sha256"] = json.dumps("wrong")
    with pytest.raises(ValueError, match="Checkpoint metadata mismatch"):
        load_phase1d_source(bad, Phase1DSourcePolicy.authoritative_production(), requirements=requirements)


def test_authoritative_source_requires_explicit_primary_requirements(monkeypatch, tmp_path):
    run_dir, config, _ = _write_primary_phase1d_fixture(tmp_path)
    data, _, _ = synthetic_dataset(config)
    monkeypatch.setattr(draws_module, "load_frozen_phase1b", lambda _: data)
    with pytest.raises(ValueError, match="requires explicit source requirements"):
        load_phase1d_source(run_dir, Phase1DSourcePolicy.authoritative_production())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("expected_run_id", "wrong"),
        ("expected_mode", "synthetic"),
        ("n_ensembles", 3),
        ("n_walkers", 24),
        ("warmup_steps", 1999),
        ("supersample_factor", 10),
        ("limb_darkening_sigma_floor", 0.07),
        ("convergence_rhat_threshold", 1.02),
        ("convergence_ess_minimum", 999.0),
        ("convergence_tau_multiple", 49.0),
        ("parameter_order", tuple(reversed(PARAMETER_ORDER))),
    ],
)
def test_primary_requirement_mismatches_are_rejected(field, value, tmp_path):
    run_dir, _, _ = _write_primary_phase1d_fixture(tmp_path)
    requirements = replace(Phase1DSourceRequirements.primary("primary"), **{field: value})
    expected_field = {"expected_run_id": "run_id", "expected_mode": "mode"}.get(field, field)
    with pytest.raises(ValueError, match=expected_field):
        load_phase1d_source(run_dir, Phase1DSourcePolicy.authoritative_production(), requirements=requirements)


def test_authoritative_primary_source_and_manifest_requirements(monkeypatch, tmp_path):
    run_dir, config, _ = _write_primary_phase1d_fixture(tmp_path)
    data, _, _ = synthetic_dataset(config)
    monkeypatch.setattr(draws_module, "load_frozen_phase1b", lambda _: data)
    source = load_phase1d_source(
        run_dir,
        Phase1DSourcePolicy.authoritative_production(),
        requirements=Phase1DSourceRequirements.primary("primary"),
    )
    selection = select_posterior_draws(source, requested_draws=4, seed=12)
    assert selection.manifest["authoritative"] is True
    assert selection.manifest["source_requirements"]["name"] == "phase1d_primary_posterior_v1"
    assert selection.manifest["source_requirements"]["expected"]["n_walkers"] == 32
    assert selection.manifest["source_requirements"]["observed"]["warmup_steps"] == 2000
    assert selection.manifest["input_identity"]["kind"] == "phase1b"
    assert selection.manifest["input_identity"]["manifest_sha256"] == "synthetic"
    assert selection.manifest["diagnostics_status"] == "converged"
    assert selection.manifest["source_git"] == {"commit": "fixture-commit", "is_dirty": False}


def test_authoritative_diagnostic_criteria_and_thresholds_are_required(monkeypatch, tmp_path):
    run_dir, config, _ = _write_primary_phase1d_fixture(tmp_path)
    data, _, _ = synthetic_dataset(config)
    monkeypatch.setattr(draws_module, "load_frozen_phase1b", lambda _: data)
    requirements = Phase1DSourceRequirements.primary("primary")

    bad = _clone_fixture(tmp_path, run_dir, "missing_criteria")
    diagnostics = json.loads((bad / "sampler_diagnostics.json").read_text(encoding="utf-8"))
    diagnostics["criteria"].pop("complete_valid_rhat")
    _write_json(bad / "sampler_diagnostics.json", diagnostics)
    with pytest.raises(ValueError, match="missing convergence criteria"):
        load_phase1d_source(bad, Phase1DSourcePolicy.authoritative_production(), requirements=requirements)

    bad = _clone_fixture(tmp_path, run_dir, "false_criteria")
    diagnostics = json.loads((bad / "sampler_diagnostics.json").read_text(encoding="utf-8"))
    diagnostics["criteria"]["complete_valid_autocorrelation"] = False
    _write_json(bad / "sampler_diagnostics.json", diagnostics)
    with pytest.raises(ValueError, match="not all true"):
        load_phase1d_source(bad, Phase1DSourcePolicy.authoritative_production(), requirements=requirements)

    bad = _clone_fixture(tmp_path, run_dir, "bad_fraction")
    diagnostics = json.loads((bad / "sampler_diagnostics.json").read_text(encoding="utf-8"))
    diagnostics["finite_log_probability_fraction"] = 0.999
    _write_json(bad / "sampler_diagnostics.json", diagnostics)
    with pytest.raises(ValueError, match="finite_log_probability_fraction"):
        load_phase1d_source(bad, Phase1DSourcePolicy.authoritative_production(), requirements=requirements)

    bad = _clone_fixture(tmp_path, run_dir, "bad_rhat")
    _write_convergence_history(bad, mode="production", run_id="primary", steps=2004, status="converged", config=config, rhat_max=1.02)
    with pytest.raises(ValueError, match="rhat_max"):
        load_phase1d_source(bad, Phase1DSourcePolicy.authoritative_production(), requirements=requirements)

    bad = _clone_fixture(tmp_path, run_dir, "bad_ess")
    _write_convergence_history(bad, mode="production", run_id="primary", steps=2004, status="converged", config=config, bulk_ess_min=999.0)
    with pytest.raises(ValueError, match="bulk_ess_min"):
        load_phase1d_source(bad, Phase1DSourcePolicy.authoritative_production(), requirements=requirements)


def test_synthetic_input_record_and_timing_are_validated_source_bound(tmp_path):
    run_dir, _, timing = _write_phase1c_hdf_fixture(tmp_path)
    record = json.loads((run_dir / "synthetic_input_record.json").read_text(encoding="utf-8"))
    record["timing_reference"]["period_reference"] += 0.01
    _write_json(run_dir / "synthetic_input_record.json", record)
    with pytest.raises(ValueError, match="synthetic_input_record"):
        load_phase1d_source(run_dir, _development_policy())

    run_dir, _, _ = _write_phase1c_hdf_fixture(tmp_path / "bound_timing")
    source = load_phase1d_source(run_dir, _development_policy())
    wrong_timing = type(source.timing)(**{**source.timing.__dict__, "period_reference": timing.period_reference + 100.0})
    assert source.timing.period_reference != wrong_timing.period_reference
    selection = select_posterior_draws(source, requested_draws=4, seed=1)
    expected = vector_to_physical(selection.selected_draws[0].vector, source.timing)
    assert selection.selected_draws[0].physical.period == pytest.approx(expected.period)
    parameters = signature(generate_replicated_flux).parameters
    assert list(parameters)[:3] == ["source", "draw", "rng"]
    assert "timing" not in parameters
    assert "data" not in parameters
    assert "config" not in parameters


def test_predictive_source_draw_mismatch_rejection(tmp_path):
    run_dir, _, _ = _write_phase1c_hdf_fixture(tmp_path / "source_a")
    other_dir, _, _ = _write_phase1c_hdf_fixture(tmp_path / "source_b", run_id="other_fixture")
    source = load_phase1d_source(run_dir, _development_policy())
    other_source = load_phase1d_source(other_dir, _development_policy())
    draw = select_posterior_draws(source, requested_draws=4, seed=3).selected_draws[0]

    with pytest.raises(ValueError, match="run ID"):
        generate_replicated_flux(other_source, draw, np.random.default_rng(1))
    with pytest.raises(ValueError, match="run ID"):
        generate_replicated_flux(source, replace(draw, run_id="other"), np.random.default_rng(1))
    with pytest.raises(ValueError, match="mode"):
        generate_replicated_flux(source, replace(draw, mode="production"), np.random.default_rng(1))
    with pytest.raises(ValueError, match="ensemble"):
        generate_replicated_flux(source, replace(draw, ensemble=99), np.random.default_rng(1))
    with pytest.raises(ValueError, match="physical transform"):
        generate_replicated_flux(source, replace(draw, physical=replace(draw.physical, period=draw.physical.period + 1.0)), np.random.default_rng(1))


def test_generate_replicated_flux_is_stable_on_valid_source_path(tmp_path):
    run_dir, _, _ = _write_phase1c_hdf_fixture(tmp_path)
    source = load_phase1d_source(run_dir, _development_policy())
    draw = select_posterior_draws(source, requested_draws=4, seed=3).selected_draws[0]
    first, _ = generate_replicated_flux(source, draw, np.random.default_rng(123), replication_index=4)
    second, _ = generate_replicated_flux(source, draw, np.random.default_rng(123), replication_index=4)
    assert np.allclose(first["replicated_flux"], second["replicated_flux"])


def test_converged_requirement_and_override_manifest_is_nonauthoritative(tmp_path):
    run_dir, _, _ = _write_phase1c_hdf_fixture(tmp_path, diagnostics_status="nonconverged")
    with pytest.raises(ValueError, match="requires converged"):
        load_phase1d_source(
            run_dir,
            Phase1DSourcePolicy(
                authoritative=False,
                require_converged=True,
                allow_nonproduction=True,
                override_reason="unit test nonproduction converged requirement",
            ),
        )

    source = load_phase1d_source(run_dir, _development_policy())
    selection = select_posterior_draws(source, requested_draws=4, seed=1)
    assert source.policy.authoritative is False
    assert selection.manifest["authoritative"] is False
    assert selection.manifest["nonproduction_override"] is True
    assert selection.manifest["override_reason"]


def test_detached_compatibility_loader_is_removed():
    assert not hasattr(draws_module, "load_phase1c_draw_sources")


def test_deterministic_equal_ensemble_draw_selection_and_provenance(tmp_path):
    run_dir, _, _ = _write_phase1c_hdf_fixture(tmp_path, n_ensembles=4)
    source = load_phase1d_source(run_dir, _development_policy())
    first = select_posterior_draws(source, requested_draws=8, seed=10)
    second = select_posterior_draws(source, requested_draws=8, seed=10)
    third = select_posterior_draws(source, requested_draws=8, seed=11)

    first_keys = [draw.key for draw in first.selected_draws]
    assert first_keys == [draw.key for draw in second.selected_draws]
    assert first_keys != [draw.key for draw in third.selected_draws]
    assert len(first_keys) == len(set(first_keys))
    assert first.manifest["selected_counts_by_ensemble"] == {"0": 2, "1": 2, "2": 2, "3": 2}
    for draw in first.selected_draws:
        assert draw.run_id == "fixture"
        assert draw.mode == "synthetic"
        assert draw.step >= 2
        assert draw.walker >= 0
        assert draw.selection_seed == 10
        assert np.all(np.isfinite(draw.vector))


def test_selection_fails_instead_of_sampling_with_replacement(tmp_path):
    run_dir, _, _ = _write_phase1c_hdf_fixture(tmp_path, n_ensembles=4, steps=4, walkers=2, warmup=2)
    source = load_phase1d_source(run_dir, _development_policy())
    with pytest.raises(ValueError, match="cannot select"):
        select_posterior_draws(source, requested_draws=20, seed=1)


def test_conditional_baseline_cholesky_monte_carlo_matches_analytic_moments(tmp_path):
    run_dir, _, _ = _write_phase1c_hdf_fixture(tmp_path, n_ensembles=4)
    source = load_phase1d_source(run_dir, _development_policy())
    draw = select_posterior_draws(source, requested_draws=4, seed=2).selected_draws[0]
    rng = np.random.default_rng(123)
    draws = np.asarray(
        [
            draw_conditional_event_baseline(source, draw, 0, rng).coefficients
            for _ in range(1200)
        ]
    )
    analytic = draw_conditional_event_baseline(source, draw, 0, np.random.default_rng(456))
    assert np.mean(draws, axis=0) == pytest.approx(analytic.conditional_mean, abs=8.0e-4)
    assert np.cov(draws.T) == pytest.approx(analytic.conditional_covariance, abs=8.0e-6)
    assert np.all(analytic.cholesky_diagonal > 0.0)


def test_baseline_covariance_validation(monkeypatch, tmp_path):
    run_dir, _, _ = _write_phase1c_hdf_fixture(tmp_path)
    source = load_phase1d_source(run_dir, _development_policy())
    draw = select_posterior_draws(source, requested_draws=4, seed=2).selected_draws[0]

    class BadResult:
        baseline_mean = np.array([1.0, 0.0])
        baseline_covariance = np.array([[1.0, 2.0], [0.0, 1.0]])
        log_likelihood = 0.0

    monkeypatch.setattr(predictive_module, "marginalized_event_log_likelihood", lambda **_: BadResult())
    with pytest.raises(ValueError, match="symmetric"):
        draw_conditional_event_baseline(source, draw, 0, np.random.default_rng(1))

    class SemidefiniteResult:
        baseline_mean = np.array([1.0, 0.0])
        baseline_covariance = np.array([[1.0, 0.0], [0.0, 0.0]])
        log_likelihood = 0.0

    monkeypatch.setattr(predictive_module, "marginalized_event_log_likelihood", lambda **_: SemidefiniteResult())
    with pytest.raises(ValueError, match="positive definite"):
        draw_conditional_event_baseline(source, draw, 0, np.random.default_rng(1))


def test_replicated_flux_alignment_variance_provenance_and_no_residual_resampling(tmp_path):
    run_dir, _, _ = _write_phase1c_hdf_fixture(tmp_path)
    source = load_phase1d_source(run_dir, _development_policy())
    draw = select_posterior_draws(source, requested_draws=4, seed=3).selected_draws[0]
    rows_a, baseline_a = generate_replicated_flux(
        source,
        draw,
        np.random.default_rng(99),
        replication_index=7,
    )
    rows_b, _ = generate_replicated_flux(source, draw, np.random.default_rng(99), replication_index=7)
    rows_c, _ = generate_replicated_flux(source, draw, np.random.default_rng(100), replication_index=8)

    assert np.array_equal(rows_a["cadence_index"], np.arange(source.data.cadence_count))
    assert np.array_equal(rows_a["time"], source.data.time)
    assert np.array_equal(rows_a["event_number"], source.data.event_number)
    assert np.allclose(rows_a["replicated_flux"], rows_a["predictive_mean"] + rows_a["predictive_noise"])
    assert np.all(rows_a["resampled_observed_residual"] == 0)
    assert not np.allclose(rows_a["predictive_noise"], source.data.flux - rows_a["predictive_mean"])
    assert np.allclose(rows_a["replicated_flux"], rows_b["replicated_flux"])
    assert not np.allclose(rows_a["replicated_flux"], rows_c["replicated_flux"])
    assert len(baseline_a) == source.data.event_count
    assert np.all(rows_a["selection_position"] == draw.selection_position)
    assert np.all(rows_a["predictive_replication_index"] == 7)
    assert np.all(rows_a["source_ensemble"] == draw.ensemble)
    assert baseline_a[0]["selection_position"] == draw.selection_position
    assert baseline_a[0]["predictive_replication_index"] == 7

    noises = []
    for seed in range(200, 450):
        rows, _ = generate_replicated_flux(source, draw, np.random.default_rng(seed))
        noises.append(rows["predictive_noise"][0])
    expected_variance = source.data.flux_uncertainty[0] ** 2 + draw.physical.jitter**2
    assert np.var(noises, ddof=1) == pytest.approx(expected_variance, rel=0.35)


def test_full_predictive_replicate_evaluates_transit_model_once(monkeypatch, tmp_path):
    run_dir, _, _ = _write_phase1c_hdf_fixture(tmp_path)
    source = load_phase1d_source(run_dir, _development_policy())
    draw = select_posterior_draws(source, requested_draws=4, seed=3).selected_draws[0]
    calls = {"count": 0}
    original = predictive_module.transit_model_for_vector

    def counted(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(predictive_module, "transit_model_for_vector", counted)
    generate_replicated_flux(source, draw, np.random.default_rng(99))
    assert calls["count"] == 1


def test_development_predictive_outputs_are_isolated_and_labeled(tmp_path):
    run_dir, _, _ = _write_phase1c_hdf_fixture(tmp_path, diagnostics_status="nonconverged")
    result = run_phase1d_development_predictive(
        Phase1DDevelopmentConfig(
            source_run_dir=run_dir,
            output_dir=tmp_path / "phase1d",
            run_id="dev",
            n_draws=2,
            selection_seed=5,
            predictive_seed=6,
        )
    )
    output_dir = Path(result["output_dir"])
    config = json.loads((output_dir / "phase1d_predictive_configuration.json").read_text(encoding="utf-8"))
    manifest = json.loads((output_dir / "posterior_draw_selection_manifest.json").read_text(encoding="utf-8"))
    assert result["authoritative"] is False
    assert config["nonproduction_label"] == "DEVELOPMENT_ONLY_NOT_AUTHORITATIVE"
    assert config["residual_resampling_used"] is False
    assert manifest["authoritative"] is False
    assert (output_dir / "development_predictive_flux.npz").exists()
    arrays = np.load(output_dir / "development_predictive_flux.npz")
    assert arrays["replicated_flux"].shape[0] == 2
    assert "selection_position" in arrays
    assert "predictive_replication_index" in arrays


@pytest.mark.parametrize("n_draws", [0, 11, 250])
def test_development_predictive_draw_cap_rejects_before_output_creation(tmp_path, n_draws):
    run_dir, _, _ = _write_phase1c_hdf_fixture(tmp_path, diagnostics_status="nonconverged")
    output_dir = tmp_path / "phase1d"
    with pytest.raises(ValueError, match="between 1 and 10"):
        run_phase1d_development_predictive(
            Phase1DDevelopmentConfig(
                source_run_dir=run_dir,
                output_dir=output_dir,
                run_id="too_many",
                n_draws=n_draws,
            )
        )
    assert not (output_dir / "development_predictive_too_many").exists()


def test_bounded_fixture_runtime_is_recorded(tmp_path):
    run_dir, _, _ = _write_phase1c_hdf_fixture(tmp_path)
    source = load_phase1d_source(run_dir, _development_policy())
    draw = select_posterior_draws(source, requested_draws=4, seed=3).selected_draws[0]
    start = time.perf_counter()
    generate_replicated_flux(source, draw, np.random.default_rng(99))
    elapsed = time.perf_counter() - start
    assert elapsed < 2.0


def test_phase1d_modules_have_no_published_value_leakage():
    package_root = Path(__file__).parents[1] / "src" / "exoplanet_search"
    for name in ("phase1d.py", "phase1d_draws.py", "phase1d_predictive.py"):
        text = (package_root / name).read_text(encoding="utf-8").upper()
        assert "KEPLER5B_" not in text
        assert "PUBLISHED" not in text


def _development_policy():
    return Phase1DSourcePolicy.development_override("unit test nonauthoritative fixture")


def _write_phase1c_hdf_fixture(
    tmp_path,
    *,
    run_id="fixture",
    n_ensembles=4,
    steps=7,
    walkers=4,
    warmup=2,
    diagnostics_status="converged",
    mode="synthetic",
    config_overrides=None,
):
    run_dir = tmp_path / "synthetic_fixture"
    run_dir.mkdir(parents=True, exist_ok=True)
    config = Phase1CConfig(
        output_dir=run_dir,
        run_id=run_id,
        n_ensembles=n_ensembles,
        n_walkers=walkers,
        warmup_steps=warmup,
        synthetic_steps=steps,
        chunk_steps=2,
    )
    if config_overrides:
        config = replace(config, **config_overrides)
    data, timing, _ = synthetic_dataset(config)
    base = np.array([-2.52572864, 2.14006616, 0.32407407, 0.3, 0.4, -9.43348392, 0.0, 0.0])
    _write_json(run_dir / "phase1c_configuration.json", config.to_dict())
    _write_json(run_dir / "sampler_diagnostics.json", _diagnostics_payload(config, mode, diagnostics_status))
    _write_json(run_dir / "provenance_manifest.json", {"git": {"commit": "fixture-commit", "is_dirty": False}})
    if mode in {"synthetic", "synthetic_recovery"}:
        _write_json(run_dir / "synthetic_input_record.json", _synthetic_input_record(data, timing))
    _write_convergence_history(run_dir, mode=mode, run_id=run_id, steps=steps, status=diagnostics_status, config=config)
    metadata = checkpoint_metadata(data, config, mode=mode)
    rng = np.random.default_rng(42)
    for ensemble in range(n_ensembles):
        path = run_dir / f"ensemble_{ensemble:02d}.h5"
        chain = base + rng.normal(0.0, 1.0e-4, size=(steps, walkers, len(PARAMETER_ORDER)))
        log_prob = rng.normal(-10.0, 0.1, size=(steps, walkers))
        with h5py.File(path, "w") as hdf:
            group = hdf.create_group("mcmc")
            group.create_dataset("chain", data=chain, maxshape=(None, walkers, len(PARAMETER_ORDER)))
            group.create_dataset("log_prob", data=log_prob, maxshape=(None, walkers))
            group.create_dataset("accepted", data=np.ones(walkers))
        write_checkpoint_metadata(path, metadata, config.random_seed + 1000 * ensemble)
    return run_dir, config, timing


def _write_primary_phase1d_fixture(tmp_path):
    return _write_phase1c_hdf_fixture(
        tmp_path,
        run_id="primary",
        mode="production",
        steps=2004,
        walkers=32,
        warmup=2000,
        config_overrides={
            "n_walkers": 32,
            "warmup_steps": 2000,
            "production_steps": 2004,
            "convergence_rhat_threshold": 1.01,
            "convergence_ess_minimum": 1000.0,
            "convergence_tau_multiple": 50.0,
            "limb_darkening_sigma_floor": 0.08,
            "supersample_factor": 11,
        },
    )


def _clone_fixture(tmp_path, run_dir, name):
    destination = tmp_path / name
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(run_dir, destination)
    return destination


def _write_convergence_history(
    run_dir,
    *,
    mode,
    run_id,
    steps,
    status,
    config=None,
    rhat_max=1.0,
    bulk_ess_min=None,
    tail_ess_min=None,
    posterior_stability_passed=True,
    independent_ensemble_agreement_passed=True,
    complete_valid_autocorrelation=True,
    chain_length_exceeds_tau_multiple=True,
):
    ess = float(config.convergence_ess_minimum + 200.0) if config is not None else 1200.0
    bulk_ess = ess if bulk_ess_min is None else bulk_ess_min
    tail_ess = ess if tail_ess_min is None else tail_ess_min
    pd.DataFrame(
        [
            {
                "mode": mode,
                "run_id": run_id,
                "completed_steps": steps,
                "retained_post_warmup_steps": max(steps - 2, 0),
                "rhat_max": rhat_max,
                "bulk_ess_min": bulk_ess,
                "tail_ess_min": tail_ess,
                "posterior_stability_passed": posterior_stability_passed,
                "independent_ensemble_agreement_passed": independent_ensemble_agreement_passed,
                "complete_valid_autocorrelation": complete_valid_autocorrelation,
                "chain_length_exceeds_tau_multiple": chain_length_exceeds_tau_multiple,
                "convergence_status": status,
            }
        ]
    ).to_csv(run_dir / "convergence_history.csv", index=False)


def _diagnostics_payload(config, mode, status):
    criteria = {
        "complete_valid_rhat": True,
        "complete_valid_bulk_ess": True,
        "complete_valid_tail_ess": True,
        "rhat_all_below_threshold": True,
        "ess_all_above_minimum": True,
        "tail_ess_all_above_minimum": True,
        "complete_valid_autocorrelation": True,
        "chain_length_exceeds_tau_multiple": True,
        "posterior_summary_stability": True,
        "independent_ensemble_agreement": True,
        "finite_log_probability_fraction_is_one": True,
    }
    return {
        "status": status,
        "mode": mode,
        "run_id": config.run_id,
        "criteria": criteria,
        "finite_log_probability_fraction": 1.0,
        "split_rhat": {name: 1.0 for name in PARAMETER_ORDER},
        "bulk_ess": {name: 1200.0 for name in PARAMETER_ORDER},
        "tail_ess": {name: 1200.0 for name in PARAMETER_ORDER},
    }


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
