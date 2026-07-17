import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from exoplanet_search.phase1c import (
    prepare_run_config,
    run_phase1c_synthetic_validation,
    summarize_phase1c_checkpoints,
    synthetic_dataset,
)
from exoplanet_search import phase1c_diagnostics as diagnostics_module
from exoplanet_search.phase1c_diagnostics import (
    convergence_diagnostics,
    independent_ensemble_agreement,
    posterior_stability_check,
    posterior_summary_frame,
)
from exoplanet_search.phase1c_inputs import build_phase1b_input_manifest, load_frozen_phase1b
from exoplanet_search.phase1c_likelihood import (
    log_likelihood,
    marginalized_event_log_likelihood,
)
from exoplanet_search.phase1c_parameters import (
    bounded_half_normal_logpdf,
    half_normal_logpdf,
    log_prior,
    physical_to_vector,
    timing_support_audit,
    truncated_normal_logpdf,
    vector_to_physical,
)
from exoplanet_search.phase1c_sampler import (
    build_initialization,
    checkpoint_metadata,
    deterministic_center_vector,
    initial_walkers,
    initialization_strategy,
    run_ensembles,
    validate_checkpoint_metadata,
)
from exoplanet_search.phase1c_types import PARAMETER_ORDER, Phase1CConfig


def test_phase1b_manifest_and_loader_validate_snapshot(tmp_path):
    phase1b_dir = _write_phase1b_snapshot(tmp_path)
    config = Phase1CConfig(phase1b_output_dir=phase1b_dir, output_dir=tmp_path / "phase1c")

    manifest = build_phase1b_input_manifest(phase1b_dir)
    data = load_frozen_phase1b(config)

    assert manifest["residuals_csv_used_as_input"] is False
    assert len(manifest["files"]) == 9
    assert data.cadence_count == 12
    assert data.event_count == 3
    assert data.deterministic_parameters["period_days"] == pytest.approx(2.0)


def test_phase1b_loader_rejects_duplicate_accepted_cadence(tmp_path):
    phase1b_dir = _write_phase1b_snapshot(tmp_path)
    accepted_path = phase1b_dir / "accepted_fit_cadences.csv"
    accepted = pd.read_csv(accepted_path)
    accepted = pd.concat([accepted, accepted.iloc[[0]]], ignore_index=True)
    accepted.to_csv(accepted_path, index=False)

    config = Phase1CConfig(phase1b_output_dir=phase1b_dir, output_dir=tmp_path / "phase1c")

    with pytest.raises(ValueError, match="duplicate"):
        load_frozen_phase1b(config)


def test_phase1b_loader_rejects_mixed_summary_timing(tmp_path):
    phase1b_dir = _write_phase1b_snapshot(tmp_path)
    summary_path = phase1b_dir / "phase1b_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["fitted_results"]["global_timing_refinement"]["period_days"] = 2.25
    _write_json(summary_path, summary)

    config = Phase1CConfig(phase1b_output_dir=phase1b_dir, output_dir=tmp_path / "phase1c")

    with pytest.raises(ValueError, match="Global timing refinement period_days"):
        load_frozen_phase1b(config)


def test_phase1b_loader_rejects_mixed_configuration_provenance(tmp_path):
    phase1b_dir = _write_phase1b_snapshot(tmp_path)
    provenance_path = phase1b_dir / "provenance_manifest.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["phase1b"]["configuration"]["supersample_factor"] = 7
    _write_json(provenance_path, provenance)

    config = Phase1CConfig(phase1b_output_dir=phase1b_dir, output_dir=tmp_path / "phase1c")

    with pytest.raises(ValueError, match="embedded configuration"):
        load_frozen_phase1b(config)


def test_phase1b_loader_rejects_event_center_mismatch(tmp_path):
    phase1b_dir = _write_phase1b_snapshot(tmp_path)
    accepted_path = phase1b_dir / "accepted_fit_cadences.csv"
    accepted = pd.read_csv(accepted_path)
    accepted.loc[0, "predicted_center"] += 0.01
    accepted.to_csv(accepted_path, index=False)

    config = Phase1CConfig(phase1b_output_dir=phase1b_dir, output_dir=tmp_path / "phase1c")

    with pytest.raises(ValueError, match="nonconstant predicted_center"):
        load_frozen_phase1b(config)


def test_timing_transform_roundtrip_across_prior_support(tmp_path):
    data, timing, _ = synthetic_dataset(Phase1CConfig(output_dir=tmp_path / "synthetic", n_ensembles=1))
    base = vector_to_physical(np.array([-2.4, 2.0, 0.4, 0.3, 0.4, -9.2, 0.0, 0.0]), timing)
    for period_offset in (-timing.period_half_width, 0.0, timing.period_half_width):
        for mid_offset in (-timing.mid_epoch_half_width, 0.0, timing.mid_epoch_half_width):
            sample = type(base)(
                rp=base.rp,
                a=base.a,
                b=base.b,
                q1=base.q1,
                q2=base.q2,
                jitter=base.jitter,
                period=timing.period_reference + period_offset,
                mid_epoch=timing.mid_epoch_reference + mid_offset,
                original_epoch=timing.mid_epoch_reference + mid_offset
                - timing.mid_epoch_cycle * (timing.period_reference + period_offset),
            )
            vector = physical_to_vector(sample, timing)
            recovered = vector_to_physical(vector, timing)
            assert recovered.original_epoch == pytest.approx(sample.original_epoch)
            assert recovered.mid_epoch == pytest.approx(sample.mid_epoch)


def test_log_prior_is_normalized_and_rejects_invalid_physics(tmp_path):
    data, timing, _ = synthetic_dataset(Phase1CConfig(output_dir=tmp_path / "synthetic", n_ensembles=1))
    sample = vector_to_physical(np.array([-2.5, 2.1, 0.3, 0.3, 0.4, -9.3, 0.0, 0.0]), timing)
    vector = physical_to_vector(sample, timing)
    assert np.isfinite(log_prior(vector, data, Phase1CConfig(), timing))
    invalid = vector.copy()
    invalid[1] = math.log(1.01)
    assert log_prior(invalid, data, Phase1CConfig(), timing) == -math.inf
    assert np.isfinite(truncated_normal_logpdf(0.5, 0.5, 0.1, (0.0, 1.0)))
    assert half_normal_logpdf(0.0, 1.0) == pytest.approx(0.5 * math.log(2.0 / math.pi))


def test_bounded_half_normal_prior_normalizes_over_finite_jitter_bounds():
    lower = 1.0e-8
    upper = 0.02
    scale = 0.0005
    grid = np.linspace(lower, upper, 100_000)
    density = np.exp([bounded_half_normal_logpdf(value, scale, lower, upper) for value in grid])
    assert np.trapezoid(density, grid) == pytest.approx(1.0, rel=2.0e-5)


def test_marginalized_event_likelihood_matches_dense_covariance():
    time = np.array([-1.0, -0.2, 0.4, 1.0])
    flux = np.array([1.001, 0.998, 0.999, 1.002])
    sigma = np.array([0.01, 0.011, 0.012, 0.013])
    model = np.array([1.0, 0.995, 0.996, 1.0])
    jitter = 0.002
    result = marginalized_event_log_likelihood(
        time=time,
        flux=flux,
        flux_uncertainty=sigma,
        transit_model=model,
        frozen_center=0.0,
        jitter=jitter,
        baseline_intercept_sigma=0.05,
        baseline_slope_sigma=0.04,
    )
    x = time / np.max(np.abs(time))
    design = np.column_stack([model, model * x])
    covariance = np.diag(sigma**2 + jitter**2) + design @ np.diag([0.05**2, 0.04**2]) @ design.T
    residual = flux - design @ np.array([1.0, 0.0])
    sign, logdet = np.linalg.slogdet(covariance)
    dense = -0.5 * (
        residual @ np.linalg.solve(covariance, residual) + logdet + len(time) * math.log(2.0 * math.pi)
    )
    assert sign > 0
    assert result.log_likelihood == pytest.approx(dense)
    assert result.baseline_covariance.shape == (2, 2)


def test_marginalized_event_likelihood_rejects_singular_cases():
    with pytest.raises(ValueError, match="positive"):
        marginalized_event_log_likelihood(
            time=np.array([0.0]),
            flux=np.array([1.0]),
            flux_uncertainty=np.array([0.0]),
            transit_model=np.array([1.0]),
            frozen_center=0.0,
            jitter=0.0,
            baseline_intercept_sigma=0.05,
            baseline_slope_sigma=0.05,
        )


def test_full_likelihood_equals_sum_of_events(tmp_path):
    data, timing, _ = synthetic_dataset(Phase1CConfig(output_dir=tmp_path / "synthetic", n_ensembles=1))
    vector = physical_to_vector(
        vector_to_physical(np.array([-2.525, 2.14, 0.324, 0.3, 0.4, -9.43, 0.0, 0.0]), timing),
        timing,
    )
    total = log_likelihood(vector, data, Phase1CConfig(), timing)
    assert np.isfinite(total)


def test_initial_walkers_are_reproducible_and_finite(tmp_path):
    config = Phase1CConfig(output_dir=tmp_path / "synthetic", n_ensembles=2, n_walkers=16)
    data, timing, _ = synthetic_dataset(config)
    first = initial_walkers(data, config, timing, np.random.default_rng(123), 0)
    second = initial_walkers(data, config, timing, np.random.default_rng(123), 0)
    assert np.array_equal(first, second)
    assert first.shape == (16, len(PARAMETER_ORDER))
    assert all(np.isfinite(log_prior(row, data, config, timing)) for row in first)


def test_local_initialization_strategies_remain_local_and_full_rank(tmp_path):
    config = Phase1CConfig(output_dir=tmp_path / "synthetic", n_ensembles=4, n_walkers=24)
    data, timing, _ = synthetic_dataset(config)
    center = deterministic_center_vector(data, config, timing)
    for index, strategy in enumerate(("local_tight", "local_moderate", "local_broad")):
        init = build_initialization(data, config, timing, np.random.default_rng(100 + index), strategy, 100 + index)
        assert init.walkers.shape == (24, len(PARAMETER_ORDER))
        assert init.summary["rank"] == len(PARAMETER_ORDER)
        assert np.max(np.abs(init.walkers[:, 6])) <= 6.0 * config.local_broad_scales[6]
        assert np.max(np.abs(init.walkers[:, 7])) <= 6.0 * config.local_broad_scales[7]
        assert abs(np.median(init.walkers[:, 6] - center[6])) < 3.0 * config.local_broad_scales[6]
        assert abs(np.median(init.walkers[:, 7] - center[7])) < 3.0 * config.local_broad_scales[7]
        assert init.summary["strategy"] == strategy


def test_initialization_is_reproducible_and_prior_informed_is_broader(tmp_path):
    config = Phase1CConfig(output_dir=tmp_path / "synthetic", n_ensembles=4, n_walkers=24)
    data, timing, _ = synthetic_dataset(config)
    first = build_initialization(data, config, timing, np.random.default_rng(123), "local_tight", 123)
    second = build_initialization(data, config, timing, np.random.default_rng(123), "local_tight", 123)
    broad = build_initialization(data, config, timing, np.random.default_rng(124), "prior_informed", 124)

    assert np.array_equal(first.walkers, second.walkers)
    assert initialization_strategy(0, 4) == "local_tight"
    assert initialization_strategy(1, 4) == "local_moderate"
    assert initialization_strategy(2, 4) == "local_broad"
    assert initialization_strategy(3, 4) == "prior_informed"
    assert broad.summary["actual_distance_from_deterministic_center"]["median"] > first.summary[
        "actual_distance_from_deterministic_center"
    ]["median"]


def test_run_directory_isolation_and_resume_selection(tmp_path):
    config = Phase1CConfig(output_dir=tmp_path / "phase1c", run_id="fixed")
    run_config = prepare_run_config(config, "pilot", resume=False)
    assert run_config.output_dir == tmp_path / "phase1c" / "pilot_fixed"
    (run_config.output_dir / "marker.txt").write_text("exists", encoding="utf-8")

    with pytest.raises(FileExistsError):
        prepare_run_config(config, "pilot", resume=False)

    resumed = prepare_run_config(config, "pilot", resume=True)
    assert resumed.output_dir == run_config.output_dir


def test_resume_requires_existing_run_directory(tmp_path):
    config = Phase1CConfig(output_dir=tmp_path / "phase1c", run_id="missing")
    with pytest.raises(FileNotFoundError):
        prepare_run_config(config, "pilot", resume=True)
    with pytest.raises(ValueError, match="run-id"):
        prepare_run_config(Phase1CConfig(output_dir=tmp_path / "phase1c"), "pilot", resume=True)


def test_timing_support_audit_records_corners(tmp_path):
    data, timing, _ = synthetic_dataset(Phase1CConfig(output_dir=tmp_path / "synthetic", n_ensembles=1))
    audit = timing_support_audit(data, timing)
    assert audit["earliest_frozen_event_number"] == 0
    assert audit["latest_frozen_event_number"] == 7
    assert len(audit["support_corners"]) == 4
    assert audit["center_remains_inside_every_frozen_window"] is True
    assert "0.95 guard factor" in audit["period_support_rule"]


def test_checkpoint_metadata_validation_and_resume_extension_equivalence(tmp_path):
    base_config = Phase1CConfig(
        output_dir=tmp_path / "one",
        n_ensembles=1,
        n_walkers=16,
        chunk_steps=2,
        warmup_steps=1,
        synthetic_steps=8,
    )
    data, timing, _ = synthetic_dataset(base_config)
    uninterrupted = run_ensembles(data, base_config, timing, steps=8, mode="synthetic", resume=False)
    metadata = checkpoint_metadata(data, base_config, mode="synthetic")
    validate_checkpoint_metadata(uninterrupted[0].backend_path, metadata, base_config.random_seed)

    resume_config = Phase1CConfig(
        output_dir=tmp_path / "two",
        n_ensembles=1,
        n_walkers=16,
        chunk_steps=2,
        warmup_steps=1,
        synthetic_steps=8,
    )
    data2, timing2, _ = synthetic_dataset(resume_config)
    run_ensembles(data2, resume_config, timing2, steps=4, mode="synthetic", resume=False)
    mutable_resume_config = Phase1CConfig(
        output_dir=tmp_path / "two",
        n_ensembles=1,
        n_walkers=16,
        chunk_steps=3,
        warmup_steps=1,
        synthetic_steps=8,
    )
    resumed = run_ensembles(data2, mutable_resume_config, timing2, steps=8, mode="synthetic", resume=True)

    import emcee

    chain_a = emcee.backends.HDFBackend(str(uninterrupted[0].backend_path), read_only=True).get_chain()
    chain_b = emcee.backends.HDFBackend(str(resumed[0].backend_path), read_only=True).get_chain()
    log_prob_a = emcee.backends.HDFBackend(str(uninterrupted[0].backend_path), read_only=True).get_log_prob()
    log_prob_b = emcee.backends.HDFBackend(str(resumed[0].backend_path), read_only=True).get_log_prob()
    assert np.allclose(chain_a, chain_b)
    assert np.allclose(log_prob_a, log_prob_b)

    immutable_change = Phase1CConfig(
        output_dir=tmp_path / "two",
        n_ensembles=1,
        n_walkers=16,
        chunk_steps=2,
        warmup_steps=2,
        synthetic_steps=8,
    )
    data3, timing3, _ = synthetic_dataset(immutable_change)
    with pytest.raises(ValueError, match="Checkpoint metadata mismatch"):
        run_ensembles(data3, immutable_change, timing3, steps=8, mode="synthetic", resume=True)


def test_posterior_summary_and_nonconvergence_status():
    rng = np.random.default_rng(1)
    config = Phase1CConfig(n_ensembles=1)
    _, timing, _ = synthetic_dataset(config)
    base = np.array([-2.525, 2.14, 0.32, 0.3, 0.4, -9.43, 0.0, 0.0])
    chain = base + rng.normal(0.0, 0.001, size=(4, 12, len(PARAMETER_ORDER)))
    log_prob = rng.normal(size=(4, 12))
    summary = posterior_summary_frame(
        chain,
        timing,
        warmup_steps=2,
    )
    diagnostics = convergence_diagnostics(
        chain,
        log_prob,
        np.full(4, 0.3),
        None,
        Phase1CConfig(convergence_ess_minimum=10_000),
        warmup_steps=2,
    )
    assert set(PARAMETER_ORDER) <= set(summary["parameter"])
    assert diagnostics["status"] == "nonconverged"


def test_convergence_requires_complete_parameter_diagnostics(monkeypatch):
    rng = np.random.default_rng(2)
    chain = rng.normal(size=(16, 20, len(PARAMETER_ORDER)))
    log_prob = rng.normal(size=(16, 20))

    def fake_arviz(_chain):
        good = {name: 1.0 for name in PARAMETER_ORDER}
        bulk = {name: 10_000.0 for name in PARAMETER_ORDER}
        tail = {name: 10_000.0 for name in PARAMETER_ORDER}
        good[PARAMETER_ORDER[0]] = math.nan
        return {"split_rhat": good, "bulk_ess": bulk, "tail_ess": tail}

    monkeypatch.setattr(diagnostics_module, "_arviz_diagnostics", fake_arviz)
    diagnostics = convergence_diagnostics(
        chain,
        log_prob,
        np.full(16, 0.3),
        _autocorr_report(ensembles=2, tau=0.1, retained_steps=19),
        Phase1CConfig(convergence_ess_minimum=100.0),
        warmup_steps=1,
        posterior_stability={"passed": True},
        ensemble_agreement={"passed": True},
    )

    assert diagnostics["status"] == "nonconverged"
    assert diagnostics["criteria"]["complete_valid_rhat"] is True
    assert diagnostics["legacy_walker_as_chain_diagnostics"]["gating"] is False
    assert math.isnan(diagnostics["legacy_walker_as_chain_diagnostics"]["split_rhat"][PARAMETER_ORDER[0]])


def test_autocorrelation_rule_requires_every_ensemble_parameter(monkeypatch):
    rng = np.random.default_rng(11)
    chain = rng.normal(size=(4, 120, len(PARAMETER_ORDER)))
    log_prob = np.zeros((4, 120))

    def fake_arviz(_chain):
        return {
            "split_rhat": {name: 1.0 for name in PARAMETER_ORDER},
            "bulk_ess": {name: 10_000.0 for name in PARAMETER_ORDER},
            "tail_ess": {name: 10_000.0 for name in PARAMETER_ORDER},
        }

    monkeypatch.setattr(diagnostics_module, "_arviz_diagnostics", fake_arviz)
    rows = [
        {
            "ensemble": ensemble,
            "parameter": parameter,
            "tau": 1.0,
            "available": True,
            "retained_steps": 119,
            "error": None,
        }
        for ensemble in range(2)
        for parameter in PARAMETER_ORDER
    ]
    rows[0] = {**rows[0], "tau": None, "available": False, "error": "missing"}
    diagnostics = convergence_diagnostics(
        chain,
        log_prob,
        np.full(4, 0.3),
        {"rows": rows, "all_available": False, "worst_tau": 1.0, "unavailable_count": 1},
        Phase1CConfig(convergence_ess_minimum=100.0, convergence_tau_multiple=50.0),
        warmup_steps=1,
        posterior_stability={"passed": True},
        ensemble_agreement={"passed": True},
    )
    assert diagnostics["status"] == "nonconverged"
    assert diagnostics["criteria"]["complete_valid_autocorrelation"] is False


def test_autocorrelation_rule_uses_worst_case_not_median(monkeypatch):
    rng = np.random.default_rng(12)
    chain = rng.normal(size=(4, 120, len(PARAMETER_ORDER)))
    log_prob = np.zeros((4, 120))

    def fake_arviz(_chain):
        return {
            "split_rhat": {name: 1.0 for name in PARAMETER_ORDER},
            "bulk_ess": {name: 10_000.0 for name in PARAMETER_ORDER},
            "tail_ess": {name: 10_000.0 for name in PARAMETER_ORDER},
        }

    monkeypatch.setattr(diagnostics_module, "_arviz_diagnostics", fake_arviz)
    rows = []
    for ensemble in range(2):
        for parameter in PARAMETER_ORDER:
            tau = 1.0
            if ensemble == 1 and parameter == PARAMETER_ORDER[-1]:
                tau = 3.0
            rows.append(
                {
                    "ensemble": ensemble,
                    "parameter": parameter,
                    "tau": tau,
                    "available": True,
                    "retained_steps": 119,
                    "error": None,
                }
            )
    diagnostics = convergence_diagnostics(
        chain,
        log_prob,
        np.full(4, 0.3),
        {"rows": rows, "all_available": True, "worst_tau": 3.0, "unavailable_count": 0},
        Phase1CConfig(convergence_ess_minimum=100.0, convergence_tau_multiple=50.0),
        warmup_steps=1,
        posterior_stability={"passed": True},
        ensemble_agreement={"passed": True},
    )
    assert diagnostics["status"] == "nonconverged"
    assert diagnostics["criteria"]["complete_valid_autocorrelation"] is True
    assert diagnostics["criteria"]["chain_length_exceeds_tau_multiple"] is False
    assert diagnostics["autocorrelation_worst_tau"] == pytest.approx(3.0)


def test_finite_log_probability_fraction_is_convergence_rule(monkeypatch):
    rng = np.random.default_rng(13)
    chain = rng.normal(size=(4, 120, len(PARAMETER_ORDER)))
    log_prob = np.zeros((4, 120))
    log_prob[0, 0] = -math.inf

    def fake_arviz(_chain):
        return {
            "split_rhat": {name: 1.0 for name in PARAMETER_ORDER},
            "bulk_ess": {name: 10_000.0 for name in PARAMETER_ORDER},
            "tail_ess": {name: 10_000.0 for name in PARAMETER_ORDER},
        }

    monkeypatch.setattr(diagnostics_module, "_arviz_diagnostics", fake_arviz)
    rows = [
        {
            "ensemble": ensemble,
            "parameter": parameter,
            "tau": 1.0,
            "available": True,
            "retained_steps": 119,
            "error": None,
        }
        for ensemble in range(2)
        for parameter in PARAMETER_ORDER
    ]
    diagnostics = convergence_diagnostics(
        chain,
        log_prob,
        np.full(4, 0.3),
        {"rows": rows, "all_available": True, "worst_tau": 1.0, "unavailable_count": 0},
        Phase1CConfig(convergence_ess_minimum=100.0, convergence_tau_multiple=50.0),
        warmup_steps=1,
        posterior_stability={"passed": True},
        ensemble_agreement={"passed": True},
    )
    assert diagnostics["status"] == "nonconverged"
    assert diagnostics["criteria"]["finite_log_probability_fraction_is_one"] is False


def test_convergence_requires_stable_intervals_and_ensemble_agreement(monkeypatch, tmp_path):
    config = Phase1CConfig(
        output_dir=tmp_path / "synthetic",
        n_ensembles=2,
        convergence_ess_minimum=100.0,
        convergence_stability_chunks=3,
        convergence_ensemble_shift_threshold=0.05,
    )
    _, timing, _ = synthetic_dataset(config)
    base = np.array([-2.525, 2.14, 0.32, 0.3, 0.4, -9.43, 0.0, 0.0])
    rng = np.random.default_rng(3)
    ensemble_a = base + rng.normal(0.0, 0.001, size=(8, 20, len(PARAMETER_ORDER)))
    ensemble_b = base + rng.normal(0.0, 0.001, size=(8, 20, len(PARAMETER_ORDER)))
    ensemble_b[:, :, 0] += 1.0
    agreement = independent_ensemble_agreement([ensemble_a, ensemble_b], timing, config, warmup_steps=1)
    assert agreement["passed"] is False

    first = posterior_summary_frame(ensemble_a, timing, warmup_steps=1)
    second = first.copy()
    third = first.copy()
    second["q16"] = second["median"] - 0.1
    second["q84"] = second["median"] + 0.1
    third["q16"] = third["median"] - 0.001
    third["q84"] = third["median"] + 0.001
    stability = posterior_stability_check([first, second, third], config)
    assert stability["passed"] is False

    def fake_arviz(_chain):
        return {
            "split_rhat": {name: 1.0 for name in PARAMETER_ORDER},
            "bulk_ess": {name: 10_000.0 for name in PARAMETER_ORDER},
            "tail_ess": {name: 10_000.0 for name in PARAMETER_ORDER},
        }

    monkeypatch.setattr(diagnostics_module, "_arviz_diagnostics", fake_arviz)
    chain = np.concatenate([ensemble_a, ensemble_b], axis=0)
    diagnostics = convergence_diagnostics(
        chain,
        np.zeros(chain.shape[:2]),
        np.full(chain.shape[0], 0.3),
        _autocorr_report(ensembles=2, tau=0.1, retained_steps=19),
        config,
        warmup_steps=1,
        posterior_stability={"passed": True},
        ensemble_agreement=agreement,
    )
    assert diagnostics["status"] == "nonconverged"
    assert diagnostics["criteria"]["independent_ensemble_agreement"] is False


def test_synthetic_run_reuses_stability_history_and_summarizes_checkpoints(tmp_path):
    config = Phase1CConfig(
        output_dir=tmp_path / "phase1c",
        run_id="stable",
        n_ensembles=2,
        n_walkers=16,
        synthetic_steps=6,
        chunk_steps=2,
        warmup_steps=1,
        convergence_stability_chunks=3,
        convergence_stability_sigma_threshold=1.0e9,
    )
    result = run_phase1c_synthetic_validation(config)
    run_dir = Path(result["run_directory"])
    diagnostics = json.loads((run_dir / "sampler_diagnostics.json").read_text(encoding="utf-8"))
    history = pd.read_csv(run_dir / "convergence_history.csv")
    run_index = json.loads((tmp_path / "phase1c" / "run_index.json").read_text(encoding="utf-8"))

    assert (run_dir / "posterior_summary_history.jsonl").exists()
    assert diagnostics["posterior_stability"]["passed"] is True
    assert bool(history.iloc[-1]["posterior_stability_passed"]) is True
    assert history.iloc[-1]["convergence_status"] == diagnostics["status"]
    assert run_index["runs"][-1]["status"] == diagnostics["status"]

    runtime = json.loads((run_dir / "sampler_runtime.json").read_text(encoding="utf-8"))
    assert runtime["latest_invocation"]["status"] == "completed"
    assert runtime["cumulative_totals"]["posterior_calls"] > 0
    assert runtime["invocation_history_path"].endswith("invocation_history.json")

    summary = summarize_phase1c_checkpoints(config, mode="synthetic")
    assert summary["checkpoint_metadata_validated"] is True
    assert summary["stored_log_probability_entries"] > 0
    invocation_history = json.loads((run_dir / "invocation_history.json").read_text(encoding="utf-8"))
    assert invocation_history["invocations"][-1]["mode"] == "synthetic_summarize"
    assert invocation_history["invocations"][-1]["status"] == "completed"


def test_generic_phase1c_modules_have_no_kepler5_planet_constant_imports():
    package_root = Path(__file__).parents[1] / "src" / "exoplanet_search"
    for name in (
        "phase1c.py",
        "phase1c_inputs.py",
        "phase1c_likelihood.py",
        "phase1c_parameters.py",
        "phase1c_sampler.py",
    ):
        text = (package_root / name).read_text(encoding="utf-8").upper()
        assert "KEPLER5B_" not in text


def _autocorr_report(*, ensembles: int, tau: float, retained_steps: int):
    rows = [
        {
            "ensemble": ensemble,
            "parameter": parameter,
            "tau": tau,
            "available": True,
            "retained_steps": retained_steps,
            "error": None,
        }
        for ensemble in range(ensembles)
        for parameter in PARAMETER_ORDER
    ]
    return {"rows": rows, "all_available": True, "worst_tau": tau, "unavailable_count": 0}


def _write_phase1b_snapshot(tmp_path):
    phase1b_dir = tmp_path / "phase1b"
    phase1b_dir.mkdir()
    times = []
    events = []
    centers = []
    for event in range(3):
        center = 1.0 + 2.0 * event
        for offset in (-0.2, -0.05, 0.05, 0.2):
            times.append(center + offset)
            events.append(event)
            centers.append(center)
    accepted = pd.DataFrame(
        {
            "time": times,
            "flux": np.ones(len(times)),
            "flux_uncertainty": np.full(len(times), 0.001),
            "event_number": events,
            "predicted_center": centers,
            "product_id": ["product"] * len(times),
            "quarter": [0] * len(times),
            "exposure_days": np.full(len(times), 0.02),
        }
    )
    accepted.to_csv(phase1b_dir / "accepted_fit_cadences.csv", index=False)
    pd.DataFrame(
        {
            "event_number": [0, 1, 2],
            "predicted_midpoint": [1.0, 3.0, 5.0],
            "included": [True, True, True],
            "total_point_count": [4, 4, 4],
        }
    ).to_csv(
        phase1b_dir / "transit_window_audit.csv",
        index=False,
    )
    pd.DataFrame(
        {
            "product_index": [0],
            "product_id": ["product"],
            "quarter": [0],
            "input_cadence_count": [12],
            "finite_cadence_count": [12],
            "exposure_duration_days": [0.02],
        }
    ).to_csv(
        phase1b_dir / "observation_product_metadata.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {
                "stage": "global_timing_refinement",
                "objective_value": -100.0,
                "rp_over_rstar": 0.08,
                "a_over_rstar": 8.0,
                "impact_parameter": 0.3,
                "q1": 0.3,
                "q2": 0.4,
                "white_noise_jitter": 0.0001,
                "period_days": 2.0,
                "transit_time": 1.0,
            }
        ]
    ).to_csv(phase1b_dir / "deterministic_fit_parameters.csv", index=False)
    _write_json(phase1b_dir / "limb_darkening_inputs.json", {"q1": 0.3, "q2": 0.4, "q1_sigma": 0.03, "q2_sigma": 0.04})
    _write_json(
        phase1b_dir / "phase1b_configuration.json",
        {"supersample_factor": 11, "timing_refinement_t0_half_width_duration_scale": 0.5},
    )
    phase1b_configuration = json.loads((phase1b_dir / "phase1b_configuration.json").read_text(encoding="utf-8"))
    phase1a_record = {"fixture": True}
    timing_refinement = {
        "objective_value": -100.0,
        "rp_over_rstar": 0.08,
        "a_over_rstar": 8.0,
        "impact_parameter": 0.3,
        "q1": 0.3,
        "q2": 0.4,
        "white_noise_jitter": 0.0001,
        "period_days": 2.0,
        "transit_time": 1.0,
    }
    _write_json(
        phase1b_dir / "phase1b_summary.json",
        {
            "established_inputs": {
                "full_mission_local_refinement": {
                    "refined_period_days": 2.0,
                    "refined_transit_time": 1.0,
                    "refined_duration_days": 0.1,
                }
            },
            "transit_windows": {"included_count": 3, "predicted_count": 3},
            "fitted_results": {"global_timing_refinement": timing_refinement},
            "diagnostic_results": {
                "residual_summary": {
                    "by_product": [{"product_id": "product", "cadence_count": len(accepted)}],
                }
            },
            "acceptance_checks": {"published_physical_planet_parameters_used_or_compared": False},
        },
    )
    _write_json(
        phase1b_dir / "provenance_manifest.json",
        {
            "cadence_counts": {"phase1b_fit_cadence_count": len(accepted)},
            "phase1b": {
                "configuration": phase1b_configuration,
                "phase1a_input_record": phase1a_record,
            },
        },
    )
    _write_json(phase1b_dir / "phase1a_input_record.json", phase1a_record)
    return phase1b_dir


def _write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
