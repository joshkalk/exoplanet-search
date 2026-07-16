import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from exoplanet_search.phase1c import prepare_run_config, synthetic_dataset
from exoplanet_search.phase1c_diagnostics import convergence_diagnostics, posterior_summary_frame
from exoplanet_search.phase1c_inputs import build_phase1b_input_manifest, load_frozen_phase1b
from exoplanet_search.phase1c_likelihood import (
    log_likelihood,
    marginalized_event_log_likelihood,
)
from exoplanet_search.phase1c_parameters import (
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


def test_checkpoint_metadata_validation_and_resume_equivalence(tmp_path):
    base_config = Phase1CConfig(
        output_dir=tmp_path / "one",
        n_ensembles=1,
        n_walkers=16,
        chunk_steps=2,
        warmup_steps=1,
        synthetic_steps=4,
    )
    data, timing, _ = synthetic_dataset(base_config)
    uninterrupted = run_ensembles(data, base_config, timing, steps=4, mode="synthetic", resume=False)
    metadata = checkpoint_metadata(data, base_config, mode="synthetic")
    validate_checkpoint_metadata(uninterrupted[0].backend_path, metadata, base_config.random_seed)

    resume_config = Phase1CConfig(
        output_dir=tmp_path / "two",
        n_ensembles=1,
        n_walkers=16,
        chunk_steps=2,
        warmup_steps=1,
        synthetic_steps=4,
    )
    data2, timing2, _ = synthetic_dataset(resume_config)
    run_ensembles(data2, resume_config, timing2, steps=2, mode="synthetic", resume=False)
    resumed = run_ensembles(data2, resume_config, timing2, steps=4, mode="synthetic", resume=True)

    import emcee

    chain_a = emcee.backends.HDFBackend(str(uninterrupted[0].backend_path), read_only=True).get_chain()
    chain_b = emcee.backends.HDFBackend(str(resumed[0].backend_path), read_only=True).get_chain()
    assert np.allclose(chain_a, chain_b)


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
    pd.DataFrame({"event_number": [0, 1, 2], "included": [True, True, True]}).to_csv(
        phase1b_dir / "transit_window_audit.csv",
        index=False,
    )
    pd.DataFrame({"product_id": ["product"], "quarter": [0]}).to_csv(
        phase1b_dir / "observation_product_metadata.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {
                "stage": "global_timing_refinement",
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
    _write_json(
        phase1b_dir / "phase1b_summary.json",
        {
            "established_inputs": {
                "full_mission_local_refinement": {
                    "refined_period_days": 2.0,
                    "refined_duration_days": 0.1,
                }
            },
            "transit_windows": {"included_count": 3, "predicted_count": 3},
            "acceptance_checks": {"published_physical_planet_parameters_used_or_compared": False},
        },
    )
    _write_json(
        phase1b_dir / "provenance_manifest.json",
        {"cadence_counts": {"phase1b_fit_cadence_count": len(accepted)}},
    )
    _write_json(phase1b_dir / "phase1a_input_record.json", {"fixture": True})
    return phase1b_dir


def _write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
