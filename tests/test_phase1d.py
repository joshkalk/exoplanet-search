import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from exoplanet_search import phase1d_predictive as predictive_module
from exoplanet_search.phase1c import synthetic_dataset
from exoplanet_search.phase1c_parameters import vector_to_physical
from exoplanet_search.phase1c_types import PARAMETER_ORDER, Phase1CConfig
from exoplanet_search.phase1d import Phase1DDevelopmentConfig, run_phase1d_development_predictive
from exoplanet_search.phase1d_draws import (
    Phase1DSourcePolicy,
    load_phase1c_draw_sources,
    select_posterior_draws,
)
from exoplanet_search.phase1d_predictive import draw_conditional_event_baseline, generate_replicated_flux


def test_hdf_shape_and_parameter_order_validation(tmp_path):
    run_dir, config, timing = _write_phase1c_hdf_fixture(tmp_path)
    load_phase1c_draw_sources(run_dir, Phase1DSourcePolicy(require_converged=True, allow_nonproduction=True))

    bad = run_dir / "ensemble_00.h5"
    with h5py.File(bad, "a") as hdf:
        hdf.attrs["phase1c_parameter_order"] = json.dumps(["wrong"] + list(PARAMETER_ORDER[1:]))

    with pytest.raises(ValueError, match="Parameter order"):
        select_posterior_draws(
            run_dir,
            timing=timing,
            requested_draws=4,
            seed=1,
            policy=Phase1DSourcePolicy(require_converged=True, allow_nonproduction=True),
        )


def test_missing_and_mixed_ensemble_rejection(tmp_path):
    run_dir, _, timing = _write_phase1c_hdf_fixture(tmp_path, n_ensembles=4)
    (run_dir / "ensemble_03.h5").unlink()
    with pytest.raises(ValueError, match="Expected 4 ensemble"):
        select_posterior_draws(
            run_dir,
            timing=timing,
            requested_draws=4,
            seed=1,
            policy=Phase1DSourcePolicy(require_converged=True, allow_nonproduction=True),
        )

    run_dir, _, timing = _write_phase1c_hdf_fixture(tmp_path / "mixed", n_ensembles=4)
    with h5py.File(run_dir / "ensemble_02.h5", "a") as hdf:
        hdf.attrs["phase1c_run_id"] = json.dumps("other")
    with pytest.raises(ValueError, match="Run ID mismatch|mixed"):
        select_posterior_draws(
            run_dir,
            timing=timing,
            requested_draws=4,
            seed=1,
            policy=Phase1DSourcePolicy(require_converged=True, allow_nonproduction=True),
        )


def test_nonfinite_retained_draw_rejection(tmp_path):
    run_dir, _, timing = _write_phase1c_hdf_fixture(tmp_path)
    with h5py.File(run_dir / "ensemble_01.h5", "a") as hdf:
        hdf["mcmc/chain"][3, 0, 0] = np.nan

    with pytest.raises(ValueError, match="Nonfinite retained vectors"):
        select_posterior_draws(
            run_dir,
            timing=timing,
            requested_draws=4,
            seed=1,
            policy=Phase1DSourcePolicy(require_converged=True, allow_nonproduction=True),
        )


def test_converged_requirement_and_nonproduction_override(tmp_path):
    run_dir, _, timing = _write_phase1c_hdf_fixture(tmp_path, diagnostics_status="nonconverged")
    with pytest.raises(ValueError, match="requires converged"):
        select_posterior_draws(
            run_dir,
            timing=timing,
            requested_draws=4,
            seed=1,
            policy=Phase1DSourcePolicy(require_converged=True, allow_nonproduction=True),
        )

    selection = select_posterior_draws(
        run_dir,
        timing=timing,
        requested_draws=4,
        seed=1,
        policy=Phase1DSourcePolicy(
            require_converged=False,
            allow_nonproduction=True,
            authoritative=False,
            override_reason="unit test",
        ),
    )
    assert selection.manifest["authoritative"] is False
    assert selection.manifest["nonproduction_override"] is True


def test_deterministic_equal_ensemble_draw_selection_and_provenance(tmp_path):
    run_dir, _, timing = _write_phase1c_hdf_fixture(tmp_path, n_ensembles=4)
    policy = Phase1DSourcePolicy(require_converged=True, allow_nonproduction=True)
    first = select_posterior_draws(run_dir, timing=timing, requested_draws=8, seed=10, policy=policy)
    second = select_posterior_draws(run_dir, timing=timing, requested_draws=8, seed=10, policy=policy)
    third = select_posterior_draws(run_dir, timing=timing, requested_draws=8, seed=11, policy=policy)

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
    run_dir, _, timing = _write_phase1c_hdf_fixture(tmp_path, n_ensembles=4, steps=4, walkers=2, warmup=2)
    with pytest.raises(ValueError, match="cannot select"):
        select_posterior_draws(
            run_dir,
            timing=timing,
            requested_draws=20,
            seed=1,
            policy=Phase1DSourcePolicy(require_converged=True, allow_nonproduction=True),
        )


def test_conditional_baseline_monte_carlo_matches_analytic_moments(tmp_path):
    run_dir, config, timing = _write_phase1c_hdf_fixture(tmp_path, n_ensembles=4)
    data, _, _ = synthetic_dataset(config)
    draw = select_posterior_draws(
        run_dir,
        timing=timing,
        requested_draws=4,
        seed=2,
        policy=Phase1DSourcePolicy(require_converged=True, allow_nonproduction=True),
    ).selected_draws[0]
    rng = np.random.default_rng(123)
    draws = np.asarray(
        [
            draw_conditional_event_baseline(draw, 0, data, config, timing, rng).coefficients
            for _ in range(1200)
        ]
    )
    analytic = draw_conditional_event_baseline(draw, 0, data, config, timing, np.random.default_rng(456))
    assert np.mean(draws, axis=0) == pytest.approx(analytic.conditional_mean, abs=8.0e-4)
    assert np.cov(draws.T) == pytest.approx(analytic.conditional_covariance, abs=8.0e-6)


def test_baseline_covariance_validation(monkeypatch, tmp_path):
    run_dir, config, timing = _write_phase1c_hdf_fixture(tmp_path)
    data, _, _ = synthetic_dataset(config)
    draw = select_posterior_draws(
        run_dir,
        timing=timing,
        requested_draws=4,
        seed=2,
        policy=Phase1DSourcePolicy(require_converged=True, allow_nonproduction=True),
    ).selected_draws[0]

    class BadResult:
        baseline_mean = np.array([1.0, 0.0])
        baseline_covariance = np.array([[1.0, 2.0], [0.0, 1.0]])
        log_likelihood = 0.0

    monkeypatch.setattr(predictive_module, "marginalized_event_log_likelihood", lambda **_: BadResult())
    with pytest.raises(ValueError, match="symmetric"):
        draw_conditional_event_baseline(draw, 0, data, config, timing, np.random.default_rng(1))


def test_replicated_flux_alignment_variance_and_no_residual_resampling(tmp_path):
    run_dir, config, timing = _write_phase1c_hdf_fixture(tmp_path)
    data, _, _ = synthetic_dataset(config)
    draw = select_posterior_draws(
        run_dir,
        timing=timing,
        requested_draws=4,
        seed=3,
        policy=Phase1DSourcePolicy(require_converged=True, allow_nonproduction=True),
    ).selected_draws[0]
    rows_a, baseline_a = generate_replicated_flux(draw, data, config, timing, np.random.default_rng(99))
    rows_b, _ = generate_replicated_flux(draw, data, config, timing, np.random.default_rng(99))
    rows_c, _ = generate_replicated_flux(draw, data, config, timing, np.random.default_rng(100))

    assert np.array_equal(rows_a["cadence_index"], np.arange(data.cadence_count))
    assert np.array_equal(rows_a["time"], data.time)
    assert np.array_equal(rows_a["event_number"], data.event_number)
    assert np.allclose(rows_a["replicated_flux"], rows_a["predictive_mean"] + rows_a["predictive_noise"])
    assert np.all(rows_a["resampled_observed_residual"] == 0)
    assert not np.allclose(rows_a["predictive_noise"], data.flux - rows_a["predictive_mean"])
    assert np.allclose(rows_a["replicated_flux"], rows_b["replicated_flux"])
    assert not np.allclose(rows_a["replicated_flux"], rows_c["replicated_flux"])
    assert len(baseline_a) == data.event_count

    noises = []
    for seed in range(200, 450):
        rows, _ = generate_replicated_flux(draw, data, config, timing, np.random.default_rng(seed))
        noises.append(rows["predictive_noise"][0])
    sample = vector_to_physical(draw.vector, timing)
    expected_variance = data.flux_uncertainty[0] ** 2 + sample.jitter**2
    assert np.var(noises, ddof=1) == pytest.approx(expected_variance, rel=0.35)


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
    assert result["authoritative"] is False
    assert config["nonproduction_label"] == "DEVELOPMENT_ONLY_NOT_AUTHORITATIVE"
    assert config["residual_resampling_used"] is False
    assert (output_dir / "development_predictive_flux.npz").exists()
    arrays = np.load(output_dir / "development_predictive_flux.npz")
    assert arrays["replicated_flux"].shape[0] == 2


def test_phase1d_modules_have_no_published_value_leakage():
    package_root = Path(__file__).parents[1] / "src" / "exoplanet_search"
    for name in ("phase1d.py", "phase1d_draws.py", "phase1d_predictive.py"):
        text = (package_root / name).read_text(encoding="utf-8").upper()
        assert "KEPLER5B_" not in text
        assert "PUBLISHED" not in text


def _write_phase1c_hdf_fixture(
    tmp_path,
    *,
    n_ensembles=4,
    steps=7,
    walkers=4,
    warmup=2,
    diagnostics_status="converged",
):
    run_dir = tmp_path / "synthetic_fixture"
    run_dir.mkdir(parents=True, exist_ok=True)
    config = Phase1CConfig(
        output_dir=run_dir,
        run_id="fixture",
        n_ensembles=n_ensembles,
        n_walkers=walkers,
        warmup_steps=warmup,
        synthetic_steps=steps,
        chunk_steps=2,
    )
    _, timing, _ = synthetic_dataset(config)
    base = np.array([-2.52572864, 2.14006616, 0.32407407, 0.3, 0.4, -9.43348392, 0.0, 0.0])
    (run_dir / "phase1c_configuration.json").write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    (run_dir / "sampler_diagnostics.json").write_text(
        json.dumps({"status": diagnostics_status, "mode": "synthetic", "run_id": "fixture"}, indent=2),
        encoding="utf-8",
    )
    rng = np.random.default_rng(42)
    for ensemble in range(n_ensembles):
        path = run_dir / f"ensemble_{ensemble:02d}.h5"
        chain = base + rng.normal(0.0, 1.0e-4, size=(steps, walkers, len(PARAMETER_ORDER)))
        log_prob = rng.normal(-10.0, 0.1, size=(steps, walkers))
        with h5py.File(path, "w") as hdf:
            group = hdf.create_group("mcmc")
            group.create_dataset("chain", data=chain)
            group.create_dataset("log_prob", data=log_prob)
            group.create_dataset("accepted", data=np.ones(walkers))
            hdf.attrs["phase1c_run_id"] = json.dumps("fixture")
            hdf.attrs["phase1c_mode"] = json.dumps("synthetic")
            hdf.attrs["phase1c_parameter_order"] = json.dumps(list(PARAMETER_ORDER))
            hdf.attrs["phase1c_ensemble_seed"] = config.random_seed + 1000 * ensemble
    return run_dir, config, timing
