import math
from pathlib import Path

import numpy as np
import pytest

from exoplanet_search import phase1c_sampler as sampler_module
from exoplanet_search.phase1c import synthetic_dataset
from exoplanet_search.phase1c_inputs import load_frozen_phase1b
from exoplanet_search.phase1c_likelihood import (
    Phase1CLikelihoodContext,
    PosteriorProfiler,
    log_likelihood_with_context,
    log_probability_with_context,
    marginalized_event_log_likelihood,
    marginalized_event_log_likelihood_from_context,
    profiled_log_probability_with_context,
    reference_log_likelihood,
    reference_log_probability,
    reference_profiled_log_probability,
    transit_model_for_sample,
    transit_model_for_vector,
)
from exoplanet_search.phase1c_parameters import build_timing_reference, vector_to_physical
from exoplanet_search.phase1c_sampler import (
    ProfiledLogPosterior,
    build_initialization,
    deterministic_center_vector,
    run_ensembles,
)
from exoplanet_search.phase1c_types import PARAMETER_ORDER, Phase1CConfig


@pytest.mark.parametrize("supersample_factor", [3, 11])
def test_context_likelihood_matches_reference_for_synthetic_vectors(tmp_path, supersample_factor):
    config = Phase1CConfig(
        output_dir=tmp_path / f"synthetic_{supersample_factor}",
        n_ensembles=1,
        n_walkers=16,
        supersample_factor=supersample_factor,
    )
    data, timing, _ = synthetic_dataset(config)
    context = Phase1CLikelihoodContext.from_data(data, config, timing)
    vectors = _synthetic_valid_vectors(data, config, timing, context)

    assert context.cadence_count == 224
    assert context.event_count == 8
    assert context.exposure_groups_safe is True
    assert all(event.is_contiguous for event in context.events)
    _assert_context_arrays_are_read_only(context)

    for vector in vectors:
        _assert_reference_equivalence(vector, data, config, timing, context)

    invalid = vectors[0].copy()
    invalid[3] = 1.2
    assert reference_log_probability(invalid, data, config, timing) == -math.inf
    assert log_probability_with_context(invalid, context) == -math.inf


def test_context_event_conditionals_match_reference_event_math(tmp_path):
    config = Phase1CConfig(output_dir=tmp_path / "synthetic", n_ensembles=1, n_walkers=16)
    data, timing, _ = synthetic_dataset(config)
    context = Phase1CLikelihoodContext.from_data(data, config, timing)
    vector = deterministic_center_vector(data, config, timing)
    sample = vector_to_physical(vector, timing)
    transit_model = transit_model_for_vector(vector, data, config, timing)
    context_model = transit_model_for_sample(sample, context)

    assert np.array_equal(transit_model, context_model)
    for event in (context.events[0], context.events[len(context.events) // 2], context.events[-1]):
        mask = data.event_number == event.event_number
        reference = marginalized_event_log_likelihood(
            time=data.time[mask],
            flux=data.flux[mask],
            flux_uncertainty=data.flux_uncertainty[mask],
            transit_model=transit_model[mask],
            frozen_center=float(np.median(data.predicted_center[mask])),
            jitter=sample.jitter,
            baseline_intercept_sigma=config.baseline_intercept_sigma,
            baseline_slope_sigma=config.baseline_slope_sigma,
        )
        optimized = marginalized_event_log_likelihood_from_context(event, context_model, sample.jitter, context)
        assert optimized.log_likelihood == pytest.approx(reference.log_likelihood, abs=1.0e-12)
        assert optimized.baseline_mean == pytest.approx(reference.baseline_mean, abs=1.0e-12)
        assert optimized.baseline_covariance == pytest.approx(reference.baseline_covariance, abs=1.0e-12)


def test_profiled_context_path_matches_reference_and_records_categories(tmp_path):
    config = Phase1CConfig(output_dir=tmp_path / "synthetic", n_ensembles=1, n_walkers=16)
    data, timing, _ = synthetic_dataset(config)
    context = Phase1CLikelihoodContext.from_data(data, config, timing)
    vector = _synthetic_valid_vectors(data, config, timing, context)[0]
    reference_profiler = PosteriorProfiler()
    optimized_profiler = PosteriorProfiler()

    reference = reference_profiled_log_probability(vector, data, config, timing, reference_profiler)
    optimized = profiled_log_probability_with_context(vector, context, optimized_profiler)

    assert optimized == pytest.approx(reference, abs=1.0e-12)
    for summary in (reference_profiler.summary(), optimized_profiler.summary()):
        assert summary["posterior_calls"] == 1
        assert set(summary) == {
            "posterior_calls",
            "prior_transform_seconds",
            "batman_model_seconds",
            "marginalized_baseline_likelihood_seconds",
            "total_log_posterior_seconds",
            "invalid_prior_count",
            "invalid_likelihood_count",
        }
        assert summary["total_log_posterior_seconds"] >= 0.0

    invalid = vector.copy()
    invalid[2] = 2.0
    assert profiled_log_probability_with_context(invalid, context, optimized_profiler) == -math.inf
    assert optimized_profiler.invalid_prior_count == 1


def test_context_constructed_once_for_bounded_sampler_setup(monkeypatch, tmp_path):
    config = Phase1CConfig(
        output_dir=tmp_path / "synthetic",
        n_ensembles=2,
        n_walkers=16,
        chunk_steps=1,
        synthetic_steps=0,
    )
    data, timing, _ = synthetic_dataset(config)
    original_from_data = sampler_module.Phase1CLikelihoodContext.from_data
    calls = 0

    def counted_from_data(data_arg, config_arg, timing_arg):
        nonlocal calls
        calls += 1
        return original_from_data(data_arg, config_arg, timing_arg)

    monkeypatch.setattr(
        sampler_module.Phase1CLikelihoodContext,
        "from_data",
        staticmethod(counted_from_data),
    )

    results = run_ensembles(data, config, timing, steps=0, mode="synthetic", resume=False)

    assert calls == 1
    assert [result.iterations for result in results] == [0, 0]


def test_prebuilt_context_is_reused_by_sampler_callable_and_initialization(monkeypatch, tmp_path):
    config = Phase1CConfig(output_dir=tmp_path / "synthetic", n_ensembles=1, n_walkers=16)
    data, timing, _ = synthetic_dataset(config)
    context = Phase1CLikelihoodContext.from_data(data, config, timing)

    def fail_rebuild(*_args, **_kwargs):
        raise AssertionError("context should not be rebuilt")

    monkeypatch.setattr(Phase1CLikelihoodContext, "from_data", staticmethod(fail_rebuild))
    init = build_initialization(
        data,
        config,
        timing,
        np.random.default_rng(123),
        "local_tight",
        123,
        context=context,
    )
    wrapper = ProfiledLogPosterior(context, PosteriorProfiler())

    assert init.walkers.shape == (16, len(PARAMETER_ORDER))
    assert np.isfinite(wrapper(init.walkers[0]))


def test_context_likelihood_matches_reference_for_real_phase1b_data():
    if not Path("data/interim/kepler5_phase1b_fit/accepted_fit_cadences.csv").exists():
        pytest.skip("Real frozen Phase 1B data is not present in this checkout.")
    config = Phase1CConfig()
    data = load_frozen_phase1b(config)
    timing = build_timing_reference(data, config)
    context = Phase1CLikelihoodContext.from_data(data, config, timing)
    center = deterministic_center_vector(data, config, timing)
    rng = np.random.default_rng(20260717)
    vectors = [center]
    scales = np.asarray(config.local_moderate_scales, dtype=float)
    attempts = 0
    while len(vectors) < 4 and attempts < 50:
        attempts += 1
        candidate = center + rng.normal(0.0, scales)
        candidate[2] = np.clip(candidate[2], 1.0e-4, 0.999)
        candidate[3] = np.clip(candidate[3], 1.0e-4, 0.999)
        candidate[4] = np.clip(candidate[4], 1.0e-4, 0.999)
        candidate[5] = np.clip(candidate[5], np.log(config.jitter_lower * 1.01), np.log(config.jitter_upper * 0.9))
        candidate[6] = np.clip(candidate[6], -0.95 * timing.period_half_width, 0.95 * timing.period_half_width)
        candidate[7] = np.clip(candidate[7], -0.95 * timing.mid_epoch_half_width, 0.95 * timing.mid_epoch_half_width)
        if np.isfinite(reference_log_probability(candidate, data, config, timing)):
            vectors.append(candidate)

    assert context.cadence_count == 18041
    assert context.event_count == 373
    assert len(vectors) == 4
    for vector in vectors:
        _assert_reference_equivalence(vector, data, config, timing, context, abs_tol=1.0e-9)


def _synthetic_valid_vectors(data, config, timing, context):
    init = build_initialization(
        data,
        config,
        timing,
        np.random.default_rng(101),
        "local_broad",
        101,
        context=context,
    )
    center = deterministic_center_vector(data, config, timing)
    return [center, *list(init.walkers[:4])]


def _assert_reference_equivalence(vector, data, config, timing, context, *, abs_tol=1.0e-12):
    reference_model = transit_model_for_vector(vector, data, config, timing)
    optimized_model = transit_model_for_sample(vector_to_physical(vector, timing), context)
    assert optimized_model == pytest.approx(reference_model, abs=abs_tol)
    assert log_likelihood_with_context(vector, context) == pytest.approx(
        reference_log_likelihood(vector, data, config, timing),
        abs=abs_tol,
    )
    assert log_probability_with_context(vector, context) == pytest.approx(
        reference_log_probability(vector, data, config, timing),
        abs=abs_tol,
    )


def _assert_context_arrays_are_read_only(context):
    assert context.time.flags.writeable is False
    assert context.exposure_days.flags.writeable is False
    assert context.event_number.flags.writeable is False
    assert context.baseline_mean.flags.writeable is False
    for event in context.events:
        assert event.time.flags.writeable is False
        assert event.flux.flags.writeable is False
        assert event.flux_uncertainty_squared.flags.writeable is False
        assert event.local_coordinate.flags.writeable is False
