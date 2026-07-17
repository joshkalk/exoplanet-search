import numpy as np
import pytest

from exoplanet_search.phase1c import synthetic_dataset
from exoplanet_search.phase1c_likelihood import Phase1CLikelihoodContext, log_probability_with_context
from exoplanet_search.phase1c_sampler import build_initialization, deterministic_center_vector
from exoplanet_search.phase1c_types import PARAMETER_ORDER, Phase1CConfig


def test_prior_informed_initialization_is_reproducible_and_coherent(tmp_path):
    config = Phase1CConfig(
        output_dir=tmp_path / "synthetic",
        n_ensembles=4,
        n_walkers=16,
        prior_informed_pool_size=96,
        prior_informed_elite_size=8,
        prior_informed_min_finite_candidates=4,
        prior_informed_cloud_logp_drop=40.0,
    )
    data, timing, _ = synthetic_dataset(config)
    context = Phase1CLikelihoodContext.from_data(data, config, timing)

    first = build_initialization(data, config, timing, np.random.default_rng(123), "prior_informed", 123, context=context)
    second = build_initialization(
        data,
        config,
        timing,
        np.random.default_rng(123),
        "prior_informed",
        123,
        context=context,
    )

    assert np.array_equal(first.walkers, second.walkers)
    assert first.summary == second.summary
    assert first.walkers.shape == (16, len(PARAMETER_ORDER))
    assert first.summary["rank"] == len(PARAMETER_ORDER)

    metadata = first.summary["prior_informed_remote_anchor"]
    center = deterministic_center_vector(data, config, timing)
    anchor = np.asarray([metadata["selected_anchor_vector"][name] for name in PARAMETER_ORDER], dtype=float)
    cloud_logp = np.asarray([log_probability_with_context(row, context) for row in first.walkers])
    distance = np.linalg.norm((anchor - center) / np.asarray(config.local_broad_scales))

    assert metadata["algorithm"] == "broad_pool_elite_remote_anchor_v1"
    assert metadata["pool_size"] == config.prior_informed_pool_size
    assert metadata["finite_candidate_count"] >= config.prior_informed_min_finite_candidates
    assert metadata["elite_size_used"] == config.prior_informed_elite_size
    assert metadata["fallback_used"] is False
    assert metadata["selected_anchor_pool_index"] is not None
    assert metadata["selected_anchor_rank_by_log_posterior"] <= config.prior_informed_elite_size
    assert distance > 5.0
    assert np.all(np.isfinite(cloud_logp))
    assert np.min(cloud_logp) >= metadata["selected_anchor_log_posterior"] - config.prior_informed_cloud_logp_drop
    assert metadata["cloud_log_posterior_range"]["min"] == pytest.approx(float(np.min(cloud_logp)))
    assert np.median(np.linalg.norm((first.walkers - anchor) / np.asarray(config.prior_informed_cloud_scales), axis=1)) < 4.0


def test_prior_informed_initialization_records_explicit_fallback(tmp_path):
    config = Phase1CConfig(
        output_dir=tmp_path / "synthetic",
        n_ensembles=4,
        n_walkers=16,
        prior_informed_pool_size=8,
        prior_informed_elite_size=4,
        prior_informed_min_finite_candidates=1000,
        prior_informed_cloud_logp_drop=40.0,
    )
    data, timing, _ = synthetic_dataset(config)
    context = Phase1CLikelihoodContext.from_data(data, config, timing)

    initialization = build_initialization(
        data,
        config,
        timing,
        np.random.default_rng(321),
        "prior_informed",
        321,
        context=context,
    )

    metadata = initialization.summary["prior_informed_remote_anchor"]
    assert metadata["fallback_used"] is True
    assert metadata["fallback_reason"] == "insufficient_finite_broad_candidates"
    assert metadata["finite_candidate_count"] < config.prior_informed_min_finite_candidates
    assert initialization.summary["rank"] == len(PARAMETER_ORDER)
    assert np.all(np.isfinite([log_probability_with_context(row, context) for row in initialization.walkers]))
