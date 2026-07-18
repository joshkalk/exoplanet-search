import json
from dataclasses import replace

import numpy as np
import pytest

from exoplanet_search import phase1c_sampler
from exoplanet_search.phase1c import _stored_phase1c_config
from exoplanet_search.phase1c import synthetic_dataset
from exoplanet_search.phase1c_likelihood import Phase1CLikelihoodContext, log_probability_with_context
from exoplanet_search.phase1c_sampler import (
    _adaptive_prior_informed_candidate_pool,
    build_initialization,
    checkpoint_metadata,
    deterministic_center_vector,
    validate_initialization_cloud,
)
from exoplanet_search.phase1c_types import PARAMETER_ORDER, Phase1CConfig
from exoplanet_search.phase1d_draws import load_phase1c_config


def test_prior_informed_initialization_is_reproducible_and_coherent(tmp_path):
    config = Phase1CConfig(
        output_dir=tmp_path / "synthetic",
        n_ensembles=4,
        n_walkers=16,
        prior_informed_pool_size=512,
        prior_informed_max_pool_size=512,
        prior_informed_elite_size=8,
        prior_informed_min_finite_candidates=4,
        maximum_initial_logp_deficit=30.0,
        prior_informed_max_logp_deficit=30.0,
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
    cloud_deficit = metadata["deterministic_center_log_posterior"] - cloud_logp
    distance = np.linalg.norm((anchor - center) / np.asarray(config.local_broad_scales))

    assert metadata["algorithm"] == "broad_pool_adaptive_expansion_v2"
    assert metadata["configured_initial_pool_size"] == config.prior_informed_pool_size
    assert metadata["configured_maximum_pool_size"] == config.prior_informed_max_pool_size
    assert metadata["configured_growth_factor"] == config.prior_informed_pool_growth_factor
    assert metadata["pool_size"] == metadata["actual_cumulative_candidates_evaluated"]
    assert metadata["expansion_count"] == 0
    assert metadata["stopping_reason"] == "eligible_requirement_met"
    assert metadata["stage_history"][-1]["stopping_requirement_met"] is True
    assert metadata["finite_candidate_count"] >= config.prior_informed_min_finite_candidates
    assert metadata["posterior_eligible_candidate_count"] >= config.prior_informed_min_finite_candidates
    assert metadata["eligible_fraction"] == pytest.approx(
        metadata["posterior_eligible_candidate_count"] / metadata["pool_size"]
    )
    assert metadata["elite_size_used"] == min(
        config.prior_informed_elite_size,
        metadata["posterior_eligible_candidate_count"],
    )
    assert metadata["fallback_used"] is False
    assert isinstance(metadata["selected_anchor_pool_index"], int)
    assert metadata["selected_anchor_rank_by_log_posterior"] <= config.prior_informed_elite_size
    assert metadata["selected_anchor_log_posterior_deficit"] <= config.prior_informed_max_logp_deficit
    assert metadata["authoritative_maximum_initial_logp_deficit"] == config.maximum_initial_logp_deficit
    assert distance > 2.0
    assert not np.array_equal(anchor, center)
    assert np.all(np.isfinite(cloud_logp))
    assert np.max(cloud_deficit) <= config.prior_informed_max_logp_deficit
    assert np.max(cloud_deficit) <= config.maximum_initial_logp_deficit
    assert first.summary["maximum_initial_logp_deficit"] == 30.0
    assert first.summary["initialization_validation"]["passed"] is True
    assert len(first.summary["walker_initialization_rows"]) == config.n_walkers
    assert metadata["cloud_log_posterior_range"]["min"] == pytest.approx(float(np.min(cloud_logp)))
    assert metadata["cloud_minimum_log_posterior_deficit"] == pytest.approx(float(np.max(cloud_deficit)))
    assert set(metadata["rejection_counts"]) == {"nonfinite", "below_floor"}
    assert np.median(np.linalg.norm((first.walkers - anchor) / np.asarray(config.prior_informed_cloud_scales), axis=1)) < 4.0


def test_prior_informed_initialization_raises_without_eligible_candidates(tmp_path):
    config = Phase1CConfig(
        output_dir=tmp_path / "synthetic",
        n_ensembles=4,
        n_walkers=16,
        prior_informed_pool_size=8,
        prior_informed_max_pool_size=8,
        prior_informed_elite_size=4,
        prior_informed_min_finite_candidates=4,
        maximum_initial_logp_deficit=1.0e-12,
        prior_informed_max_logp_deficit=1.0e-12,
    )
    data, timing, _ = synthetic_dataset(config)
    context = Phase1CLikelihoodContext.from_data(data, config, timing)

    with pytest.raises(RuntimeError, match="Insufficient posterior-eligible"):
        build_initialization(
            data,
            config,
            timing,
            np.random.default_rng(321),
            "prior_informed",
            321,
            context=context,
        )


def test_prior_informed_initialization_raises_when_eligible_count_is_too_low(tmp_path):
    config = Phase1CConfig(
        output_dir=tmp_path / "synthetic",
        n_ensembles=4,
        n_walkers=16,
        prior_informed_pool_size=64,
        prior_informed_max_pool_size=64,
        prior_informed_elite_size=4,
        prior_informed_min_finite_candidates=1000,
        maximum_initial_logp_deficit=30.0,
        prior_informed_max_logp_deficit=30.0,
    )
    data, timing, _ = synthetic_dataset(config)
    context = Phase1CLikelihoodContext.from_data(data, config, timing)

    with pytest.raises(RuntimeError, match="Insufficient posterior-eligible"):
        build_initialization(
            data,
            config,
            timing,
            np.random.default_rng(321),
            "prior_informed",
            321,
            context=context,
        )


def test_failed_pilot_seed_expands_to_2048_and_keeps_thresholds(tmp_path):
    config = Phase1CConfig(
        output_dir=tmp_path / "synthetic",
        random_seed=20260715,
        n_ensembles=4,
        n_walkers=32,
        prior_informed_pool_size=1024,
        prior_informed_max_pool_size=8192,
        prior_informed_pool_growth_factor=2,
        prior_informed_elite_size=16,
        prior_informed_min_finite_candidates=8,
        maximum_initial_logp_deficit=30.0,
        prior_informed_max_logp_deficit=30.0,
    )
    data, timing, _ = synthetic_dataset(config)
    context = Phase1CLikelihoodContext.from_data(data, config, timing)
    seed = config.random_seed + 3000

    initialization = build_initialization(
        data,
        config,
        timing,
        np.random.default_rng(seed),
        "prior_informed",
        seed,
        context=context,
    )

    metadata = initialization.summary["prior_informed_remote_anchor"]
    cloud_logp = np.asarray([log_probability_with_context(row, context) for row in initialization.walkers])
    cloud_deficit = metadata["deterministic_center_log_posterior"] - cloud_logp

    assert metadata["stage_history"][0]["cumulative_pool_size"] == 1024
    assert metadata["stage_history"][0]["cumulative_eligible_count"] == 3
    assert metadata["stage_history"][0]["stopping_requirement_met"] is False
    assert metadata["actual_cumulative_candidates_evaluated"] == 2048
    assert metadata["expansion_count"] == 1
    assert metadata["stage_history"][1]["cumulative_pool_size"] == 2048
    assert metadata["posterior_eligible_candidate_count"] >= 8
    assert metadata["maximum_log_posterior_deficit"] == 30.0
    assert metadata["required_eligible_candidate_count"] == 8
    assert metadata["fallback_used"] is False
    assert metadata["fallback_reason"] is None
    assert 0 <= metadata["selected_anchor_pool_index"] < 2048
    assert np.max(cloud_deficit) <= 30.0
    assert metadata["full_rank"] is True
    assert initialization.summary["rank"] == len(PARAMETER_ORDER)


def test_prior_informed_initial_stage_success_does_not_expand(tmp_path):
    config = Phase1CConfig(
        output_dir=tmp_path / "synthetic",
        random_seed=20260715,
        n_ensembles=4,
        n_walkers=32,
        prior_informed_pool_size=1024,
        prior_informed_max_pool_size=8192,
        prior_informed_min_finite_candidates=8,
    )
    data, timing, _ = synthetic_dataset(config)
    context = Phase1CLikelihoodContext.from_data(data, config, timing)
    seed = 20264724

    initialization = build_initialization(
        data,
        config,
        timing,
        np.random.default_rng(seed),
        "prior_informed",
        seed,
        context=context,
    )

    metadata = initialization.summary["prior_informed_remote_anchor"]
    assert metadata["actual_cumulative_candidates_evaluated"] == 1024
    assert metadata["expansion_count"] == 0
    assert len(metadata["stage_history"]) == 1
    assert metadata["stage_history"][0]["cumulative_eligible_count"] >= 8
    assert metadata["stopping_reason"] == "eligible_requirement_met"


def test_adaptive_pool_preserves_nested_candidate_sequence(monkeypatch):
    config = Phase1CConfig(
        prior_informed_pool_size=1024,
        prior_informed_max_pool_size=2048,
        prior_informed_min_finite_candidates=8,
    )
    state = _patch_indexed_candidate_pool(monkeypatch, eligible_indices=range(1500, 1508))

    result = _adaptive_prior_informed_candidate_pool(
        object(),
        config,
        object(),
        np.random.default_rng(10),
        object(),
        center_logp=0.0,
    )

    expected_first_pool = np.asarray([_indexed_vector(index) for index in range(1024)])
    assert np.array_equal(result.pool_vectors[:1024], expected_first_pool)
    assert np.array_equal(result.pool_log_prob[:1024], np.full(1024, -100.0))
    assert np.array_equal(result.pool_vectors[:, 0], np.arange(2048, dtype=float))
    assert np.array_equal(result.eligible_indices, np.arange(1500, 1508))
    assert state["draws"] == 2048
    assert state["posterior_calls"] == 2048
    assert result.stage_history[0]["cumulative_eligible_count"] == 0
    assert result.stage_history[1]["cumulative_eligible_count"] == 8


def test_adaptive_pool_records_multiple_expansion_stages(monkeypatch):
    config = Phase1CConfig(
        prior_informed_pool_size=1024,
        prior_informed_max_pool_size=4096,
        prior_informed_pool_growth_factor=2,
        prior_informed_min_finite_candidates=8,
    )
    _patch_indexed_candidate_pool(monkeypatch, eligible_indices=range(3000, 3008))

    result = _adaptive_prior_informed_candidate_pool(
        object(),
        config,
        object(),
        np.random.default_rng(11),
        object(),
        center_logp=0.0,
    )

    assert [row["cumulative_pool_size"] for row in result.stage_history] == [1024, 2048, 4096]
    assert [row["candidates_added"] for row in result.stage_history] == [1024, 1024, 2048]
    assert [row["cumulative_eligible_count"] for row in result.stage_history] == [0, 0, 8]
    assert result.expansion_count == 2
    assert result.stopping_reason == "eligible_requirement_met"


def test_adaptive_pool_fails_closed_at_configured_cap(monkeypatch):
    config = Phase1CConfig(
        prior_informed_pool_size=1024,
        prior_informed_max_pool_size=8192,
        prior_informed_pool_growth_factor=2,
        prior_informed_min_finite_candidates=8,
    )
    state = _patch_indexed_candidate_pool(monkeypatch, eligible_indices=())

    with pytest.raises(RuntimeError, match="maximum pool size=8192") as excinfo:
        _adaptive_prior_informed_candidate_pool(
            object(),
            config,
            object(),
            np.random.default_rng(12),
            object(),
            center_logp=0.0,
        )

    message = str(excinfo.value)
    assert "cumulative candidates evaluated=8192" in message
    assert "finite candidate count=8192" in message
    assert "eligible candidate count=0" in message
    assert "required eligible count=8" in message
    assert "center-relative log-posterior floor=-30.0" in message
    assert "expansion stage history=" in message
    assert state["draws"] == 8192
    assert state["posterior_calls"] == 8192


def test_phase1c_config_validation_and_legacy_loading(tmp_path):
    with pytest.raises(ValueError, match="max_pool_size"):
        Phase1CConfig(prior_informed_pool_size=1024, prior_informed_max_pool_size=512)
    with pytest.raises(ValueError, match="growth_factor"):
        Phase1CConfig(prior_informed_pool_growth_factor=1)
    with pytest.raises(ValueError, match="growth_factor"):
        Phase1CConfig(prior_informed_pool_growth_factor=2.5)
    with pytest.raises(ValueError, match="min_finite_candidates"):
        Phase1CConfig(prior_informed_min_finite_candidates=0)
    with pytest.raises(ValueError, match="elite_size"):
        Phase1CConfig(prior_informed_elite_size=0)
    with pytest.raises(ValueError, match="maximum_initial_logp_deficit"):
        Phase1CConfig(maximum_initial_logp_deficit=np.inf, prior_informed_max_logp_deficit=np.inf)
    with pytest.raises(ValueError, match="exactly match"):
        Phase1CConfig(maximum_initial_logp_deficit=30.0, prior_informed_max_logp_deficit=29.0)

    run_dir = tmp_path / "old_config"
    run_dir.mkdir()
    payload = Phase1CConfig(output_dir=run_dir, run_id="legacy").to_dict()
    payload.pop("parameter_order")
    payload.pop("notes")
    payload.pop("prior_informed_max_pool_size")
    payload.pop("prior_informed_pool_growth_factor")
    payload["prior_informed_cloud_logp_drop"] = payload.pop("prior_informed_max_logp_deficit")
    (run_dir / "phase1c_configuration.json").write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_phase1c_config(run_dir)
    stored = _stored_phase1c_config(Phase1CConfig(output_dir=run_dir, run_id="legacy"))
    assert loaded.prior_informed_max_pool_size == 8192
    assert loaded.prior_informed_pool_growth_factor == 2
    assert loaded.prior_informed_max_logp_deficit == 30.0
    assert stored.prior_informed_max_pool_size == 8192
    assert stored.prior_informed_pool_growth_factor == 2
    assert loaded.sampler_move_strategy == "stretch_v1"
    assert stored.sampler_move_strategy == "stretch_v1"

    serialized = loaded.to_dict()
    assert serialized["prior_informed_max_pool_size"] == 8192
    assert serialized["prior_informed_pool_growth_factor"] == 2
    data, _, _ = synthetic_dataset(loaded)
    identity = checkpoint_metadata(data, loaded, mode="synthetic")["immutable_scientific_identity"]
    identity_config = identity["priors_and_transforms"]["configuration"]
    assert identity_config["prior_informed_max_pool_size"] == 8192
    assert identity_config["prior_informed_pool_growth_factor"] == 2
    assert identity_config["sampler_move_strategy"] == "stretch_v1"

    bad_dir = tmp_path / "bad_config"
    bad_dir.mkdir()
    bad_payload = Phase1CConfig(output_dir=bad_dir, run_id="bad").to_dict()
    bad_payload["sampler_move_strategy"] = "not_real"
    (bad_dir / "phase1c_configuration.json").write_text(json.dumps(bad_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="sampler_move_strategy"):
        _stored_phase1c_config(Phase1CConfig(output_dir=bad_dir, run_id="bad"))


def test_local_initialization_strategies_keep_existing_behavior(tmp_path):
    config = Phase1CConfig(output_dir=tmp_path / "synthetic", n_ensembles=4, n_walkers=24)
    data, timing, _ = synthetic_dataset(config)
    center = deterministic_center_vector(data, config, timing)

    for index, strategy in enumerate(("local_tight", "local_moderate", "local_broad")):
        initialization = build_initialization(
            data,
            config,
            timing,
            np.random.default_rng(100 + index),
            strategy,
            100 + index,
        )
        context = Phase1CLikelihoodContext.from_data(data, config, timing)
        logp = np.asarray([log_probability_with_context(row, context) for row in initialization.walkers])
        deficits = initialization.summary["deterministic_center_log_posterior"] - logp
        assert initialization.walkers.shape == (24, len(PARAMETER_ORDER))
        assert initialization.summary["strategy"] == strategy
        assert "prior_informed_remote_anchor" not in initialization.summary
        assert initialization.summary["rank"] == len(PARAMETER_ORDER)
        assert np.max(deficits) <= config.maximum_initial_logp_deficit
        assert set(initialization.summary["rejection_counts"]) == {"nonfinite", "below_floor"}
        assert initialization.summary["initialization_validation"]["passed"] is True
        assert abs(np.median(initialization.walkers[:, 6] - center[6])) < 3.0e-4
        assert abs(np.median(initialization.walkers[:, 7] - center[7])) < 7.5e-3


def test_initialization_uses_no_injected_truth_record(tmp_path):
    config = Phase1CConfig(output_dir=tmp_path / "synthetic", n_ensembles=4, n_walkers=16)
    data, timing, _ = synthetic_dataset(config)
    altered_manifest = json.loads(json.dumps(data.input_manifest))
    altered_manifest.setdefault("synthetic_dataset_identity", {})["injected_parameters"] = {"rp_over_rstar": 999.0}
    altered_data = replace(data, input_manifest=altered_manifest)

    first = build_initialization(data, config, timing, np.random.default_rng(123), "local_tight", 123)
    second = build_initialization(altered_data, config, timing, np.random.default_rng(123), "local_tight", 123)

    assert np.array_equal(first.walkers, second.walkers)
    assert first.summary == second.summary


def test_local_initialization_distinguishes_rejection_reasons(monkeypatch, tmp_path):
    config = Phase1CConfig(output_dir=tmp_path / "synthetic", n_ensembles=1, n_walkers=16)
    data, timing, _ = synthetic_dataset(config)
    values = iter([0.0, -np.inf, -100.0, *([0.0] * 16)])

    def fake_log_probability_with_context(vector, context):
        del vector, context
        return next(values)

    monkeypatch.setattr(phase1c_sampler, "log_probability_with_context", fake_log_probability_with_context)
    initialization = build_initialization(data, config, timing, np.random.default_rng(1), "local_tight", 1)

    assert initialization.summary["rejection_counts"] == {"nonfinite": 1, "below_floor": 1}


def test_local_initialization_fails_without_eligible_cloud(monkeypatch, tmp_path):
    config = Phase1CConfig(output_dir=tmp_path / "synthetic", n_ensembles=1, n_walkers=4)
    data, timing, _ = synthetic_dataset(config)
    calls = {"count": 0}

    def fake_log_probability_with_context(vector, context):
        del vector, context
        calls["count"] += 1
        return 0.0 if calls["count"] == 1 else -100.0

    monkeypatch.setattr(phase1c_sampler, "log_probability_with_context", fake_log_probability_with_context)
    with pytest.raises(RuntimeError, match="posterior-eligible"):
        build_initialization(data, config, timing, np.random.default_rng(1), "local_tight", 1)


def test_initialization_validation_rejects_duplicate_and_near_duplicate_clouds(tmp_path):
    config = Phase1CConfig(output_dir=tmp_path / "synthetic", n_ensembles=1, n_walkers=16)
    data, timing, _ = synthetic_dataset(config)
    context = Phase1CLikelihoodContext.from_data(data, config, timing)
    center = deterministic_center_vector(data, config, timing)
    center_logp = float(log_probability_with_context(center, context))
    scales = np.asarray(config.local_tight_scales, dtype=float)
    rng = np.random.default_rng(42)
    cloud = center + rng.normal(0.0, scales, size=(config.n_walkers, len(PARAMETER_ORDER)))
    cloud = np.asarray([phase1c_sampler._clip_local_candidate(row, config, timing) for row in cloud])
    logp = np.asarray([log_probability_with_context(row, context) for row in cloud])

    duplicate = cloud.copy()
    duplicate[1] = duplicate[0]
    with pytest.raises(RuntimeError, match="initialization validation failed"):
        validate_initialization_cloud(
            duplicate,
            logp,
            data,
            config,
            timing,
            center,
            center_logp,
            strategy="local_tight",
            scales=scales,
            rejection_counts={"nonfinite": 0, "below_floor": 0},
        )

    near = cloud.copy()
    near[1] = near[0] + scales * 1.0e-10
    with pytest.raises(RuntimeError, match="initialization validation failed"):
        validate_initialization_cloud(
            near,
            logp,
            data,
            config,
            timing,
            center,
            center_logp,
            strategy="local_tight",
            scales=scales,
            rejection_counts={"nonfinite": 0, "below_floor": 0},
        )


def _patch_indexed_candidate_pool(monkeypatch, *, eligible_indices):
    eligible = set(eligible_indices)
    state = {"draws": 0, "posterior_calls": 0}

    def fake_broad_prior_candidate(data, config, timing, rng):
        del data, config, timing, rng
        index = state["draws"]
        state["draws"] += 1
        return _indexed_vector(index)

    def fake_log_probability_with_context(vector, context):
        del context
        state["posterior_calls"] += 1
        return 0.0 if int(vector[0]) in eligible else -100.0

    monkeypatch.setattr(phase1c_sampler, "broad_prior_candidate", fake_broad_prior_candidate)
    monkeypatch.setattr(phase1c_sampler, "log_probability_with_context", fake_log_probability_with_context)
    return state


def _indexed_vector(index):
    vector = np.zeros(len(PARAMETER_ORDER), dtype=float)
    vector[0] = float(index)
    return vector
