import hashlib
import json
import os
from pathlib import Path

import emcee
import h5py
import numpy as np
import pytest

from exoplanet_search.cli import build_parser
from exoplanet_search.phase1c import _stored_phase1c_config, run_phase1c_synthetic_validation, synthetic_dataset
from exoplanet_search.phase1c_sampler import (
    THREAD_LIMIT_ENV_VARS,
    canonical_sampler_move_configuration,
    checkpoint_metadata,
    execution_provenance,
    immutable_checkpoint_identity,
    _run_ensemble_chunk_worker,
    run_ensembles,
)
from exoplanet_search.phase1c_types import Phase1CConfig


def test_config_and_cli_accept_and_reject_ensemble_processes(tmp_path):
    assert Phase1CConfig().ensemble_processes == 1
    assert Phase1CConfig(n_ensembles=4, ensemble_processes=1).ensemble_processes == 1
    assert Phase1CConfig(n_ensembles=4, ensemble_processes=2).ensemble_processes == 2
    assert Phase1CConfig(n_ensembles=4, ensemble_processes=4).ensemble_processes == 4

    args = build_parser().parse_args(["--phase1c-synthetic-validation", "--phase1c-ensemble-processes", "4"])
    assert args.phase1c_ensemble_processes == 4
    args = build_parser().parse_args(
        ["--phase1c-synthetic-validation", "--phase1c-sampler-move-strategy", "de_snooker_v1"]
    )
    assert args.phase1c_sampler_move_strategy == "de_snooker_v1"
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--phase1c-synthetic-validation", "--phase1c-sampler-move-strategy", "bad"])

    for value in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            Phase1CConfig(n_ensembles=4, ensemble_processes=value)
    with pytest.raises(ValueError, match="cannot exceed n_ensembles"):
        Phase1CConfig(n_ensembles=2, ensemble_processes=4)

    for value in ("0", "-1", "true", "1.5"):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["--phase1c-synthetic-validation", "--phase1c-ensemble-processes", value])

    payload = Phase1CConfig(output_dir=tmp_path / "old").to_dict()
    payload.pop("ensemble_processes")
    (tmp_path / "old").mkdir()
    (tmp_path / "old" / "phase1c_configuration.json").write_text(json.dumps(payload), encoding="utf-8")
    stored = _stored_phase1c_config(Phase1CConfig(output_dir=tmp_path / "old"))
    assert stored.ensemble_processes == 1


def test_ensemble_processes_is_not_part_of_scientific_checkpoint_identity(tmp_path):
    config_one = _parallel_test_config(tmp_path / "one", ensemble_processes=1)
    config_two = _parallel_test_config(tmp_path / "two", ensemble_processes=2)
    data, _, _ = synthetic_dataset(config_one)

    identity_one = immutable_checkpoint_identity(data, config_one, mode="synthetic")
    identity_two = immutable_checkpoint_identity(data, config_two, mode="synthetic")
    assert identity_one == identity_two
    assert _identity_sha(identity_one) == _identity_sha(identity_two)
    assert "ensemble_processes" not in identity_one["priors_and_transforms"]["configuration"]

    scientific_change = _parallel_test_config(tmp_path / "changed", supersample_factor=7)
    changed_identity = immutable_checkpoint_identity(data, scientific_change, mode="synthetic")
    assert _identity_sha(identity_one) != _identity_sha(changed_identity)


def test_exact_sequential_parallel_equivalence_for_bounded_synthetic_run(tmp_path):
    seq = _run_synthetic_ensembles(tmp_path, "seq", ensemble_processes=1, steps=4)
    par = _run_synthetic_ensembles(tmp_path, "par", ensemble_processes=2, steps=4)

    assert [result.ensemble_index for result in par.results] == [0, 1]
    for seq_result, par_result in zip(seq.results, par.results, strict=True):
        _assert_result_equivalent(seq_result, par_result)
        _assert_backend_equivalent(seq_result.backend_path, par_result.backend_path)

    assert seq.callback_iterations == [[2, 2], [4, 4]]
    assert par.callback_iterations == [[2, 2], [4, 4]]


def test_two_process_parallel_run_is_exactly_reproducible(tmp_path):
    first = _run_synthetic_ensembles(tmp_path, "first", ensemble_processes=2, steps=4)
    second = _run_synthetic_ensembles(tmp_path, "second", ensemble_processes=2, steps=4)

    for left, right in zip(first.results, second.results, strict=True):
        _assert_result_equivalent(left, right)
        _assert_backend_equivalent(left.backend_path, right.backend_path)


def test_cross_mode_resume_matches_uninterrupted_chain_exactly(tmp_path):
    full = _run_synthetic_ensembles(tmp_path, "full", ensemble_processes=1, steps=6)

    partial_config = _parallel_test_config(tmp_path / "resume", ensemble_processes=1, chunk_steps=2)
    data, timing, _ = synthetic_dataset(partial_config)
    run_ensembles(data, partial_config, timing, steps=2, mode="synthetic", resume=False)

    resume_config = _parallel_test_config(tmp_path / "resume", ensemble_processes=2, chunk_steps=2)
    resumed = run_ensembles(data, resume_config, timing, steps=6, mode="synthetic", resume=True)

    for full_result, resumed_result in zip(full.results, resumed, strict=True):
        _assert_backend_equivalent(full_result.backend_path, resumed_result.backend_path)


def test_process_parallel_resume_from_uneven_checkpoints_to_current_target(tmp_path):
    reference = _run_synthetic_ensembles(tmp_path, "reference4", ensemble_processes=1, steps=4)
    config, data, timing = _build_uneven_checkpoint_state(tmp_path, "uneven4")
    complete_before = _backend_snapshot(config.output_dir / "ensemble_00.h5")
    callbacks = []

    resume_config = _parallel_test_config(tmp_path / "uneven4", ensemble_processes=2, chunk_steps=2)
    results = run_ensembles(
        data,
        resume_config,
        timing,
        steps=4,
        mode="synthetic",
        resume=True,
        chunk_callback=lambda chunk_results, _elapsed, _profiler: callbacks.append(chunk_results),
    )

    assert [result.ensemble_index for result in results] == [0, 1]
    assert [result.iterations for result in results] == [4, 4]
    assert len(callbacks) == 1
    assert [result.ensemble_index for result in callbacks[0]] == [0, 1]
    assert [result.iterations for result in callbacks[0]] == [4, 4]
    assert results[0].process_ids == ()
    assert results[0].profiler_summary["posterior_calls"] == 0
    _assert_backend_matches_snapshot(config.output_dir / "ensemble_00.h5", complete_before)
    for reference_result, resumed_result in zip(reference.results, results, strict=True):
        _assert_backend_equivalent(reference_result.backend_path, resumed_result.backend_path)


def test_process_parallel_resume_from_uneven_checkpoints_to_higher_target(tmp_path):
    reference = _run_synthetic_ensembles(tmp_path, "reference6", ensemble_processes=1, steps=6)
    config, data, timing = _build_uneven_checkpoint_state(tmp_path, "uneven6")
    complete_before = _backend_snapshot(config.output_dir / "ensemble_00.h5")

    resume_config = _parallel_test_config(tmp_path / "uneven6", ensemble_processes=2, chunk_steps=2)
    results = run_ensembles(data, resume_config, timing, steps=6, mode="synthetic", resume=True)

    assert [result.iterations for result in results] == [6, 6]
    assert results[0].process_ids
    assert results[0].profiler_summary["posterior_calls"] > 0
    assert not np.array_equal(
        complete_before["chain"],
        emcee.backends.HDFBackend(str(config.output_dir / "ensemble_00.h5"), read_only=True).get_chain(),
    )
    for reference_result, resumed_result in zip(reference.results, results, strict=True):
        _assert_backend_equivalent(reference_result.backend_path, resumed_result.backend_path)


def test_process_parallel_noop_resume_hydrates_without_sampling_or_callback(tmp_path):
    complete = _run_synthetic_ensembles(tmp_path, "complete", ensemble_processes=1, steps=4)
    snapshots = [_backend_snapshot(result.backend_path) for result in complete.results]
    callbacks = []

    resume_config = _parallel_test_config(tmp_path / "complete", ensemble_processes=2, chunk_steps=2)
    data, timing, _ = synthetic_dataset(resume_config)
    results = run_ensembles(
        data,
        resume_config,
        timing,
        steps=4,
        mode="synthetic",
        resume=True,
        chunk_callback=lambda chunk_results, _elapsed, _profiler: callbacks.append(chunk_results),
    )

    assert [result.ensemble_index for result in results] == [0, 1]
    assert [result.iterations for result in results] == [4, 4]
    assert callbacks == []
    for result, snapshot in zip(results, snapshots, strict=True):
        assert result.process_ids == ()
        assert result.runtime_seconds == 0.0
        assert result.profiler_summary["posterior_calls"] == 0
        _assert_backend_matches_snapshot(result.backend_path, snapshot)


def test_parallel_chunk_callbacks_are_global_barriers_and_ordered(tmp_path):
    run = _run_synthetic_ensembles(tmp_path, "callbacks", ensemble_processes=2, steps=6)

    assert run.callback_indices == [[0, 1], [0, 1], [0, 1]]
    assert run.callback_iterations == [[2, 2], [4, 4], [6, 6]]
    for callback_index, iterations in enumerate(run.callback_iterations, start=1):
        assert iterations == [2 * callback_index, 2 * callback_index]


def test_parallel_worker_failure_can_resume_preserved_uneven_checkpoints(tmp_path):
    reference = _run_synthetic_ensembles(tmp_path, "failure_reference4", ensemble_processes=1, steps=4)
    config, data, timing = _build_uneven_checkpoint_state(tmp_path, "failure")
    complete_before = _backend_snapshot(config.output_dir / "ensemble_00.h5")
    callbacks = []

    resume_config = _parallel_test_config(tmp_path / "failure", ensemble_processes=2, chunk_steps=2)
    with pytest.raises(RuntimeError) as excinfo:
        run_ensembles(
            data,
            resume_config,
            timing,
            steps=4,
            mode="synthetic",
            resume=True,
            chunk_callback=lambda results, _elapsed, _profiler: callbacks.append(results),
            _failure_injection={"ensemble_index": 1, "message": "controlled failure"},
        )

    message = str(excinfo.value)
    assert "ensemble_index=1" in message
    assert "seed=" in message
    assert "strategy=" in message
    assert "backend_path=" in message
    assert "requested_chunk=2" in message
    assert callbacks == []
    _assert_backend_matches_snapshot(config.output_dir / "ensemble_00.h5", complete_before)

    recovered = run_ensembles(data, resume_config, timing, steps=4, mode="synthetic", resume=True)
    for reference_result, recovered_result in zip(reference.results, recovered, strict=True):
        _assert_backend_equivalent(reference_result.backend_path, recovered_result.backend_path)


def test_parallel_provenance_records_spawn_processes_and_thread_policy(tmp_path):
    config = _parallel_test_config(
        tmp_path / "phase1c",
        run_id="parallel",
        ensemble_processes=2,
        synthetic_steps=4,
        chunk_steps=2,
    )
    result = run_phase1c_synthetic_validation(config)
    run_dir = Path(result["run_directory"])

    runtime = json.loads((run_dir / "sampler_runtime.json").read_text(encoding="utf-8"))
    provenance = json.loads((run_dir / "provenance_manifest.json").read_text(encoding="utf-8"))
    saved_config = json.loads((run_dir / "phase1c_configuration.json").read_text(encoding="utf-8"))
    history = json.loads((run_dir / "invocation_history.json").read_text(encoding="utf-8"))["invocations"]
    expected_move = canonical_sampler_move_configuration("stretch_v1")

    execution = runtime["execution_parallelism"]
    assert execution["requested_ensemble_processes"] == 2
    assert execution["effective_ensemble_processes"] == 2
    assert execution["execution_mode"] == "process_parallel"
    assert execution["sampler_move_configuration"] == expected_move
    assert runtime["sampler_move_configuration"] == expected_move
    assert saved_config["sampler_move_configuration"] == expected_move
    assert provenance["sampler_move_configuration"] == expected_move
    assert execution["multiprocessing_start_method"] == "spawn"
    assert execution["thread_limit_active_enforcement"] is True
    assert set(execution["thread_limit_environment"].values()) == {"1"}
    assert set(execution["configured_thread_limit_environment"].values()) == {"1"}
    assert execution["worker_process_ids"]
    assert runtime["latest_invocation"]["mutable_execution_controls"]["ensemble_processes"] == 2
    assert history[-1]["mutable_execution_controls"]["ensemble_processes"] == 2
    assert provenance["execution_parallelism"]["requested_ensemble_processes"] == 2
    data, _, _ = synthetic_dataset(config)
    metadata = checkpoint_metadata(data, config, mode="synthetic")
    assert "ensemble_processes" not in metadata["immutable_scientific_identity"]["priors_and_transforms"]["configuration"]


def test_sequential_execution_provenance_reports_inherited_thread_environment(monkeypatch):
    for name in THREAD_LIMIT_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "3")

    execution = execution_provenance(Phase1CConfig(n_ensembles=2, ensemble_processes=1))

    assert execution["execution_mode"] == "sequential"
    assert execution["thread_limit_active_enforcement"] is False
    assert execution["thread_limit_environment"]["OPENBLAS_NUM_THREADS"] == "3"
    assert execution["thread_limit_environment"]["OMP_NUM_THREADS"] is None
    assert set(execution["configured_thread_limit_environment"].values()) == {"1"}


class _SyntheticRun:
    def __init__(self, results, callback_indices, callback_iterations):
        self.results = results
        self.callback_indices = callback_indices
        self.callback_iterations = callback_iterations


def _run_synthetic_ensembles(tmp_path, name: str, *, ensemble_processes: int, steps: int) -> _SyntheticRun:
    config = _parallel_test_config(tmp_path / name, ensemble_processes=ensemble_processes, chunk_steps=2)
    data, timing, _ = synthetic_dataset(config)
    callback_indices = []
    callback_iterations = []

    def on_chunk(results, _elapsed_seconds, _profiler_summary):
        callback_indices.append([result.ensemble_index for result in results])
        callback_iterations.append([result.iterations for result in results])

    results = run_ensembles(
        data,
        config,
        timing,
        steps=steps,
        mode="synthetic",
        resume=False,
        chunk_callback=on_chunk,
    )
    return _SyntheticRun(results, callback_indices, callback_iterations)


def _parallel_test_config(output_dir, **overrides) -> Phase1CConfig:
    values = {
        "output_dir": output_dir,
        "random_seed": 24680,
        "n_ensembles": 2,
        "ensemble_processes": 1,
        "n_walkers": 16,
        "synthetic_steps": 4,
        "chunk_steps": 2,
        "warmup_steps": 1,
        "prior_informed_pool_size": 16,
        "prior_informed_max_pool_size": 16,
        "prior_informed_elite_size": 1,
        "prior_informed_min_finite_candidates": 1,
        "maximum_initial_logp_deficit": 1.0e9,
        "prior_informed_max_logp_deficit": 1.0e9,
    }
    values.update(overrides)
    return Phase1CConfig(**values)


def _build_uneven_checkpoint_state(tmp_path, name: str):
    config = _parallel_test_config(tmp_path / name, ensemble_processes=1, chunk_steps=2)
    data, timing, _ = synthetic_dataset(config)
    run_ensembles(data, config, timing, steps=2, mode="synthetic", resume=False)
    _advance_single_ensemble(data, config, timing, ensemble_index=0, target_steps=4, chunk_steps=2)
    assert emcee.backends.HDFBackend(str(config.output_dir / "ensemble_00.h5"), read_only=True).iteration == 4
    assert emcee.backends.HDFBackend(str(config.output_dir / "ensemble_01.h5"), read_only=True).iteration == 2
    return config, data, timing


def _advance_single_ensemble(
    data,
    config: Phase1CConfig,
    timing,
    *,
    ensemble_index: int,
    target_steps: int,
    chunk_steps: int,
) -> None:
    previous = {name: os.environ.get(name) for name in THREAD_LIMIT_ENV_VARS}
    try:
        _run_ensemble_chunk_worker(
            {
                "data": data,
                "config": config,
                "timing": timing,
                "mode": "synthetic",
                "resume": True,
                "ensemble_index": ensemble_index,
                "target_steps": target_steps,
                "chunk_steps": chunk_steps,
                "failure_injection": None,
            }
        )
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _assert_result_equivalent(left, right) -> None:
    assert left.ensemble_index == right.ensemble_index
    assert left.seed == right.seed
    assert left.strategy == right.strategy
    assert left.iterations == right.iterations
    assert np.array_equal(left.acceptance_fraction, right.acceptance_fraction)
    assert left.initialization_summary == right.initialization_summary
    assert left.profiler_summary["posterior_calls"] == right.profiler_summary["posterior_calls"]
    assert left.profiler_summary["invalid_prior_count"] == right.profiler_summary["invalid_prior_count"]
    assert left.profiler_summary["invalid_likelihood_count"] == right.profiler_summary["invalid_likelihood_count"]


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
            "phase1c_run_id",
            "phase1c_immutable_scientific_identity_sha256",
            "phase1c_immutable_scientific_identity",
            "phase1c_ensemble_seed",
        ):
            assert left_hdf.attrs[key] == right_hdf.attrs[key]


def _backend_snapshot(path: Path) -> dict:
    backend = emcee.backends.HDFBackend(str(path), read_only=True)
    return {
        "iteration": int(backend.iteration),
        "chain": backend.get_chain().copy(),
        "log_prob": backend.get_log_prob().copy(),
        "accepted": np.asarray(backend.accepted).copy(),
        "coords": backend.get_last_sample().coords.copy(),
        "last_log_prob": backend.get_last_sample().log_prob.copy(),
        "random_state": backend.random_state,
    }


def _assert_backend_matches_snapshot(path: Path, snapshot: dict) -> None:
    backend = emcee.backends.HDFBackend(str(path), read_only=True)
    assert int(backend.iteration) == snapshot["iteration"]
    assert np.array_equal(backend.get_chain(), snapshot["chain"])
    assert np.array_equal(backend.get_log_prob(), snapshot["log_prob"])
    assert np.array_equal(backend.accepted, snapshot["accepted"])
    assert np.array_equal(backend.get_last_sample().coords, snapshot["coords"])
    assert np.array_equal(backend.get_last_sample().log_prob, snapshot["last_log_prob"])
    _assert_random_states_equal(backend.random_state, snapshot["random_state"])


def _assert_random_states_equal(left, right) -> None:
    assert left[0] == right[0]
    assert np.array_equal(left[1], right[1])
    assert left[2:] == right[2:]


def _identity_sha(identity: dict) -> str:
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
