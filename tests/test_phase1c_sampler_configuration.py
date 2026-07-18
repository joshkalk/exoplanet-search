import hashlib
import json
from pathlib import Path

import emcee
import h5py
import numpy as np
import pytest

from exoplanet_search import phase1c_sampler
from exoplanet_search.phase1c import synthetic_dataset
from exoplanet_search.phase1c_sampler import (
    _canonical_sampler_move_record,
    canonical_sampler_move_configuration,
    checkpoint_metadata,
    legacy_sampler_move_configuration,
    run_ensembles,
    sampler_move_specification,
    validate_checkpoint_metadata,
)
from exoplanet_search.phase1c_types import Phase1CConfig


def test_exact_sampler_move_records():
    stretch_moves, stretch = sampler_move_specification("stretch_v1")
    de_moves, de_snooker = sampler_move_specification("de_snooker_v1")

    assert [type(move).__name__ for move, _weight in stretch_moves] == ["StretchMove"]
    assert stretch["strategy"] == "stretch_v1"
    assert stretch["schema_version"] == "phase1c_sampler_move_strategy_v1"
    assert stretch["ordered_move_classes"] == ["emcee.moves.StretchMove"]
    assert stretch["weights"] == [1.0]
    assert stretch["weights_sum"] == pytest.approx(1.0)
    assert stretch["weights_sum_is_one"] is True
    assert stretch["moves"][0]["constructor_parameters"] == {"a": 2.0}

    assert [type(move).__name__ for move, _weight in de_moves] == ["DEMove", "DESnookerMove"]
    assert de_snooker["strategy"] == "de_snooker_v1"
    assert de_snooker["ordered_move_classes"] == ["emcee.moves.DEMove", "emcee.moves.DESnookerMove"]
    assert de_snooker["weights"] == [0.8, 0.2]
    assert [weight for _move, weight in de_moves] == [0.8, 0.2]
    assert de_snooker["weights_sum"] == pytest.approx(1.0)
    assert de_snooker["weights_sum_is_one"] is True
    assert de_snooker["emcee_version"] == emcee.__version__


def test_unknown_and_invalid_move_strategies_are_rejected():
    with pytest.raises(ValueError, match="Unknown"):
        sampler_move_specification("mystery")
    with pytest.raises(ValueError, match="sampler_move_strategy"):
        Phase1CConfig(sampler_move_strategy="mystery")
    with pytest.raises(ValueError, match="positive"):
        _canonical_sampler_move_record("bad", [{"class": "x", "weight": 0.0, "constructor_parameters": {}}])
    with pytest.raises(ValueError, match="finite"):
        _canonical_sampler_move_record("bad", [{"class": "x", "weight": np.inf, "constructor_parameters": {}}])
    with pytest.raises(ValueError, match="sum"):
        _canonical_sampler_move_record("bad", [{"class": "x", "weight": 0.5, "constructor_parameters": {}}])


def test_sequential_and_worker_sampler_receive_configured_move_mixture(monkeypatch, tmp_path):
    seen = []

    class RecordingSampler(_FakeSampler):
        def __init__(self, *args, **kwargs):
            seen.append(_move_names_and_weights(kwargs["moves"]))
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(phase1c_sampler.emcee, "EnsembleSampler", RecordingSampler)
    config = _sampler_test_config(tmp_path / "seq", sampler_move_strategy="de_snooker_v1")
    data, timing, _ = synthetic_dataset(config)
    run_ensembles(data, config, timing, steps=0, mode="synthetic", resume=False)

    worker_config = _sampler_test_config(tmp_path / "worker", sampler_move_strategy="de_snooker_v1")
    worker_data, worker_timing, _ = synthetic_dataset(worker_config)
    worker_config.output_dir.mkdir(parents=True)
    phase1c_sampler._run_ensemble_chunk_worker(
        {
            "data": worker_data,
            "config": worker_config,
            "timing": worker_timing,
            "mode": "synthetic",
            "resume": False,
            "ensemble_index": 0,
            "target_steps": 0,
            "chunk_steps": 1,
            "failure_injection": None,
        }
    )

    assert seen == [
        [("DEMove", 0.8), ("DESnookerMove", 0.2)],
        [("DEMove", 0.8), ("DESnookerMove", 0.2)],
    ]


def test_checkpoint_metadata_records_and_rejects_changed_move_strategy(tmp_path):
    config = _sampler_test_config(tmp_path / "run", sampler_move_strategy="stretch_v1")
    data, timing, _ = synthetic_dataset(config)
    result = run_ensembles(data, config, timing, steps=1, mode="synthetic", resume=False)[0]
    metadata = checkpoint_metadata(data, config, mode="synthetic")

    validate_checkpoint_metadata(result.backend_path, metadata, config.random_seed)
    with h5py.File(result.backend_path, "r") as hdf:
        recorded = json.loads(hdf.attrs["phase1c_sampler_move_configuration"])
    assert recorded == canonical_sampler_move_configuration("stretch_v1")

    changed = _sampler_test_config(tmp_path / "run", sampler_move_strategy="de_snooker_v1")
    changed_data, _, _ = synthetic_dataset(changed)
    changed_metadata = checkpoint_metadata(changed_data, changed, mode="synthetic")
    with pytest.raises(ValueError, match="sampler_move_configuration"):
        validate_checkpoint_metadata(result.backend_path, changed_metadata, changed.random_seed)


def test_legacy_checkpoint_metadata_is_explicit_stretch_only(tmp_path):
    config = _sampler_test_config(tmp_path / "legacy", sampler_move_strategy="stretch_v1")
    data, timing, _ = synthetic_dataset(config)
    result = run_ensembles(data, config, timing, steps=1, mode="synthetic", resume=False)[0]
    _rewrite_checkpoint_as_legacy_stretch(result.backend_path)

    validate_checkpoint_metadata(
        result.backend_path,
        checkpoint_metadata(data, config, mode="synthetic"),
        config.random_seed,
    )
    assert legacy_sampler_move_configuration()["strategy"] == "stretch_v1"

    changed = _sampler_test_config(tmp_path / "legacy", sampler_move_strategy="de_snooker_v1")
    changed_data, _, _ = synthetic_dataset(changed)
    with pytest.raises(ValueError, match="sampler_move_configuration"):
        validate_checkpoint_metadata(
            result.backend_path,
            checkpoint_metadata(changed_data, changed, mode="synthetic"),
            changed.random_seed,
        )


def test_fresh_first_chunk_uses_initial_state_check_and_later_chunks_skip(monkeypatch, tmp_path):
    calls = []

    class RecordingSampler(_FakeSampler):
        def run_mcmc(self, initial_state, nsteps, **kwargs):
            calls.append(
                {
                    "initial_state_is_none": initial_state is None,
                    "skip_initial_state_check": kwargs["skip_initial_state_check"],
                }
            )
            return super().run_mcmc(initial_state, nsteps, **kwargs)

    monkeypatch.setattr(phase1c_sampler.emcee, "EnsembleSampler", RecordingSampler)
    config = _sampler_test_config(tmp_path / "fresh", chunk_steps=1)
    data, timing, _ = synthetic_dataset(config)
    run_ensembles(data, config, timing, steps=2, mode="synthetic", resume=False)

    assert calls == [
        {"initial_state_is_none": False, "skip_initial_state_check": False},
        {"initial_state_is_none": True, "skip_initial_state_check": True},
    ]


def test_resume_chunk_uses_skip_initial_state_check(monkeypatch, tmp_path):
    config = _sampler_test_config(tmp_path / "resume", chunk_steps=1)
    data, timing, _ = synthetic_dataset(config)
    run_ensembles(data, config, timing, steps=1, mode="synthetic", resume=False)

    calls = []

    class RecordingSampler(_FakeSampler):
        def run_mcmc(self, initial_state, nsteps, **kwargs):
            calls.append(
                {
                    "initial_state_is_none": initial_state is None,
                    "skip_initial_state_check": kwargs["skip_initial_state_check"],
                }
            )
            return super().run_mcmc(initial_state, nsteps, **kwargs)

    monkeypatch.setattr(phase1c_sampler.emcee, "EnsembleSampler", RecordingSampler)
    run_ensembles(data, config, timing, steps=2, mode="synthetic", resume=True)

    assert calls == [{"initial_state_is_none": True, "skip_initial_state_check": True}]


class _FakeSampler:
    def __init__(self, nwalkers, ndim, log_prob_fn, *, backend, moves, **kwargs):
        del log_prob_fn, kwargs
        self.nwalkers = nwalkers
        self.ndim = ndim
        self.backend = _FakeBackend(backend)
        self.moves = moves
        self.acceptance_fraction = np.zeros(nwalkers, dtype=float)

    def run_mcmc(self, initial_state, nsteps, **kwargs):
        del initial_state, kwargs
        self.backend.iteration += int(nsteps)
        return None


class _FakeBackend:
    def __init__(self, backend):
        self.iteration = int(backend.iteration)


def _move_names_and_weights(moves):
    return [(type(move).__name__, float(weight)) for move, weight in moves]


def _sampler_test_config(output_dir: Path, **overrides) -> Phase1CConfig:
    values = {
        "output_dir": output_dir,
        "random_seed": 13579,
        "n_ensembles": 1,
        "n_walkers": 16,
        "synthetic_steps": 2,
        "chunk_steps": 1,
        "warmup_steps": 0,
        "prior_informed_pool_size": 16,
        "prior_informed_max_pool_size": 16,
        "prior_informed_elite_size": 1,
        "prior_informed_min_finite_candidates": 1,
    }
    values.update(overrides)
    return Phase1CConfig(**values)


def _rewrite_checkpoint_as_legacy_stretch(path: Path) -> None:
    with h5py.File(path, "a") as hdf:
        attrs = hdf.attrs
        identity = json.loads(attrs["phase1c_immutable_scientific_identity"])
        identity["sampler"].pop("move_configuration", None)
        configuration = identity["priors_and_transforms"]["configuration"]
        configuration.pop("sampler_move_strategy", None)
        configuration.pop("maximum_initial_logp_deficit", None)
        attrs["phase1c_immutable_scientific_identity"] = json.dumps(identity, sort_keys=True)
        attrs["phase1c_immutable_scientific_identity_sha256"] = _identity_sha(identity)
        del attrs["phase1c_sampler_move_configuration"]


def _identity_sha(identity: dict) -> str:
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
