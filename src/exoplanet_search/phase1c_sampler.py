"""emcee sampling, initialization, and HDF checkpoint management for Phase 1C."""

from __future__ import annotations

import hashlib
import json
import platform
import time
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable

import emcee
import h5py
import numpy as np

from .phase1c_likelihood import (
    Phase1CLikelihoodContext,
    PosteriorProfiler,
    log_probability_with_context,
    profiled_log_probability_with_context,
)
from .phase1c_parameters import (
    deterministic_physical_sample,
    jitter_prior_scale,
    physical_to_vector,
)
from .phase1c_types import FrozenPhase1BData, PARAMETER_ORDER, Phase1CConfig, TimingReference

LOCAL_STRATEGIES = ("local_tight", "local_moderate", "local_broad")
PRIOR_STRATEGY = "prior_informed"


@dataclass(frozen=True)
class EnsembleRunResult:
    """Recorded result of one independent emcee ensemble."""

    ensemble_index: int
    seed: int
    strategy: str
    backend_path: Path
    iterations: int
    runtime_seconds: float
    acceptance_fraction: np.ndarray
    initialization_summary: dict[str, Any]
    profiler_summary: dict[str, Any]


@dataclass(frozen=True)
class InitializationResult:
    """Initial walker cloud and recorded diagnostics."""

    walkers: np.ndarray
    summary: dict[str, Any]


class ProfiledLogPosterior:
    """Callable log-posterior wrapper used by emcee."""

    def __init__(
        self,
        context: Phase1CLikelihoodContext,
        profiler: PosteriorProfiler,
    ) -> None:
        self.context = context
        self.profiler = profiler

    def __call__(self, vector: np.ndarray) -> float:
        return profiled_log_probability_with_context(vector, self.context, self.profiler)


def run_ensembles(
    data: FrozenPhase1BData,
    config: Phase1CConfig,
    timing: TimingReference,
    *,
    steps: int,
    mode: str,
    resume: bool = False,
    chunk_callback: Callable[[list[EnsembleRunResult], float, dict[str, Any]], None] | None = None,
) -> list[EnsembleRunResult]:
    """Run independent chunked emcee ensembles with HDF checkpoint backends."""
    if config.n_walkers < 2 * len(PARAMETER_ORDER):
        raise ValueError("Phase 1C requires at least 2 * ndim emcee walkers.")
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = checkpoint_metadata(data, config, mode=mode)
    likelihood_context = Phase1CLikelihoodContext.from_data(data, config, timing)

    samplers = []
    profilers = []
    initial_states: list[np.ndarray | None] = []
    initialization_summaries = []
    runtime_starts = []
    backend_paths = []
    seeds = []
    strategies = []
    for ensemble_index in range(config.n_ensembles):
        seed = int(config.random_seed + 1000 * ensemble_index)
        strategy = initialization_strategy(ensemble_index, config.n_ensembles)
        backend_path = output_dir / f"ensemble_{ensemble_index:02d}.h5"
        backend = emcee.backends.HDFBackend(str(backend_path))
        if resume:
            validate_checkpoint_metadata(backend_path, metadata, seed)
        else:
            backend.reset(config.n_walkers, len(PARAMETER_ORDER))
            write_checkpoint_metadata(backend_path, metadata, seed)
        profiler = PosteriorProfiler()
        sampler = emcee.EnsembleSampler(
            config.n_walkers,
            len(PARAMETER_ORDER),
            ProfiledLogPosterior(likelihood_context, profiler),
            backend=backend,
        )
        rng = np.random.default_rng(seed)
        if resume and backend.iteration > 0:
            initial_state = None
            initialization_summary = _resume_initialization_summary(strategy, seed)
        else:
            initialization = build_initialization(
                data,
                config,
                timing,
                rng,
                strategy,
                seed,
                context=likelihood_context,
            )
            initial_state = initialization.walkers
            initialization_summary = initialization.summary
        samplers.append(sampler)
        profilers.append(profiler)
        initial_states.append(initial_state)
        initialization_summaries.append(initialization_summary)
        runtime_starts.append(time.perf_counter())
        backend_paths.append(backend_path)
        seeds.append(seed)
        strategies.append(strategy)

    remaining = max(int(steps) - min(int(sampler.backend.iteration) for sampler in samplers), 0)
    while remaining > 0:
        chunk = min(config.chunk_steps, remaining)
        for index, sampler in enumerate(samplers):
            current_remaining = max(int(steps) - int(sampler.backend.iteration), 0)
            if current_remaining <= 0:
                continue
            sampler.run_mcmc(
                initial_states[index],
                min(chunk, current_remaining),
                progress=False,
                skip_initial_state_check=True,
            )
            initial_states[index] = None
        remaining = max(int(steps) - min(int(sampler.backend.iteration) for sampler in samplers), 0)
        if chunk_callback is not None:
            chunk_callback(
                _current_results(
                    samplers,
                    profilers,
                    backend_paths,
                    seeds,
                    strategies,
                    initialization_summaries,
                    runtime_starts,
                ),
                max(time.perf_counter() - min(runtime_starts), 0.0),
                aggregate_profiler_summary(profilers),
            )

    return _current_results(
        samplers,
        profilers,
        backend_paths,
        seeds,
        strategies,
        initialization_summaries,
        runtime_starts,
    )


def initialization_strategy(ensemble_index: int, n_ensembles: int) -> str:
    """Return the deterministic strategy assigned to an ensemble."""
    if n_ensembles <= 1:
        return "local_tight"
    if n_ensembles == 2:
        return "local_tight" if ensemble_index == 0 else PRIOR_STRATEGY
    if n_ensembles == 3:
        return ("local_tight", "local_moderate", PRIOR_STRATEGY)[ensemble_index]
    if ensemble_index == 0:
        return "local_tight"
    if ensemble_index == 1:
        return "local_moderate"
    if ensemble_index == 2:
        return "local_broad"
    return PRIOR_STRATEGY if ensemble_index == 3 else f"{PRIOR_STRATEGY}_{ensemble_index}"


def build_initialization(
    data: FrozenPhase1BData,
    config: Phase1CConfig,
    timing: TimingReference,
    rng: np.random.Generator,
    strategy: str,
    seed: int,
    *,
    context: Phase1CLikelihoodContext | None = None,
) -> InitializationResult:
    """Generate initial walkers and diagnostics for one strategy."""
    likelihood_context = context or Phase1CLikelihoodContext.from_data(data, config, timing)
    center = deterministic_center_vector(data, config, timing)
    walkers: list[np.ndarray] = []
    initial_log_prob: list[float] = []
    redraws = 0
    tries = 0
    while len(walkers) < config.n_walkers and tries < config.n_walkers * 1000:
        tries += 1
        if strategy.startswith(PRIOR_STRATEGY):
            candidate = broad_prior_candidate(data, config, timing, rng)
        else:
            scales = initialization_scales(config, strategy)
            candidate = center + rng.normal(0.0, scales)
            candidate = _clip_local_candidate(candidate, config, timing)
        value = log_probability_with_context(candidate, likelihood_context)
        if np.isfinite(value):
            walkers.append(candidate)
            initial_log_prob.append(float(value))
        else:
            redraws += 1
    if len(walkers) != config.n_walkers:
        raise RuntimeError(f"Could not generate enough finite Phase 1C initial walkers for {strategy}.")
    walker_array = np.asarray(walkers, dtype=float)
    log_prob_array = np.asarray(initial_log_prob, dtype=float)
    distances = np.linalg.norm(walker_array - center, axis=1)
    timing_offsets = {
        "period_offset": _min_median_max(walker_array[:, 6]),
        "mid_epoch_offset": _min_median_max(walker_array[:, 7]),
    }
    summary = {
        "strategy": strategy,
        "seed": int(seed),
        "center": _parameter_dict(center),
        "configured_scales": None
        if strategy.startswith(PRIOR_STRATEGY)
        else _parameter_dict(initialization_scales(config, strategy)),
        "actual_distance_from_deterministic_center": _min_median_max(distances),
        "initial_finite_log_probability_fraction": float(np.mean(np.isfinite(log_prob_array))),
        "initial_log_posterior": _min_median_max(log_prob_array),
        "redraws": int(redraws),
        "timing_offsets": timing_offsets,
        "rank": int(np.linalg.matrix_rank(walker_array - np.mean(walker_array, axis=0))),
    }
    return InitializationResult(walkers=walker_array, summary=summary)


def initial_walkers(
    data: FrozenPhase1BData,
    config: Phase1CConfig,
    timing: TimingReference,
    rng: np.random.Generator,
    ensemble_index: int,
) -> np.ndarray:
    """Generate dispersed finite initial walkers for one ensemble."""
    strategy = initialization_strategy(ensemble_index, config.n_ensembles)
    return build_initialization(
        data,
        config,
        timing,
        rng,
        strategy,
        config.random_seed + 1000 * ensemble_index,
    ).walkers


def deterministic_center_vector(
    data: FrozenPhase1BData,
    config: Phase1CConfig,
    timing: TimingReference,
) -> np.ndarray:
    """Return deterministic center used by all local initialization clouds."""
    del config
    return physical_to_vector(
        deterministic_physical_sample(data, timing, jitter_floor=1.0e-8),
        timing,
    )


def initialization_scales(config: Phase1CConfig, strategy: str) -> np.ndarray:
    """Return configured transformed-coordinate Gaussian scales for a local strategy."""
    scale_map = {
        "local_tight": config.local_tight_scales,
        "local_moderate": config.local_moderate_scales,
        "local_broad": config.local_broad_scales,
    }
    if strategy not in scale_map:
        raise ValueError(f"Strategy {strategy!r} does not use local Gaussian scales.")
    scales = np.asarray(scale_map[strategy], dtype=float)
    if scales.shape != (len(PARAMETER_ORDER),):
        raise ValueError(f"Initialization scales for {strategy} must have {len(PARAMETER_ORDER)} entries.")
    return scales


def broad_prior_candidate(
    data: FrozenPhase1BData,
    config: Phase1CConfig,
    timing: TimingReference,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw a broad prior-informed candidate in transformed coordinates."""
    rp = rng.uniform(*config.rp_bounds)
    lower_a = max(config.a_bounds[0], 1.0 + rp + 1.0e-4)
    log_a = rng.uniform(np.log(lower_a), np.log(config.a_bounds[1]))
    jitter_scale = 0.5 * (np.log(config.jitter_upper) - np.log(config.jitter_lower))
    log_jitter = np.clip(
        np.log(jitter_prior_scale(data, config)) + rng.normal(0.0, jitter_scale),
        np.log(config.jitter_lower * 1.01),
        np.log(config.jitter_upper * 0.9),
    )
    return np.asarray(
        [
            np.log(rp),
            log_a,
            rng.uniform(0.0, 1.0),
            rng.uniform(*config.q_bounds),
            rng.uniform(*config.q_bounds),
            log_jitter,
            rng.uniform(-0.9 * timing.period_half_width, 0.9 * timing.period_half_width),
            rng.uniform(-0.9 * timing.mid_epoch_half_width, 0.9 * timing.mid_epoch_half_width),
        ],
        dtype=float,
    )


def load_backend_chains(results: list[EnsembleRunResult]) -> tuple[np.ndarray, np.ndarray]:
    """Load HDF backend chains as diagnostics arrays shaped chains, draws, ndim."""
    chain_parts = []
    log_prob_parts = []
    for result in results:
        backend = emcee.backends.HDFBackend(str(result.backend_path), read_only=True)
        chain = backend.get_chain()
        log_prob = backend.get_log_prob()
        chain_parts.append(np.transpose(chain, (1, 0, 2)))
        log_prob_parts.append(np.transpose(log_prob, (1, 0)))
    return np.concatenate(chain_parts, axis=0), np.concatenate(log_prob_parts, axis=0)


def load_backend_chains_by_ensemble(results: list[EnsembleRunResult]) -> list[np.ndarray]:
    chains = []
    for result in results:
        backend = emcee.backends.HDFBackend(str(result.backend_path), read_only=True)
        chains.append(np.transpose(backend.get_chain(), (1, 0, 2)))
    return chains


def estimate_autocorrelation_time(results: list[EnsembleRunResult], warmup_steps: int) -> dict[str, Any]:
    """Return per-ensemble integrated autocorrelation estimates for all sampled parameters."""
    rows = []
    for result in results:
        backend = emcee.backends.HDFBackend(str(result.backend_path), read_only=True)
        retained_steps = max(int(backend.iteration) - int(warmup_steps), 0)
        estimate = None
        error = None
        try:
            estimate = backend.get_autocorr_time(discard=warmup_steps, quiet=True)
        except (emcee.autocorr.AutocorrError, ValueError, FloatingPointError, IndexError) as exc:
            error = str(exc)
        for parameter_index, parameter in enumerate(PARAMETER_ORDER):
            value = None
            available = False
            if estimate is not None:
                raw_value = float(np.asarray(estimate, dtype=float)[parameter_index])
                if np.isfinite(raw_value):
                    value = raw_value
                    available = True
                else:
                    error = "nonfinite_autocorrelation_time"
            rows.append(
                {
                    "ensemble": int(result.ensemble_index),
                    "parameter": parameter,
                    "tau": value,
                    "available": bool(available),
                    "retained_steps": int(retained_steps),
                    "error": error,
                }
            )
    valid = [float(row["tau"]) for row in rows if row["available"] and row["tau"] is not None]
    unavailable = [row for row in rows if not row["available"]]
    return {
        "rows": rows,
        "all_available": not unavailable and len(rows) == len(results) * len(PARAMETER_ORDER),
        "worst_tau": max(valid) if valid else None,
        "unavailable_count": len(unavailable),
    }


def aggregate_profiler_summary(profilers: list[PosteriorProfiler]) -> dict[str, Any]:
    aggregate = PosteriorProfiler()
    for profiler in profilers:
        aggregate.add(profiler)
    return aggregate.summary()


def checkpoint_metadata(data: FrozenPhase1BData, config: Phase1CConfig, *, mode: str) -> dict[str, Any]:
    """Return metadata that binds checkpoints to inputs, config, and dependency versions."""
    identity = immutable_checkpoint_identity(data, config, mode=mode)
    identity_sha = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "mode": mode,
        "run_id": config.run_id,
        "immutable_scientific_identity_sha256": identity_sha,
        "immutable_scientific_identity": identity,
    }


def immutable_checkpoint_identity(
    data: FrozenPhase1BData,
    config: Phase1CConfig,
    *,
    mode: str,
) -> dict[str, Any]:
    """Return checkpoint identity excluding mutable execution controls."""
    payload = config.to_dict()
    for key in (
        "output_dir",
        "run_id",
        "pilot_steps",
        "synthetic_steps",
        "synthetic_recovery_steps",
        "production_steps",
        "target_total_steps",
        "additional_steps",
        "chunk_steps",
        "max_pilot_seconds",
        "minimum_meaningful_summary_draws",
    ):
        payload.pop(key, None)
    return {
        "mode": mode,
        "phase1b_input_manifest_sha256": data.input_manifest["manifest_sha256"],
        "model": {
            "transit": "batman circular one-planet quadratic limb-darkening",
            "likelihood": "Gaussian cadence noise with exact Gaussian local-baseline marginalization",
            "baseline_treatment": {
                "intercept_mean": 1.0,
                "intercept_sigma": config.baseline_intercept_sigma,
                "slope_mean": 0.0,
                "slope_sigma": config.baseline_slope_sigma,
            },
            "supersample_factor": config.supersample_factor,
        },
        "priors_and_transforms": {
            "parameter_order": list(PARAMETER_ORDER),
            "transform_record": "log_rp, log_a, z_b, q1, q2, log_jitter, period_offset, mid_epoch_offset",
            "configuration": payload,
            "a_over_rstar_prior_interpretation": (
                "independent log-uniform draw over configured bounds followed by physical geometry rejection"
            ),
        },
        "sampler": {
            "n_walkers": config.n_walkers,
            "n_ensembles": config.n_ensembles,
            "seeds": [int(config.random_seed + 1000 * index) for index in range(config.n_ensembles)],
        },
        "dependencies": dependency_versions(),
    }


def write_checkpoint_metadata(path: Path, metadata: dict[str, Any], seed: int) -> None:
    with h5py.File(path, "a") as hdf:
        attrs = hdf.attrs
        for key, value in metadata.items():
            attrs[f"phase1c_{key}"] = json.dumps(value, sort_keys=True)
        attrs["phase1c_ensemble_seed"] = int(seed)


def validate_checkpoint_metadata(path: Path, metadata: dict[str, Any], seed: int) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Cannot resume missing checkpoint: {path}")
    with h5py.File(path, "r") as hdf:
        attrs = hdf.attrs
        for key, value in metadata.items():
            recorded = attrs.get(f"phase1c_{key}")
            expected = json.dumps(value, sort_keys=True)
            if recorded != expected:
                raise ValueError(f"Checkpoint metadata mismatch for {key}: {path}")
        if int(attrs.get("phase1c_ensemble_seed", -1)) != int(seed):
            raise ValueError(f"Checkpoint ensemble seed mismatch: {path}")


def dependency_versions() -> dict[str, str]:
    packages = {
        "emcee": "emcee",
        "arviz": "arviz",
        "h5py": "h5py",
        "batman": "batman-package",
        "numpy": "numpy",
        "scipy": "scipy",
        "astropy": "astropy",
        "pandas": "pandas",
    }
    versions = {"python": platform.python_version()}
    for label, package in packages.items():
        try:
            versions[label] = version(package)
        except PackageNotFoundError:
            versions[label] = "not-installed"
    return versions


def _current_results(
    samplers,
    profilers: list[PosteriorProfiler],
    backend_paths: list[Path],
    seeds: list[int],
    strategies: list[str],
    initialization_summaries: list[dict[str, Any]],
    runtime_starts: list[float],
) -> list[EnsembleRunResult]:
    results = []
    now = time.perf_counter()
    for index, sampler in enumerate(samplers):
        results.append(
            EnsembleRunResult(
                ensemble_index=index,
                seed=seeds[index],
                strategy=strategies[index],
                backend_path=backend_paths[index],
                iterations=int(sampler.backend.iteration),
                runtime_seconds=float(now - runtime_starts[index]),
                acceptance_fraction=np.asarray(sampler.acceptance_fraction, dtype=float),
                initialization_summary=initialization_summaries[index],
                profiler_summary=profilers[index].summary(),
            )
        )
    return results


def _clip_local_candidate(
    candidate: np.ndarray,
    config: Phase1CConfig,
    timing: TimingReference,
) -> np.ndarray:
    clipped = np.asarray(candidate, dtype=float).copy()
    clipped[2] = np.clip(clipped[2], 1.0e-4, 0.999)
    clipped[3] = np.clip(clipped[3], 1.0e-4, 0.999)
    clipped[4] = np.clip(clipped[4], 1.0e-4, 0.999)
    clipped[5] = np.clip(clipped[5], np.log(config.jitter_lower * 1.01), np.log(config.jitter_upper * 0.9))
    clipped[6] = np.clip(clipped[6], -0.95 * timing.period_half_width, 0.95 * timing.period_half_width)
    clipped[7] = np.clip(clipped[7], -0.95 * timing.mid_epoch_half_width, 0.95 * timing.mid_epoch_half_width)
    return clipped


def _parameter_dict(values: np.ndarray) -> dict[str, float]:
    return {name: float(values[index]) for index, name in enumerate(PARAMETER_ORDER)}


def _min_median_max(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "max": float(np.max(values)),
    }


def _resume_initialization_summary(strategy: str, seed: int) -> dict[str, Any]:
    return {
        "strategy": strategy,
        "seed": int(seed),
        "resume": True,
        "message": "Initial walkers are loaded from existing HDF checkpoint state.",
    }
