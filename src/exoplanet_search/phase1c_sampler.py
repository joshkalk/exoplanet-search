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
    if strategy.startswith(PRIOR_STRATEGY):
        return _build_prior_informed_initialization(data, config, timing, rng, seed, likelihood_context, center)
    walkers: list[np.ndarray] = []
    initial_log_prob: list[float] = []
    redraws = 0
    tries = 0
    while len(walkers) < config.n_walkers and tries < config.n_walkers * 1000:
        tries += 1
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


def _build_prior_informed_initialization(
    data: FrozenPhase1BData,
    config: Phase1CConfig,
    timing: TimingReference,
    rng: np.random.Generator,
    seed: int,
    context: Phase1CLikelihoodContext,
    center: np.ndarray,
) -> InitializationResult:
    """Build a coherent remote-start cloud selected from broad posterior-screened draws."""
    center_logp = float(log_probability_with_context(center, context))
    maximum_deficit = float(config.prior_informed_max_logp_deficit)
    logp_floor = center_logp - maximum_deficit
    pool_vectors = np.asarray(
        [broad_prior_candidate(data, config, timing, rng) for _ in range(config.prior_informed_pool_size)],
        dtype=float,
    )
    pool_log_prob = np.asarray([log_probability_with_context(candidate, context) for candidate in pool_vectors])
    finite_mask = np.isfinite(pool_log_prob)
    finite_indices = np.flatnonzero(finite_mask)
    deficits = np.full(pool_log_prob.shape, np.inf, dtype=float)
    deficits[finite_mask] = center_logp - pool_log_prob[finite_mask]
    eligible_mask = finite_mask & (pool_log_prob >= logp_floor)
    eligible_indices = np.flatnonzero(eligible_mask)
    if eligible_indices.size < int(config.prior_informed_min_finite_candidates):
        raise RuntimeError(
            "Insufficient posterior-eligible broad prior-informed candidates: "
            f"{eligible_indices.size} found, "
            f"{config.prior_informed_min_finite_candidates} required."
        )
    eligible_order = eligible_indices[np.argsort(pool_log_prob[eligible_indices])[::-1]]
    elite_size = min(int(config.prior_informed_elite_size), int(eligible_order.size))
    if elite_size <= 0:
        raise RuntimeError("Prior-informed elite set is empty after posterior eligibility screening.")
    elite_indices = eligible_order[:elite_size]
    normalized = _normalized_distance(pool_vectors[elite_indices], center, np.asarray(config.local_broad_scales))
    best_local = int(np.argmax(normalized))
    anchor_index = int(elite_indices[best_local])
    anchor = pool_vectors[anchor_index].copy()
    anchor_logp = float(pool_log_prob[anchor_index])
    anchor_deficit = float(center_logp - anchor_logp)
    if anchor_deficit > maximum_deficit or not np.isfinite(anchor_logp):
        raise RuntimeError("Selected prior-informed anchor failed posterior eligibility screening.")
    if np.array_equal(anchor, center):
        raise RuntimeError("Prior-informed anchor unexpectedly equals deterministic center.")

    scales = np.asarray(config.prior_informed_cloud_scales, dtype=float)
    if scales.shape != (len(PARAMETER_ORDER),):
        raise ValueError(f"prior_informed_cloud_scales must have {len(PARAMETER_ORDER)} entries.")
    walkers: list[np.ndarray] = [anchor.copy()]
    initial_log_prob: list[float] = [anchor_logp]
    rejection_counts = {"nonfinite": 0, "below_floor": 0}
    tries = 0
    max_tries = config.n_walkers * 2000
    while len(walkers) < config.n_walkers and tries < max_tries:
        tries += 1
        candidate = _clip_local_candidate(anchor + rng.normal(0.0, scales), config, timing)
        value = log_probability_with_context(candidate, context)
        if np.isfinite(value) and float(value) >= logp_floor:
            walkers.append(candidate)
            initial_log_prob.append(float(value))
        else:
            if not np.isfinite(value):
                rejection_counts["nonfinite"] += 1
            else:
                rejection_counts["below_floor"] += 1
    if len(walkers) != config.n_walkers:
        raise RuntimeError("Could not generate enough posterior-screened prior-informed walkers.")
    walker_array = np.asarray(walkers, dtype=float)
    log_prob_array = np.asarray(initial_log_prob, dtype=float)
    rank = int(np.linalg.matrix_rank(walker_array - np.mean(walker_array, axis=0)))
    if rank != len(PARAMETER_ORDER):
        raise RuntimeError("Prior-informed walker cloud is not full rank.")
    distances = _normalized_distance(walker_array, center, np.asarray(config.local_broad_scales))
    anchor_rank = None
    if eligible_indices.size:
        anchor_rank = int(np.where(eligible_order == anchor_index)[0][0] + 1)
    log_prob_deficits = center_logp - log_prob_array
    timing_offsets = {
        "period_offset": _min_median_max(walker_array[:, 6]),
        "mid_epoch_offset": _min_median_max(walker_array[:, 7]),
    }
    summary = {
        "strategy": PRIOR_STRATEGY,
        "seed": int(seed),
        "center": _parameter_dict(center),
        "configured_scales": _parameter_dict(scales),
        "actual_distance_from_deterministic_center": _min_median_max(distances),
        "initial_finite_log_probability_fraction": float(np.mean(np.isfinite(log_prob_array))),
        "initial_log_posterior": _min_median_max(log_prob_array),
        "redraws": int(sum(rejection_counts.values())),
        "timing_offsets": timing_offsets,
        "rank": rank,
        "prior_informed_remote_anchor": {
            "algorithm": "broad_pool_elite_remote_anchor_v1",
            "pool_size": int(config.prior_informed_pool_size),
            "pool_scale_multiplier": float(config.prior_informed_pool_scale_multiplier),
            "finite_candidate_count": int(finite_indices.size),
            "posterior_eligible_candidate_count": int(eligible_indices.size),
            "finite_candidate_log_posterior_quantiles": _quantiles(pool_log_prob[finite_mask]),
            "broad_candidate_log_posterior_deficits": [
                None if not np.isfinite(value) else float(value) for value in deficits
            ],
            "deterministic_center_log_posterior": center_logp,
            "maximum_log_posterior_deficit": maximum_deficit,
            "elite_rule": "highest posterior finite candidates, then largest normalized distance from deterministic center",
            "elite_size_configured": int(config.prior_informed_elite_size),
            "elite_size_used": int(elite_size),
            "selected_anchor_pool_index": anchor_index,
            "selected_anchor_rank_by_log_posterior": anchor_rank,
            "selected_anchor_vector": _parameter_dict(anchor),
            "selected_anchor_log_posterior": anchor_logp,
            "selected_anchor_log_posterior_deficit": anchor_deficit,
            "selected_anchor_normalized_distance_from_center": float(
                _normalized_distance(anchor[None, :], center, np.asarray(config.local_broad_scales))[0]
            ),
            "cloud_log_posterior_floor": float(logp_floor),
            "cloud_minimum_log_posterior_deficit": float(np.max(log_prob_deficits)),
            "cloud_log_posterior_range": _min_median_max(log_prob_array),
            "cloud_scales": _parameter_dict(scales),
            "rejection_counts": {key: int(value) for key, value in rejection_counts.items()},
            "fallback_used": False,
            "fallback_reason": None,
            "full_rank": bool(rank == len(PARAMETER_ORDER)),
        },
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
    center = deterministic_center_vector(data, config, timing)
    scales = float(config.prior_informed_pool_scale_multiplier) * np.asarray(config.local_broad_scales, dtype=float)
    return _clip_local_candidate(center + rng.normal(0.0, scales), config, timing)


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
            estimate = backend.get_autocorr_time(discard=warmup_steps, quiet=False)
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
    rp = float(np.clip(np.exp(clipped[0]), config.rp_bounds[0], config.rp_bounds[1]))
    clipped[0] = np.log(rp)
    lower_a = max(config.a_bounds[0], 1.0 + rp + 1.0e-4)
    clipped[1] = np.clip(clipped[1], np.log(lower_a), np.log(config.a_bounds[1]))
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


def _quantiles(values: np.ndarray) -> dict[str, float | None]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"q05": None, "q16": None, "q50": None, "q84": None, "q95": None}
    q05, q16, q50, q84, q95 = np.quantile(finite, [0.05, 0.16, 0.50, 0.84, 0.95])
    return {
        "q05": float(q05),
        "q16": float(q16),
        "q50": float(q50),
        "q84": float(q84),
        "q95": float(q95),
    }


def _normalized_distance(values: np.ndarray, center: np.ndarray, scales: np.ndarray) -> np.ndarray:
    safe_scales = np.maximum(np.asarray(scales, dtype=float), 1.0e-12)
    return np.linalg.norm((np.asarray(values, dtype=float) - np.asarray(center, dtype=float)) / safe_scales, axis=1)


def _resume_initialization_summary(strategy: str, seed: int) -> dict[str, Any]:
    return {
        "strategy": strategy,
        "seed": int(seed),
        "resume": True,
        "message": "Initial walkers are loaded from existing HDF checkpoint state.",
    }
