"""Phase 1D ensemble-aware posterior draw access and deterministic selection."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .phase1c_parameters import vector_to_physical
from .phase1c_types import PARAMETER_ORDER, Phase1CConfig, PhysicalSample, TimingReference


@dataclass(frozen=True)
class Phase1DSourcePolicy:
    """Controls whether a posterior source can be used authoritatively."""

    require_converged: bool = True
    allow_nonproduction: bool = False
    authoritative: bool = True
    override_reason: str | None = None


@dataclass(frozen=True)
class EnsembleSource:
    """One validated Phase 1C HDF ensemble source."""

    ensemble: int
    seed: int
    path: Path
    mode: str
    run_id: str
    steps: int
    walkers: int
    ndim: int
    warmup_steps: int

    @property
    def retained_steps(self) -> int:
        return max(self.steps - self.warmup_steps, 0)

    @property
    def eligible_count(self) -> int:
        return self.retained_steps * self.walkers


@dataclass(frozen=True)
class SelectedPosteriorDraw:
    """A selected posterior draw with ensemble, walker, and step provenance."""

    run_id: str
    mode: str
    ensemble: int
    ensemble_seed: int
    walker: int
    step: int
    log_posterior: float
    vector: np.ndarray
    physical: PhysicalSample
    selection_seed: int
    selection_position: int

    @property
    def key(self) -> tuple[int, int, int]:
        return (self.ensemble, self.walker, self.step)

    def audit_record(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "mode": self.mode,
            "ensemble": self.ensemble,
            "ensemble_seed": self.ensemble_seed,
            "walker": self.walker,
            "step": self.step,
            "log_posterior": self.log_posterior,
            "vector": [float(value) for value in self.vector],
            "physical": asdict(self.physical),
            "selection_seed": self.selection_seed,
            "selection_position": self.selection_position,
        }


@dataclass(frozen=True)
class DrawSelection:
    """Selected posterior draws and their auditable manifest."""

    source_run_dir: Path
    run_id: str
    mode: str
    selection_seed: int
    requested_draws: int
    selected_draws: tuple[SelectedPosteriorDraw, ...]
    manifest: dict[str, Any]


def load_phase1c_config(run_dir: Path) -> Phase1CConfig:
    """Load the saved Phase 1C configuration without mutating it."""
    payload = _read_json(run_dir / "phase1c_configuration.json")
    payload.pop("parameter_order", None)
    payload.pop("notes", None)
    for key in ("phase1b_output_dir", "output_dir"):
        if key in payload:
            payload[key] = Path(payload[key])
    for key in (
        "rp_bounds",
        "a_bounds",
        "q_bounds",
        "local_tight_scales",
        "local_moderate_scales",
        "local_broad_scales",
    ):
        if key in payload:
            payload[key] = tuple(payload[key])
    config = Phase1CConfig(**payload)
    return type(config)(**{**config.__dict__, "output_dir": run_dir})


def load_phase1c_draw_sources(run_dir: Path, policy: Phase1DSourcePolicy) -> tuple[Phase1CConfig, tuple[EnsembleSource, ...]]:
    """Validate Phase 1C HDF files and return ensemble sources.

    Scientific Phase 1D posterior predictive execution requires a converged
    production source. Development-only overrides must be explicit and are
    recorded in selection manifests.
    """
    config = load_phase1c_config(run_dir)
    diagnostics = _read_json(run_dir / "sampler_diagnostics.json")
    status = str(diagnostics.get("status", ""))
    if policy.require_converged and status != "converged":
        raise ValueError(f"Authoritative Phase 1D draw access requires converged diagnostics, found {status!r}.")

    sources = []
    seen_run_ids = set()
    seen_modes = set()
    for path in sorted(run_dir.glob("ensemble_*.h5")):
        source = _validate_hdf_source(path, config)
        sources.append(source)
        seen_run_ids.add(source.run_id)
        seen_modes.add(source.mode)
    if not sources:
        raise FileNotFoundError(f"No Phase 1C ensemble HDF files found in {run_dir}.")
    if len(sources) != config.n_ensembles:
        raise ValueError(f"Expected {config.n_ensembles} ensemble files, found {len(sources)}.")
    expected = set(range(config.n_ensembles))
    actual = {source.ensemble for source in sources}
    if actual != expected:
        raise ValueError(f"Phase 1C ensemble set mismatch: expected {sorted(expected)}, found {sorted(actual)}.")
    if len(seen_run_ids) != 1 or config.run_id not in seen_run_ids:
        raise ValueError("Phase 1C HDF files contain mixed or unexpected run IDs.")
    if len(seen_modes) != 1:
        raise ValueError("Phase 1C HDF files contain mixed modes.")
    mode = next(iter(seen_modes))
    if mode != "production" and not policy.allow_nonproduction:
        raise ValueError(f"Authoritative Phase 1D draw access requires production mode, found {mode!r}.")
    return config, tuple(sorted(sources, key=lambda source: source.ensemble))


def select_posterior_draws(
    run_dir: Path,
    *,
    timing: TimingReference,
    requested_draws: int,
    seed: int,
    policy: Phase1DSourcePolicy,
) -> DrawSelection:
    """Select deterministic ensemble-aware posterior draws without replacement.

    The selector samples retained post-warmup HDF positions only. It does not
    thin the posterior as an inference procedure; it merely chooses a smaller
    Monte Carlo subset for posterior-predictive replication while preserving
    ensemble, walker, and stored-step provenance.
    """
    if requested_draws <= 0:
        raise ValueError("requested_draws must be positive.")
    config, sources = load_phase1c_draw_sources(run_dir, policy)
    allocation = _draw_allocation(sources, requested_draws)
    rng = np.random.default_rng(seed)
    selected: list[SelectedPosteriorDraw] = []
    selected_by_ensemble = {}
    selected_by_walker: dict[str, int] = {}
    for source in sources:
        count = allocation[source.ensemble]
        if source.eligible_count < count:
            raise ValueError(
                f"Ensemble {source.ensemble} has {source.eligible_count} eligible draws, "
                f"cannot select {count} without replacement."
            )
        flat_choices = rng.choice(source.eligible_count, size=count, replace=False)
        selected_by_ensemble[str(source.ensemble)] = int(count)
        with h5py.File(source.path, "r") as hdf:
            chain = hdf["mcmc/chain"]
            log_prob = hdf["mcmc/log_prob"]
            for local_position in np.sort(flat_choices):
                retained_step = int(local_position // source.walkers)
                walker = int(local_position % source.walkers)
                step = int(source.warmup_steps + retained_step)
                vector = np.asarray(chain[step, walker, :], dtype=float)
                value = float(log_prob[step, walker])
                if not np.all(np.isfinite(vector)) or not np.isfinite(value):
                    raise ValueError(f"Selected nonfinite draw from {source.path}: step={step}, walker={walker}.")
                physical = vector_to_physical(vector, timing)
                selected_by_walker[f"{source.ensemble}:{walker}"] = selected_by_walker.get(
                    f"{source.ensemble}:{walker}", 0
                ) + 1
                selected.append(
                    SelectedPosteriorDraw(
                        run_id=source.run_id,
                        mode=source.mode,
                        ensemble=source.ensemble,
                        ensemble_seed=source.seed,
                        walker=walker,
                        step=step,
                        log_posterior=value,
                        vector=vector,
                        physical=physical,
                        selection_seed=seed,
                        selection_position=len(selected),
                    )
                )
    if len({draw.key for draw in selected}) != len(selected):
        raise ValueError("Draw selection produced duplicate HDF positions.")
    manifest = {
        "schema_version": "phase1d_draw_selection_v1",
        "source_run_dir": str(run_dir),
        "source_run_id": config.run_id,
        "source_mode": selected[0].mode if selected else None,
        "authoritative": policy.authoritative,
        "nonproduction_override": policy.allow_nonproduction,
        "override_reason": policy.override_reason,
        "selection_seed": seed,
        "requested_draws": requested_draws,
        "selected_draws": len(selected),
        "eligible_counts_by_ensemble": {str(source.ensemble): source.eligible_count for source in sources},
        "selected_counts_by_ensemble": selected_by_ensemble,
        "selected_counts_by_walker": selected_by_walker,
        "parameter_order": list(PARAMETER_ORDER),
    }
    return DrawSelection(
        source_run_dir=run_dir,
        run_id=str(config.run_id),
        mode=str(selected[0].mode if selected else ""),
        selection_seed=seed,
        requested_draws=requested_draws,
        selected_draws=tuple(selected),
        manifest=manifest,
    )


def write_draw_selection(output_dir: Path, selection: DrawSelection) -> None:
    """Write a compact selection manifest and JSONL draw audit."""
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "posterior_draw_selection_manifest.json", selection.manifest)
    with (output_dir / "selected_draw_audit.jsonl").open("w", encoding="utf-8") as output_file:
        for draw in selection.selected_draws:
            output_file.write(json.dumps(draw.audit_record(), sort_keys=True, separators=(",", ":")) + "\n")


def _validate_hdf_source(path: Path, config: Phase1CConfig) -> EnsembleSource:
    with h5py.File(path, "r") as hdf:
        for name in ("mcmc/chain", "mcmc/log_prob", "mcmc/accepted"):
            if name not in hdf:
                raise ValueError(f"Missing HDF dataset {name!r}: {path}")
        chain = hdf["mcmc/chain"]
        log_prob = hdf["mcmc/log_prob"]
        if len(chain.shape) != 3:
            raise ValueError(f"Expected mcmc/chain to have shape (step, walker, ndim): {path}")
        steps, walkers, ndim = map(int, chain.shape)
        if log_prob.shape != (steps, walkers):
            raise ValueError(f"mcmc/log_prob shape does not match chain shape: {path}")
        if ndim != len(PARAMETER_ORDER):
            raise ValueError(f"Expected {len(PARAMETER_ORDER)} parameters, found {ndim}: {path}")
        if walkers != config.n_walkers:
            raise ValueError(f"Walker count mismatch for {path}.")
        parameter_order = _parameter_order_from_attrs(hdf)
        if parameter_order != list(PARAMETER_ORDER):
            raise ValueError(f"Parameter order mismatch for {path}: {parameter_order}")
        run_id = str(_json_attr(hdf.attrs, "phase1c_run_id"))
        if run_id != str(config.run_id):
            raise ValueError(f"Run ID mismatch for {path}: {run_id!r}")
        mode = str(_json_attr(hdf.attrs, "phase1c_mode"))
        seed = int(hdf.attrs.get("phase1c_ensemble_seed", -1))
        if seed < 0 or (seed - config.random_seed) % 1000 != 0:
            raise ValueError(f"Cannot infer ensemble from seed {seed}: {path}")
        ensemble = (seed - config.random_seed) // 1000
        if ensemble < 0:
            raise ValueError(f"Invalid ensemble index inferred from seed {seed}: {path}")
        if config.warmup_steps >= steps:
            raise ValueError(f"No retained post-warmup draws in {path}.")
        _validate_retained_finite(chain, log_prob, config.warmup_steps, steps, path)
    return EnsembleSource(
        ensemble=int(ensemble),
        seed=seed,
        path=path,
        mode=mode,
        run_id=run_id,
        steps=steps,
        walkers=walkers,
        ndim=ndim,
        warmup_steps=config.warmup_steps,
    )


def _parameter_order_from_attrs(hdf: h5py.File) -> list[str]:
    direct = hdf.attrs.get("phase1c_parameter_order")
    if direct is not None:
        return list(_loads_attr(direct))
    identity = hdf.attrs.get("phase1c_immutable_scientific_identity")
    if identity is not None:
        payload = _loads_attr(identity)
        return list(payload["priors_and_transforms"]["parameter_order"])
    raise ValueError("Missing Phase 1C parameter-order metadata.")


def _draw_allocation(sources: tuple[EnsembleSource, ...], requested_draws: int) -> dict[int, int]:
    base = requested_draws // len(sources)
    remainder = requested_draws % len(sources)
    allocation = {}
    for index, source in enumerate(sorted(sources, key=lambda item: item.ensemble)):
        allocation[source.ensemble] = base + (1 if index < remainder else 0)
    return allocation


def _validate_retained_finite(chain, log_prob, start: int, stop: int, path: Path) -> None:
    chunk_size = 256
    for lower in range(start, stop, chunk_size):
        upper = min(lower + chunk_size, stop)
        if not np.all(np.isfinite(chain[lower:upper])):
            raise ValueError(f"Nonfinite retained vectors in {path}.")
        if not np.all(np.isfinite(log_prob[lower:upper])):
            raise ValueError(f"Nonfinite retained log posterior values in {path}.")


def _json_attr(attrs, key: str) -> Any:
    if key not in attrs:
        raise ValueError(f"Missing HDF attribute {key!r}.")
    return _loads_attr(attrs[key])


def _loads_attr(value: Any) -> Any:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return value


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as input_file:
        return json.load(input_file)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2)
