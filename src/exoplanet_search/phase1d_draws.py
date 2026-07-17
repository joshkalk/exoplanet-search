"""Phase 1D ensemble-aware posterior draw access and deterministic selection."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

from .phase1c import _validate_synthetic_input_record_if_needed, checkpoint_metadata, synthetic_dataset
from .phase1c_inputs import load_frozen_phase1b
from .phase1c_parameters import build_timing_reference, vector_to_physical
from .phase1c_sampler import validate_checkpoint_metadata
from .phase1c_types import PARAMETER_ORDER, FrozenPhase1BData, Phase1CConfig, PhysicalSample, TimingReference


@dataclass(frozen=True)
class Phase1DSourcePolicy:
    """Controls whether a posterior source can be used authoritatively."""

    require_converged: bool = True
    allow_nonproduction: bool = False
    authoritative: bool = True
    override_reason: str | None = None

    def __post_init__(self) -> None:
        reason = self.override_reason.strip() if isinstance(self.override_reason, str) else self.override_reason
        object.__setattr__(self, "override_reason", reason)
        if self.authoritative:
            if not self.require_converged:
                raise ValueError("Authoritative Phase 1D access requires require_converged=True.")
            if self.allow_nonproduction:
                raise ValueError("Authoritative Phase 1D access cannot allow nonproduction sources.")
            if self.override_reason is not None:
                raise ValueError("Authoritative Phase 1D access cannot include an override reason.")
        if not self.require_converged or self.allow_nonproduction:
            if self.authoritative:
                raise ValueError("Nonconverged or nonproduction overrides must be nonauthoritative.")
            if not self.override_reason:
                raise ValueError("Nonconverged or nonproduction overrides require a nonempty reason.")
        if self.override_reason == "":
            raise ValueError("Override reason must be nonempty when provided.")

    @classmethod
    def authoritative_production(cls) -> "Phase1DSourcePolicy":
        return cls(require_converged=True, allow_nonproduction=False, authoritative=True, override_reason=None)

    @classmethod
    def development_override(cls, reason: str) -> "Phase1DSourcePolicy":
        return cls(require_converged=False, allow_nonproduction=True, authoritative=False, override_reason=reason)


@dataclass(frozen=True)
class Phase1DSourceRequirements:
    """Declared identity requirements for a Phase 1D source class."""

    name: str
    expected_run_id: str
    expected_mode: str = "production"
    n_ensembles: int = 4
    n_walkers: int = 32
    warmup_steps: int = 2000
    supersample_factor: int = 11
    limb_darkening_sigma_floor: float = 0.08
    convergence_rhat_threshold: float = 1.01
    convergence_ess_minimum: float = 1000.0
    convergence_tau_multiple: float = 50.0
    parameter_order: tuple[str, ...] = PARAMETER_ORDER

    @classmethod
    def primary(cls, expected_run_id: str) -> "Phase1DSourceRequirements":
        return cls(name="phase1d_primary_posterior_v1", expected_run_id=expected_run_id)

    def validate(self, config: Phase1CConfig, mode: str) -> dict[str, Any]:
        expected = {
            "run_id": self.expected_run_id,
            "mode": self.expected_mode,
            "n_ensembles": self.n_ensembles,
            "n_walkers": self.n_walkers,
            "warmup_steps": self.warmup_steps,
            "supersample_factor": self.supersample_factor,
            "limb_darkening_sigma_floor": self.limb_darkening_sigma_floor,
            "convergence_rhat_threshold": self.convergence_rhat_threshold,
            "convergence_ess_minimum": self.convergence_ess_minimum,
            "convergence_tau_multiple": self.convergence_tau_multiple,
            "parameter_order": list(self.parameter_order),
        }
        observed = {
            "run_id": str(config.run_id),
            "mode": mode,
            "n_ensembles": int(config.n_ensembles),
            "n_walkers": int(config.n_walkers),
            "warmup_steps": int(config.warmup_steps),
            "supersample_factor": int(config.supersample_factor),
            "limb_darkening_sigma_floor": float(config.limb_darkening_sigma_floor),
            "convergence_rhat_threshold": float(config.convergence_rhat_threshold),
            "convergence_ess_minimum": float(config.convergence_ess_minimum),
            "convergence_tau_multiple": float(config.convergence_tau_multiple),
            "parameter_order": list(PARAMETER_ORDER),
        }
        mismatches = []
        for key, expected_value in expected.items():
            observed_value = observed[key]
            if isinstance(expected_value, float):
                matches = np.isclose(float(observed_value), expected_value, rtol=0.0, atol=1.0e-12)
            else:
                matches = observed_value == expected_value
            if not matches:
                mismatches.append({"field": key, "expected": expected_value, "observed": observed_value})
        record = {"name": self.name, "expected": expected, "observed": observed, "mismatches": mismatches}
        if mismatches:
            fields = ", ".join(item["field"] for item in mismatches)
            raise ValueError(f"Phase 1D source requirements mismatch for {self.name}: {fields}")
        return record


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
class ValidatedPhase1DSource:
    """Phase 1D source with bound configuration, data, timing, HDFs, and diagnostics."""

    run_dir: Path
    config: Phase1CConfig
    mode: str
    data: FrozenPhase1BData
    timing: TimingReference
    ensembles: tuple[EnsembleSource, ...]
    diagnostics: dict[str, Any]
    checkpoint_iterations: dict[str, int]
    input_provenance: dict[str, Any]
    policy: Phase1DSourcePolicy
    requirements: dict[str, Any] | None

    @property
    def run_id(self) -> str:
        return str(self.config.run_id)


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


def load_phase1d_source(
    run_dir: Path,
    policy: Phase1DSourcePolicy,
    *,
    requirements: Phase1DSourceRequirements | None = None,
) -> ValidatedPhase1DSource:
    """Validate and bind a Phase 1C source for Phase 1D draw access.

    Scientific Phase 1D posterior predictive execution requires a converged
    production source with current diagnostics. Development-only overrides are
    nonauthoritative and recorded in selection manifests.
    """
    config = load_phase1c_config(run_dir)
    diagnostics = _read_json(run_dir / "sampler_diagnostics.json")
    mode = str(diagnostics.get("mode", ""))
    _validate_diagnostics_identity(diagnostics, config)
    if policy.authoritative and requirements is None:
        raise ValueError("Authoritative Phase 1D source loading requires explicit source requirements.")
    requirements_record = requirements.validate(config, mode) if requirements is not None else None
    data, timing, input_provenance = _load_bound_data_and_timing(config, mode)

    metadata = checkpoint_metadata(data, config, mode=mode)
    sources = []
    seen_run_ids = set()
    seen_modes = set()
    for ensemble_index in range(config.n_ensembles):
        path = run_dir / f"ensemble_{ensemble_index:02d}.h5"
        validate_checkpoint_metadata(path, metadata, config.random_seed + 1000 * ensemble_index)
        source = _validate_hdf_source(path, config, expected_mode=mode)
        sources.append(source)
        seen_run_ids.add(source.run_id)
        seen_modes.add(source.mode)

    expected = set(range(config.n_ensembles))
    actual = {source.ensemble for source in sources}
    if actual != expected:
        raise ValueError(f"Phase 1C ensemble set mismatch: expected {sorted(expected)}, found {sorted(actual)}.")
    if seen_run_ids != {str(config.run_id)}:
        raise ValueError("Phase 1C HDF files contain mixed or unexpected run IDs.")
    if seen_modes != {mode}:
        raise ValueError("Phase 1C HDF files contain mixed or unexpected modes.")

    if mode != "production" and not policy.allow_nonproduction:
        raise ValueError(f"Authoritative Phase 1D draw access requires production mode, found {mode!r}.")
    status = str(diagnostics.get("status", ""))
    if policy.require_converged and status != "converged":
        raise ValueError(f"Authoritative Phase 1D draw access requires converged diagnostics, found {status!r}.")
    checkpoint_iterations = {source.path.name: source.steps for source in sources}
    if policy.authoritative:
        _validate_authoritative_diagnostics(run_dir, config, mode, diagnostics, sources)

    return ValidatedPhase1DSource(
        run_dir=run_dir,
        config=config,
        mode=mode,
        data=data,
        timing=timing,
        ensembles=tuple(sorted(sources, key=lambda source: source.ensemble)),
        diagnostics=diagnostics,
        checkpoint_iterations=checkpoint_iterations,
        input_provenance=input_provenance,
        policy=policy,
        requirements=requirements_record,
    )


def select_posterior_draws(
    source: ValidatedPhase1DSource,
    *,
    requested_draws: int,
    seed: int,
) -> DrawSelection:
    """Select deterministic ensemble-aware posterior draws without replacement.

    The selector samples retained post-warmup HDF positions only. It does not
    thin the posterior as an inference procedure; it merely chooses a smaller
    Monte Carlo subset for posterior-predictive replication while preserving
    ensemble, walker, and stored-step provenance.
    """
    if requested_draws <= 0:
        raise ValueError("requested_draws must be positive.")
    config = source.config
    sources = source.ensembles
    allocation = _draw_allocation(sources, requested_draws)
    rng = np.random.default_rng(seed)
    selected: list[SelectedPosteriorDraw] = []
    selected_by_ensemble = {}
    selected_by_walker: dict[str, int] = {}
    for ensemble_source in sources:
        count = allocation[ensemble_source.ensemble]
        if ensemble_source.eligible_count < count:
            raise ValueError(
                f"Ensemble {ensemble_source.ensemble} has {ensemble_source.eligible_count} eligible draws, "
                f"cannot select {count} without replacement."
            )
        flat_choices = rng.choice(ensemble_source.eligible_count, size=count, replace=False)
        selected_by_ensemble[str(ensemble_source.ensemble)] = int(count)
        with h5py.File(ensemble_source.path, "r") as hdf:
            chain = hdf["mcmc/chain"]
            log_prob = hdf["mcmc/log_prob"]
            for local_position in np.sort(flat_choices):
                retained_step = int(local_position // ensemble_source.walkers)
                walker = int(local_position % ensemble_source.walkers)
                step = int(ensemble_source.warmup_steps + retained_step)
                vector = np.asarray(chain[step, walker, :], dtype=float)
                value = float(log_prob[step, walker])
                if not np.all(np.isfinite(vector)) or not np.isfinite(value):
                    raise ValueError(
                        f"Selected nonfinite draw from {ensemble_source.path}: step={step}, walker={walker}."
                    )
                physical = vector_to_physical(vector, source.timing)
                selected_by_walker[f"{ensemble_source.ensemble}:{walker}"] = selected_by_walker.get(
                    f"{ensemble_source.ensemble}:{walker}", 0
                ) + 1
                selected.append(
                    SelectedPosteriorDraw(
                        run_id=ensemble_source.run_id,
                        mode=ensemble_source.mode,
                        ensemble=ensemble_source.ensemble,
                        ensemble_seed=ensemble_source.seed,
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
        "source_run_dir": str(source.run_dir),
        "source_run_id": config.run_id,
        "source_mode": selected[0].mode if selected else None,
        "authoritative": source.policy.authoritative,
        "nonproduction_override": source.policy.allow_nonproduction or not source.policy.require_converged,
        "override_reason": source.policy.override_reason,
        "source_requirements": source.requirements,
        "input_identity": source.input_provenance,
        "diagnostics_status": source.diagnostics.get("status"),
        "source_git": _source_git_summary(source.run_dir),
        "selection_seed": seed,
        "requested_draws": requested_draws,
        "selected_draws": len(selected),
        "eligible_counts_by_ensemble": {str(source.ensemble): source.eligible_count for source in sources},
        "selected_counts_by_ensemble": selected_by_ensemble,
        "selected_counts_by_walker": selected_by_walker,
        "checkpoint_iterations": source.checkpoint_iterations,
        "parameter_order": list(PARAMETER_ORDER),
    }
    return DrawSelection(
        source_run_dir=source.run_dir,
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


def _validate_hdf_source(path: Path, config: Phase1CConfig, *, expected_mode: str) -> EnsembleSource:
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
        accepted = hdf["mcmc/accepted"]
        if accepted.shape != (walkers,):
            raise ValueError(f"mcmc/accepted shape does not match walker count: {path}")
        accepted_values = np.asarray(accepted[:], dtype=float)
        if not np.all(np.isfinite(accepted_values)) or np.any(accepted_values < 0.0):
            raise ValueError(f"mcmc/accepted contains nonfinite or negative values: {path}")
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
        if mode != expected_mode:
            raise ValueError(f"Mode mismatch for {path}: {mode!r}")
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


def _validate_diagnostics_identity(diagnostics: dict[str, Any], config: Phase1CConfig) -> None:
    if str(diagnostics.get("run_id", "")) != str(config.run_id):
        raise ValueError("sampler_diagnostics.json run ID does not match Phase 1C configuration.")
    if str(diagnostics.get("mode", "")) not in {"pilot", "production", "synthetic", "synthetic_recovery"}:
        raise ValueError("sampler_diagnostics.json has missing or invalid mode.")


def _load_bound_data_and_timing(
    config: Phase1CConfig,
    mode: str,
) -> tuple[FrozenPhase1BData, TimingReference, dict[str, Any]]:
    if mode in {"synthetic", "synthetic_recovery"}:
        data, timing, _ = synthetic_dataset(config)
        _validate_synthetic_input_record_if_needed(data, timing, config, mode)
        return data, timing, {
            "kind": "synthetic",
            "manifest_sha256": data.input_manifest.get("manifest_sha256"),
            "synthetic_input_record": str(config.output_dir / "synthetic_input_record.json"),
        }
    data = load_frozen_phase1b(config)
    timing = build_timing_reference(data, config)
    return data, timing, {
        "kind": "phase1b",
        "manifest_sha256": data.input_manifest.get("manifest_sha256"),
        "phase1b_input_manifest": data.input_manifest,
    }


def _validate_authoritative_diagnostics(
    run_dir: Path,
    config: Phase1CConfig,
    mode: str,
    diagnostics: dict[str, Any],
    sources: list[EnsembleSource],
) -> None:
    if mode != "production":
        raise ValueError(f"Authoritative Phase 1D source mode must be production, found {mode!r}.")
    if str(diagnostics.get("status", "")) != "converged":
        raise ValueError("Authoritative Phase 1D source requires converged sampler_diagnostics.json.")
    iterations = {source.steps for source in sources}
    if len(iterations) != 1:
        raise ValueError("Authoritative Phase 1D source has unequal ensemble iteration counts.")
    current_steps = next(iter(iterations))
    row = _final_convergence_history_row(run_dir)
    if str(row.get("run_id", "")) != str(config.run_id):
        raise ValueError("convergence_history.csv run ID does not match Phase 1C configuration.")
    if str(row.get("mode", "")) != mode:
        raise ValueError("convergence_history.csv mode does not match Phase 1C source mode.")
    if int(row.get("completed_steps", -1)) != int(current_steps):
        raise ValueError("convergence_history.csv is stale relative to current HDF checkpoints.")
    if str(row.get("convergence_status", "")) != "converged":
        raise ValueError("Final convergence_history.csv row is not converged.")
    _validate_authoritative_diagnostic_criteria(config, diagnostics, row)


def _validate_authoritative_diagnostic_criteria(
    config: Phase1CConfig,
    diagnostics: dict[str, Any],
    final_history_row: dict[str, Any],
) -> None:
    criteria = diagnostics.get("criteria")
    if not isinstance(criteria, dict):
        raise ValueError("sampler_diagnostics.json lacks convergence criteria mapping.")
    required = (
        "complete_valid_rhat",
        "complete_valid_bulk_ess",
        "complete_valid_tail_ess",
        "rhat_all_below_threshold",
        "ess_all_above_minimum",
        "tail_ess_all_above_minimum",
        "complete_valid_autocorrelation",
        "chain_length_exceeds_tau_multiple",
        "posterior_summary_stability",
        "independent_ensemble_agreement",
        "finite_log_probability_fraction_is_one",
    )
    missing = [key for key in required if key not in criteria]
    if missing:
        raise ValueError(f"sampler_diagnostics.json missing convergence criteria: {missing}")
    false = [key for key in required if criteria[key] is not True]
    if false:
        raise ValueError(f"sampler_diagnostics.json convergence criteria are not all true: {false}")
    if float(diagnostics.get("finite_log_probability_fraction", -1.0)) != 1.0:
        raise ValueError("Authoritative diagnostics require finite_log_probability_fraction == 1.0.")
    if float(final_history_row.get("rhat_max", np.inf)) >= float(config.convergence_rhat_threshold):
        raise ValueError("Final convergence-history rhat_max does not satisfy saved threshold.")
    if float(final_history_row.get("bulk_ess_min", -np.inf)) < float(config.convergence_ess_minimum):
        raise ValueError("Final convergence-history bulk_ess_min does not satisfy saved minimum.")
    if float(final_history_row.get("tail_ess_min", -np.inf)) < float(config.convergence_ess_minimum):
        raise ValueError("Final convergence-history tail_ess_min does not satisfy saved minimum.")
    history_flags = {
        "posterior_stability_passed": "posterior stability",
        "independent_ensemble_agreement_passed": "independent ensemble agreement",
        "complete_valid_autocorrelation": "complete autocorrelation",
        "chain_length_exceeds_tau_multiple": "chain length autocorrelation multiple",
    }
    for field, label in history_flags.items():
        if _history_bool(final_history_row.get(field)) is not True:
            raise ValueError(f"Final convergence-history row failed {label}.")


def _final_convergence_history_row(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "convergence_history.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing convergence history: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Empty convergence history: {path}")
    return frame.iloc[-1].to_dict()


def _history_bool(value: Any) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def _source_git_summary(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "provenance_manifest.json"
    if not path.exists():
        return None
    try:
        git = _read_json(path).get("git")
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(git, dict):
        return None
    return {
        "commit": git.get("commit"),
        "is_dirty": git.get("is_dirty"),
    }


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
