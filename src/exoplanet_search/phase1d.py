"""Phase 1D development-only posterior-predictive foundation orchestration."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .phase1c import synthetic_dataset
from .phase1c_inputs import load_frozen_phase1b
from .phase1c_parameters import build_timing_reference
from .phase1d_draws import (
    Phase1DSourcePolicy,
    load_phase1c_config,
    select_posterior_draws,
    write_draw_selection,
)
from .phase1d_predictive import generate_replicated_flux, write_development_predictive_output


@dataclass(frozen=True)
class Phase1DDevelopmentConfig:
    """Bounded nonauthoritative Phase 1D development predictive configuration."""

    source_run_dir: Path
    output_dir: Path = Path("data/interim/kepler5_phase1d")
    run_id: str | None = None
    n_draws: int = 2
    selection_seed: int = 2026071701
    predictive_seed: int = 2026071702
    allow_nonproduction_source: bool = True


def run_phase1d_development_predictive(config: Phase1DDevelopmentConfig) -> dict[str, Any]:
    """Run a tiny nonauthoritative posterior-predictive development check.

    This path exists only to validate Phase 1D draw plumbing, conditional
    baseline draws, and cadence-aligned replicated flux. It cannot produce an
    authoritative Phase 1D posterior-predictive result.
    """
    run_id = _run_id(config)
    output_dir = config.output_dir / f"development_predictive_{run_id}"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing Phase 1D development output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    phase1c_config = load_phase1c_config(config.source_run_dir)
    data, timing = _data_for_source(phase1c_config)
    policy = Phase1DSourcePolicy(
        require_converged=False,
        allow_nonproduction=config.allow_nonproduction_source,
        authoritative=False,
        override_reason="development-only predictive plumbing validation",
    )
    selection = select_posterior_draws(
        config.source_run_dir,
        timing=timing,
        requested_draws=config.n_draws,
        seed=config.selection_seed,
        policy=policy,
    )
    write_draw_selection(output_dir, selection)
    rng = np.random.default_rng(config.predictive_seed)
    predictive_rows = []
    baseline_rows = []
    for draw in selection.selected_draws:
        rows, baseline = generate_replicated_flux(draw, data, phase1c_config, timing, rng)
        predictive_rows.append(rows)
        baseline_rows.extend(baseline)
    config_payload = {
        "schema_version": "phase1d_development_predictive_v1",
        "authoritative": False,
        "nonproduction_label": "DEVELOPMENT_ONLY_NOT_AUTHORITATIVE",
        "source_run_dir": str(config.source_run_dir),
        "source_run_id": selection.run_id,
        "source_mode": selection.mode,
        "output_dir": str(output_dir),
        "run_id": run_id,
        "n_draws": config.n_draws,
        "selection_seed": config.selection_seed,
        "predictive_seed": config.predictive_seed,
        "cadence_count": data.cadence_count,
        "event_count": data.event_count,
        "residual_resampling_used": False,
        "predictive_equation": "replicated_flux = [m, m*x] beta_draw + Normal(0, sigma_i^2 + jitter^2)",
        "baseline_distribution": "beta | y, theta ~ Normal(baseline_mean, baseline_covariance)",
    }
    write_development_predictive_output(
        output_dir,
        predictive_rows=predictive_rows,
        baseline_audit=baseline_rows,
        config_payload=config_payload,
    )
    result = {
        "status": "development_predictive_complete",
        "authoritative": False,
        "run_id": run_id,
        "output_dir": str(output_dir),
        "selected_draws": len(selection.selected_draws),
        "cadence_count": data.cadence_count,
        "baseline_draw_count": len(baseline_rows),
        "files": {
            "configuration": str(output_dir / "phase1d_predictive_configuration.json"),
            "selection_manifest": str(output_dir / "posterior_draw_selection_manifest.json"),
            "selected_draw_audit": str(output_dir / "selected_draw_audit.jsonl"),
            "baseline_audit": str(output_dir / "event_baseline_draw_audit.jsonl"),
            "predictive_flux": str(output_dir / "development_predictive_flux.npz"),
        },
    }
    with (output_dir / "development_predictive_summary.json").open("w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2)
    return result


def _data_for_source(phase1c_config):
    mode = _read_source_mode(phase1c_config.output_dir)
    if mode in {"synthetic", "synthetic_recovery"}:
        data, timing, _ = synthetic_dataset(phase1c_config)
        return data, timing
    data = load_frozen_phase1b(phase1c_config)
    return data, build_timing_reference(data, phase1c_config)


def _read_source_mode(run_dir: Path) -> str:
    path = run_dir / "sampler_diagnostics.json"
    with path.open("r", encoding="utf-8") as input_file:
        payload = json.load(input_file)
    return str(payload.get("mode", ""))


def _run_id(config: Phase1DDevelopmentConfig) -> str:
    if config.run_id:
        cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", config.run_id.strip())
        if cleaned:
            return cleaned
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
