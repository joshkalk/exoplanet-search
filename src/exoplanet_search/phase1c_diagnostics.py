"""Phase 1C posterior summaries and convergence diagnostics."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from .phase1c_parameters import physical_parameter_row, vector_to_physical
from .phase1c_types import PARAMETER_ORDER, Phase1CConfig, TimingReference


SUMMARY_QUANTILES = (0.025, 0.16, 0.5, 0.84, 0.975)


def posterior_summary_frame(
    chain: np.ndarray,
    timing: TimingReference,
    *,
    warmup_steps: int,
) -> pd.DataFrame:
    """Summarize transformed and physical posterior coordinates."""
    flat = post_warmup_flat_chain(chain, warmup_steps)
    columns: dict[str, np.ndarray] = {name: flat[:, index] for index, name in enumerate(PARAMETER_ORDER)}
    physical_rows = [physical_parameter_row(vector_to_physical(row, timing)) for row in flat]
    for key in physical_rows[0]:
        columns[key] = np.asarray([row[key] for row in physical_rows], dtype=float)
    rows = []
    for name, values in columns.items():
        quantiles = np.quantile(values, SUMMARY_QUANTILES)
        rows.append(
            {
                "parameter": name,
                "median": float(quantiles[2]),
                "q02_5": float(quantiles[0]),
                "q16": float(quantiles[1]),
                "q84": float(quantiles[3]),
                "q97_5": float(quantiles[4]),
                "mean": float(np.mean(values)),
                "sd": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def post_warmup_chain(chain: np.ndarray, warmup_steps: int) -> np.ndarray:
    """Return chain with explicit warmup discarded; input shape is chains, draws, ndim."""
    draws = chain.shape[1]
    discard = min(max(int(warmup_steps), 0), max(draws - 1, 0))
    return chain[:, discard:, :]


def post_warmup_flat_chain(chain: np.ndarray, warmup_steps: int) -> np.ndarray:
    kept = post_warmup_chain(chain, warmup_steps)
    return kept.reshape((-1, kept.shape[-1]))


def split_rhat(chain: np.ndarray) -> dict[str, float]:
    """Compute split R-hat for each sampled coordinate."""
    split = _split_chain(chain)
    result = {}
    for index, name in enumerate(PARAMETER_ORDER):
        values = split[:, :, index]
        result[name] = _rhat_1d(values)
    return result


def effective_sample_size(chain: np.ndarray) -> dict[str, float]:
    """Estimate bulk ESS with an initial-positive autocorrelation sum."""
    result = {}
    chains, draws, _ = chain.shape
    for index, name in enumerate(PARAMETER_ORDER):
        values = chain[:, :, index]
        tau = _integrated_time(values)
        result[name] = float(chains * draws / max(tau, 1.0))
    return result


def tail_effective_sample_size(chain: np.ndarray) -> dict[str, float]:
    """Approximate tail ESS by applying ESS to lower/upper tail indicators."""
    result = {}
    for index, name in enumerate(PARAMETER_ORDER):
        values = chain[:, :, index]
        low = np.quantile(values, 0.05)
        high = np.quantile(values, 0.95)
        low_ess = _indicator_ess(values <= low)
        high_ess = _indicator_ess(values >= high)
        result[name] = float(min(low_ess, high_ess))
    return result


def convergence_diagnostics(
    chain: np.ndarray,
    log_prob: np.ndarray,
    acceptance_fraction: np.ndarray,
    autocorr_time: np.ndarray | None,
    config: Phase1CConfig,
    *,
    warmup_steps: int,
) -> dict[str, Any]:
    """Return machine-readable convergence diagnostics for a combined run."""
    kept = post_warmup_chain(chain, warmup_steps)
    too_short = kept.shape[1] < 4 or kept.shape[0] < 2
    arviz_diagnostics = None if too_short else _arviz_diagnostics(kept)
    if too_short:
        rhat = {name: None for name in PARAMETER_ORDER}
        ess = {name: None for name in PARAMETER_ORDER}
        tail_ess = {name: None for name in PARAMETER_ORDER}
        diagnostic_backend = "unavailable_too_short"
    elif arviz_diagnostics is None:
        rhat = split_rhat(kept)
        ess = effective_sample_size(kept)
        tail_ess = tail_effective_sample_size(kept)
        diagnostic_backend = "internal_fallback"
    else:
        rhat = arviz_diagnostics["split_rhat"]
        ess = arviz_diagnostics["bulk_ess"]
        tail_ess = arviz_diagnostics["tail_ess"]
        diagnostic_backend = "arviz"
    finite_log_prob_fraction = float(np.mean(np.isfinite(log_prob)))
    tau = None if autocorr_time is None else [float(value) for value in np.ravel(autocorr_time)]
    tau_ok = False
    if autocorr_time is not None and np.all(np.isfinite(autocorr_time)):
        tau_ok = bool(kept.shape[1] > config.convergence_tau_multiple * float(np.nanmax(autocorr_time)))
    rhat_values = [value for value in rhat.values() if value is not None and np.isfinite(value)]
    ess_values = [value for value in ess.values() if value is not None and np.isfinite(value)]
    tail_ess_values = [value for value in tail_ess.values() if value is not None and np.isfinite(value)]
    criteria = {
        "rhat_all_below_threshold": bool(rhat_values)
        and all(value < config.convergence_rhat_threshold for value in rhat_values),
        "ess_all_above_minimum": bool(ess_values)
        and all(value >= config.convergence_ess_minimum for value in ess_values),
        "tail_ess_all_above_minimum": bool(tail_ess_values)
        and all(value >= config.convergence_ess_minimum for value in tail_ess_values),
        "chain_length_exceeds_tau_multiple": tau_ok,
        "finite_log_probability_fraction": finite_log_prob_fraction,
    }
    converged = all(value for key, value in criteria.items() if key != "finite_log_probability_fraction")
    return {
        "status": "converged" if converged else "nonconverged",
        "criteria": criteria,
        "split_rhat": rhat,
        "bulk_ess": ess,
        "tail_ess": tail_ess,
        "emcee_autocorrelation_time": tau,
        "acceptance_fraction": {
            "min": float(np.min(acceptance_fraction)),
            "median": float(np.median(acceptance_fraction)),
            "max": float(np.max(acceptance_fraction)),
        },
        "finite_log_probability_fraction": finite_log_prob_fraction,
        "warmup_steps_excluded": int(warmup_steps),
        "standard_diagnostic_backend": diagnostic_backend,
        "diagnostic_note": (
            "emcee walkers are treated as diagnostic chains for split R-hat/ESS; "
            "independent ensemble summaries are also recorded."
        ),
    }


def ensemble_summary_frame(chain_by_ensemble: list[np.ndarray], warmup_steps: int) -> pd.DataFrame:
    """Summarize independent ensemble agreement by transformed-coordinate medians."""
    rows = []
    for ensemble_index, chain in enumerate(chain_by_ensemble):
        flat = post_warmup_flat_chain(chain, warmup_steps)
        row = {"ensemble": ensemble_index, "draws": int(flat.shape[0])}
        for index, name in enumerate(PARAMETER_ORDER):
            row[f"{name}_median"] = float(np.median(flat[:, index]))
            row[f"{name}_q16"] = float(np.quantile(flat[:, index], 0.16))
            row[f"{name}_q84"] = float(np.quantile(flat[:, index], 0.84))
        rows.append(row)
    return pd.DataFrame(rows)


def _split_chain(chain: np.ndarray) -> np.ndarray:
    chains, draws, ndim = chain.shape
    even_draws = draws - draws % 2
    if even_draws < 4:
        return chain
    first = chain[:, : even_draws // 2, :]
    second = chain[:, even_draws // 2 : even_draws, :]
    return np.concatenate([first, second], axis=0).reshape((2 * chains, even_draws // 2, ndim))


def _rhat_1d(values: np.ndarray) -> float:
    chains, draws = values.shape
    if chains < 2 or draws < 2:
        return math.inf
    chain_means = np.mean(values, axis=1)
    chain_vars = np.var(values, axis=1, ddof=1)
    within = float(np.mean(chain_vars))
    if within <= 0.0:
        return 1.0
    between = draws * float(np.var(chain_means, ddof=1))
    var_hat = ((draws - 1.0) / draws) * within + between / draws
    return float(math.sqrt(max(var_hat / within, 0.0)))


def _integrated_time(values: np.ndarray) -> float:
    chains, draws = values.shape
    if draws < 3:
        return math.inf
    centered = values - np.mean(values, axis=1, keepdims=True)
    variance = float(np.var(centered))
    if variance <= 0.0:
        return 1.0
    rho_sum = 0.0
    max_lag = min(draws - 1, 100)
    for lag in range(1, max_lag + 1):
        acov = np.mean(centered[:, :-lag] * centered[:, lag:])
        rho = float(acov / variance)
        if rho <= 0.0:
            break
        rho_sum += rho
    return 1.0 + 2.0 * rho_sum


def _indicator_ess(indicator: np.ndarray) -> float:
    return float(np.prod(indicator.shape) / max(_integrated_time(indicator.astype(float)), 1.0))


def _arviz_diagnostics(chain: np.ndarray) -> dict[str, dict[str, float]] | None:
    try:
        import arviz as az
    except ImportError:
        return None
    posterior = {
        name: chain[:, :, index]
        for index, name in enumerate(PARAMETER_ORDER)
    }
    try:
        inference_data = az.from_dict(posterior=posterior)
        rhat = az.rhat(inference_data, method="rank")
        bulk = az.ess(inference_data, method="bulk")
        tail = az.ess(inference_data, method="tail")
    except (TypeError, ValueError, RuntimeError, AttributeError):
        return None
    return {
        "split_rhat": _dataset_to_dict(rhat),
        "bulk_ess": _dataset_to_dict(bulk),
        "tail_ess": _dataset_to_dict(tail),
    }


def _dataset_to_dict(dataset) -> dict[str, float]:
    values = {}
    for name in PARAMETER_ORDER:
        item = dataset[name].values
        values[name] = float(np.asarray(item).reshape(-1)[0])
    return values
