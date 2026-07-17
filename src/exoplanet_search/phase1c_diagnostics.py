"""Phase 1C posterior summaries and convergence diagnostics."""

from __future__ import annotations

import math
from typing import Any

import emcee
import numpy as np
import pandas as pd
from scipy.special import ndtri

from .phase1c_parameters import physical_parameter_row, vector_to_physical
from .phase1c_types import DIAGNOSTIC_METHODOLOGY_VERSION, PARAMETER_ORDER, Phase1CConfig, TimingReference


SUMMARY_QUANTILES = (0.025, 0.05, 0.16, 0.5, 0.84, 0.95, 0.975)
ENSEMBLE_STATE_STATISTICS = ("mean", "median", "sd", "q05", "q95")


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
                "median": float(quantiles[3]),
                "q02_5": float(quantiles[0]),
                "q05": float(quantiles[1]),
                "q16": float(quantiles[2]),
                "q84": float(quantiles[4]),
                "q95": float(quantiles[5]),
                "q97_5": float(quantiles[6]),
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


def post_warmup_log_prob(log_prob: np.ndarray, warmup_steps: int) -> np.ndarray:
    draws = log_prob.shape[1]
    discard = min(max(int(warmup_steps), 0), max(draws - 1, 0))
    return log_prob[:, discard:]


def independent_ensemble_state_rhat(ensemble_chain: np.ndarray) -> dict[str, Any]:
    """Rank-normalized split R-hat on independent ensemble state summaries."""
    rows = []
    worst_by_parameter = {}
    complete = True
    for parameter_index, parameter in enumerate(PARAMETER_ORDER):
        values = ensemble_chain[:, :, :, parameter_index]
        statistic_values = {}
        for statistic in ENSEMBLE_STATE_STATISTICS:
            trajectories = _ensemble_state_trajectories(values, statistic)
            value = _rank_normalized_split_rhat(trajectories)
            statistic_values[statistic] = value
            rows.append(
                {
                    "parameter": parameter,
                    "statistic": statistic,
                    "rhat": value,
                    "available": bool(value is not None and np.isfinite(value)),
                }
            )
        valid = [float(value) for value in statistic_values.values() if value is not None and np.isfinite(value)]
        worst_by_parameter[parameter] = max(valid) if len(valid) == len(ENSEMBLE_STATE_STATISTICS) else None
        complete = complete and worst_by_parameter[parameter] is not None
    return {
        "method": (
            "rank-normalized split R-hat across independent ensembles after reducing each "
            "ensemble state at each iteration to walker-permutation-invariant summaries"
        ),
        "statistics": list(ENSEMBLE_STATE_STATISTICS),
        "rows": rows,
        "worst_by_parameter": worst_by_parameter,
        "maximum": _max_or_none(worst_by_parameter.values()),
        "complete": bool(complete),
    }


def emcee_rank_bulk_ess(
    ensemble_chain: np.ndarray,
    *,
    min_usable_walkers: int = 2,
) -> dict[str, Any]:
    """Estimate bulk ESS from rank-normalized emcee ensemble autocorrelation times."""
    rows = []
    combined = {}
    all_available = True
    for parameter_index, parameter in enumerate(PARAMETER_ORDER):
        ranked = _rank_normalize_values(ensemble_chain[:, :, :, parameter_index])
        parameter_ess = []
        for ensemble_index in range(ranked.shape[0]):
            estimate = _emcee_tau(
                ranked[ensemble_index].T,
                min_usable_walkers=min_usable_walkers,
            )
            row = {
                "ensemble": int(ensemble_index),
                "parameter": parameter,
                "tau": estimate["tau"],
                "available": estimate["available"],
                "retained_steps": estimate["retained_steps"],
                "total_walkers": estimate["total_walkers"],
                "usable_walkers": estimate["usable_walkers"],
                "retained_samples": estimate["retained_samples_used"],
                "total_retained_samples": int(ranked.shape[1] * ranked.shape[2]),
                "ess": None,
                "error": estimate["error"],
            }
            if estimate["available"]:
                samples_used = int(estimate["retained_samples_used"])
                row["ess"] = float(min(samples_used / float(estimate["tau"]), samples_used))
                parameter_ess.append(float(row["ess"]))
            rows.append(row)
        if len(parameter_ess) == ranked.shape[0]:
            combined[parameter] = float(np.sum(parameter_ess))
        else:
            combined[parameter] = None
            all_available = False
    return {
        "method": "rank-normalized parameter values with emcee integrated autocorrelation per independent ensemble",
        "rows": rows,
        "combined_by_parameter": combined,
        "minimum": _min_or_none(combined.values()),
        "all_available": bool(all_available),
    }


def emcee_tail_ess(
    ensemble_chain: np.ndarray,
    *,
    min_usable_walkers: int = 2,
) -> dict[str, Any]:
    """Estimate lower/upper tail ESS from pooled tail-indicator processes."""
    rows = []
    combined = {}
    all_available = True
    for parameter_index, parameter in enumerate(PARAMETER_ORDER):
        values = ensemble_chain[:, :, :, parameter_index]
        lower_threshold = float(np.quantile(values, 0.05))
        upper_threshold = float(np.quantile(values, 0.95))
        lower_ess = []
        upper_ess = []
        for ensemble_index in range(values.shape[0]):
            for tail, indicator in (
                ("lower", values[ensemble_index] <= lower_threshold),
                ("upper", values[ensemble_index] >= upper_threshold),
            ):
                estimate = _emcee_tau(
                    indicator.T.astype(float),
                    min_usable_walkers=min_usable_walkers,
                )
                row = {
                    "ensemble": int(ensemble_index),
                    "parameter": parameter,
                    "tail": tail,
                    "threshold": lower_threshold if tail == "lower" else upper_threshold,
                    "tau": estimate["tau"],
                    "available": estimate["available"],
                    "retained_steps": estimate["retained_steps"],
                    "total_walkers": estimate["total_walkers"],
                    "usable_walkers": estimate["usable_walkers"],
                    "retained_samples": estimate["retained_samples_used"],
                    "total_retained_samples": int(values.shape[1] * values.shape[2]),
                    "ess": None,
                    "error": estimate["error"],
                }
                if estimate["available"]:
                    samples_used = int(estimate["retained_samples_used"])
                    row["ess"] = float(min(samples_used / float(estimate["tau"]), samples_used))
                    if tail == "lower":
                        lower_ess.append(float(row["ess"]))
                    else:
                        upper_ess.append(float(row["ess"]))
                rows.append(row)
        if len(lower_ess) == values.shape[0] and len(upper_ess) == values.shape[0]:
            combined[parameter] = float(min(np.sum(lower_ess), np.sum(upper_ess)))
        else:
            combined[parameter] = None
            all_available = False
    return {
        "method": "pooled 5th/95th percentile tail indicators with emcee integrated autocorrelation per ensemble",
        "rows": rows,
        "combined_by_parameter": combined,
        "minimum": _min_or_none(combined.values()),
        "all_available": bool(all_available),
    }


def walker_health_diagnostics(
    ensemble_chain: np.ndarray,
    ensemble_log_prob: np.ndarray,
    acceptance_fraction: np.ndarray,
    config: Phase1CConfig,
    *,
    warmup_steps: int,
) -> dict[str, Any]:
    """Return per-walker pathology diagnostics and severe-walker gating summary."""
    kept_chain = _post_warmup_ensemble_array(ensemble_chain, warmup_steps)
    kept_log_prob = _post_warmup_ensemble_array(ensemble_log_prob, warmup_steps)
    final_states = kept_chain[:, :, -1, :].reshape((-1, kept_chain.shape[-1]))
    center = np.median(final_states, axis=0)
    scale = 1.4826 * np.median(np.abs(final_states - center), axis=0)
    scale = np.maximum(scale, 1.0e-8)
    rows = []
    severe = []
    for ensemble_index in range(kept_chain.shape[0]):
        finite_logp = kept_log_prob[ensemble_index][np.isfinite(kept_log_prob[ensemble_index])]
        ensemble_reference = float(np.percentile(finite_logp, 90)) if finite_logp.size else math.inf
        for walker_index in range(kept_chain.shape[1]):
            path = kept_chain[ensemble_index, walker_index]
            logp_path = kept_log_prob[ensemble_index, walker_index]
            unchanged = (
                np.all(np.diff(path, axis=0) == 0.0, axis=1)
                if path.shape[0] > 1
                else np.asarray([], dtype=bool)
            )
            repeated_fraction = float(np.mean(unchanged)) if unchanged.size else 0.0
            longest = _longest_unchanged_run(unchanged)
            finite = logp_path[np.isfinite(logp_path)]
            median_logp = float(np.median(finite)) if finite.size else -math.inf
            max_logp = float(np.max(finite)) if finite.size else -math.inf
            deficit = float(ensemble_reference - median_logp) if np.isfinite(median_logp) else math.inf
            transformed_medians = np.median(path, axis=0)
            final_distance = float(np.linalg.norm((path[-1] - center) / scale))
            acceptance = float(acceptance_fraction[ensemble_index, walker_index])
            low_acceptance = acceptance <= config.severe_walker_acceptance_max
            nearly_unchanged = repeated_fraction >= config.severe_walker_repeated_fraction_min
            extreme_deficit = deficit >= config.severe_walker_logp_deficit_min
            extreme_distance = final_distance >= config.severe_walker_final_distance_min
            severe_rule = bool(
                (low_acceptance and nearly_unchanged)
                or (nearly_unchanged and extreme_deficit and extreme_distance)
            )
            row = {
                "ensemble": int(ensemble_index),
                "walker": int(walker_index),
                "acceptance_fraction": acceptance,
                "repeated_state_fraction": repeated_fraction,
                "longest_unchanged_run": int(longest),
                "median_log_posterior": median_logp,
                "maximum_log_posterior": max_logp,
                "log_posterior_deficit_from_ensemble_p90": deficit,
                "transformed_parameter_medians": {
                    name: float(transformed_medians[index]) for index, name in enumerate(PARAMETER_ORDER)
                },
                "final_state_robust_distance": final_distance,
                "flags": {
                    "low_acceptance": bool(low_acceptance),
                    "nearly_unchanged": bool(nearly_unchanged),
                    "extreme_log_posterior_deficit": bool(extreme_deficit),
                    "extreme_final_distance": bool(extreme_distance),
                },
                "severe": severe_rule,
            }
            rows.append(row)
            if severe_rule:
                severe.append({"ensemble": int(ensemble_index), "walker": int(walker_index), "reason": row["flags"]})
    return {
        "method": (
            "severe pathology is flagged for frozen low-acceptance walkers, or for "
            "combined stasis/log-posterior-deficit/final-distance evidence"
        ),
        "thresholds": {
            "acceptance_fraction_max": config.severe_walker_acceptance_max,
            "repeated_state_fraction_min": config.severe_walker_repeated_fraction_min,
            "log_posterior_deficit_min": config.severe_walker_logp_deficit_min,
            "final_state_robust_distance_min": config.severe_walker_final_distance_min,
        },
        "rows": rows,
        "severe_walkers": severe,
        "severe_walker_count": int(len(severe)),
        "passed": len(severe) == 0,
    }


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
    autocorr_time: dict[str, Any] | None,
    config: Phase1CConfig,
    *,
    warmup_steps: int,
    posterior_stability: dict[str, Any] | None = None,
    ensemble_agreement: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return machine-readable convergence diagnostics for a combined run."""
    _validate_diagnostic_array_shapes(chain, log_prob, acceptance_fraction, config)
    kept = post_warmup_chain(chain, warmup_steps)
    chain_by_ensemble = _reshape_walkers_by_ensemble(chain, config)
    kept_by_ensemble = _reshape_walkers_by_ensemble(kept, config)
    log_prob_by_ensemble = _reshape_walkers_by_ensemble(log_prob, config)
    acceptance_by_ensemble = _reshape_acceptance_by_ensemble(acceptance_fraction, config)
    too_short = kept.shape[1] < 4 or kept_by_ensemble.shape[0] < 2
    rhat_report = independent_ensemble_state_rhat(kept_by_ensemble)
    ess_report = emcee_rank_bulk_ess(
        kept_by_ensemble,
        min_usable_walkers=config.autocorrelation_min_usable_walkers,
    )
    tail_ess_report = emcee_tail_ess(
        kept_by_ensemble,
        min_usable_walkers=config.autocorrelation_min_usable_walkers,
    )
    walker_health = walker_health_diagnostics(
        chain_by_ensemble,
        log_prob_by_ensemble,
        acceptance_by_ensemble,
        config,
        warmup_steps=warmup_steps,
    )
    if too_short:
        rhat = {name: None for name in PARAMETER_ORDER}
        ess = {name: None for name in PARAMETER_ORDER}
        tail_ess = {name: None for name in PARAMETER_ORDER}
    else:
        rhat = rhat_report["worst_by_parameter"]
        ess = ess_report["combined_by_parameter"]
        tail_ess = tail_ess_report["combined_by_parameter"]
    legacy = None if too_short else _legacy_walker_as_chain_diagnostics(kept)
    finite_log_prob_fraction = float(np.mean(np.isfinite(log_prob)))
    tau_rows = [] if autocorr_time is None else list(autocorr_time.get("rows", []))
    worst_tau = None if autocorr_time is None else autocorr_time.get("worst_tau")
    tau_ok = False
    tau_complete = bool(autocorr_time and autocorr_time.get("all_available") and worst_tau is not None)
    if tau_complete and np.isfinite(float(worst_tau)):
        tau_ok = all(
            int(row.get("retained_steps", 0)) > config.convergence_tau_multiple * float(worst_tau)
            for row in tau_rows
        )
    rhat_values = [value for value in rhat.values() if value is not None and np.isfinite(value)]
    ess_values = [value for value in ess.values() if value is not None and np.isfinite(value)]
    tail_ess_values = [value for value in tail_ess.values() if value is not None and np.isfinite(value)]
    rhat_complete = len(rhat_values) == len(PARAMETER_ORDER)
    ess_complete = len(ess_values) == len(PARAMETER_ORDER)
    tail_ess_complete = len(tail_ess_values) == len(PARAMETER_ORDER)
    stability_pass = bool(posterior_stability and posterior_stability.get("passed"))
    ensemble_pass = bool(ensemble_agreement and ensemble_agreement.get("passed"))
    walker_health_pass = int(walker_health["severe_walker_count"]) == 0
    criteria = {
        "complete_valid_rhat": rhat_complete,
        "complete_valid_bulk_ess": ess_complete,
        "complete_valid_tail_ess": tail_ess_complete,
        "rhat_all_below_threshold": rhat_complete
        and all(value < config.convergence_rhat_threshold for value in rhat_values),
        "ess_all_above_minimum": ess_complete
        and all(value >= config.convergence_ess_minimum for value in ess_values),
        "tail_ess_all_above_minimum": tail_ess_complete
        and all(value >= config.convergence_ess_minimum for value in tail_ess_values),
        "complete_valid_autocorrelation": tau_complete,
        "chain_length_exceeds_tau_multiple": tau_ok,
        "posterior_summary_stability": stability_pass,
        "independent_ensemble_agreement": ensemble_pass,
        "finite_log_probability_fraction_is_one": finite_log_prob_fraction == 1.0,
        "no_severe_walker_pathology": walker_health_pass,
    }
    converged = all(criteria.values())
    return {
        "status": "converged" if converged else "nonconverged",
        "diagnostic_methodology_version": DIAGNOSTIC_METHODOLOGY_VERSION,
        "criteria": criteria,
        "split_rhat": rhat,
        "bulk_ess": ess,
        "tail_ess": tail_ess,
        "ensemble_state_rhat": rhat_report,
        "emcee_bulk_ess": ess_report,
        "emcee_tail_ess": tail_ess_report,
        "legacy_walker_as_chain_diagnostics": legacy,
        "walker_health": walker_health,
        "emcee_autocorrelation_time": autocorr_time
        or {"rows": [], "all_available": False, "worst_tau": None, "unavailable_count": len(PARAMETER_ORDER)},
        "autocorrelation_worst_tau": None if worst_tau is None else float(worst_tau),
        "acceptance_fraction": {
            "min": float(np.min(acceptance_fraction)),
            "median": float(np.median(acceptance_fraction)),
            "max": float(np.max(acceptance_fraction)),
        },
        "finite_log_probability_fraction": finite_log_prob_fraction,
        "warmup_steps_excluded": int(warmup_steps),
        "standard_diagnostic_backend": DIAGNOSTIC_METHODOLOGY_VERSION,
        "posterior_stability": posterior_stability or {"passed": False, "reason": "not_evaluated"},
        "independent_ensemble_agreement": ensemble_agreement or {"passed": False, "reason": "not_evaluated"},
        "diagnostic_note": (
            "Gating R-hat uses rank-normalized split R-hat on permutation-invariant "
            "independent-ensemble state summaries. Gating ESS uses emcee integrated "
            "autocorrelation estimates within independent ensembles. Walker-as-chain "
            "diagnostics are retained only as non-gating legacy comparisons."
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


def posterior_stability_check(
    summary_history: list[pd.DataFrame],
    config: Phase1CConfig,
) -> dict[str, Any]:
    """Check stability of medians and interval endpoints over final monitoring chunks."""
    if len(summary_history) < config.convergence_stability_chunks:
        return {"passed": False, "reason": "insufficient_history", "chunks_available": len(summary_history)}
    recent = summary_history[-config.convergence_stability_chunks :]
    failures = []
    parameters = set(recent[-1]["parameter"])
    fields = ("median", "q16", "q84", "q02_5", "q97_5")
    for parameter in sorted(parameters):
        latest_row = recent[-1][recent[-1]["parameter"] == parameter]
        if latest_row.empty:
            failures.append({"parameter": parameter, "reason": "missing_latest"})
            continue
        latest = latest_row.iloc[0]
        scale = max(float(abs(latest["q84"] - latest["q16"])), float(latest.get("sd", 0.0)), 1.0e-12)
        for frame in recent[:-1]:
            row = frame[frame["parameter"] == parameter]
            if row.empty:
                failures.append({"parameter": parameter, "reason": "missing_history"})
                continue
            previous = row.iloc[0]
            for field in fields:
                standardized_change = abs(float(latest[field]) - float(previous[field])) / scale
                if standardized_change > config.convergence_stability_sigma_threshold:
                    failures.append(
                        {
                            "parameter": parameter,
                            "field": field,
                            "standardized_change": float(standardized_change),
                        }
                    )
    return {
        "passed": not failures,
        "failures": failures,
        "chunks_evaluated": len(recent),
        "threshold": config.convergence_stability_sigma_threshold,
    }


def independent_ensemble_agreement(
    chain_by_ensemble: list[np.ndarray],
    timing: TimingReference,
    config: Phase1CConfig,
    *,
    warmup_steps: int,
) -> dict[str, Any]:
    """Check independent ensembles for shifted or poorly overlapping populations."""
    if len(chain_by_ensemble) < 2:
        return {"passed": False, "reason": "fewer_than_two_ensembles"}
    summaries = [
        posterior_summary_frame(chain, timing, warmup_steps=warmup_steps)
        for chain in chain_by_ensemble
    ]
    combined = posterior_summary_frame(np.concatenate(chain_by_ensemble, axis=0), timing, warmup_steps=warmup_steps)
    failures = []
    for _, combined_row in combined.iterrows():
        parameter = combined_row["parameter"]
        scale = max(float(combined_row["q84"] - combined_row["q16"]), float(combined_row["sd"]), 1.0e-12)
        combined_median = float(combined_row["median"])
        for ensemble_index, summary in enumerate(summaries):
            row = summary[summary["parameter"] == parameter].iloc[0]
            shift = abs(float(row["median"]) - combined_median) / scale
            overlap = _interval_overlap_fraction(
                float(row["q16"]),
                float(row["q84"]),
                float(combined_row["q16"]),
                float(combined_row["q84"]),
            )
            tail_overlap = _interval_overlap_fraction(
                float(row["q05"]),
                float(row["q95"]),
                float(combined_row["q05"]),
                float(combined_row["q95"]),
            )
            scale_ratio = _scale_ratio(float(row["sd"]), float(combined_row["sd"]))
            if shift > config.convergence_ensemble_shift_threshold:
                failures.append(
                    {
                        "ensemble": ensemble_index,
                        "parameter": parameter,
                        "reason": "median_shift",
                        "standardized_shift": float(shift),
                    }
                )
            if overlap < config.convergence_interval_overlap_minimum:
                failures.append(
                    {
                        "ensemble": ensemble_index,
                        "parameter": parameter,
                        "reason": "interval_overlap",
                        "overlap_fraction": float(overlap),
                    }
                )
            if tail_overlap < config.convergence_tail_interval_overlap_minimum:
                failures.append(
                    {
                        "ensemble": ensemble_index,
                        "parameter": parameter,
                        "reason": "tail_interval_overlap",
                        "overlap_fraction": float(tail_overlap),
                    }
                )
            if scale_ratio > config.convergence_ensemble_scale_ratio_max:
                failures.append(
                    {
                        "ensemble": ensemble_index,
                        "parameter": parameter,
                        "reason": "scale_mismatch",
                        "scale_ratio": float(scale_ratio),
                    }
                )
    return {
        "passed": not failures,
        "failures": failures,
        "median_shift_threshold": config.convergence_ensemble_shift_threshold,
        "interval_overlap_minimum": config.convergence_interval_overlap_minimum,
        "tail_interval_overlap_minimum": config.convergence_tail_interval_overlap_minimum,
        "scale_ratio_maximum": config.convergence_ensemble_scale_ratio_max,
    }


def _reshape_walkers_by_ensemble(values: np.ndarray, config: Phase1CConfig) -> np.ndarray:
    array = np.asarray(values)
    n_ensembles = int(config.n_ensembles)
    n_walkers = int(config.n_walkers)
    expected_walkers = n_ensembles * n_walkers
    if n_ensembles <= 0 or n_walkers <= 0:
        raise ValueError("Phase 1C diagnostic configuration must declare positive ensembles and walkers.")
    if array.ndim not in {1, 2, 3}:
        raise ValueError(f"Diagnostic array must be 1D, 2D, or 3D; got shape {array.shape}.")
    if array.shape[0] != expected_walkers:
        raise ValueError(
            "Diagnostic array walker axis does not match configuration: "
            f"got {array.shape[0]}, expected {expected_walkers} "
            f"({n_ensembles} ensembles x {n_walkers} walkers)."
        )
    if array.ndim == 3 and array.shape[2] != len(PARAMETER_ORDER):
        raise ValueError(
            "Diagnostic chain parameter dimension does not match PARAMETER_ORDER: "
            f"got {array.shape[2]}, expected {len(PARAMETER_ORDER)}."
        )
    return array.reshape((n_ensembles, n_walkers, *array.shape[1:]))


def _reshape_acceptance_by_ensemble(values: np.ndarray, config: Phase1CConfig) -> np.ndarray:
    return _reshape_walkers_by_ensemble(np.asarray(values, dtype=float), config)


def _validate_diagnostic_array_shapes(
    chain: np.ndarray,
    log_prob: np.ndarray,
    acceptance_fraction: np.ndarray,
    config: Phase1CConfig,
) -> None:
    chain_array = np.asarray(chain)
    log_prob_array = np.asarray(log_prob)
    acceptance_array = np.asarray(acceptance_fraction)
    expected_walkers = int(config.n_ensembles) * int(config.n_walkers)
    expected_ndim = len(PARAMETER_ORDER)
    if chain_array.ndim != 3:
        raise ValueError(f"Diagnostic chain must have shape (walkers, draws, ndim); got {chain_array.shape}.")
    if chain_array.shape[0] != expected_walkers:
        raise ValueError(
            "Diagnostic chain walker count does not match configuration: "
            f"got {chain_array.shape[0]}, expected {expected_walkers}."
        )
    if chain_array.shape[2] != expected_ndim:
        raise ValueError(
            "Diagnostic chain parameter dimension does not match PARAMETER_ORDER: "
            f"got {chain_array.shape[2]}, expected {expected_ndim}."
        )
    if log_prob_array.shape != chain_array.shape[:2]:
        raise ValueError(
            "Diagnostic log-probability shape must match chain walker/draw axes: "
            f"got {log_prob_array.shape}, expected {chain_array.shape[:2]}."
        )
    if acceptance_array.shape != (expected_walkers,):
        raise ValueError(
            "Diagnostic acceptance-fraction shape does not match configuration: "
            f"got {acceptance_array.shape}, expected {(expected_walkers,)}."
        )


def _post_warmup_ensemble_array(values: np.ndarray, warmup_steps: int) -> np.ndarray:
    draws = values.shape[2]
    discard = min(max(int(warmup_steps), 0), max(draws - 1, 0))
    return values[:, :, discard:, ...]


def _ensemble_state_trajectories(values: np.ndarray, statistic: str) -> np.ndarray:
    if statistic == "mean":
        return np.mean(values, axis=1)
    if statistic == "median":
        return np.median(values, axis=1)
    if statistic == "sd":
        return np.std(values, axis=1, ddof=1) if values.shape[1] > 1 else np.zeros(values.shape[::2])
    if statistic == "q05":
        return np.quantile(values, 0.05, axis=1)
    if statistic == "q95":
        return np.quantile(values, 0.95, axis=1)
    raise ValueError(f"Unknown ensemble-state statistic {statistic!r}.")


def _rank_normalized_split_rhat(values: np.ndarray) -> float | None:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[0] < 2 or array.shape[1] < 4 or not np.all(np.isfinite(array)):
        return None
    ranked = _rank_normalize_values(array)
    chains, draws = ranked.shape
    even_draws = draws - draws % 2
    if even_draws < 4:
        return None
    split = np.concatenate(
        [ranked[:, : even_draws // 2], ranked[:, even_draws // 2 : even_draws]],
        axis=0,
    )
    split_variances = np.var(split, axis=1, ddof=1)
    if not np.all(np.isfinite(split_variances)) or np.any(split_variances <= 0.0):
        return None
    return _rhat_1d(split)


def _rank_normalize_values(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    flat = array.reshape(-1)
    if not np.all(np.isfinite(flat)):
        return np.full_like(array, np.nan, dtype=float)
    order = np.argsort(flat, kind="mergesort")
    ranks = np.empty(flat.size, dtype=float)
    sorted_values = flat[order]
    start = 0
    while start < flat.size:
        stop = start + 1
        while stop < flat.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        average_rank = 0.5 * (start + stop - 1) + 1.0
        ranks[order[start:stop]] = average_rank
        start = stop
    probabilities = (ranks - 0.375) / (flat.size + 0.25)
    return ndtri(probabilities).reshape(array.shape)


def _emcee_tau(values: np.ndarray, *, min_usable_walkers: int) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    base = {
        "tau": None,
        "available": False,
        "error": None,
        "retained_steps": int(array.shape[0]) if array.ndim >= 1 else 0,
        "total_walkers": int(array.shape[1]) if array.ndim == 2 else 0,
        "usable_walkers": 0,
        "retained_samples_used": 0,
    }
    if array.ndim != 2 or array.shape[0] < 3 or not np.all(np.isfinite(array)):
        return {**base, "error": "invalid_or_too_short"}
    column_variance = np.var(array, axis=0)
    usable = np.isfinite(column_variance) & (column_variance > 1.0e-14)
    usable_count = int(np.sum(usable))
    base["usable_walkers"] = usable_count
    base["retained_samples_used"] = int(array.shape[0] * usable_count)
    if usable_count < int(min_usable_walkers):
        return {**base, "error": "too_few_varying_walkers"}
    array = array[:, usable]
    if float(np.var(array)) <= 0.0:
        return {**base, "error": "zero_variance_process"}
    try:
        tau = float(np.asarray(emcee.autocorr.integrated_time(array, quiet=False, has_walkers=True)).reshape(-1)[0])
    except (emcee.autocorr.AutocorrError, ValueError, FloatingPointError, IndexError) as exc:
        return {**base, "error": str(exc)}
    if not np.isfinite(tau) or tau <= 0.0:
        return {**base, "error": "nonfinite_or_nonpositive_tau"}
    return {**base, "tau": tau, "available": True, "error": None}


def _longest_unchanged_run(unchanged: np.ndarray) -> int:
    longest = 1
    current = 1
    for value in unchanged:
        if bool(value):
            current += 1
            longest = max(longest, current)
        else:
            current = 1
    return int(longest)


def _max_or_none(values) -> float | None:
    valid = [float(value) for value in values if value is not None and np.isfinite(value)]
    return max(valid) if valid else None


def _min_or_none(values) -> float | None:
    valid = [float(value) for value in values if value is not None and np.isfinite(value)]
    return min(valid) if valid else None


def _legacy_walker_as_chain_diagnostics(chain: np.ndarray) -> dict[str, Any]:
    arviz = _arviz_diagnostics(chain)
    if arviz is None:
        arviz = {
            "split_rhat": split_rhat(chain),
            "bulk_ess": effective_sample_size(chain),
            "tail_ess": tail_effective_sample_size(chain),
        }
        backend = "internal_walker_as_chain_fallback"
    else:
        backend = "arviz_walker_as_chain"
    return {
        "gating": False,
        "backend": backend,
        "warning": "Interacting emcee walkers are not independent chains; these values are non-gating.",
        **arviz,
    }


def _split_chain(chain: np.ndarray) -> np.ndarray:
    chains, draws, ndim = chain.shape
    even_draws = draws - draws % 2
    if even_draws < 4:
        return chain
    first = chain[:, : even_draws // 2, :]
    second = chain[:, even_draws // 2 : even_draws, :]
    return np.concatenate([first, second], axis=0).reshape((2 * chains, even_draws // 2, ndim))


def _interval_overlap_fraction(a_low: float, a_high: float, b_low: float, b_high: float) -> float:
    width = max(min(a_high, b_high) - max(a_low, b_low), 0.0)
    denominator = max(min(a_high - a_low, b_high - b_low), 1.0e-12)
    return float(width / denominator)


def _scale_ratio(a_scale: float, b_scale: float) -> float:
    a = max(abs(float(a_scale)), 1.0e-12)
    b = max(abs(float(b_scale)), 1.0e-12)
    return float(max(a / b, b / a))


def _rhat_1d(values: np.ndarray) -> float | None:
    chains, draws = values.shape
    if chains < 2 or draws < 2:
        return None
    chain_means = np.mean(values, axis=1)
    chain_vars = np.var(values, axis=1, ddof=1)
    within = float(np.mean(chain_vars))
    if not np.isfinite(within) or within <= 0.0:
        return None
    between = draws * float(np.var(chain_means, ddof=1))
    var_hat = ((draws - 1.0) / draws) * within + between / draws
    if not np.isfinite(var_hat):
        return None
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
