"""Generic blind Box Least Squares discovery, refinement, and holdout validation."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.timeseries import BoxLeastSquares

from .provenance import write_json

ANTI_LEAKAGE_STATEMENT = (
    "Published ephemeris values were not used for preprocessing, broad BLS discovery, "
    "local refinement, candidate selection, harmonic grouping, odd-even diagnostics, "
    "holdout evaluation, or full-mission local refinement."
)

HARMONIC_RELATIONS = (
    ("1/4", 0.25),
    ("1/3", 1.0 / 3.0),
    ("1/2", 0.5),
    ("1", 1.0),
    ("2", 2.0),
    ("3", 3.0),
    ("4", 4.0),
)


@dataclass(frozen=True)
class BLSSearchConfig:
    """Generic BLS settings not tuned to a specific target."""

    minimum_period_days: float = 0.5
    maximum_period_days: float = 100.0
    minimum_duration_hours: float = 1.0
    maximum_duration_hours: float = 12.0
    n_durations: int = 12
    n_periods: int = 5000
    frequency_factor: float = 1.0
    oversample: int = 10
    objective: str = "likelihood"
    top_n_peaks: int = 10
    training_fraction: float = 0.70
    minimum_cadences: int = 100
    minimum_train_cadences: int = 50
    minimum_holdout_cadences: int = 20
    minimum_transits_in_train: int = 3
    peak_period_separation_fraction: float = 0.01
    local_window_coarse_steps: int = 3
    allowed_phase_drift_fraction: float = 0.10
    local_max_period_samples: int = 8000
    local_duration_step_hours: float = 0.25
    harmonic_fractional_tolerance: float = 0.02
    odd_even_depth_ratio_warning: float = 2.0
    odd_even_ratio_min_scatter_multiple: float = 1.0
    odd_even_depth_difference_sigma: float = 3.0
    holdout_window_duration_scale: float = 1.0
    holdout_baseline_duration_scale: float = 3.0
    minimum_event_in_transit_cadences: int = 1
    minimum_event_baseline_cadences: int = 2


def run_phase1a_search(
    light_curve,
    output_dir: Path,
    target: str,
    config: BLSSearchConfig | None = None,
    provenance: dict[str, Any] | None = None,
    published_ephemeris: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Run broad discovery, local training refinement, and locked holdout evaluation."""
    config = config or BLSSearchConfig()
    output_dir.mkdir(parents=True, exist_ok=True)

    time, flux = _time_flux_arrays(light_curve)
    train_mask, holdout_mask, split = chronological_split(time, config.training_fraction)
    train_time = time[train_mask]
    train_flux = flux[train_mask]
    holdout_time = time[holdout_mask]
    holdout_flux = flux[holdout_mask]

    broad_training = run_bls_search(train_time, train_flux, config)
    coarse_candidates = distinct_candidates(broad_training, train_time, config)
    refined_candidates = refine_candidates(train_time, train_flux, coarse_candidates, broad_training, config)
    refined_candidates = assign_harmonic_families(refined_candidates, config)
    locked_candidate = refined_candidates[0]

    odd_even_rows = [
        odd_even_diagnostic(train_time, train_flux, candidate, config)
        for candidate in refined_candidates
    ]
    harmonic_rows = harmonic_family_rows(refined_candidates)
    alias_rows = alias_diagnostics(train_time, train_flux, locked_candidate, config)

    holdout_summary, holdout_events = evaluate_holdout(
        holdout_time,
        holdout_flux,
        locked_candidate,
        config,
    )

    full_global = run_bls_search(time, flux, config)
    full_global_candidates = distinct_candidates(full_global, time, config)
    full_global_winner = assign_harmonic_families([locked_candidate, full_global_candidates[0]], config)[1]
    full_local_refinement = refine_single_candidate(
        time,
        flux,
        locked_candidate,
        full_global,
        config,
        coarse_rank=int(locked_candidate["coarse_rank"]),
        refined_rank=1,
        local_window=(
            float(locked_candidate["local_period_window_min_days"]),
            float(locked_candidate["local_period_window_max_days"]),
        ),
    )
    full_local_refinement["shift_from_training_period_days"] = (
        float(full_local_refinement["refined_period_days"]) - float(locked_candidate["refined_period_days"])
    )
    full_local_refinement["shift_from_training_transit_time_days"] = (
        float(full_local_refinement["refined_transit_time"]) - float(locked_candidate["refined_transit_time"])
    )
    full_local_refinement["shift_from_training_duration_hours"] = (
        float(full_local_refinement["refined_duration_hours"])
        - float(locked_candidate["refined_duration_hours"])
    )

    summary = {
        "target": target,
        "anti_leakage_statement": ANTI_LEAKAGE_STATEMENT,
        "search_method": {
            "library": "astropy.timeseries.BoxLeastSquares",
            "objective": config.objective,
        },
        "broad_training_search": {
            "settings": search_settings_summary(config, train_time),
            "evaluated_period_count": int(len(broad_training["periods"])),
            "evaluated_duration_count": int(len(broad_training["durations_days"])),
        },
        "coarse_training_winner": coarse_candidates[0],
        "local_refinement_configuration": local_refinement_summary(config),
        "refined_training_candidates": refined_candidates,
        "locked_refined_training_candidate": locked_candidate,
        "harmonic_family_diagnostics": harmonic_rows,
        "odd_even_diagnostics": odd_even_rows,
        "alias_diagnostics": alias_rows,
        "chronological_split": split,
        "holdout_summary": holdout_summary,
        "full_mission_global_search_diagnostic": {
            "winner": full_global_winner,
            "evaluated_period_count": int(len(full_global["periods"])),
            "evaluated_duration_count": int(len(full_global["durations_days"])),
            "label": "global_search_stability_and_alias_diagnostic_not_refinement",
        },
        "full_mission_local_refinement": full_local_refinement,
        "candidate_selection_rule": (
            "Select the highest refined training-data BLS objective after broad-peak "
            "deduplication and local training-only refinement."
        ),
        "peak_deduplication_rule": peak_deduplication_rule(config),
        "warnings": collect_warnings(refined_candidates, full_local_refinement),
        "known_ephemeris_used_during_search": False,
    }

    write_json(output_dir / "search_summary.json", summary)
    write_csv(output_dir / "top_period_candidates.csv", refined_candidates)
    write_csv(output_dir / "alias_diagnostics.csv", alias_rows)
    write_csv(output_dir / "holdout_event_diagnostics.csv", holdout_events)
    write_csv(output_dir / "odd_even_diagnostics.csv", odd_even_rows)
    write_csv(output_dir / "harmonic_family_diagnostics.csv", harmonic_rows)
    save_periodogram_plot(
        output_dir / "periodogram.png",
        broad_training["periods"],
        broad_training["power"],
        coarse_period=float(locked_candidate["coarse_period_days"]),
        refined_period=float(locked_candidate["refined_period_days"]),
        harmonic_periods=[float(row["refined_period_days"]) for row in refined_candidates[:5]],
    )
    save_folded_plot(
        output_dir / "recovered_folded_light_curve.png",
        time,
        flux,
        full_local_refinement,
        period_key="refined_period_days",
        transit_time_key="refined_transit_time",
        duration_key="refined_duration_days",
        title="Full data folded on full-mission local refinement",
    )
    save_folded_plot(
        output_dir / "holdout_folded_light_curve.png",
        holdout_time,
        holdout_flux,
        locked_candidate,
        period_key="refined_period_days",
        transit_time_key="refined_transit_time",
        duration_key="refined_duration_days",
        title="Holdout data folded on locked refined training ephemeris",
    )

    if provenance is not None:
        phase1a_provenance = dict(provenance)
        phase1a_provenance["phase1a_search"] = {
            "search_method": summary["search_method"],
            "broad_search_configuration": summary["broad_training_search"],
            "local_refinement_configuration": summary["local_refinement_configuration"],
            "chronological_split": split,
            "candidate_selection_rule": summary["candidate_selection_rule"],
            "peak_deduplication_rule": summary["peak_deduplication_rule"],
            "harmonic_grouping_rule": harmonic_grouping_rule(config),
            "odd_even_diagnostic_rule": odd_even_rule(config),
            "holdout_locking_rule": "Holdout uses locked refined training period, transit time, and duration without retuning.",
            "full_mission_global_diagnostic": "Stored separately from local refinement.",
            "full_mission_local_refinement": "Centered on the locked training family after holdout outputs are computed.",
            "anti_leakage_statement": ANTI_LEAKAGE_STATEMENT,
        }
        write_json(output_dir / "provenance_manifest.json", phase1a_provenance)
    if published_ephemeris is not None:
        comparison = published_comparison(locked_candidate, full_local_refinement, published_ephemeris)
        write_json(output_dir / "published_comparison.json", comparison)
    return summary


def run_bls_search(time: np.ndarray, flux: np.ndarray, config: BLSSearchConfig) -> dict[str, Any]:
    """Run Astropy BLS on a broad generic period/duration grid."""
    _validate_search_inputs(time, config)
    durations = broad_durations(config)
    periods = period_grid(time, config)
    result = BoxLeastSquares(time, flux).power(
        periods,
        durations,
        objective=config.objective,
        method="fast",
        oversample=config.oversample,
    )
    return bls_result_dict(result, durations)


def refine_candidates(
    time: np.ndarray,
    flux: np.ndarray,
    coarse_candidates: list[dict[str, float | int]],
    broad_result: dict[str, Any],
    config: BLSSearchConfig,
) -> list[dict[str, Any]]:
    """Refine leading broad peaks using training data only."""
    refined = [
        refine_single_candidate(
            time,
            flux,
            candidate,
            broad_result,
            config,
            coarse_rank=int(candidate["rank"]),
            refined_rank=0,
        )
        for candidate in coarse_candidates
    ]
    refined.sort(key=lambda row: float(row["refined_bls_power"]), reverse=True)
    for rank, row in enumerate(refined, start=1):
        row["refined_rank"] = rank
    return refined


def refine_single_candidate(
    time: np.ndarray,
    flux: np.ndarray,
    candidate: dict[str, float | int],
    broad_result: dict[str, Any],
    config: BLSSearchConfig,
    coarse_rank: int,
    refined_rank: int,
    local_window: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Run a local BLS search around one period family."""
    baseline = float(np.nanmax(time) - np.nanmin(time))
    coarse_period = candidate_period(candidate)
    coarse_duration = candidate_duration(candidate)
    if local_window is None:
        local_window = local_period_window(coarse_period, broad_result["periods"], config)
    precision_duration = config.minimum_duration_hours / 24.0
    local_periods, achieved_step, capped = local_period_grid(
        local_window,
        coarse_period,
        precision_duration,
        baseline,
        config,
    )
    durations = local_durations(config)
    result = BoxLeastSquares(time, flux).power(
        local_periods,
        durations,
        objective=config.objective,
        method="fast",
        oversample=config.oversample,
    )
    best = int(np.nanargmax(result.power))
    refined_period = float(result.period[best])
    refined_duration = float(result.duration[best])
    refined_transit_time = float(result.transit_time[best])
    requested_step = local_period_step_requirement(coarse_period, precision_duration, baseline, config)
    drift = accumulated_timing_drift_days(achieved_step, baseline, refined_period)
    warnings = []
    if capped:
        warnings.append("local_period_sample_cap_prevented_requested_resolution")
    if np.isclose(refined_duration, durations[0]):
        warnings.append("refined_duration_at_minimum_boundary")
    if np.isclose(refined_duration, durations[-1]):
        warnings.append("refined_duration_at_maximum_boundary")
    return {
        "coarse_rank": coarse_rank,
        "refined_rank": refined_rank,
        "coarse_period_days": coarse_period,
        "refined_period_days": refined_period,
        "period_shift_days": refined_period - coarse_period,
        "coarse_duration_days": coarse_duration,
        "coarse_duration_hours": coarse_duration * 24.0,
        "refined_duration_days": refined_duration,
        "refined_duration_hours": refined_duration * 24.0,
        "duration_shift_hours": refined_duration * 24.0 - coarse_duration * 24.0,
        "coarse_transit_time": candidate_transit_time(candidate),
        "refined_transit_time": refined_transit_time,
        "transit_time_shift_days": refined_transit_time - candidate_transit_time(candidate),
        "coarse_bls_power": candidate_power(candidate),
        "refined_bls_power": float(result.power[best]),
        "power_shift": float(result.power[best]) - candidate_power(candidate),
        "refined_depth": float(result.depth[best]),
        "refined_depth_ppm": float(result.depth[best] * 1.0e6),
        "refined_depth_snr": float(result.depth_snr[best]),
        "estimated_transits_in_baseline": estimated_transit_count(time, refined_period, refined_transit_time),
        "local_period_window_min_days": float(local_window[0]),
        "local_period_window_max_days": float(local_window[1]),
        "requested_local_period_step_days": requested_step,
        "precision_duration_used_for_local_period_step_days": precision_duration,
        "achieved_local_period_step_days": achieved_step,
        "implied_training_baseline_timing_drift_days": drift,
        "implied_training_baseline_timing_drift_fraction_of_duration": drift / refined_duration,
        "local_period_sample_count": int(len(local_periods)),
        "local_duration_sample_count": int(len(durations)),
        "local_sample_cap_hit": bool(capped),
        "duration_boundary_warning": ";".join(warnings),
        "warnings": ";".join(warnings),
    }


def chronological_split(
    time: np.ndarray,
    training_fraction: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Split by observed time baseline, not randomized cadence index."""
    start = float(np.nanmin(time))
    end = float(np.nanmax(time))
    split_time = start + training_fraction * (end - start)
    train_mask = time <= split_time
    holdout_mask = time > split_time
    split = {
        "method": "chronological_time_baseline_fraction",
        "training_fraction": float(training_fraction),
        "split_time": float(split_time),
        "training_time_min": float(np.nanmin(time[train_mask])),
        "training_time_max": float(np.nanmax(time[train_mask])),
        "holdout_time_min": float(np.nanmin(time[holdout_mask])),
        "holdout_time_max": float(np.nanmax(time[holdout_mask])),
        "training_cadence_count": int(np.count_nonzero(train_mask)),
        "holdout_cadence_count": int(np.count_nonzero(holdout_mask)),
        "gap_across_split_days": _gap_across_split(time, split_time),
    }
    return train_mask, holdout_mask, split


def distinct_candidates(
    search_result: dict[str, Any],
    time: np.ndarray,
    config: BLSSearchConfig,
) -> list[dict[str, float | int]]:
    """Return leading broad periodogram peaks after period-neighborhood deduplication."""
    order = np.argsort(search_result["power"])[::-1]
    candidates: list[dict[str, float | int]] = []
    for index in order:
        period = float(search_result["periods"][index])
        if not np.isfinite(period):
            continue
        if any(_same_peak(period, float(row["period_days"]), config) for row in candidates):
            continue
        duration = float(search_result["duration"][index])
        transit_time = float(search_result["transit_time"][index])
        candidates.append(
            {
                "rank": len(candidates) + 1,
                "period_days": period,
                "transit_time": transit_time,
                "duration_days": duration,
                "duration_hours": duration * 24.0,
                "depth": float(search_result["depth"][index]),
                "depth_ppm": float(search_result["depth"][index] * 1.0e6),
                "bls_power": float(search_result["power"][index]),
                "depth_snr": float(search_result["depth_snr"][index]),
                "estimated_transits_in_baseline": estimated_transit_count(time, period, transit_time),
                "duration_boundary_warning": _duration_boundary_warning(duration, broad_durations(config)),
            }
        )
        if len(candidates) >= config.top_n_peaks:
            break
    if not candidates:
        raise ValueError("BLS search produced no valid candidates.")
    return candidates


def assign_harmonic_families(
    candidates: list[dict[str, Any]],
    config: BLSSearchConfig,
) -> list[dict[str, Any]]:
    """Group candidates by small-integer period relationships without literature periods."""
    family_refs: list[tuple[int, float]] = []
    assigned: list[dict[str, Any]] = []
    for candidate in candidates:
        period = candidate_period(candidate)
        match = None
        for family_id, reference_period in family_refs:
            relation = harmonic_relation(period, reference_period, config)
            if relation is not None:
                match = (family_id, reference_period, relation)
                break
        row = dict(candidate)
        if match is None:
            family_id = len(family_refs) + 1
            family_refs.append((family_id, period))
            row.update(
                {
                    "harmonic_family_id": family_id,
                    "family_reference_period_days": period,
                    "relationship_to_family_reference": "1",
                    "period_ratio_to_family_reference": 1.0,
                    "harmonic_fractional_mismatch": 0.0,
                }
            )
        else:
            family_id, reference_period, relation = match
            label, expected_ratio, mismatch = relation
            row.update(
                {
                    "harmonic_family_id": family_id,
                    "family_reference_period_days": reference_period,
                    "relationship_to_family_reference": label,
                    "period_ratio_to_family_reference": period / reference_period,
                    "harmonic_fractional_mismatch": mismatch,
                }
            )
        assigned.append(row)
    return assigned


def harmonic_relation(
    period: float,
    reference_period: float,
    config: BLSSearchConfig,
) -> tuple[str, float, float] | None:
    """Return the small-integer relationship to a reference period, if any."""
    ratio = period / reference_period
    best: tuple[str, float, float] | None = None
    for label, expected in HARMONIC_RELATIONS:
        mismatch = abs(ratio - expected) / expected
        if mismatch <= config.harmonic_fractional_tolerance and (
            best is None or mismatch < best[2]
        ):
            best = (label, expected, mismatch)
    return best


def odd_even_diagnostic(
    time: np.ndarray,
    flux: np.ndarray,
    candidate: dict[str, Any],
    config: BLSSearchConfig,
) -> dict[str, Any]:
    """Estimate odd/even transit depth consistency for a fixed candidate."""
    period = candidate_period(candidate)
    transit_time = candidate_transit_time(candidate)
    duration = candidate_duration(candidate)
    centers = predicted_centers(time, period, transit_time)
    odd_depths: list[float] = []
    even_depths: list[float] = []
    usable_odd = usable_even = 0
    for sequence, center in enumerate(centers):
        event = _evaluate_event(time, flux, center, duration, config, sequence)
        if not event["sufficient_coverage"]:
            continue
        if sequence % 2:
            odd_depths.append(float(event["local_depth"]))
            usable_odd += 1
        else:
            even_depths.append(float(event["local_depth"]))
            usable_even += 1
    odd_depth = float(np.nanmedian(odd_depths)) if odd_depths else float("nan")
    even_depth = float(np.nanmedian(even_depths)) if even_depths else float("nan")
    all_depths = np.asarray(odd_depths + even_depths, dtype=float)
    scatter = float(np.nanstd(all_depths)) if len(all_depths) > 1 else float("nan")
    difference = abs(odd_depth - even_depth)
    ratio = _safe_depth_ratio(odd_depth, even_depth, scatter)
    warning_trigger = odd_even_warning_trigger(ratio, difference, scatter, config)
    return {
        "refined_rank": candidate.get("refined_rank", candidate.get("rank", 0)),
        "period_days": period,
        "event_count": int(len(centers)),
        "usable_odd_event_count": usable_odd,
        "usable_even_event_count": usable_even,
        "odd_depth": odd_depth,
        "even_depth": even_depth,
        "odd_depth_ppm": odd_depth * 1.0e6,
        "even_depth_ppm": even_depth * 1.0e6,
        "absolute_depth_difference": difference,
        "absolute_depth_difference_ppm": difference * 1.0e6,
        "odd_even_depth_ratio": ratio,
        "local_depth_scatter": scatter,
        "local_depth_scatter_ppm": scatter * 1.0e6 if np.isfinite(scatter) else float("nan"),
        "odd_even_warning_trigger": warning_trigger,
        "strong_odd_even_inconsistency": warning_trigger != "none",
        "rule": odd_even_rule(config),
    }


def odd_even_warning_trigger(
    ratio: float,
    absolute_difference: float,
    local_depth_scatter: float,
    config: BLSSearchConfig,
) -> str:
    """Classify odd/even inconsistency without allowing ratio-only warnings."""
    if not (
        np.isfinite(absolute_difference)
        and np.isfinite(local_depth_scatter)
        and local_depth_scatter > 0
    ):
        return "none"
    ratio_warning = (
        np.isfinite(ratio)
        and ratio >= config.odd_even_depth_ratio_warning
        and absolute_difference >= config.odd_even_ratio_min_scatter_multiple * local_depth_scatter
    )
    difference_warning = (
        absolute_difference >= config.odd_even_depth_difference_sigma * local_depth_scatter
    )
    if ratio_warning:
        return "ratio_and_absolute_difference"
    if difference_warning:
        return "absolute_difference_only"
    return "none"


def alias_diagnostics(
    time: np.ndarray,
    flux: np.ndarray,
    candidate: dict[str, Any],
    config: BLSSearchConfig,
) -> list[dict[str, float | int | str]]:
    """Evaluate P/2, P, and 2P around the locked refined candidate."""
    model = BoxLeastSquares(time, flux)
    durations = local_durations(config)
    aliases = [("P/2", 0.5, "half_period_alias"), ("P", 1.0, "selected_period"), ("2P", 2.0, "double_period_alias")]
    rows: list[dict[str, float | int | str]] = []
    max_period = effective_maximum_period(time, config)
    for label, factor, interpretation in aliases:
        period = candidate_period(candidate) * factor
        if period < effective_minimum_period(config) or period > max_period:
            rows.append({"alias": label, "period_days": period, "valid": 0, "harmonic_interpretation": interpretation})
            continue
        result = model.power(period, durations, objective=config.objective, method="fast", oversample=config.oversample)
        best = int(np.nanargmax(result.power))
        alias_candidate = {
            "period_days": period,
            "duration_days": float(result.duration[best]),
            "transit_time": float(result.transit_time[best]),
        }
        odd_even = odd_even_diagnostic(time, flux, alias_candidate, config)
        rows.append(
            {
                "alias": label,
                "valid": 1,
                "period_days": period,
                "duration_days": float(result.duration[best]),
                "duration_hours": float(result.duration[best] * 24.0),
                "transit_time": float(result.transit_time[best]),
                "depth": float(result.depth[best]),
                "depth_ppm": float(result.depth[best] * 1.0e6),
                "bls_power": float(result.power[best]),
                "depth_snr": float(result.depth_snr[best]),
                "estimated_transits_in_baseline": estimated_transit_count(
                    time, period, float(result.transit_time[best])
                ),
                "odd_depth_ppm": odd_even["odd_depth_ppm"],
                "even_depth_ppm": odd_even["even_depth_ppm"],
                "odd_even_depth_ratio": odd_even["odd_even_depth_ratio"],
                "absolute_depth_difference_ppm": odd_even["absolute_depth_difference_ppm"],
                "local_depth_scatter_ppm": odd_even["local_depth_scatter_ppm"],
                "odd_even_warning_trigger": odd_even["odd_even_warning_trigger"],
                "strong_odd_even_inconsistency": odd_even["strong_odd_even_inconsistency"],
                "odd_even_rule": odd_even["rule"],
                "harmonic_interpretation": interpretation,
            }
        )
    return rows


def evaluate_holdout(
    time: np.ndarray,
    flux: np.ndarray,
    candidate: dict[str, Any],
    config: BLSSearchConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate locked refined training ephemeris in holdout without retuning."""
    period = candidate_period(candidate)
    transit_time = candidate_transit_time(candidate)
    duration = candidate_duration(candidate)
    centers = predicted_centers(time, period, transit_time)
    event_rows = [
        _evaluate_event(time, flux, center, duration, config, sequence)
        for sequence, center in enumerate(centers)
    ]
    usable = [row for row in event_rows if row["sufficient_coverage"]]
    if usable:
        in_fluxes = []
        baseline_fluxes = []
        for row in usable:
            center = float(row["predicted_transit_center"])
            event_mask = np.abs(time - center) <= duration / 2.0
            base_mask = (
                (np.abs(time - center) > config.holdout_window_duration_scale * duration)
                & (np.abs(time - center) <= config.holdout_baseline_duration_scale * duration)
            )
            in_fluxes.append(flux[event_mask])
            baseline_fluxes.append(flux[base_mask])
        in_values = np.concatenate(in_fluxes)
        baseline_values = np.concatenate(baseline_fluxes)
        baseline_median = float(np.nanmedian(baseline_values))
        in_median = float(np.nanmedian(in_values))
        depth = baseline_median - in_median
        baseline_std = float(np.nanstd(baseline_values))
        snr = depth / (baseline_std * np.sqrt(1.0 / len(in_values) + 1.0 / len(baseline_values)))
    else:
        baseline_median = in_median = depth = baseline_std = snr = float("nan")
    holdout_baseline = float(np.nanmax(time) - np.nanmin(time)) if len(time) else 0.0
    period_step = float(candidate.get("achieved_local_period_step_days", 0.0))
    expected_drift = accumulated_timing_drift_days(period_step, holdout_baseline, period)
    return (
        {
            "predicted_event_count": int(len(event_rows)),
            "usable_event_count": int(len(usable)),
            "aggregate_in_transit_flux_median": in_median,
            "aggregate_baseline_flux_median": baseline_median,
            "aggregate_depth": depth,
            "aggregate_depth_ppm": depth * 1.0e6,
            "aggregate_baseline_flux_std": baseline_std,
            "diagnostic_snr_proxy": snr,
            "expected_accumulated_timing_uncertainty_days": expected_drift,
            "expected_accumulated_timing_uncertainty_fraction_of_duration": expected_drift / duration,
            "ephemeris_locked_from_training": True,
            "holdout_retuning_performed": False,
            "coverage_exclusion_rule": (
                "Events are excluded only when in-transit or local-baseline cadence counts "
                "fall below configured minima; event depth is not used for exclusion."
            ),
        },
        event_rows,
    )


def predicted_centers(time: np.ndarray, period: float, transit_time: float) -> np.ndarray:
    """Predict all centers in a time interval from a fixed ephemeris."""
    first = int(np.ceil((np.nanmin(time) - transit_time) / period))
    last = int(np.floor((np.nanmax(time) - transit_time) / period))
    if last < first:
        return np.asarray([], dtype=float)
    return transit_time + np.arange(first, last + 1) * period


def phase_fold(time: np.ndarray, period: float, transit_time: float) -> np.ndarray:
    """Return phase in days centered on transit."""
    return ((time - transit_time + 0.5 * period) % period) - 0.5 * period


def published_comparison(
    locked_candidate: dict[str, Any],
    full_local_refinement: dict[str, Any],
    published_ephemeris: dict[str, float],
) -> dict[str, Any]:
    """Compare recovered values with published values after search finalization."""
    return {
        "comparison_stage": "post_search_validation_only",
        "known_values_used_during_search": False,
        "published": published_ephemeris,
        "refined_training_candidate": _compare_candidate(locked_candidate, published_ephemeris),
        "full_mission_local_refinement": _compare_candidate(full_local_refinement, published_ephemeris),
    }


def search_settings_summary(config: BLSSearchConfig, train_time: np.ndarray) -> dict[str, Any]:
    """Return JSON-ready broad-search settings including derived bounds."""
    settings = asdict(config)
    settings["trial_durations_hours"] = list(
        np.linspace(config.minimum_duration_hours, config.maximum_duration_hours, config.n_durations)
    )
    settings["period_grid"] = "uniform_frequency_grid"
    settings["effective_period_samples"] = int(max(2, round(config.n_periods * config.frequency_factor)))
    settings["effective_minimum_period_days"] = effective_minimum_period(config)
    settings["effective_maximum_period_days"] = effective_maximum_period(train_time, config)
    settings["rationale"] = (
        "Broad generic discovery search: periods span sub-day to long-period signals, "
        "durations span 1-12 hours, and maximum period is limited by the training "
        "baseline to require multiple possible transits. Coarse samples are discovery "
        "locations, not final ephemerides."
    )
    return settings


def local_refinement_summary(config: BLSSearchConfig) -> dict[str, Any]:
    """Return JSON-ready local-refinement settings."""
    return {
        "period_step_formula": "delta_P <= allowed_phase_drift_fraction * duration * period / training_baseline",
        "allowed_phase_drift_fraction": config.allowed_phase_drift_fraction,
        "local_window_rule": (
            "Center on each coarse peak and search +/- local_window_coarse_steps times "
            "the local broad-grid period spacing."
        ),
        "local_window_coarse_steps": config.local_window_coarse_steps,
        "local_max_period_samples": config.local_max_period_samples,
        "local_duration_step_hours": config.local_duration_step_hours,
        "duration_range_hours": [config.minimum_duration_hours, config.maximum_duration_hours],
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows to CSV, preserving field order from all rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_periodogram_plot(
    path: Path,
    periods: np.ndarray,
    power: np.ndarray,
    coarse_period: float,
    refined_period: float,
    harmonic_periods: list[float],
) -> None:
    """Save a two-panel blind training BLS periodogram without published markers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 1, figsize=(10, 7))
    axes[0].plot(periods, power, color="black", linewidth=0.8)
    axes[0].axvline(coarse_period, color="tab:orange", linestyle=":", linewidth=1.2, label="coarse peak")
    axes[0].axvline(refined_period, color="tab:red", linestyle="--", linewidth=1.2, label="refined")
    axes[0].set_xlabel("Period [days]")
    axes[0].set_ylabel("BLS power")
    axes[0].set_title("Broad training BLS periodogram")
    axes[0].legend()

    zoom_half_width = max(0.2, 0.08 * refined_period)
    zoom_mask = np.abs(periods - coarse_period) <= zoom_half_width
    if not np.any(zoom_mask):
        zoom_mask = np.ones_like(periods, dtype=bool)
    axes[1].plot(periods[zoom_mask], power[zoom_mask], color="black", linewidth=0.9)
    axes[1].axvline(coarse_period, color="tab:orange", linestyle=":", linewidth=1.2)
    axes[1].axvline(refined_period, color="tab:red", linestyle="--", linewidth=1.2)
    for harmonic in harmonic_periods:
        if abs(harmonic - coarse_period) <= zoom_half_width:
            axes[1].axvline(harmonic, color="0.55", linestyle="-.", linewidth=0.8, alpha=0.6)
    axes[1].set_xlabel("Period [days]")
    axes[1].set_ylabel("BLS power")
    axes[1].set_title("Zoom around selected recovered period family")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def save_folded_plot(
    path: Path,
    time: np.ndarray,
    flux: np.ndarray,
    candidate: dict[str, Any],
    period_key: str,
    transit_time_key: str,
    duration_key: str,
    title: str,
) -> None:
    """Save a folded diagnostic plot using recovered parameters only."""
    path.parent.mkdir(parents=True, exist_ok=True)
    period = float(candidate[period_key])
    transit_time = float(candidate[transit_time_key])
    duration = float(candidate[duration_key])
    phase = phase_fold(time, period, transit_time)
    order = np.argsort(phase)
    figure, axis = plt.subplots(figsize=(10, 4))
    axis.scatter(phase[order], flux[order], s=3, color="0.65", alpha=0.35, linewidths=0)
    axis.axvspan(-duration / 2.0, duration / 2.0, color="tab:blue", alpha=0.12)
    axis.axvline(0.0, color="tab:blue", linestyle="--", linewidth=1.0)
    axis.set_xlim(-4.0 * duration, 4.0 * duration)
    axis.set_xlabel("Phase [days from recovered transit time]")
    axis.set_ylabel("Relative Flux")
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def harmonic_family_rows(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = [
        "refined_rank",
        "refined_period_days",
        "harmonic_family_id",
        "family_reference_period_days",
        "relationship_to_family_reference",
        "period_ratio_to_family_reference",
        "harmonic_fractional_mismatch",
    ]
    return [{key: candidate[key] for key in keys} for candidate in candidates]


def collect_warnings(candidates: list[dict[str, Any]], full_local_refinement: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    for candidate in [*candidates, full_local_refinement]:
        for warning in str(candidate.get("warnings", "")).split(";"):
            if warning and warning not in warnings:
                warnings.append(warning)
    return warnings


def bls_result_dict(result, durations: np.ndarray) -> dict[str, Any]:
    return {
        "periods": np.asarray(result.period, dtype=float),
        "power": np.asarray(result.power, dtype=float),
        "duration": np.asarray(result.duration, dtype=float),
        "transit_time": np.asarray(result.transit_time, dtype=float),
        "depth": np.asarray(result.depth, dtype=float),
        "depth_snr": np.asarray(result.depth_snr, dtype=float),
        "durations_days": durations,
    }


def broad_durations(config: BLSSearchConfig) -> np.ndarray:
    return np.linspace(
        config.minimum_duration_hours / 24.0,
        config.maximum_duration_hours / 24.0,
        config.n_durations,
    )


def local_durations(config: BLSSearchConfig) -> np.ndarray:
    return np.arange(
        config.minimum_duration_hours,
        config.maximum_duration_hours + 0.5 * config.local_duration_step_hours,
        config.local_duration_step_hours,
    ) / 24.0


def local_period_window(
    coarse_period: float,
    broad_periods: np.ndarray,
    config: BLSSearchConfig,
) -> tuple[float, float]:
    sorted_periods = np.sort(np.asarray(broad_periods, dtype=float))
    index = int(np.argmin(np.abs(sorted_periods - coarse_period)))
    left_step = coarse_period - sorted_periods[max(index - 1, 0)]
    right_step = sorted_periods[min(index + 1, len(sorted_periods) - 1)] - coarse_period
    step = max(abs(left_step), abs(right_step), coarse_period * 1.0e-4)
    half_width = config.local_window_coarse_steps * step
    return max(effective_minimum_period(config), coarse_period - half_width), coarse_period + half_width


def local_period_step_requirement(
    period: float,
    duration: float,
    baseline: float,
    config: BLSSearchConfig,
) -> float:
    return config.allowed_phase_drift_fraction * duration * period / baseline


def local_period_grid(
    local_window: tuple[float, float],
    period: float,
    duration: float,
    baseline: float,
    config: BLSSearchConfig,
) -> tuple[np.ndarray, float, bool]:
    requested_step = local_period_step_requirement(period, duration, baseline, config)
    width = max(local_window[1] - local_window[0], requested_step)
    requested_samples = int(np.ceil(width / requested_step)) + 1
    capped = requested_samples > config.local_max_period_samples
    samples = min(max(requested_samples, 3), config.local_max_period_samples)
    periods = np.linspace(local_window[0], local_window[1], samples)
    achieved_step = float(np.nanmax(np.diff(periods))) if len(periods) > 1 else 0.0
    return periods, achieved_step, capped


def accumulated_timing_drift_days(period_step: float, baseline: float, period: float) -> float:
    return abs(period_step) * baseline / period


def _time_flux_arrays(light_curve) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(light_curve.time.value, dtype=float), np.asarray(light_curve.flux.value, dtype=float)


def _validate_search_inputs(time: np.ndarray, config: BLSSearchConfig) -> None:
    if len(time) < config.minimum_cadences:
        raise ValueError("Not enough cadences for BLS search.")
    if effective_maximum_period(time, config) <= effective_minimum_period(config):
        raise ValueError("Training baseline is too short for the configured period search.")


def effective_minimum_period(config: BLSSearchConfig) -> float:
    max_duration_days = config.maximum_duration_hours / 24.0
    return max(float(config.minimum_period_days), 1.01 * max_duration_days)


def effective_maximum_period(time: np.ndarray, config: BLSSearchConfig) -> float:
    baseline = float(np.nanmax(time) - np.nanmin(time))
    return min(float(config.maximum_period_days), baseline / float(config.minimum_transits_in_train))


def period_grid(time: np.ndarray, config: BLSSearchConfig) -> np.ndarray:
    minimum_period = effective_minimum_period(config)
    maximum_period = effective_maximum_period(time, config)
    n_periods = int(max(2, round(config.n_periods * config.frequency_factor)))
    frequencies = np.linspace(1.0 / maximum_period, 1.0 / minimum_period, n_periods)
    return 1.0 / frequencies[::-1]


def estimated_transit_count(time: np.ndarray, period: float, transit_time: float) -> int:
    return int(len(predicted_centers(time, period, transit_time)))


def candidate_period(candidate: dict[str, Any]) -> float:
    return float(candidate.get("refined_period_days", candidate.get("period_days")))


def candidate_duration(candidate: dict[str, Any]) -> float:
    return float(candidate.get("refined_duration_days", candidate.get("duration_days")))


def candidate_transit_time(candidate: dict[str, Any]) -> float:
    return float(candidate.get("refined_transit_time", candidate.get("transit_time")))


def candidate_power(candidate: dict[str, Any]) -> float:
    return float(candidate.get("refined_bls_power", candidate.get("bls_power")))


def _same_peak(period: float, accepted_period: float, config: BLSSearchConfig) -> bool:
    return abs(period - accepted_period) < config.peak_period_separation_fraction * max(
        period, accepted_period
    )


def _evaluate_event(
    time: np.ndarray,
    flux: np.ndarray,
    center: float,
    duration: float,
    config: BLSSearchConfig,
    sequence: int,
) -> dict[str, Any]:
    phase = time - center
    in_mask = np.abs(phase) <= duration / 2.0
    baseline_mask = (
        (np.abs(phase) > config.holdout_window_duration_scale * duration)
        & (np.abs(phase) <= config.holdout_baseline_duration_scale * duration)
    )
    in_count = int(np.count_nonzero(in_mask))
    baseline_count = int(np.count_nonzero(baseline_mask))
    sufficient = (
        in_count >= config.minimum_event_in_transit_cadences
        and baseline_count >= config.minimum_event_baseline_cadences
    )
    in_median = float(np.nanmedian(flux[in_mask])) if in_count else float("nan")
    baseline_median = float(np.nanmedian(flux[baseline_mask])) if baseline_count else float("nan")
    return {
        "event_sequence_number": int(sequence),
        "event_parity": "odd" if sequence % 2 else "even",
        "predicted_transit_center": float(center),
        "n_in_transit_cadences": in_count,
        "n_local_baseline_cadences": baseline_count,
        "local_in_transit_flux_median": in_median,
        "local_baseline_flux_median": baseline_median,
        "local_depth": baseline_median - in_median if sufficient else float("nan"),
        "local_depth_ppm": (baseline_median - in_median) * 1.0e6 if sufficient else float("nan"),
        "sufficient_coverage": bool(sufficient),
        "nearby_large_gap_days": _nearby_gap_days(time, center, config.holdout_baseline_duration_scale * duration),
    }


def _duration_boundary_warning(duration: float, durations: np.ndarray) -> str:
    if np.isclose(duration, durations[0]):
        return "duration_at_minimum_boundary"
    if np.isclose(duration, durations[-1]):
        return "duration_at_maximum_boundary"
    return ""


def _safe_depth_ratio(odd_depth: float, even_depth: float, local_depth_scatter: float) -> float:
    denominator = min(abs(odd_depth), abs(even_depth))
    numerator = max(abs(odd_depth), abs(even_depth))
    denominator_floor = 1.0e-12
    if np.isfinite(local_depth_scatter):
        denominator_floor = max(denominator_floor, 1.0e-3 * abs(local_depth_scatter))
    if denominator <= denominator_floor or not np.isfinite(denominator):
        return float("nan")
    return numerator / denominator


def _nearby_gap_days(time: np.ndarray, center: float, half_width: float) -> float:
    local = np.sort(time[np.abs(time - center) <= half_width])
    if len(local) < 2:
        return float("nan")
    return float(np.nanmax(np.diff(local)))


def _gap_across_split(time: np.ndarray, split_time: float) -> float:
    before = time[time <= split_time]
    after = time[time > split_time]
    if len(before) == 0 or len(after) == 0:
        return float("nan")
    return float(np.nanmin(after) - np.nanmax(before))


def _compare_candidate(candidate: dict[str, Any], published: dict[str, float]) -> dict[str, Any]:
    period = candidate_period(candidate)
    transit_time = candidate_transit_time(candidate)
    duration_hours = candidate_duration(candidate) * 24.0
    period_delta = period - published["period_days"]
    epoch_delta = transit_time - published["epoch_bkjd"]
    duration_delta = duration_hours - published["duration_hours"]
    return {
        "recovered": candidate,
        "absolute_differences": {
            "period_days": abs(period_delta),
            "epoch_days": abs(epoch_delta),
            "duration_hours": abs(duration_delta),
        },
        "signed_differences": {
            "period_days": period_delta,
            "epoch_days": epoch_delta,
            "duration_hours": duration_delta,
        },
        "relative_differences": {
            "period": abs(period_delta) / published["period_days"],
            "duration": abs(duration_delta) / published["duration_hours"],
        },
    }


def peak_deduplication_rule(config: BLSSearchConfig) -> str:
    return (
        "Broad candidates are accepted in descending BLS power only when their periods "
        f"differ from already accepted peaks by at least "
        f"{config.peak_period_separation_fraction:.3f} times the larger period."
    )


def harmonic_grouping_rule(config: BLSSearchConfig) -> str:
    return (
        "Refined candidates are grouped when their period ratio to a stronger family "
        "reference matches one of 1/4, 1/3, 1/2, 1, 2, 3, or 4 within fractional "
        f"tolerance {config.harmonic_fractional_tolerance:.3f}."
    )


def odd_even_rule(config: BLSSearchConfig) -> str:
    return (
        "Flag strong odd-even inconsistency when the odd/even depth ratio is at least "
        f"{config.odd_even_depth_ratio_warning:.2f} and the absolute difference is at least "
        f"{config.odd_even_ratio_min_scatter_multiple:.2f} times the local event-depth scatter, "
        "or when the absolute difference alone is at least "
        f"{config.odd_even_depth_difference_sigma:.2f} times that scatter. Ratio alone never triggers."
    )
