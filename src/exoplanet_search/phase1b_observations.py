"""Observation metadata and transit-window construction for Phase 1B."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .phase1b_types import ObservationSet, Phase1BConfig, TransitWindows

KEPLER_LONG_CADENCE_EXPOSURE_DAYS = 1765.5 / 86400.0


def observations_from_bundle(bundle, flux_column: str) -> tuple[ObservationSet, list[dict[str, Any]]]:
    """Build per-cadence arrays while preserving source-product labels."""
    if not bundle.product_light_curves:
        return observations_from_light_curve(bundle.light_curve, flux_column), []

    time_parts: list[np.ndarray] = []
    flux_parts: list[np.ndarray] = []
    err_parts: list[np.ndarray] = []
    product_parts: list[np.ndarray] = []
    quarter_parts: list[np.ndarray] = []
    cadence_parts: list[np.ndarray] = []
    exposure_parts: list[np.ndarray] = []
    mission_parts: list[np.ndarray] = []
    flux_product_parts: list[np.ndarray] = []
    metadata_rows: list[dict[str, Any]] = []

    for index, product in enumerate(bundle.product_light_curves):
        normalized = product.normalize()
        time = np.asarray(normalized.time.value, dtype=float)
        flux = np.asarray(normalized.flux.value, dtype=float)
        flux_err = _flux_err_array(normalized, len(time))
        finite = np.isfinite(time) & np.isfinite(flux)
        product_id = _product_id(product, index)
        quarter = _meta_first(product.meta, ("QUARTER", "quarter"), "")
        exposure_days, exposure_record = _exposure_days(product.meta, bundle.cadence)

        time_parts.append(time[finite])
        flux_parts.append(flux[finite])
        err_parts.append(flux_err[finite])
        product_parts.append(np.full(np.count_nonzero(finite), product_id, dtype=object))
        quarter_parts.append(np.full(np.count_nonzero(finite), str(quarter), dtype=object))
        cadence_parts.append(np.full(np.count_nonzero(finite), bundle.cadence, dtype=object))
        exposure_parts.append(np.full(np.count_nonzero(finite), exposure_days, dtype=float))
        mission_parts.append(np.full(np.count_nonzero(finite), bundle.mission, dtype=object))
        flux_product_parts.append(np.full(np.count_nonzero(finite), flux_column, dtype=object))
        metadata_rows.append(
            {
                "product_index": index,
                "product_id": product_id,
                "quarter": quarter,
                "cadence": bundle.cadence,
                "mission": bundle.mission,
                "flux_product": flux_column,
                "input_cadence_count": int(len(time)),
                "finite_cadence_count": int(np.count_nonzero(finite)),
                **exposure_record,
            }
        )

    observations = ObservationSet(
        time=np.concatenate(time_parts),
        flux=np.concatenate(flux_parts),
        flux_err=np.concatenate(err_parts),
        product_id=np.concatenate(product_parts),
        quarter=np.concatenate(quarter_parts),
        cadence=np.concatenate(cadence_parts),
        exposure_days=np.concatenate(exposure_parts),
        mission=np.concatenate(mission_parts),
        flux_product=np.concatenate(flux_product_parts),
    )
    order = np.argsort(observations.time)
    observations = _reorder(observations, order)
    median_flux = float(np.nanmedian(observations.flux))
    return _scale_flux(observations, median_flux), metadata_rows


def observations_from_light_curve(light_curve, flux_column: str) -> ObservationSet:
    """Fallback observation set for synthetic tests and stitched-only inputs."""
    time = np.asarray(light_curve.time.value, dtype=float)
    flux = np.asarray(light_curve.flux.value, dtype=float)
    finite = np.isfinite(time) & np.isfinite(flux)
    err = _flux_err_array(light_curve, len(time))
    observations = ObservationSet(
        time=time[finite],
        flux=flux[finite],
        flux_err=err[finite],
        product_id=np.full(np.count_nonzero(finite), "synthetic_or_stitched", dtype=object),
        quarter=np.full(np.count_nonzero(finite), "", dtype=object),
        cadence=np.full(np.count_nonzero(finite), "long", dtype=object),
        exposure_days=np.full(np.count_nonzero(finite), KEPLER_LONG_CADENCE_EXPOSURE_DAYS, dtype=float),
        mission=np.full(np.count_nonzero(finite), "synthetic_or_stitched", dtype=object),
        flux_product=np.full(np.count_nonzero(finite), flux_column, dtype=object),
    )
    return _scale_flux(observations, float(np.nanmedian(observations.flux)))


def build_transit_windows(
    observations: ObservationSet,
    *,
    period_days: float,
    transit_time: float,
    duration_days: float,
    config: Phase1BConfig,
) -> TransitWindows:
    """Predict, audit, and select all transit windows using objective coverage rules."""
    centers = predicted_centers(observations.time, period_days, transit_time)
    accepted_mask = np.zeros(len(observations), dtype=bool)
    event_numbers = np.full(len(observations), -1, dtype=int)
    center_by_cadence = np.full(len(observations), np.nan, dtype=float)
    audit_rows: list[dict[str, Any]] = []
    requested_half_width = config.window_half_width_duration_scale * duration_days
    cap = config.window_period_cap_fraction * period_days
    half_width = min(requested_half_width, cap)
    cap_applied = half_width < requested_half_width
    baseline_inner = config.baseline_inner_duration_scale * duration_days

    for sequence, center in enumerate(centers):
        phase = observations.time - center
        window_mask = np.abs(phase) <= half_width
        expected_mask = np.abs(phase) <= duration_days / 2.0
        pre_mask = (phase >= -half_width) & (phase <= -baseline_inner)
        post_mask = (phase >= baseline_inner) & (phase <= half_width)
        reasons = _coverage_reasons(
            observations.time,
            phase,
            expected_mask,
            pre_mask,
            post_mask,
            duration_days,
            config,
        )
        included = len(reasons) == 0
        if included:
            indices = np.flatnonzero(window_mask)
            accepted_mask[indices] = True
            event_numbers[indices] = sequence
            center_by_cadence[indices] = center
        local_indices = np.flatnonzero(window_mask)
        audit_rows.append(
            {
                "event_number": int(sequence),
                "predicted_midpoint": float(center),
                "source_products": _joined_mode(observations.product_id[local_indices]),
                "quarters": _joined_mode(observations.quarter[local_indices]),
                "window_start": float(center - half_width),
                "window_end": float(center + half_width),
                "requested_half_width_days": float(requested_half_width),
                "effective_half_width_days": float(half_width),
                "half_width_period_cap_applied": bool(cap_applied),
                "expected_transit_start": float(center - duration_days / 2.0),
                "expected_transit_end": float(center + duration_days / 2.0),
                "pre_baseline_start": float(center - half_width),
                "pre_baseline_end": float(center - baseline_inner),
                "post_baseline_start": float(center + baseline_inner),
                "post_baseline_end": float(center + half_width),
                "total_point_count": int(np.count_nonzero(window_mask)),
                "expected_in_transit_point_count": int(np.count_nonzero(expected_mask)),
                "left_in_transit_point_count": int(np.count_nonzero(expected_mask & (phase < 0.0))),
                "right_in_transit_point_count": int(np.count_nonzero(expected_mask & (phase >= 0.0))),
                "pre_transit_baseline_count": int(np.count_nonzero(pre_mask)),
                "post_transit_baseline_count": int(np.count_nonzero(post_mask)),
                "largest_local_gap_days": _largest_gap(observations.time[window_mask]),
                "largest_expected_transit_gap_days": _largest_gap(observations.time[expected_mask]),
                "exposure_duration_days": _unique_or_nan(observations.exposure_days[local_indices]),
                "included": bool(included),
                "exclusion_reasons": ";".join(reasons),
            }
        )
    return TransitWindows(audit_rows, accepted_mask, event_numbers, center_by_cadence)


def predicted_centers(time: np.ndarray, period: float, transit_time: float) -> np.ndarray:
    first = int(np.ceil((np.nanmin(time) - transit_time) / period))
    last = int(np.floor((np.nanmax(time) - transit_time) / period))
    if last < first:
        return np.asarray([], dtype=float)
    return transit_time + np.arange(first, last + 1) * period


def _coverage_reasons(
    time: np.ndarray,
    phase: np.ndarray,
    expected_mask: np.ndarray,
    pre_mask: np.ndarray,
    post_mask: np.ndarray,
    duration_days: float,
    config: Phase1BConfig,
) -> list[str]:
    reasons: list[str] = []
    in_count = int(np.count_nonzero(expected_mask))
    if in_count < config.minimum_in_transit_points:
        reasons.append("too_few_expected_in_transit_points")
    if np.count_nonzero(expected_mask & (phase < 0.0)) < config.minimum_points_each_side_in_transit:
        reasons.append("missing_pre_midpoint_in_transit_point")
    if np.count_nonzero(expected_mask & (phase >= 0.0)) < config.minimum_points_each_side_in_transit:
        reasons.append("missing_post_midpoint_in_transit_point")
    if np.count_nonzero(pre_mask) < config.minimum_pre_baseline_points:
        reasons.append("too_few_pre_transit_baseline_points")
    if np.count_nonzero(post_mask) < config.minimum_post_baseline_points:
        reasons.append("too_few_post_transit_baseline_points")
    expected_times = np.sort(time[expected_mask])
    if len(expected_times) >= 2 and np.nanmax(np.diff(expected_times)) > duration_days / 2.0:
        reasons.append("gap_through_expected_transit")
    return reasons


def _flux_err_array(light_curve, length: int) -> np.ndarray:
    try:
        values = np.asarray(light_curve.flux_err.value, dtype=float)
    except (AttributeError, TypeError, ValueError):
        values = np.full(length, np.nan, dtype=float)
    if len(values) != length:
        values = np.full(length, np.nan, dtype=float)
    valid = np.isfinite(values) & (values > 0.0)
    if np.any(valid):
        fill = float(np.nanmedian(values[valid]))
        values = np.where(valid, values, fill)
    else:
        values = np.full(length, 1.0, dtype=float)
    return values


def _scale_flux(observations: ObservationSet, scale: float) -> ObservationSet:
    if not np.isfinite(scale) or scale == 0.0:
        scale = 1.0
    return ObservationSet(
        time=observations.time,
        flux=observations.flux / scale,
        flux_err=observations.flux_err / abs(scale),
        product_id=observations.product_id,
        quarter=observations.quarter,
        cadence=observations.cadence,
        exposure_days=observations.exposure_days,
        mission=observations.mission,
        flux_product=observations.flux_product,
    )


def _reorder(observations: ObservationSet, order: np.ndarray) -> ObservationSet:
    return ObservationSet(
        time=observations.time[order],
        flux=observations.flux[order],
        flux_err=observations.flux_err[order],
        product_id=observations.product_id[order],
        quarter=observations.quarter[order],
        cadence=observations.cadence[order],
        exposure_days=observations.exposure_days[order],
        mission=observations.mission[order],
        flux_product=observations.flux_product[order],
    )


def _exposure_days(meta: dict[str, Any], cadence: str) -> tuple[float, dict[str, Any]]:
    if "INT_TIME" in meta and "NUM_FRM" in meta:
        integration_seconds = float(meta["INT_TIME"]) * float(meta["NUM_FRM"])
        days = integration_seconds / 86400.0
        return days, {
            "exposure_metadata_key": "INT_TIME;NUM_FRM",
            "exposure_raw_value": f"INT_TIME={meta['INT_TIME']};NUM_FRM={meta['NUM_FRM']}",
            "exposure_conversion_formula": "INT_TIME_seconds * NUM_FRM / 86400",
            "exposure_duration_days": days,
            "exposure_fallback": "",
        }
    if "FRAMETIM" in meta and "NUM_FRM" in meta:
        frame_seconds = float(meta["FRAMETIM"]) * float(meta["NUM_FRM"])
        days = frame_seconds / 86400.0
        return days, {
            "exposure_metadata_key": "FRAMETIM;NUM_FRM",
            "exposure_raw_value": f"FRAMETIM={meta['FRAMETIM']};NUM_FRM={meta['NUM_FRM']}",
            "exposure_conversion_formula": "FRAMETIM_seconds * NUM_FRM / 86400",
            "exposure_duration_days": days,
            "exposure_fallback": "frame_time_used_when_integration_time_missing",
        }
    if "TIMEDEL" in meta:
        days = float(meta["TIMEDEL"])
        return days, {
            "exposure_metadata_key": "TIMEDEL",
            "exposure_raw_value": days,
            "exposure_conversion_formula": "TIMEDEL_days",
            "exposure_duration_days": days,
            "exposure_fallback": "cadence_interval_used_when_integration_metadata_missing",
        }
    if str(cadence).lower().startswith("long"):
        return KEPLER_LONG_CADENCE_EXPOSURE_DAYS, {
            "exposure_metadata_key": "",
            "exposure_raw_value": "",
            "exposure_conversion_formula": "1765.5 seconds / 86400",
            "exposure_duration_days": KEPLER_LONG_CADENCE_EXPOSURE_DAYS,
            "exposure_fallback": "documented_kepler_long_cadence_integration_time",
        }
    raise ValueError("Exposure duration metadata unavailable for non-long cadence product.")


def _meta_first(meta: dict[str, Any], keys: tuple[str, ...], default: Any) -> Any:
    for key in keys:
        if key in meta:
            return meta[key]
    return default


def _product_id(product, index: int) -> str:
    for key in ("FILENAME", "filename", "FILE", "LABEL", "OBJECT"):
        value = product.meta.get(key)
        if value:
            path = Path(str(value))
            return path.name if path.name else str(value)
    return f"product_{index:02d}"


def _joined_mode(values: np.ndarray) -> str:
    if len(values) == 0:
        return ""
    counts = Counter(str(value) for value in values)
    return ";".join(value for value, _ in counts.most_common())


def _largest_gap(values: np.ndarray) -> float:
    if len(values) < 2:
        return float("nan")
    return float(np.nanmax(np.diff(np.sort(values))))


def _unique_or_nan(values: np.ndarray) -> float:
    if len(values) == 0:
        return float("nan")
    unique = np.unique(values.astype(float))
    if len(unique) == 1:
        return float(unique[0])
    return float("nan")
