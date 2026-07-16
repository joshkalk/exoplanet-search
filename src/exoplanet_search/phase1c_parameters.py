"""Phase 1C parameter transformations, timing references, and priors."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.special import ndtr

from .phase1b_model import inclination_degrees, q_to_u, validate_physical
from .phase1b_model import PhysicalParameters as Phase1BPhysicalParameters
from .phase1c_types import FrozenPhase1BData, PARAMETER_ORDER, Phase1CConfig, PhysicalSample, TimingReference


def build_timing_reference(data: FrozenPhase1BData, config: Phase1CConfig) -> TimingReference:
    """Build the mid-mission timing reference from the Phase 1B timing-refined solution."""
    period = float(data.deterministic_parameters["period_days"])
    original_epoch = float(data.deterministic_parameters["transit_time"])
    median_time = float(np.median(data.time))
    mid_cycle = int(np.rint((median_time - original_epoch) / period))
    mid_epoch = original_epoch + mid_cycle * period
    duration = float(
        data.phase1b_summary["established_inputs"]["full_mission_local_refinement"][
            "refined_duration_days"
        ]
    )
    phase1a_period = float(
        data.phase1b_summary["established_inputs"]["full_mission_local_refinement"][
            "refined_period_days"
        ]
    )
    period_half_width = max(0.02 * phase1a_period, duration)
    scale = float(
        data.phase1b_configuration.get(
            "timing_refinement_t0_half_width_duration_scale",
            config.mid_epoch_half_width_duration_scale,
        )
    )
    mid_half_width = scale * duration
    period_half_width = min(
        period_half_width,
        _fixed_event_compatible_period_half_width(data, mid_cycle, mid_half_width),
    )
    reference = TimingReference(
        period_reference=period,
        original_epoch_reference=original_epoch,
        mid_epoch_reference=mid_epoch,
        mid_epoch_cycle=mid_cycle,
        period_half_width=period_half_width,
        mid_epoch_half_width=mid_half_width,
    )
    validate_timing_support(data, reference)
    return reference


def _fixed_event_compatible_period_half_width(
    data: FrozenPhase1BData,
    mid_cycle: int,
    mid_epoch_half_width: float,
) -> float:
    """Return the largest symmetric period offset preserving frozen event windows."""
    widths = []
    for event in np.unique(data.event_number):
        mask = data.event_number == event
        frozen_center = float(np.median(data.predicted_center[mask]))
        lower_margin = frozen_center - float(np.min(data.time[mask]))
        upper_margin = float(np.max(data.time[mask])) - frozen_center
        nearest_window_edge_margin = min(lower_margin, upper_margin)
        remaining = nearest_window_edge_margin - mid_epoch_half_width
        if remaining <= 0.0:
            raise ValueError("Phase 1C mid-epoch support exceeds at least one frozen event window.")
        cycle_distance = abs(int(event) - int(mid_cycle))
        if cycle_distance > 0:
            widths.append(remaining / cycle_distance)
    if not widths:
        raise ValueError("Cannot derive period support from frozen Phase 1B event labels.")
    return float(0.95 * min(widths))


def validate_timing_support(data: FrozenPhase1BData, timing: TimingReference) -> None:
    """Ensure full timing support does not relabel Phase 1B fixed event assignments."""
    event_ids = np.unique(data.event_number)
    for period in (
        timing.period_reference - timing.period_half_width,
        timing.period_reference + timing.period_half_width,
    ):
        if not np.isfinite(period) or period <= 0.0:
            raise ValueError("Phase 1C timing support includes a nonpositive period.")
    for period_offset in (-timing.period_half_width, timing.period_half_width):
        for mid_offset in (-timing.mid_epoch_half_width, timing.mid_epoch_half_width):
            period = timing.period_reference + period_offset
            original_epoch = timing.mid_epoch_reference + mid_offset - timing.mid_epoch_cycle * period
            for event in event_ids:
                mask = data.event_number == event
                proposed_center = original_epoch + int(event) * period
                frozen_center = float(np.median(data.predicted_center[mask]))
                window_half_width = float(np.max(np.abs(data.time[mask] - frozen_center)))
                if abs(proposed_center - frozen_center) > window_half_width:
                    raise ValueError(
                        "Phase 1C timing prior can move a fixed event outside its frozen "
                        f"accepted window: event={int(event)}"
                    )


def timing_support_audit(data: FrozenPhase1BData, timing: TimingReference) -> dict[str, Any]:
    """Return a machine-readable audit of fixed-event timing support."""
    event_ids = np.unique(data.event_number).astype(int)
    duration = float(
        data.phase1b_summary["established_inputs"]["full_mission_local_refinement"][
            "refined_duration_days"
        ]
    )
    rows = []
    all_centers_inside = True
    all_complete_inside = True
    max_center_displacement = 0.0
    min_center_margin = math.inf
    min_complete_margin = math.inf
    for period_offset in (-timing.period_half_width, timing.period_half_width):
        for mid_offset in (-timing.mid_epoch_half_width, timing.mid_epoch_half_width):
            period = timing.period_reference + period_offset
            original_epoch = timing.mid_epoch_reference + mid_offset - timing.mid_epoch_cycle * period
            corner_rows = []
            for event in event_ids:
                mask = data.event_number == event
                proposed_center = original_epoch + int(event) * period
                frozen_center = float(np.median(data.predicted_center[mask]))
                window_min = float(np.min(data.time[mask]))
                window_max = float(np.max(data.time[mask]))
                displacement = abs(proposed_center - frozen_center)
                center_margin = min(proposed_center - window_min, window_max - proposed_center)
                complete_margin = min(
                    proposed_center - 0.5 * duration - window_min,
                    window_max - (proposed_center + 0.5 * duration),
                )
                center_inside = center_margin >= 0.0
                complete_inside = complete_margin >= 0.0
                all_centers_inside = all_centers_inside and center_inside
                all_complete_inside = all_complete_inside and complete_inside
                max_center_displacement = max(max_center_displacement, float(displacement))
                min_center_margin = min(min_center_margin, float(center_margin))
                min_complete_margin = min(min_complete_margin, float(complete_margin))
                corner_rows.append(
                    {
                        "event_number": int(event),
                        "proposed_center": float(proposed_center),
                        "frozen_center": frozen_center,
                        "center_displacement_days": float(displacement),
                        "center_to_window_edge_margin_days": float(center_margin),
                        "complete_transit_margin_days": float(complete_margin),
                        "center_inside_frozen_window": bool(center_inside),
                        "complete_nominal_transit_inside_frozen_window": bool(complete_inside),
                    }
                )
            rows.append(
                {
                    "period_offset_days": float(period_offset),
                    "mid_epoch_offset_days": float(mid_offset),
                    "period_days": float(period),
                    "original_epoch_days": float(original_epoch),
                    "maximum_center_displacement_days": max(
                        row["center_displacement_days"] for row in corner_rows
                    ),
                    "minimum_center_to_window_edge_margin_days": min(
                        row["center_to_window_edge_margin_days"] for row in corner_rows
                    ),
                    "minimum_complete_transit_margin_days": min(
                        row["complete_transit_margin_days"] for row in corner_rows
                    ),
                    "center_remains_inside_every_frozen_window": all(
                        row["center_inside_frozen_window"] for row in corner_rows
                    ),
                    "complete_nominal_transit_remains_inside_every_frozen_window": all(
                        row["complete_nominal_transit_inside_frozen_window"] for row in corner_rows
                    ),
                    "event_rows": corner_rows,
                }
            )
    return {
        "earliest_frozen_event_number": int(np.min(event_ids)),
        "latest_frozen_event_number": int(np.max(event_ids)),
        "timing_reference": timing.__dict__,
        "nominal_duration_days": duration,
        "support_corners": rows,
        "maximum_center_displacement_days": float(max_center_displacement),
        "minimum_center_to_window_edge_margin_days": float(min_center_margin),
        "minimum_complete_transit_margin_days": float(min_complete_margin),
        "center_remains_inside_every_frozen_window": bool(all_centers_inside),
        "complete_nominal_transit_remains_inside_every_frozen_window": bool(all_complete_inside),
        "period_support_rule": (
            "symmetric Phase 1B timing-refinement period support narrowed only as needed "
            "to keep fixed event centers inside the actual nearest edge of every frozen "
            "accepted cadence window at all joint period/mid-epoch support corners; "
            "a 0.95 guard factor is applied to the minimum compatible period half-width."
        ),
    }


def vector_to_physical(vector: np.ndarray, timing: TimingReference) -> PhysicalSample:
    """Convert transformed coordinates to physical transit parameters."""
    if len(vector) != len(PARAMETER_ORDER):
        raise ValueError(f"Expected {len(PARAMETER_ORDER)} parameters, received {len(vector)}.")
    log_rp, log_a, z_b, q1, q2, log_jitter, period_offset, mid_epoch_offset = map(float, vector)
    rp = math.exp(log_rp)
    a = math.exp(log_a)
    b = z_b * (1.0 + rp)
    jitter = math.exp(log_jitter)
    period = timing.period_reference + period_offset
    mid_epoch = timing.mid_epoch_reference + mid_epoch_offset
    original_epoch = mid_epoch - timing.mid_epoch_cycle * period
    return PhysicalSample(
        rp=rp,
        a=a,
        b=b,
        q1=q1,
        q2=q2,
        jitter=jitter,
        period=period,
        mid_epoch=mid_epoch,
        original_epoch=original_epoch,
    )


def physical_to_vector(sample: PhysicalSample, timing: TimingReference) -> np.ndarray:
    """Convert physical transit parameters to transformed Phase 1C coordinates."""
    if sample.rp <= 0.0 or sample.a <= 0.0 or sample.jitter <= 0.0:
        raise ValueError("Cannot transform nonpositive radius, semimajor axis, or jitter.")
    return np.asarray(
        [
            math.log(sample.rp),
            math.log(sample.a),
            sample.b / (1.0 + sample.rp),
            sample.q1,
            sample.q2,
            math.log(sample.jitter),
            sample.period - timing.period_reference,
            sample.mid_epoch - timing.mid_epoch_reference,
        ],
        dtype=float,
    )


def deterministic_physical_sample(
    data: FrozenPhase1BData,
    timing: TimingReference,
    *,
    jitter_floor: float | None = None,
) -> PhysicalSample:
    """Return the Phase 1B timing-refined solution as a Phase 1C initialization point."""
    params = data.deterministic_parameters
    jitter = float(params["white_noise_jitter"])
    if jitter_floor is not None:
        jitter = max(jitter, float(jitter_floor))
    period = float(params["period_days"])
    original_epoch = float(params["transit_time"])
    mid_epoch = original_epoch + timing.mid_epoch_cycle * period
    return PhysicalSample(
        rp=float(params["rp_over_rstar"]),
        a=float(params["a_over_rstar"]),
        b=float(params["impact_parameter"]),
        q1=float(params["q1"]),
        q2=float(params["q2"]),
        jitter=jitter,
        period=period,
        mid_epoch=mid_epoch,
        original_epoch=original_epoch,
    )


def log_prior(
    vector: np.ndarray,
    data: FrozenPhase1BData,
    config: Phase1CConfig,
    timing: TimingReference,
) -> float:
    """Return the normalized transformed-coordinate log-prior density."""
    if not np.all(np.isfinite(vector)):
        return -math.inf
    sample = vector_to_physical(vector, timing)
    failures = validate_physical(
        Phase1BPhysicalParameters(
            rp=sample.rp,
            a=sample.a,
            b=sample.b,
            q1=sample.q1,
            q2=sample.q2,
            jitter=sample.jitter,
            period=sample.period,
            t0=sample.original_epoch,
        )
    )
    if failures or sample.period <= 0.0:
        return -math.inf
    if not config.rp_bounds[0] <= sample.rp <= config.rp_bounds[1]:
        return -math.inf
    if not config.a_bounds[0] <= sample.a <= config.a_bounds[1]:
        return -math.inf
    if not 0.0 <= vector[2] <= 1.0:
        return -math.inf
    if not config.q_bounds[0] <= sample.q1 <= config.q_bounds[1]:
        return -math.inf
    if not config.q_bounds[0] <= sample.q2 <= config.q_bounds[1]:
        return -math.inf
    if not config.jitter_lower <= sample.jitter <= config.jitter_upper:
        return -math.inf
    if abs(vector[6]) > timing.period_half_width or abs(vector[7]) > timing.mid_epoch_half_width:
        return -math.inf

    rp_width = config.rp_bounds[1] - config.rp_bounds[0]
    logp = -math.log(rp_width) + math.log(sample.rp)
    logp += -math.log(math.log(config.a_bounds[1] / config.a_bounds[0]))
    logp += truncated_normal_logpdf(
        sample.q1,
        float(data.limb_darkening["q1"]),
        max(float(data.limb_darkening["q1_sigma"]), config.limb_darkening_sigma_floor),
        config.q_bounds,
    )
    logp += truncated_normal_logpdf(
        sample.q2,
        float(data.limb_darkening["q2"]),
        max(float(data.limb_darkening["q2_sigma"]), config.limb_darkening_sigma_floor),
        config.q_bounds,
    )
    jitter_scale = jitter_prior_scale(data, config)
    logp += bounded_half_normal_logpdf(
        sample.jitter,
        jitter_scale,
        config.jitter_lower,
        config.jitter_upper,
    ) + math.log(sample.jitter)
    logp += -math.log(2.0 * timing.period_half_width)
    logp += -math.log(2.0 * timing.mid_epoch_half_width)
    return float(logp)


def truncated_normal_logpdf(
    value: float,
    mean: float,
    sigma: float,
    bounds: tuple[float, float],
) -> float:
    """Normalized normal log density truncated to finite bounds."""
    lower, upper = bounds
    if sigma <= 0.0 or not lower <= value <= upper:
        return -math.inf
    alpha = (lower - mean) / sigma
    beta = (upper - mean) / sigma
    normalization = ndtr(beta) - ndtr(alpha)
    if normalization <= 0.0:
        raise ValueError("Truncated-normal normalization is nonpositive.")
    z = (value - mean) / sigma
    return float(-0.5 * z * z - math.log(sigma) - 0.5 * math.log(2.0 * math.pi) - math.log(normalization))


def half_normal_logpdf(value: float, scale: float) -> float:
    """Normalized half-normal density for positive physical jitter."""
    if scale <= 0.0:
        raise ValueError("Half-normal scale must be positive.")
    if value < 0.0:
        return -math.inf
    return float(0.5 * math.log(2.0 / math.pi) - math.log(scale) - 0.5 * (value / scale) ** 2)


def bounded_half_normal_logpdf(value: float, scale: float, lower: float, upper: float) -> float:
    """Half-normal density normalized over finite configured jitter bounds."""
    if not lower <= value <= upper:
        return -math.inf
    if lower < 0.0 or upper <= lower:
        raise ValueError("Bounded half-normal requires 0 <= lower < upper.")
    normalization = _half_normal_cdf(upper, scale) - _half_normal_cdf(lower, scale)
    if normalization <= 0.0:
        raise ValueError("Bounded half-normal normalization is nonpositive.")
    return half_normal_logpdf(value, scale) - math.log(normalization)


def _half_normal_cdf(value: float, scale: float) -> float:
    if value < 0.0:
        return 0.0
    return float(2.0 * ndtr(value / scale) - 1.0)


def jitter_prior_scale(data: FrozenPhase1BData, config: Phase1CConfig) -> float:
    positive = data.flux_uncertainty[data.flux_uncertainty > 0.0]
    if positive.size == 0:
        raise ValueError("Cannot derive jitter prior scale without positive cadence uncertainties.")
    return float(config.jitter_prior_median_uncertainty_multiple * np.median(positive))


def physical_parameter_row(sample: PhysicalSample) -> dict[str, float]:
    """Return output-friendly physical coordinates derived from a sample."""
    u1, u2 = q_to_u(sample.q1, sample.q2)
    return {
        "rp_over_rstar": float(sample.rp),
        "a_over_rstar": float(sample.a),
        "impact_parameter": float(sample.b),
        "inclination_degrees": inclination_degrees(sample.a, sample.b),
        "q1": float(sample.q1),
        "q2": float(sample.q2),
        "u1": float(u1),
        "u2": float(u2),
        "white_noise_jitter": float(sample.jitter),
        "period_days": float(sample.period),
        "transit_time_original_reference": float(sample.original_epoch),
        "transit_time_mid_mission_reference": float(sample.mid_epoch),
    }


def prior_description(data: FrozenPhase1BData, config: Phase1CConfig, timing: TimingReference) -> dict[str, Any]:
    """Machine-readable prior and transformation record."""
    return {
        "parameter_order": list(PARAMETER_ORDER),
        "transforms": {
            "rp_over_rstar": "sampled as log_rp with uniform physical prior and log-rp Jacobian",
            "a_over_rstar": "sampled as log_a with log-uniform physical prior",
            "impact_parameter": "b = z_b * (1 + rp), z_b uniform on [0, 1]",
            "jitter": "sampled as log_jitter with half-normal physical prior and log-jitter Jacobian",
            "timing": "period offset and mid-mission epoch offset from Phase 1B timing-refined solution",
        },
        "bounds": {
            "rp_over_rstar": list(config.rp_bounds),
            "a_over_rstar": list(config.a_bounds),
            "z_b": [0.0, 1.0],
            "q1": list(config.q_bounds),
            "q2": list(config.q_bounds),
            "jitter": [config.jitter_lower, config.jitter_upper],
            "period_offset_days": [-timing.period_half_width, timing.period_half_width],
            "mid_epoch_offset_days": [-timing.mid_epoch_half_width, timing.mid_epoch_half_width],
        },
        "limb_darkening_priors": {
            "q1_center": float(data.limb_darkening["q1"]),
            "q1_sigma": max(float(data.limb_darkening["q1_sigma"]), config.limb_darkening_sigma_floor),
            "q2_center": float(data.limb_darkening["q2"]),
            "q2_sigma": max(float(data.limb_darkening["q2_sigma"]), config.limb_darkening_sigma_floor),
            "sigma_floor": config.limb_darkening_sigma_floor,
        },
        "jitter_prior": {
            "family": "half-normal in physical jitter, normalized over finite configured bounds",
            "scale": jitter_prior_scale(data, config),
            "scale_source": "configured multiple of median positive cadence flux uncertainty",
            "median_uncertainty_multiple": config.jitter_prior_median_uncertainty_multiple,
        },
        "a_over_rstar_prior_interpretation": (
            "independent log-uniform prior over configured bounds followed by physical geometry rejection; "
            "not conditionally renormalized above 1 + Rp/Rstar"
        ),
        "baseline_priors": {
            "intercept": {"mean": 1.0, "sigma": config.baseline_intercept_sigma},
            "slope": {"mean": 0.0, "sigma": config.baseline_slope_sigma},
        },
        "timing_reference": timing.__dict__,
        "anti_leakage": {
            "published_planet_parameters_used": False,
            "residuals_csv_used_as_input": False,
        },
    }
