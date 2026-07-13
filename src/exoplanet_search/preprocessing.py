"""Transit-preserving preprocessing modes and cadence accounting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .recovery import compute_phase_days

PREPROCESSING_MODES = (
    "none",
    "positive_only",
    "symmetric",
    "transit_protected_symmetric",
)


@dataclass(frozen=True)
class PreprocessingConfig:
    """Configuration for light-curve preprocessing."""

    mode: str = "none"
    sigma: float = 5.0
    normalize: bool = True
    period_days: float | None = None
    epoch_bkjd: float | None = None
    duration_hours: float | None = None
    transit_mask_scale: float = 1.25


@dataclass(frozen=True)
class PreprocessingResult:
    """A processed light curve with auditable cadence masks and counts."""

    light_curve: Any
    config: PreprocessingConfig
    input_count: int
    finite_count: int
    quality_filtered_count: int | None
    clipped_count: int
    output_count: int
    finite_mask: np.ndarray
    retained_mask: np.ndarray
    clipped_mask: np.ndarray
    normalized_by: float | None

    @property
    def removed_count(self) -> int:
        return self.input_count - self.output_count

    @property
    def removed_fraction(self) -> float:
        if self.input_count == 0:
            return 0.0
        return self.removed_count / self.input_count

    def summary(self) -> dict[str, Any]:
        return {
            "mode": self.config.mode,
            "parameters": preprocessing_parameters(self.config),
            "cadence_count_before_preprocessing": self.input_count,
            "cadence_count_after_nonfinite_removal": self.finite_count,
            "cadence_count_after_quality_filtering": self.quality_filtered_count,
            "cadence_count_after_flux_clipping": self.output_count,
            "nonfinite_removed_cadence_count": int(np.count_nonzero(~self.finite_mask)),
            "clipped_cadence_count": self.clipped_count,
            "total_removed_cadence_count": self.removed_count,
            "total_removed_fraction": self.removed_fraction,
            "normalization": {
                "enabled": self.config.normalize,
                "method": "divide_by_median_flux" if self.config.normalize else "none",
                "median_flux_before_normalization": self.normalized_by,
            },
        }


def validate_preprocessing_mode(mode: str) -> str:
    """Return a valid preprocessing mode or raise a helpful error."""
    if mode not in PREPROCESSING_MODES:
        choices = ", ".join(PREPROCESSING_MODES)
        raise ValueError(f"Unknown preprocessing mode {mode!r}. Expected one of: {choices}.")
    return mode


def preprocessing_parameters(config: PreprocessingConfig) -> dict[str, Any]:
    """Return JSON-serializable preprocessing parameters."""
    parameters: dict[str, Any] = {
        "sigma": float(config.sigma),
        "normalize": bool(config.normalize),
    }
    if config.mode == "none":
        parameters["flux_clipping"] = "none"
    elif config.mode == "positive_only":
        parameters["flux_clipping"] = "sigma_clip_upper_only"
        parameters["sigma_upper"] = float(config.sigma)
        parameters["sigma_lower"] = None
    elif config.mode == "symmetric":
        parameters["flux_clipping"] = "symmetric_sigma_clip"
        parameters["sigma_lower"] = float(config.sigma)
        parameters["sigma_upper"] = float(config.sigma)
    elif config.mode == "transit_protected_symmetric":
        parameters["flux_clipping"] = "symmetric_sigma_clip_outside_known_transit_window"
        parameters["sigma_lower"] = float(config.sigma)
        parameters["sigma_upper"] = float(config.sigma)
        parameters["period_days"] = config.period_days
        parameters["epoch_bkjd"] = config.epoch_bkjd
        parameters["duration_hours"] = config.duration_hours
        parameters["transit_mask_scale"] = float(config.transit_mask_scale)
    return parameters


def preprocess_light_curve(light_curve, config: PreprocessingConfig | None = None) -> PreprocessingResult:
    """Preprocess a light curve using a named, transit-aware mode.

    The default scientific mode performs no generic flux-amplitude clipping:
    it removes non-finite values and normalizes by the median flux only.
    """
    config = config or PreprocessingConfig()
    validate_preprocessing_mode(config.mode)

    input_count = len(light_curve)
    finite_mask = _finite_mask(light_curve)
    finite_curve = light_curve[finite_mask]

    clip_mask_finite = _clip_mask(finite_curve, config)
    retained_finite_mask = ~clip_mask_finite
    clipped_curve = finite_curve[retained_finite_mask]

    normalized_by: float | None = None
    processed_curve = clipped_curve
    if config.normalize:
        flux = np.asarray(clipped_curve.flux.value, dtype=float)
        normalized_by = float(np.nanmedian(flux))
        processed_curve = clipped_curve.normalize()

    retained_mask = np.zeros(input_count, dtype=bool)
    finite_indices = np.flatnonzero(finite_mask)
    retained_mask[finite_indices[retained_finite_mask]] = True

    clipped_mask = np.zeros(input_count, dtype=bool)
    clipped_mask[finite_indices[clip_mask_finite]] = True

    return PreprocessingResult(
        light_curve=processed_curve,
        config=config,
        input_count=input_count,
        finite_count=int(np.count_nonzero(finite_mask)),
        quality_filtered_count=None,
        clipped_count=int(np.count_nonzero(clip_mask_finite)),
        output_count=len(processed_curve),
        finite_mask=finite_mask,
        retained_mask=retained_mask,
        clipped_mask=clipped_mask,
        normalized_by=normalized_by,
    )


def lightly_preprocess_light_curve(light_curve, sigma: float = 5.0, normalize: bool = True):
    """Backward-compatible wrapper using the new default no-clipping mode."""
    result = preprocess_light_curve(
        light_curve,
        PreprocessingConfig(mode="none", sigma=sigma, normalize=normalize),
    )
    return result.light_curve


def transit_window_mask(
    light_curve,
    period_days: float,
    epoch_bkjd: float,
    duration_hours: float,
    transit_mask_scale: float = 1.25,
) -> np.ndarray:
    """Return cadences inside a known transit window for diagnostics only."""
    phase_days = compute_phase_days(light_curve, period_days=period_days, epoch_bkjd=epoch_bkjd)
    half_duration_days = duration_hours / 24.0 / 2.0
    return np.abs(phase_days) <= transit_mask_scale * half_duration_days


def removal_diagnostics(
    original_light_curve,
    result: PreprocessingResult,
    period_days: float,
    epoch_bkjd: float,
    duration_hours: float,
    transit_mask_scale: float = 1.25,
    phase_bins: int = 20,
) -> tuple[dict[str, int], list[dict[str, float | int]]]:
    """Count retained and removed cadences by known-transit phase diagnostics."""
    diagnostic_transit_mask = transit_window_mask(
        original_light_curve,
        period_days=period_days,
        epoch_bkjd=epoch_bkjd,
        duration_hours=duration_hours,
        transit_mask_scale=transit_mask_scale,
    )
    nonfinite_removed_mask = ~result.finite_mask
    clipped_removed_mask = result.clipped_mask
    removed_mask = nonfinite_removed_mask | clipped_removed_mask

    counts = {
        "retained_inside_known_transit_window": int(
            np.count_nonzero(result.retained_mask & diagnostic_transit_mask)
        ),
        "retained_outside_known_transit_window": int(
            np.count_nonzero(result.retained_mask & ~diagnostic_transit_mask)
        ),
        "nonfinite_removed_inside_known_transit_window": int(
            np.count_nonzero(nonfinite_removed_mask & diagnostic_transit_mask)
        ),
        "nonfinite_removed_outside_known_transit_window": int(
            np.count_nonzero(nonfinite_removed_mask & ~diagnostic_transit_mask)
        ),
        "flux_clipped_inside_known_transit_window": int(
            np.count_nonzero(clipped_removed_mask & diagnostic_transit_mask)
        ),
        "flux_clipped_outside_known_transit_window": int(
            np.count_nonzero(clipped_removed_mask & ~diagnostic_transit_mask)
        ),
        "total_removed_inside_known_transit_window": int(
            np.count_nonzero(removed_mask & diagnostic_transit_mask)
        ),
        "total_removed_outside_known_transit_window": int(
            np.count_nonzero(removed_mask & ~diagnostic_transit_mask)
        ),
    }

    phase_days = compute_phase_days(
        original_light_curve,
        period_days=period_days,
        epoch_bkjd=epoch_bkjd,
    )
    phase_fraction = (phase_days / period_days) + 0.5
    edges = np.linspace(0.0, 1.0, phase_bins + 1)
    rows: list[dict[str, float | int]] = []
    for index in range(phase_bins):
        if index == phase_bins - 1:
            bin_mask = (phase_fraction >= edges[index]) & (phase_fraction <= edges[index + 1])
        else:
            bin_mask = (phase_fraction >= edges[index]) & (phase_fraction < edges[index + 1])
        total = int(np.count_nonzero(bin_mask))
        nonfinite_removed = int(np.count_nonzero(bin_mask & nonfinite_removed_mask))
        flux_clipped = int(np.count_nonzero(bin_mask & clipped_removed_mask))
        total_removed = nonfinite_removed + flux_clipped
        rows.append(
            {
                "phase_bin": index,
                "phase_start": float(edges[index]),
                "phase_end": float(edges[index + 1]),
                "cadence_count": total,
                "nonfinite_removed_count": nonfinite_removed,
                "flux_clipped_count": flux_clipped,
                "total_removed_count": total_removed,
                "total_removed_fraction": float(total_removed / total) if total else 0.0,
            }
        )
    return counts, rows


def _finite_mask(light_curve) -> np.ndarray:
    time = np.asarray(light_curve.time.value, dtype=float)
    flux = np.asarray(light_curve.flux.value, dtype=float)
    return np.isfinite(time) & np.isfinite(flux)


def _clip_mask(light_curve, config: PreprocessingConfig) -> np.ndarray:
    if config.mode == "none":
        return np.zeros(len(light_curve), dtype=bool)
    if config.mode == "positive_only":
        return _lightkurve_outlier_mask(
            light_curve,
            sigma=config.sigma,
            sigma_lower=np.inf,
            sigma_upper=config.sigma,
        )
    if config.mode == "symmetric":
        return _lightkurve_outlier_mask(
            light_curve,
            sigma=config.sigma,
            sigma_lower=config.sigma,
            sigma_upper=config.sigma,
        )
    if config.mode == "transit_protected_symmetric":
        if (
            config.period_days is None
            or config.epoch_bkjd is None
            or config.duration_hours is None
        ):
            raise ValueError(
                "transit_protected_symmetric requires period_days, epoch_bkjd, "
                "and duration_hours."
            )
        protected = transit_window_mask(
            light_curve,
            period_days=config.period_days,
            epoch_bkjd=config.epoch_bkjd,
            duration_hours=config.duration_hours,
            transit_mask_scale=config.transit_mask_scale,
        )
        outside = ~protected
        mask = np.zeros(len(light_curve), dtype=bool)
        if np.any(outside):
            outside_mask = _lightkurve_outlier_mask(
                light_curve[outside],
                sigma=config.sigma,
                sigma_lower=config.sigma,
                sigma_upper=config.sigma,
            )
            mask[np.flatnonzero(outside)[outside_mask]] = True
        return mask
    raise ValueError(f"Unknown preprocessing mode {config.mode!r}.")


def _lightkurve_outlier_mask(
    light_curve,
    sigma: float,
    sigma_lower: float | None,
    sigma_upper: float | None,
) -> np.ndarray:
    _, outlier_mask = light_curve.remove_outliers(
        sigma=sigma,
        sigma_lower=sigma_lower,
        sigma_upper=sigma_upper,
        return_mask=True,
    )
    return np.asarray(outlier_mask, dtype=bool)
