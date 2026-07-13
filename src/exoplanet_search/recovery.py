"""Minimal known-transit recovery helpers for Kepler-5."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def compute_phase_days(light_curve, period_days: float, epoch_bkjd: float) -> np.ndarray:
    """Return wrapped orbital phase in days centered on the expected transit."""
    time_days = np.asarray(light_curve.time.value, dtype=float)
    return ((time_days - epoch_bkjd + 0.5 * period_days) % period_days) - 0.5 * period_days


def estimate_known_transit_signal(
    light_curve,
    period_days: float,
    epoch_bkjd: float,
    duration_hours: float,
) -> dict[str, float]:
    """Estimate the known transit depth using a simple phase-window comparison.

    This intentionally avoids aggressive detrending or fitting. It phase-folds the
    light curve on the expected ephemeris and compares the in-transit median flux
    to the out-of-transit median flux.
    """
    phase_days = compute_phase_days(light_curve, period_days=period_days, epoch_bkjd=epoch_bkjd)
    flux = np.asarray(light_curve.flux.value, dtype=float)

    half_duration_days = duration_hours / 24.0 / 2.0
    in_transit_mask = np.abs(phase_days) <= half_duration_days
    out_of_transit_mask = ~in_transit_mask

    if not np.any(in_transit_mask):
        raise ValueError("No in-transit cadences found for the provided ephemeris.")
    if np.count_nonzero(out_of_transit_mask) < 2:
        raise ValueError("Not enough out-of-transit cadences to estimate a baseline.")

    in_transit_flux = flux[in_transit_mask]
    out_of_transit_flux = flux[out_of_transit_mask]

    in_transit_median = float(np.nanmedian(in_transit_flux))
    out_of_transit_median = float(np.nanmedian(out_of_transit_flux))
    transit_depth = out_of_transit_median - in_transit_median

    out_of_transit_std = float(np.nanstd(out_of_transit_flux))
    depth_uncertainty = out_of_transit_std * np.sqrt(
        (1.0 / np.count_nonzero(in_transit_mask)) + (1.0 / np.count_nonzero(out_of_transit_mask))
    )
    if depth_uncertainty > 0:
        depth_snr_proxy = transit_depth / depth_uncertainty
    elif transit_depth > 0:
        depth_snr_proxy = float("inf")
    else:
        depth_snr_proxy = 0.0

    return {
        "period_days": float(period_days),
        "epoch_bkjd": float(epoch_bkjd),
        "duration_hours": float(duration_hours),
        "n_in_transit": int(np.count_nonzero(in_transit_mask)),
        "n_out_of_transit": int(np.count_nonzero(out_of_transit_mask)),
        "in_transit_flux_median": in_transit_median,
        "out_of_transit_flux_median": out_of_transit_median,
        "transit_depth": float(transit_depth),
        "transit_depth_ppm": float(transit_depth * 1.0e6),
        "out_of_transit_flux_std": out_of_transit_std,
        "depth_snr_proxy": float(depth_snr_proxy),
    }


def _compute_transit_centers(
    time_days: np.ndarray,
    period_days: float,
    epoch_bkjd: float,
) -> np.ndarray:
    first_index = int(np.ceil((np.nanmin(time_days) - epoch_bkjd) / period_days))
    last_index = int(np.floor((np.nanmax(time_days) - epoch_bkjd) / period_days))
    if last_index < first_index:
        return np.asarray([], dtype=float)
    indices = np.arange(first_index, last_index + 1, dtype=int)
    return epoch_bkjd + indices * period_days


def estimate_windowed_known_transit_signal(
    light_curve,
    period_days: float,
    epoch_bkjd: float,
    duration_hours: float,
    window_half_width_days: float,
    transit_mask_scale: float = 1.25,
    baseline_mask_scale: float = 2.0,
) -> dict[str, float]:
    """Recover a known transit using local windows around each expected event.

    Each transit window is normalized by the median flux in the local out-of-transit
    wings, which keeps the preprocessing conservative while reducing slow baseline
    variations that can dilute a full-mission folded estimate.
    """
    time_days = np.asarray(light_curve.time.value, dtype=float)
    flux = np.asarray(light_curve.flux.value, dtype=float)
    half_duration_days = duration_hours / 24.0 / 2.0
    in_transit_half_width_days = transit_mask_scale * half_duration_days
    baseline_min_phase_days = baseline_mask_scale * half_duration_days

    event_centers = _compute_transit_centers(time_days, period_days=period_days, epoch_bkjd=epoch_bkjd)
    window_phases: list[np.ndarray] = []
    window_fluxes: list[np.ndarray] = []
    used_event_centers: list[float] = []

    for center in event_centers:
        phase_days = time_days - center
        window_mask = np.abs(phase_days) <= window_half_width_days
        if np.count_nonzero(window_mask) < 10:
            continue

        local_phase = phase_days[window_mask]
        local_flux = flux[window_mask]
        in_transit_mask = np.abs(local_phase) <= in_transit_half_width_days
        wing_mask = np.abs(local_phase) > baseline_min_phase_days

        if np.count_nonzero(in_transit_mask) < 1 or np.count_nonzero(wing_mask) < 4:
            continue

        wing_median = float(np.nanmedian(local_flux[wing_mask]))
        if not np.isfinite(wing_median) or wing_median == 0.0:
            continue

        normalized_flux = local_flux / wing_median
        window_phases.append(local_phase)
        window_fluxes.append(normalized_flux)
        used_event_centers.append(float(center))

    if not window_phases:
        raise ValueError("No usable transit windows found for the provided ephemeris.")

    stacked_phase = np.concatenate(window_phases)
    stacked_flux = np.concatenate(window_fluxes)
    in_transit_mask = np.abs(stacked_phase) <= in_transit_half_width_days
    wing_mask = np.abs(stacked_phase) > baseline_min_phase_days

    in_transit_flux = stacked_flux[in_transit_mask]
    wing_flux = stacked_flux[wing_mask]

    in_transit_median = float(np.nanmedian(in_transit_flux))
    wing_median = float(np.nanmedian(wing_flux))
    transit_depth = wing_median - in_transit_median

    wing_std = float(np.nanstd(wing_flux))
    depth_uncertainty = wing_std * np.sqrt(
        (1.0 / np.count_nonzero(in_transit_mask)) + (1.0 / np.count_nonzero(wing_mask))
    )
    if depth_uncertainty > 0:
        depth_snr_proxy = transit_depth / depth_uncertainty
    elif transit_depth > 0:
        depth_snr_proxy = float("inf")
    else:
        depth_snr_proxy = 0.0

    return {
        "period_days": float(period_days),
        "epoch_bkjd": float(epoch_bkjd),
        "duration_hours": float(duration_hours),
        "window_half_width_days": float(window_half_width_days),
        "transit_mask_scale": float(transit_mask_scale),
        "baseline_mask_scale": float(baseline_mask_scale),
        "n_transit_windows": int(len(window_phases)),
        "n_stacked_in_transit": int(np.count_nonzero(in_transit_mask)),
        "n_stacked_out_of_transit": int(np.count_nonzero(wing_mask)),
        "windowed_in_transit_flux_median": in_transit_median,
        "windowed_out_of_transit_flux_median": wing_median,
        "windowed_transit_depth": float(transit_depth),
        "windowed_transit_depth_ppm": float(transit_depth * 1.0e6),
        "windowed_out_of_transit_flux_std": wing_std,
        "windowed_depth_snr_proxy": float(depth_snr_proxy),
        "event_centers_bkjd": used_event_centers,
    }


def _bin_folded_light_curve(phase_days: np.ndarray, flux: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(np.nanmin(phase_days), np.nanmax(phase_days), bins + 1)
    indices = np.digitize(phase_days, edges) - 1

    phase_centers: list[float] = []
    binned_flux: list[float] = []
    for idx in range(bins):
        mask = indices == idx
        if not np.any(mask):
            continue
        phase_centers.append(float(np.nanmean(phase_days[mask])))
        binned_flux.append(float(np.nanmean(flux[mask])))

    return np.asarray(phase_centers), np.asarray(binned_flux)


def save_folded_transit_plot(
    light_curve,
    output_path: Path,
    target_name: str,
    period_days: float,
    epoch_bkjd: float,
    duration_hours: float,
    bins: int = 150,
) -> None:
    """Save a phase-folded plot around the known transit ephemeris."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    phase_days = compute_phase_days(light_curve, period_days=period_days, epoch_bkjd=epoch_bkjd)
    flux = np.asarray(light_curve.flux.value, dtype=float)
    binned_phase, binned_flux = _bin_folded_light_curve(phase_days, flux, bins=bins)
    half_duration_days = duration_hours / 24.0 / 2.0

    figure, axis = plt.subplots(figsize=(10, 4))
    axis.scatter(phase_days, flux, color="0.75", s=2, alpha=0.4, linewidths=0)
    axis.plot(binned_phase, binned_flux, color="black", linewidth=1.5)
    axis.axvspan(-half_duration_days, half_duration_days, color="tab:blue", alpha=0.12)
    axis.axvline(0.0, color="tab:blue", linestyle="--", linewidth=1)
    axis.set_title(f"{target_name} phase-folded around known transit")
    axis.set_xlabel("Phase [days from expected mid-transit]")
    axis.set_ylabel("Relative Flux")
    axis.set_xlim(-0.35, 0.35)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_windowed_transit_plot(
    light_curve,
    output_path: Path,
    target_name: str,
    period_days: float,
    epoch_bkjd: float,
    duration_hours: float,
    window_half_width_days: float,
    transit_mask_scale: float = 1.25,
    baseline_mask_scale: float = 2.0,
    bins: int = 120,
) -> None:
    """Save a stacked, locally normalized transit-window plot."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    time_days = np.asarray(light_curve.time.value, dtype=float)
    flux = np.asarray(light_curve.flux.value, dtype=float)
    half_duration_days = duration_hours / 24.0 / 2.0
    in_transit_half_width_days = transit_mask_scale * half_duration_days
    baseline_min_phase_days = baseline_mask_scale * half_duration_days
    event_centers = _compute_transit_centers(time_days, period_days=period_days, epoch_bkjd=epoch_bkjd)

    all_phase: list[np.ndarray] = []
    all_flux: list[np.ndarray] = []
    for center in event_centers:
        phase_days = time_days - center
        window_mask = np.abs(phase_days) <= window_half_width_days
        if np.count_nonzero(window_mask) < 10:
            continue

        local_phase = phase_days[window_mask]
        local_flux = flux[window_mask]
        wing_mask = np.abs(local_phase) > baseline_min_phase_days
        if np.count_nonzero(wing_mask) < 4:
            continue

        wing_median = float(np.nanmedian(local_flux[wing_mask]))
        if not np.isfinite(wing_median) or wing_median == 0.0:
            continue

        all_phase.append(local_phase)
        all_flux.append(local_flux / wing_median)

    if not all_phase:
        raise ValueError("No usable transit windows found for plotting.")

    stacked_phase = np.concatenate(all_phase)
    stacked_flux = np.concatenate(all_flux)
    binned_phase, binned_flux = _bin_folded_light_curve(stacked_phase, stacked_flux, bins=bins)

    figure, axis = plt.subplots(figsize=(10, 4))
    axis.scatter(stacked_phase, stacked_flux, color="0.75", s=3, alpha=0.35, linewidths=0)
    axis.plot(binned_phase, binned_flux, color="black", linewidth=1.5)
    axis.axvspan(-in_transit_half_width_days, in_transit_half_width_days, color="tab:green", alpha=0.12)
    axis.axvline(0.0, color="tab:green", linestyle="--", linewidth=1)
    axis.set_title(f"{target_name} stacked known-period transit windows")
    axis.set_xlabel("Days from expected mid-transit")
    axis.set_ylabel("Locally normalized flux")
    axis.set_xlim(-window_half_width_days, window_half_width_days)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
