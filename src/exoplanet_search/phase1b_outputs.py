"""Output writers and residual diagnostics for Phase 1B."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows to CSV, preserving union field order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        if not fieldnames:
            output_file.write("")
            return
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def residual_rows(data, fit, stage: str) -> list[dict[str, Any]]:
    """Per-cadence residual table for a fit stage."""
    period = fit.parameters["period_days"]
    t0 = fit.parameters["transit_time"]
    phase = ((data.time - t0 + 0.5 * period) % period) - 0.5 * period
    rows = []
    for index in range(len(data.time)):
        rows.append(
            {
                "fit_stage": stage,
                "time": float(data.time[index]),
                "phase": float(phase[index]),
                "event_number": int(data.event_number[index]),
                "product_id": str(data.product_id[index]),
                "quarter": str(data.quarter[index]),
                "observed_flux": float(data.flux[index]),
                "flux_uncertainty": float(data.flux_err[index]),
                "local_baseline": float(fit.local_baseline[index]),
                "exposure_integrated_transit_model": float(fit.transit_model[index]),
                "combined_model": float(fit.combined_model[index]),
                "residual": float(fit.residuals[index]),
            }
        )
    return rows


def residual_summary(data, fit) -> dict[str, Any]:
    residuals = np.asarray(fit.residuals, dtype=float)
    scatter = float(np.nanstd(residuals, ddof=1)) if len(residuals) > 1 else float("nan")
    by_quarter = []
    for quarter in sorted(set(map(str, data.quarter))):
        mask = data.quarter.astype(str) == quarter
        by_quarter.append(
            {
                "quarter": quarter,
                "cadence_count": int(np.count_nonzero(mask)),
                "residual_std": float(np.nanstd(residuals[mask], ddof=1)) if np.count_nonzero(mask) > 1 else float("nan"),
            }
        )
    by_product = []
    for product in sorted(set(map(str, data.product_id))):
        mask = data.product_id.astype(str) == product
        by_product.append(
            {
                "product_id": product,
                "cadence_count": int(np.count_nonzero(mask)),
                "residual_std": float(np.nanstd(residuals[mask], ddof=1)) if np.count_nonzero(mask) > 1 else float("nan"),
            }
        )
    return {
        "cadence_count": int(len(residuals)),
        "residual_mean": float(np.nanmean(residuals)),
        "residual_std": scatter,
        "residual_mad": float(1.4826 * np.nanmedian(np.abs(residuals - np.nanmedian(residuals)))),
        "by_quarter": by_quarter,
        "by_product": by_product,
    }


def acf_rows(residuals: np.ndarray, max_lag: int = 20) -> list[dict[str, Any]]:
    residuals = np.asarray(residuals, dtype=float) - float(np.nanmean(residuals))
    denom = float(np.nansum(residuals**2))
    rows = []
    for lag in range(1, max_lag + 1):
        if len(residuals) <= lag or denom == 0.0:
            value = float("nan")
        else:
            value = float(np.nansum(residuals[:-lag] * residuals[lag:]) / denom)
        rows.append({"lag_cadences": lag, "acf": value})
    return rows


def rms_binning_rows(data, residuals: np.ndarray, bin_sizes: tuple[int, ...] = (1, 2, 4, 8, 16, 32)) -> list[dict[str, Any]]:
    rows = []
    base = float(np.nanstd(residuals, ddof=1))
    for size in bin_sizes:
        binned = []
        for event in np.unique(data.event_number):
            values = residuals[data.event_number == event]
            usable = (len(values) // size) * size
            if usable:
                binned.extend(np.nanmean(values[:usable].reshape(-1, size), axis=1))
        rms = float(np.nanstd(binned, ddof=1)) if len(binned) > 1 else float("nan")
        rows.append(
            {
                "bin_size_cadences": size,
                "binned_sample_count": int(len(binned)),
                "rms": rms,
                "independent_noise_expectation": base / np.sqrt(size) if np.isfinite(base) else float("nan"),
            }
        )
    return rows


def save_diagnostic_plots(output_dir: Path, data, fit, acf: list[dict[str, Any]], rms: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    phase = ((data.time - fit.parameters["transit_time"] + 0.5 * fit.parameters["period_days"]) % fit.parameters["period_days"]) - (
        0.5 * fit.parameters["period_days"]
    )
    order = np.argsort(phase)
    _folded_plot(output_dir / "folded_transit_model.png", phase, data.flux, fit.combined_model, order)
    _folded_plot(
        output_dir / "ingress_egress_model.png",
        phase,
        data.flux,
        fit.combined_model,
        order,
        xlim=1.2 * data.phase1a_duration_days,
    )
    _scatter_plot(output_dir / "residuals_by_phase.png", phase, fit.residuals, "Phase [days]", "Residual")
    _scatter_plot(output_dir / "residuals_by_time.png", data.time, fit.residuals, "Time", "Residual")
    _line_plot(output_dir / "residual_acf.png", [row["lag_cadences"] for row in acf], [row["acf"] for row in acf], "Lag [cadences]", "ACF")
    _line_plot(
        output_dir / "residual_rms_binning.png",
        [row["bin_size_cadences"] for row in rms],
        [row["rms"] for row in rms],
        "Bin size [cadences]",
        "Residual RMS",
        expected=[row["independent_noise_expectation"] for row in rms],
    )
    _parameter_stability_placeholder(output_dir / "parameter_stability.png")


def _folded_plot(path: Path, phase: np.ndarray, flux: np.ndarray, model: np.ndarray, order: np.ndarray, xlim: float | None = None) -> None:
    figure, axis = plt.subplots(figsize=(9, 4))
    axis.scatter(phase[order], flux[order], s=4, color="0.60", alpha=0.35, linewidths=0)
    axis.plot(phase[order], model[order], color="tab:red", linewidth=1.0)
    if xlim:
        axis.set_xlim(-xlim, xlim)
    axis.set_xlabel("Phase [days]")
    axis.set_ylabel("Normalized flux")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _scatter_plot(path: Path, x: np.ndarray, y: np.ndarray, xlabel: str, ylabel: str) -> None:
    figure, axis = plt.subplots(figsize=(9, 4))
    axis.scatter(x, y, s=4, color="black", alpha=0.35, linewidths=0)
    axis.axhline(0.0, color="tab:red", linewidth=1.0)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _line_plot(path: Path, x: list[float], y: list[float], xlabel: str, ylabel: str, expected: list[float] | None = None) -> None:
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(x, y, marker="o", label="measured")
    if expected is not None:
        axis.plot(x, expected, marker="s", label="1/sqrt(N)")
        axis.legend()
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _parameter_stability_placeholder(path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7, 3))
    axis.text(0.5, 0.5, "Stability diagnostics are recorded in CSV", ha="center", va="center")
    axis.set_axis_off()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
