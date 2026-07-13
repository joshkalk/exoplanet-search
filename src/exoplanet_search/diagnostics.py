"""Preprocessing comparison diagnostics for known Kepler-5 checks."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from .config import (
    DEFAULT_TIME_SYSTEM,
    KEPLER5B_BASELINE_MASK_SCALE,
    KEPLER5B_DURATION_HOURS,
    KEPLER5B_EPOCH_BKJD,
    KEPLER5B_PERIOD_DAYS,
    KEPLER5B_TRANSIT_MASK_SCALE,
    KEPLER5B_WINDOW_HALF_WIDTH_DAYS,
)
from .preprocessing import (
    PREPROCESSING_MODES,
    PreprocessingConfig,
    preprocess_light_curve,
    removal_diagnostics,
)
from .provenance import build_provenance_manifest, write_json
from .recovery import estimate_known_transit_signal, estimate_windowed_known_transit_signal


def run_preprocessing_comparison(
    *,
    light_curve,
    output_dir: Path,
    target: str,
    mission: str,
    author: str,
    cadence: str,
    flux_product: str,
    quality_bitmask: str | int,
    downloaded_paths: tuple[Path, ...] = (),
) -> dict[str, Any]:
    """Run all preprocessing modes and save diagnostic comparison products."""
    output_dir.mkdir(parents=True, exist_ok=True)
    mode_summaries: list[dict[str, Any]] = []
    phase_rows: list[dict[str, Any]] = []

    for mode in PREPROCESSING_MODES:
        config = _comparison_config(mode)
        result = preprocess_light_curve(light_curve, config)
        recovery = estimate_known_transit_signal(
            result.light_curve,
            period_days=KEPLER5B_PERIOD_DAYS,
            epoch_bkjd=KEPLER5B_EPOCH_BKJD,
            duration_hours=KEPLER5B_DURATION_HOURS,
        )
        windowed_recovery = estimate_windowed_known_transit_signal(
            result.light_curve,
            period_days=KEPLER5B_PERIOD_DAYS,
            epoch_bkjd=KEPLER5B_EPOCH_BKJD,
            duration_hours=KEPLER5B_DURATION_HOURS,
            window_half_width_days=KEPLER5B_WINDOW_HALF_WIDTH_DAYS,
            transit_mask_scale=KEPLER5B_TRANSIT_MASK_SCALE,
            baseline_mask_scale=KEPLER5B_BASELINE_MASK_SCALE,
        )
        transit_counts, binned_rows = removal_diagnostics(
            light_curve,
            result,
            period_days=KEPLER5B_PERIOD_DAYS,
            epoch_bkjd=KEPLER5B_EPOCH_BKJD,
            duration_hours=KEPLER5B_DURATION_HOURS,
            transit_mask_scale=KEPLER5B_TRANSIT_MASK_SCALE,
        )

        mode_summary = {
            "mode": mode,
            "preprocessing": result.summary(),
            "known_transit_window_counts": transit_counts,
            "known_period_recovery": recovery,
            "windowed_known_period_recovery": windowed_recovery,
        }
        mode_summaries.append(mode_summary)

        for row in binned_rows:
            phase_rows.append({"mode": mode, **row})

    summary = {
        "target": target,
        "time_system": DEFAULT_TIME_SYSTEM,
        "known_ephemeris_use": "diagnostic_only",
        "modes": mode_summaries,
    }
    write_json(output_dir / "preprocessing_comparison_summary.json", summary)
    _write_phase_rows(output_dir / "phase_binned_removed_cadences.csv", phase_rows)
    _save_comparison_plot(output_dir / "preprocessing_comparison.png", mode_summaries)

    manifest = build_provenance_manifest(
        target=target,
        mission=mission,
        author=author,
        cadence=cadence,
        flux_product=flux_product,
        time_system=DEFAULT_TIME_SYSTEM,
        quality_bitmask=quality_bitmask,
        preprocessing={
            "comparison_modes": list(PREPROCESSING_MODES),
            "known_ephemeris_use": "diagnostic_only",
        },
        downloaded_paths=downloaded_paths,
        cadence_counts={
            mode_summary["mode"]: mode_summary["preprocessing"]
            for mode_summary in mode_summaries
        },
    )
    write_json(output_dir / "provenance_manifest.json", manifest)
    return summary


def _comparison_config(mode: str) -> PreprocessingConfig:
    if mode == "transit_protected_symmetric":
        return PreprocessingConfig(
            mode=mode,
            period_days=KEPLER5B_PERIOD_DAYS,
            epoch_bkjd=KEPLER5B_EPOCH_BKJD,
            duration_hours=KEPLER5B_DURATION_HOURS,
            transit_mask_scale=KEPLER5B_TRANSIT_MASK_SCALE,
        )
    return PreprocessingConfig(mode=mode)


def _write_phase_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "mode",
        "phase_bin",
        "phase_start",
        "phase_end",
        "cadence_count",
        "removed_count",
        "removed_fraction",
    ]
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _save_comparison_plot(path: Path, mode_summaries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    modes = [summary["mode"] for summary in mode_summaries]
    removed = [
        summary["preprocessing"]["total_removed_cadence_count"]
        for summary in mode_summaries
    ]
    depths = [
        summary["known_period_recovery"]["transit_depth_ppm"]
        for summary in mode_summaries
    ]
    windowed_depths = [
        summary["windowed_known_period_recovery"]["windowed_transit_depth_ppm"]
        for summary in mode_summaries
    ]

    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].bar(modes, removed, color="tab:gray")
    axes[0].set_ylabel("Removed cadences")
    axes[0].set_title("Effect of preprocessing mode on Kepler-5 diagnostics")

    axes[1].plot(modes, depths, marker="o", label="Folded depth proxy")
    axes[1].plot(modes, windowed_depths, marker="s", label="Windowed depth proxy")
    axes[1].set_ylabel("Recovered depth [ppm]")
    axes[1].set_xlabel("Preprocessing mode")
    axes[1].legend()
    axes[1].tick_params(axis="x", rotation=20)

    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
