"""CLI for first-pass Kepler target validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import (
    DATA_RAW_DIR,
    DEFAULT_AUTHOR,
    DEFAULT_CADENCE,
    DEFAULT_COMPARISON_DIR,
    DEFAULT_FLUX_COLUMN,
    DEFAULT_INSPECTION_DIR,
    DEFAULT_MISSION,
    DEFAULT_QUALITY_BITMASK,
    DEFAULT_RECOVERY_DIR,
    DEFAULT_TIME_SYSTEM,
    DEFAULT_WINDOWED_RECOVERY_DIR,
    DEFAULT_TARGET,
    KEPLER5B_BASELINE_MASK_SCALE,
    KEPLER5B_DURATION_HOURS,
    KEPLER5B_EPOCH_BKJD,
    KEPLER5B_PERIOD_DAYS,
    KEPLER5B_TRANSIT_MASK_SCALE,
    KEPLER5B_WINDOW_HALF_WIDTH_DAYS,
)
from .data_access import download_kepler_light_curve_bundle
from .diagnostics import run_preprocessing_comparison
from .inspection import (
    save_light_curve_plot,
    summarize_light_curve,
)
from .preprocessing import PREPROCESSING_MODES, PreprocessingConfig, preprocess_light_curve
from .provenance import build_provenance_manifest, write_json
from .recovery import (
    estimate_known_transit_signal,
    estimate_windowed_known_transit_signal,
    save_folded_transit_plot,
    save_windowed_transit_plot,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download and inspect a Kepler light curve with light preprocessing "
            "(default target: Kepler-5)."
        )
    )
    parser.add_argument("--target", default=DEFAULT_TARGET, help="Kepler target name (default: Kepler-5)")
    parser.add_argument("--mission", default=DEFAULT_MISSION, help="Mission query (default: Kepler)")
    parser.add_argument("--author", default=DEFAULT_AUTHOR, help="Author query (default: Kepler)")
    parser.add_argument("--cadence", default=DEFAULT_CADENCE, help="Cadence query (default: long)")
    parser.add_argument(
        "--flux-column",
        default=DEFAULT_FLUX_COLUMN,
        help="FITS flux column to analyze (default: pdcsap_flux)",
    )
    parser.add_argument(
        "--quality-bitmask",
        default=DEFAULT_QUALITY_BITMASK,
        help="Lightkurve quality_bitmask used while reading FITS products (default: default)",
    )
    parser.add_argument(
        "--preprocessing-mode",
        choices=PREPROCESSING_MODES,
        default="none",
        help="Preprocessing mode for ordinary inspection/recovery outputs (default: none)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_INSPECTION_DIR,
        help="Directory for inspection outputs (default: data/interim/kepler5_inspection)",
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=DATA_RAW_DIR,
        help="Directory where Lightkurve download cache/files are stored (default: data/raw)",
    )
    parser.add_argument(
        "--recover",
        action="store_true",
        help="Also run a minimal known-transit recovery step for Kepler-5 b.",
    )
    parser.add_argument(
        "--recovery-output-dir",
        type=Path,
        default=DEFAULT_RECOVERY_DIR,
        help="Directory for recovery outputs (default: data/interim/kepler5_recovery)",
    )
    parser.add_argument(
        "--windowed-recovery",
        action="store_true",
        help="Run a local-window known-period recovery check around each expected transit.",
    )
    parser.add_argument(
        "--windowed-recovery-output-dir",
        type=Path,
        default=DEFAULT_WINDOWED_RECOVERY_DIR,
        help="Directory for windowed recovery outputs (default: data/interim/kepler5_windowed_recovery)",
    )
    parser.add_argument(
        "--compare-preprocessing",
        action="store_true",
        help="Run all four preprocessing modes and write comparison diagnostics.",
    )
    parser.add_argument(
        "--comparison-output-dir",
        type=Path,
        default=DEFAULT_COMPARISON_DIR,
        help=(
            "Directory for preprocessing comparison outputs "
            "(default: data/interim/kepler5_preprocessing_comparison)"
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _validate_target_specific_options(args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.download_dir.mkdir(parents=True, exist_ok=True)

    try:
        bundle = download_kepler_light_curve_bundle(
            target=args.target,
            mission=args.mission,
            author=args.author,
            cadence=args.cadence,
            flux_column=args.flux_column,
            quality_bitmask=args.quality_bitmask,
            download_dir=args.download_dir,
        )
    except Exception as exc:  # pragma: no cover - depends on remote service/network state
        message = (
            f"Failed to download light curve for {args.target!r}. "
            "Confirm internet/proxy access to MAST and retry."
        )
        raise SystemExit(f"{message}\nOriginal error: {exc}") from exc

    preprocessing_config = _preprocessing_config_from_args(args.preprocessing_mode)
    preprocessing_result = preprocess_light_curve(bundle.light_curve, preprocessing_config)
    processed_curve = preprocessing_result.light_curve

    summary = summarize_light_curve(processed_curve)
    summary.update(
        {
            "target": args.target,
            "time_system": DEFAULT_TIME_SYSTEM,
            "source_fits_flux_column": args.flux_column,
            "quality_bitmask": args.quality_bitmask,
            "preprocessing": preprocessing_result.summary(),
        }
    )

    summary_path = args.output_dir / "summary.json"
    plot_path = args.output_dir / "light_curve.png"
    provenance_path = args.output_dir / "provenance_manifest.json"

    write_json(summary_path, summary)

    save_light_curve_plot(
        processed_curve,
        plot_path,
        target_name=args.target,
        time_system=DEFAULT_TIME_SYSTEM,
    )
    provenance = build_provenance_manifest(
        target=args.target,
        mission=args.mission,
        author=args.author,
        cadence=args.cadence,
        flux_product=args.flux_column,
        time_system=DEFAULT_TIME_SYSTEM,
        quality_bitmask=args.quality_bitmask,
        preprocessing=preprocessing_result.summary(),
        stitching_policy=bundle.stitching_policy,
        downloaded_paths=bundle.downloaded_paths,
        cadence_counts=preprocessing_result.summary(),
    )
    write_json(provenance_path, provenance)

    print(f"Wrote summary: {summary_path}")
    print(f"Wrote plot: {plot_path}")
    print(f"Wrote provenance manifest: {provenance_path}")
    print(json.dumps(summary, indent=2))

    if args.recover:
        args.recovery_output_dir.mkdir(parents=True, exist_ok=True)
        recovery_summary = estimate_known_transit_signal(
            processed_curve,
            period_days=KEPLER5B_PERIOD_DAYS,
            epoch_bkjd=KEPLER5B_EPOCH_BKJD,
            duration_hours=KEPLER5B_DURATION_HOURS,
        )
        recovery_summary.update(
            {
                "target": args.target,
                "time_system": DEFAULT_TIME_SYSTEM,
                "preprocessing": preprocessing_result.summary(),
            }
        )

        recovery_summary_path = args.recovery_output_dir / "recovery_summary.json"
        folded_plot_path = args.recovery_output_dir / "folded_light_curve.png"

        write_json(recovery_summary_path, recovery_summary)

        save_folded_transit_plot(
            processed_curve,
            folded_plot_path,
            target_name=args.target,
            period_days=KEPLER5B_PERIOD_DAYS,
            epoch_bkjd=KEPLER5B_EPOCH_BKJD,
            duration_hours=KEPLER5B_DURATION_HOURS,
        )

        print(f"Wrote recovery summary: {recovery_summary_path}")
        print(f"Wrote folded plot: {folded_plot_path}")
        print(json.dumps(recovery_summary, indent=2))

    if args.windowed_recovery:
        args.windowed_recovery_output_dir.mkdir(parents=True, exist_ok=True)
        windowed_summary = estimate_windowed_known_transit_signal(
            processed_curve,
            period_days=KEPLER5B_PERIOD_DAYS,
            epoch_bkjd=KEPLER5B_EPOCH_BKJD,
            duration_hours=KEPLER5B_DURATION_HOURS,
            window_half_width_days=KEPLER5B_WINDOW_HALF_WIDTH_DAYS,
            transit_mask_scale=KEPLER5B_TRANSIT_MASK_SCALE,
            baseline_mask_scale=KEPLER5B_BASELINE_MASK_SCALE,
        )
        windowed_summary.update(
            {
                "target": args.target,
                "time_system": DEFAULT_TIME_SYSTEM,
                "preprocessing": preprocessing_result.summary(),
            }
        )

        windowed_summary_path = args.windowed_recovery_output_dir / "windowed_recovery_summary.json"
        windowed_plot_path = args.windowed_recovery_output_dir / "windowed_folded_light_curve.png"

        write_json(windowed_summary_path, windowed_summary)

        save_windowed_transit_plot(
            processed_curve,
            windowed_plot_path,
            target_name=args.target,
            period_days=KEPLER5B_PERIOD_DAYS,
            epoch_bkjd=KEPLER5B_EPOCH_BKJD,
            duration_hours=KEPLER5B_DURATION_HOURS,
            window_half_width_days=KEPLER5B_WINDOW_HALF_WIDTH_DAYS,
            transit_mask_scale=KEPLER5B_TRANSIT_MASK_SCALE,
            baseline_mask_scale=KEPLER5B_BASELINE_MASK_SCALE,
        )

        print(f"Wrote windowed recovery summary: {windowed_summary_path}")
        print(f"Wrote windowed folded plot: {windowed_plot_path}")
        print(json.dumps(windowed_summary, indent=2))

    if args.compare_preprocessing:
        comparison_summary = run_preprocessing_comparison(
            light_curve=bundle.light_curve,
            output_dir=args.comparison_output_dir,
            target=args.target,
            mission=args.mission,
            author=args.author,
            cadence=args.cadence,
            flux_product=args.flux_column,
            quality_bitmask=args.quality_bitmask,
            stitching_policy=bundle.stitching_policy,
            downloaded_paths=bundle.downloaded_paths,
        )
        print(f"Wrote preprocessing comparison outputs: {args.comparison_output_dir}")
        print(json.dumps(comparison_summary, indent=2))


def _preprocessing_config_from_args(mode: str) -> PreprocessingConfig:
    if mode == "transit_protected_symmetric":
        return PreprocessingConfig(
            mode=mode,
            period_days=KEPLER5B_PERIOD_DAYS,
            epoch_bkjd=KEPLER5B_EPOCH_BKJD,
            duration_hours=KEPLER5B_DURATION_HOURS,
            transit_mask_scale=KEPLER5B_TRANSIT_MASK_SCALE,
        )
    return PreprocessingConfig(mode=mode)


def _validate_target_specific_options(args) -> None:
    target_is_kepler5 = _is_kepler5_target(args.target)
    requested_known_ephemeris = (
        args.recover
        or args.windowed_recovery
        or args.compare_preprocessing
        or args.preprocessing_mode == "transit_protected_symmetric"
    )
    if requested_known_ephemeris and not target_is_kepler5:
        raise SystemExit(
            "Known-ephemeris recovery, preprocessing comparison, and "
            "transit_protected_symmetric mode are currently implemented only for "
            "Kepler-5. Run ordinary inspection without those flags, or add a "
            "target-specific ephemeris before using these diagnostics."
        )


def _is_kepler5_target(target: str) -> bool:
    normalized = target.lower().replace(" ", "").replace("-", "")
    return normalized in {"kepler5", "kic8191672", "8191672"}


if __name__ == "__main__":
    main()
