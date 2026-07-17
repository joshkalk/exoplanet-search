"""CLI for first-pass Kepler target validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import (
    DATA_INTERIM_DIR,
    DATA_RAW_DIR,
    DEFAULT_AUTHOR,
    DEFAULT_CADENCE,
    DEFAULT_COMPARISON_DIR,
    DEFAULT_FLUX_COLUMN,
    DEFAULT_INSPECTION_DIR,
    DEFAULT_MISSION,
    DEFAULT_PHASE1A_DIR,
    DEFAULT_PHASE1B_DIR,
    DEFAULT_PHASE1C_DIR,
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
from .phase1a import BLSSearchConfig, run_phase1a_search
from .phase1b import run_phase1b_fit
from .phase1b_types import Phase1BConfig
from .phase1c import (
    run_phase1c_pilot,
    run_phase1c_production,
    run_phase1c_synthetic_validation,
    summarize_phase1c_checkpoints,
    validate_phase1c_inputs,
)
from .phase1c_types import Phase1CConfig
from .phase1d import Phase1DDevelopmentConfig, run_phase1d_development_predictive
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
    parser.add_argument(
        "--blind-period-search",
        action="store_true",
        help="Run Phase 1A blind BLS period search with chronological holdout validation.",
    )
    parser.add_argument(
        "--phase1a-output-dir",
        type=Path,
        default=DEFAULT_PHASE1A_DIR,
        help="Directory for Phase 1A blind-search outputs (default: data/interim/kepler5_phase1a_search)",
    )
    parser.add_argument("--bls-min-period", type=float, default=0.5, help="Minimum BLS period in days.")
    parser.add_argument("--bls-max-period", type=float, default=100.0, help="Maximum BLS period in days.")
    parser.add_argument("--bls-min-duration-hours", type=float, default=1.0)
    parser.add_argument("--bls-max-duration-hours", type=float, default=12.0)
    parser.add_argument("--bls-n-durations", type=int, default=12)
    parser.add_argument("--bls-n-periods", type=int, default=5000)
    parser.add_argument("--bls-frequency-factor", type=float, default=1.0)
    parser.add_argument("--bls-oversample", type=int, default=10)
    parser.add_argument("--bls-local-duration-step-hours", type=float, default=0.25)
    parser.add_argument("--bls-local-max-period-samples", type=int, default=8000)
    parser.add_argument("--bls-allowed-drift-fraction", type=float, default=0.10)
    parser.add_argument("--training-fraction", type=float, default=0.70)
    parser.add_argument(
        "--physical-transit-fit",
        action="store_true",
        help="Run Phase 1B deterministic BATMAN physical transit fit from Phase 1A outputs.",
    )
    parser.add_argument(
        "--phase1a-summary-path",
        type=Path,
        default=DEFAULT_PHASE1A_DIR / "search_summary.json",
        help="Path to Phase 1A search_summary.json.",
    )
    parser.add_argument(
        "--phase1a-provenance-path",
        type=Path,
        default=DEFAULT_PHASE1A_DIR / "provenance_manifest.json",
        help="Path to Phase 1A provenance_manifest.json.",
    )
    parser.add_argument(
        "--stellar-inputs-path",
        type=Path,
        default=Path("data/interim/kepler5_phase1b_stellar_inputs.json"),
        help="JSON file containing reproducible stellar and limb-darkening inputs.",
    )
    parser.add_argument(
        "--phase1b-output-dir",
        type=Path,
        default=DEFAULT_PHASE1B_DIR,
        help="Directory for Phase 1B deterministic fit outputs.",
    )
    parser.add_argument("--phase1b-n-starts", type=int, default=14)
    parser.add_argument("--phase1b-random-seed", type=int, default=481516)
    parser.add_argument("--phase1b-supersample-factor", type=int, default=11)
    parser.add_argument("--phase1b-high-supersample-factor", type=int, default=21)
    parser.add_argument(
        "--phase1c-validate-inputs",
        action="store_true",
        help="Validate frozen Phase 1B inputs for Phase 1C without downloading or rebuilding.",
    )
    parser.add_argument(
        "--phase1c-synthetic-validation",
        action="store_true",
        help="Run the reproducible synthetic Phase 1C sampler validation.",
    )
    parser.add_argument(
        "--phase1c-synthetic-recovery",
        action="store_true",
        help="Run a longer synthetic Phase 1C recovery attempt; recovery is claimed only if converged.",
    )
    parser.add_argument(
        "--phase1c-pilot",
        action="store_true",
        help="Run a short nonproduction real-data Phase 1C pilot from frozen Phase 1B outputs.",
    )
    parser.add_argument(
        "--phase1c-production",
        action="store_true",
        help="Run production Phase 1C posterior sampling from frozen Phase 1B outputs.",
    )
    parser.add_argument(
        "--phase1c-summarize",
        action="store_true",
        help="Summarize existing Phase 1C checkpoints without sampling.",
    )
    parser.add_argument(
        "--phase1c-resume",
        action="store_true",
        help="Resume Phase 1C HDF checkpoints after validating checkpoint metadata.",
    )
    parser.add_argument(
        "--phase1c-phase1b-dir",
        type=Path,
        default=DEFAULT_PHASE1B_DIR,
        help="Frozen Phase 1B output directory consumed by Phase 1C.",
    )
    parser.add_argument(
        "--phase1c-output-dir",
        type=Path,
        default=DEFAULT_PHASE1C_DIR,
        help="Directory for Phase 1C posterior outputs.",
    )
    parser.add_argument("--phase1c-random-seed", type=int, default=20260715)
    parser.add_argument("--phase1c-run-id", default=None, help="Deterministic Phase 1C run ID for isolated outputs.")
    parser.add_argument("--phase1c-n-ensembles", type=int, default=4)
    parser.add_argument("--phase1c-n-walkers", type=int, default=24)
    parser.add_argument("--phase1c-pilot-steps", type=int, default=24)
    parser.add_argument("--phase1c-synthetic-steps", type=int, default=80)
    parser.add_argument("--phase1c-synthetic-recovery-steps", type=int, default=2000)
    parser.add_argument("--phase1c-production-steps", type=int, default=2000)
    parser.add_argument("--phase1c-target-total-steps", type=int, default=None)
    parser.add_argument("--phase1c-additional-steps", type=int, default=None)
    parser.add_argument("--phase1c-chunk-steps", type=int, default=12)
    parser.add_argument("--phase1c-warmup-steps", type=int, default=8)
    parser.add_argument(
        "--phase1c-summarize-mode",
        choices=("pilot", "production", "synthetic", "synthetic_recovery"),
        default="pilot",
    )
    parser.add_argument(
        "--phase1d-development-predictive",
        action="store_true",
        help="Run a tiny nonauthoritative Phase 1D posterior-predictive development check.",
    )
    parser.add_argument("--phase1d-source-run-dir", type=Path, default=None)
    parser.add_argument("--phase1d-output-dir", type=Path, default=DATA_INTERIM_DIR / "kepler5_phase1d")
    parser.add_argument("--phase1d-run-id", default=None)
    parser.add_argument("--phase1d-n-draws", type=int, default=2)
    parser.add_argument("--phase1d-selection-seed", type=int, default=2026071701)
    parser.add_argument("--phase1d-predictive-seed", type=int, default=2026071702)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _validate_target_specific_options(args)

    if _phase1c_requested(args):
        _run_phase1c_from_args(args)
        return
    if args.phase1d_development_predictive:
        _run_phase1d_from_args(args)
        return

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

    if args.blind_period_search:
        phase1a_config = BLSSearchConfig(
            minimum_period_days=args.bls_min_period,
            maximum_period_days=args.bls_max_period,
            minimum_duration_hours=args.bls_min_duration_hours,
            maximum_duration_hours=args.bls_max_duration_hours,
            n_durations=args.bls_n_durations,
            n_periods=args.bls_n_periods,
            frequency_factor=args.bls_frequency_factor,
            oversample=args.bls_oversample,
            local_duration_step_hours=args.bls_local_duration_step_hours,
            local_max_period_samples=args.bls_local_max_period_samples,
            allowed_phase_drift_fraction=args.bls_allowed_drift_fraction,
            training_fraction=args.training_fraction,
        )
        phase1a_provenance = build_provenance_manifest(
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
        published_ephemeris = _published_kepler5_ephemeris() if _is_kepler5_target(args.target) else None
        phase1a_summary = run_phase1a_search(
            processed_curve,
            output_dir=args.phase1a_output_dir,
            target=args.target,
            config=phase1a_config,
            provenance=phase1a_provenance,
            published_ephemeris=published_ephemeris,
        )
        locked = phase1a_summary["locked_refined_training_candidate"]
        holdout = phase1a_summary["holdout_summary"]
        print(f"Wrote Phase 1A blind-search outputs: {args.phase1a_output_dir}")
        print(
            "Locked refined training candidate: "
            f"period={locked['refined_period_days']:.8f} d, "
            f"transit_time={locked['refined_transit_time']:.8f}, "
            f"duration={locked['refined_duration_hours']:.3f} h, "
            f"depth={locked['refined_depth_ppm']:.1f} ppm, "
            f"BLS power={locked['refined_bls_power']:.6g}"
        )
        print(
            "Holdout: "
            f"{holdout['usable_event_count']}/{holdout['predicted_event_count']} usable events, "
            f"depth={holdout['aggregate_depth_ppm']:.1f} ppm"
        )

    if args.physical_transit_fit:
        phase1b_config = Phase1BConfig(
            phase1a_summary_path=args.phase1a_summary_path,
            phase1a_provenance_path=args.phase1a_provenance_path,
            stellar_inputs_path=args.stellar_inputs_path,
            output_dir=args.phase1b_output_dir,
            cadence=args.cadence,
            random_seed=args.phase1b_random_seed,
            n_starts=args.phase1b_n_starts,
            supersample_factor=args.phase1b_supersample_factor,
            high_supersample_factor=args.phase1b_high_supersample_factor,
        )
        try:
            phase1b_summary = run_phase1b_fit(
                bundle=bundle,
                output_dir=args.phase1b_output_dir,
                config=phase1b_config,
                target=args.target,
                mission=args.mission,
                author=args.author,
                cadence=args.cadence,
                flux_product=args.flux_column,
                quality_bitmask=args.quality_bitmask,
                preprocessing=preprocessing_result.summary(),
            )
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            raise SystemExit(f"Phase 1B physical transit fit failed: {exc}") from exc
        result = phase1b_summary["fitted_results"]["global_timing_refinement"]
        windows = phase1b_summary["transit_windows"]
        print(f"Wrote Phase 1B deterministic fit outputs: {args.phase1b_output_dir}")
        print(
            "Phase 1B timing-refined fit: "
            f"Rp/Rstar={result['rp_over_rstar']:.6f}, "
            f"a/Rstar={result['a_over_rstar']:.6f}, "
            f"b={result['impact_parameter']:.6f}, "
            f"objective={result['objective_value']:.6g}"
        )
        print(
            "Transit windows: "
            f"{windows['included_count']}/{windows['predicted_count']} included, "
            f"{windows['excluded_count']} excluded"
        )


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


def _phase1c_requested(args) -> bool:
    return any(
        (
            args.phase1c_validate_inputs,
            args.phase1c_synthetic_validation,
            args.phase1c_synthetic_recovery,
            args.phase1c_pilot,
            args.phase1c_production,
            args.phase1c_summarize,
        )
    )


def _phase1c_config_from_args(args) -> Phase1CConfig:
    return Phase1CConfig(
        phase1b_output_dir=args.phase1c_phase1b_dir,
        output_dir=args.phase1c_output_dir,
        run_id=args.phase1c_run_id,
        random_seed=args.phase1c_random_seed,
        n_ensembles=args.phase1c_n_ensembles,
        n_walkers=args.phase1c_n_walkers,
        pilot_steps=args.phase1c_pilot_steps,
        synthetic_steps=args.phase1c_synthetic_steps,
        synthetic_recovery_steps=args.phase1c_synthetic_recovery_steps,
        production_steps=args.phase1c_production_steps,
        target_total_steps=args.phase1c_target_total_steps,
        additional_steps=args.phase1c_additional_steps,
        chunk_steps=args.phase1c_chunk_steps,
        warmup_steps=args.phase1c_warmup_steps,
    )


def _run_phase1c_from_args(args) -> None:
    config = _phase1c_config_from_args(args)
    requested = [
        name
        for name, enabled in (
            ("validate-inputs", args.phase1c_validate_inputs),
            ("synthetic-validation", args.phase1c_synthetic_validation),
            ("synthetic-recovery", args.phase1c_synthetic_recovery),
            ("pilot", args.phase1c_pilot),
            ("production", args.phase1c_production),
            ("summarize", args.phase1c_summarize),
        )
        if enabled
    ]
    if len(requested) != 1:
        raise SystemExit(f"Choose exactly one Phase 1C mode, got: {', '.join(requested)}")
    try:
        if args.phase1c_validate_inputs:
            result = validate_phase1c_inputs(config)
        elif args.phase1c_synthetic_validation:
            result = run_phase1c_synthetic_validation(config, resume=args.phase1c_resume)
        elif args.phase1c_synthetic_recovery:
            result = run_phase1c_synthetic_validation(config, resume=args.phase1c_resume, recovery=True)
        elif args.phase1c_pilot:
            result = run_phase1c_pilot(config, resume=args.phase1c_resume)
        elif args.phase1c_production:
            result = run_phase1c_production(config, resume=args.phase1c_resume)
        else:
            result = summarize_phase1c_checkpoints(config, mode=args.phase1c_summarize_mode)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"Phase 1C {requested[0]} failed: {exc}") from exc
    print(f"Wrote Phase 1C {requested[0]} outputs under: {config.output_dir}")
    print(json.dumps(result, indent=2))


def _run_phase1d_from_args(args) -> None:
    if args.phase1d_source_run_dir is None:
        raise SystemExit("--phase1d-development-predictive requires --phase1d-source-run-dir.")
    config = Phase1DDevelopmentConfig(
        source_run_dir=args.phase1d_source_run_dir,
        output_dir=args.phase1d_output_dir,
        run_id=args.phase1d_run_id,
        n_draws=args.phase1d_n_draws,
        selection_seed=args.phase1d_selection_seed,
        predictive_seed=args.phase1d_predictive_seed,
    )
    try:
        result = run_phase1d_development_predictive(config)
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"Phase 1D development predictive failed: {exc}") from exc
    print(f"Wrote Phase 1D development predictive outputs under: {config.output_dir}")
    print(json.dumps(result, indent=2))


def _is_kepler5_target(target: str) -> bool:
    normalized = target.lower().replace(" ", "").replace("-", "")
    return normalized in {"kepler5", "kic8191672", "8191672"}


def _published_kepler5_ephemeris() -> dict[str, float]:
    return {
        "period_days": KEPLER5B_PERIOD_DAYS,
        "epoch_bkjd": KEPLER5B_EPOCH_BKJD,
        "duration_hours": KEPLER5B_DURATION_HOURS,
    }


if __name__ == "__main__":
    main()
