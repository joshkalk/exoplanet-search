"""CLI for first-pass Kepler target validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import (
    DATA_RAW_DIR,
    DEFAULT_INSPECTION_DIR,
    DEFAULT_RECOVERY_DIR,
    DEFAULT_TARGET,
    KEPLER5B_DURATION_HOURS,
    KEPLER5B_EPOCH_BKJD,
    KEPLER5B_PERIOD_DAYS,
)
from .data_access import download_kepler_light_curve
from .inspection import (
    lightly_preprocess_light_curve,
    save_light_curve_plot,
    summarize_light_curve,
)
from .recovery import estimate_known_transit_signal, save_folded_transit_plot


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download and inspect a Kepler light curve with light preprocessing "
            "(default target: Kepler-5)."
        )
    )
    parser.add_argument("--target", default=DEFAULT_TARGET, help="Kepler target name (default: Kepler-5)")
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
    return parser


def main() -> None:
    args = build_parser().parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.download_dir.mkdir(parents=True, exist_ok=True)

    try:
        light_curve = download_kepler_light_curve(target=args.target, download_dir=args.download_dir)
    except Exception as exc:  # pragma: no cover - depends on remote service/network state
        message = (
            f"Failed to download light curve for {args.target!r}. "
            "Confirm internet/proxy access to MAST and retry."
        )
        raise SystemExit(f"{message}\nOriginal error: {exc}") from exc

    processed_curve = lightly_preprocess_light_curve(light_curve)

    summary = summarize_light_curve(processed_curve)
    summary.update(
        {
            "target": args.target,
            "preprocessing": "remove_nans + remove_outliers(sigma=5) + normalize",
        }
    )

    summary_path = args.output_dir / "summary.json"
    plot_path = args.output_dir / "light_curve.png"

    with summary_path.open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2)

    save_light_curve_plot(processed_curve, plot_path, target_name=args.target)

    print(f"Wrote summary: {summary_path}")
    print(f"Wrote plot: {plot_path}")
    print(json.dumps(summary, indent=2))

    if args.recover:
        args.recovery_output_dir.mkdir(parents=True, exist_ok=True)
        recovery_summary = estimate_known_transit_signal(
            processed_curve,
            period_days=KEPLER5B_PERIOD_DAYS,
            epoch_bkjd=KEPLER5B_EPOCH_BKJD,
            duration_hours=KEPLER5B_DURATION_HOURS,
        )
        recovery_summary.update({"target": args.target, "time_system": "BKJD"})

        recovery_summary_path = args.recovery_output_dir / "recovery_summary.json"
        folded_plot_path = args.recovery_output_dir / "folded_light_curve.png"

        with recovery_summary_path.open("w", encoding="utf-8") as output_file:
            json.dump(recovery_summary, output_file, indent=2)

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


if __name__ == "__main__":
    main()
