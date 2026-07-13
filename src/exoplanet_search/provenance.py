"""Run provenance helpers for reproducible light-curve diagnostics."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from astropy.io import fits


PACKAGE_DISTRIBUTIONS = (
    "numpy",
    "scipy",
    "pandas",
    "matplotlib",
    "astropy",
    "lightkurve",
    "batman-package",
)


def build_provenance_manifest(
    *,
    target: str,
    mission: str,
    author: str,
    cadence: str,
    flux_product: str,
    time_system: str,
    quality_bitmask: str | int,
    preprocessing: dict[str, Any],
    stitching_policy: dict[str, Any] | None = None,
    downloaded_paths: tuple[Path, ...] = (),
    cadence_counts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a JSON-serializable run provenance manifest."""
    raw_files = [_raw_file_record(path) for path in downloaded_paths]
    limitations = [
        (
            "Per-cadence counts before Lightkurve quality masking are not available from the "
            "stitched LightCurve returned by the current download path."
        )
    ]
    if not raw_files:
        limitations.append(
            "Exact downloaded FITS paths were not exposed by the Lightkurve objects; raw input "
            "file checksums and FITS header metadata were not recorded for this run."
        )
    return {
        "run_timestamp_utc": datetime.now(UTC).isoformat(),
        "git": {
            "commit": _git_output("rev-parse", "HEAD"),
            "status": _git_output("status", "--short"),
            "is_dirty": bool(_git_output("status", "--short")),
        },
        "python": {
            "version": sys.version,
            "implementation": platform.python_implementation(),
        },
        "packages": _package_versions(),
        "target_query": target,
        "mission": mission,
        "author": author,
        "cadence": cadence,
        "source_fits_flux_column": flux_product,
        "analyzed_flux": (
            "stitched flux from the source FITS column after explicit per-product "
            "LightCurve.normalize(); preprocessing may apply an additional global median "
            "normalization."
        ),
        "time_system": time_system,
        "stitching_policy": stitching_policy or {},
        "quality_mask_policy": {
            "lightkurve_quality_bitmask": quality_bitmask,
            "application_stage": "Lightkurve SearchResult.download_all while reading FITS products",
            "separately_observable_after_download": False,
        },
        "preprocessing": preprocessing,
        "raw_inputs": raw_files,
        "cadence_counts": cadence_counts or {},
        "limitations": limitations,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write indented JSON with stable UTF-8 encoding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2)


def _raw_file_record(path: Path) -> dict[str, Any]:
    header_metadata = _fits_header_metadata(path)
    record: dict[str, Any] = {
        "path": str(path),
        "name": path.name,
        "sha256": _sha256(path),
        "fits_header_metadata": header_metadata,
    }
    if "quarter" in header_metadata:
        record["kepler_quarter"] = header_metadata["quarter"]
    return record


def _fits_header_metadata(path: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    try:
        with fits.open(path, memmap=False) as hdul:
            headers = [hdu.header for hdu in hdul]
            _copy_first_header_value(headers, metadata, "quarter", ("QUARTER",))
            _copy_first_header_value(
                headers,
                metadata,
                "data_release",
                ("DATA_REL", "DATREL", "DRN"),
            )
            _copy_first_header_value(
                headers,
                metadata,
                "pipeline_version",
                ("PROCVER", "PIPEVER", "SOCVER"),
            )
    except OSError:
        metadata["header_read_error"] = True
    return metadata


def _copy_first_header_value(
    headers,
    metadata: dict[str, Any],
    output_key: str,
    header_keys: tuple[str, ...],
) -> None:
    for header in headers:
        for header_key in header_keys:
            if header_key in header:
                value = header[header_key]
                if hasattr(value, "item"):
                    value = value.item()
                metadata[output_key] = value
                return


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_output(*args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    return completed.stdout.strip()


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in PACKAGE_DISTRIBUTIONS:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return versions
