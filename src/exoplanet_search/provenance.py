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
    downloaded_paths: tuple[Path, ...] = (),
    cadence_counts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a JSON-serializable run provenance manifest."""
    raw_files = [_raw_file_record(path) for path in downloaded_paths]
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
        "flux_product": flux_product,
        "time_system": time_system,
        "quality_mask_policy": {
            "lightkurve_quality_bitmask": quality_bitmask,
            "application_stage": "Lightkurve SearchResult.download_all while reading FITS products",
            "separately_observable_after_download": False,
        },
        "preprocessing": preprocessing,
        "raw_inputs": raw_files,
        "cadence_counts": cadence_counts or {},
        "limitations": [
            (
                "Per-cadence counts before Lightkurve quality masking are not available from the "
                "stitched LightCurve returned by the current download path."
            )
        ],
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write indented JSON with stable UTF-8 encoding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2)


def _raw_file_record(path: Path) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": str(path),
        "name": path.name,
        "sha256": _sha256(path),
    }
    quarter = _quarter_from_name(path.name)
    if quarter is not None:
        record["kepler_quarter"] = quarter
    return record


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _quarter_from_name(name: str) -> int | None:
    marker = "_q"
    lower_name = name.lower()
    if marker not in lower_name:
        return None
    after_marker = lower_name.split(marker, maxsplit=1)[1]
    digits = ""
    for char in after_marker:
        if char.isdigit():
            digits += char
        else:
            break
    return int(digits) if digits else None


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
