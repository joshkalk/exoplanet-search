"""Helpers for downloading Kepler light-curve data."""

from dataclasses import dataclass
from pathlib import Path

import lightkurve as lk

from .config import (
    DEFAULT_AUTHOR,
    DEFAULT_CADENCE,
    DEFAULT_FLUX_COLUMN,
    DEFAULT_MISSION,
    DEFAULT_QUALITY_BITMASK,
    DEFAULT_TARGET,
)

STITCHING_POLICY = {
    "method": "LightCurveCollection.stitch",
    "corrector_func": "per_product_light_curve_normalize",
    "description": (
        "Each downloaded light curve is normalized with LightCurve.normalize() "
        "before stitching, matching Lightkurve's historical default explicitly."
    ),
}


@dataclass(frozen=True)
class KeplerLightCurveBundle:
    """Downloaded Kepler light curve plus auditable retrieval settings."""

    light_curve: object
    target: str
    mission: str
    author: str
    cadence: str
    source_flux_column: str
    quality_bitmask: str | int
    stitching_policy: dict[str, str]
    downloaded_paths: tuple[Path, ...]
    n_products: int
    product_light_curves: tuple[object, ...] = ()


def download_kepler_light_curve(
    target: str = DEFAULT_TARGET,
    mission: str = DEFAULT_MISSION,
    author: str = DEFAULT_AUTHOR,
    cadence: str = DEFAULT_CADENCE,
    flux_column: str = DEFAULT_FLUX_COLUMN,
    quality_bitmask: str | int = DEFAULT_QUALITY_BITMASK,
    download_dir: Path | None = None,
):
    """Download and stitch a Kepler light curve for a target."""
    return download_kepler_light_curve_bundle(
        target=target,
        mission=mission,
        author=author,
        cadence=cadence,
        flux_column=flux_column,
        quality_bitmask=quality_bitmask,
        download_dir=download_dir,
    ).light_curve


def download_kepler_light_curve_bundle(
    target: str = DEFAULT_TARGET,
    mission: str = DEFAULT_MISSION,
    author: str = DEFAULT_AUTHOR,
    cadence: str = DEFAULT_CADENCE,
    flux_column: str = DEFAULT_FLUX_COLUMN,
    quality_bitmask: str | int = DEFAULT_QUALITY_BITMASK,
    download_dir: Path | None = None,
) -> KeplerLightCurveBundle:
    """Download and stitch a Kepler light curve with retrieval provenance.

    Lightkurve applies the requested quality bitmask while reading each FITS
    product. Non-finite flux removal is intentionally left to preprocessing so
    cadence accounting stays explicit.
    """
    search_result = lk.search_lightcurve(
        target,
        mission=mission,
        author=author,
        cadence=cadence,
    )
    if len(search_result) == 0:
        raise ValueError(
            f"No light curves found for target={target!r}, mission={mission!r}, "
            f"author={author!r}, cadence={cadence!r}."
        )

    collection = search_result.download_all(
        quality_bitmask=quality_bitmask,
        download_dir=str(download_dir) if download_dir else None,
        flux_column=flux_column,
    )
    if collection is None or len(collection) == 0:
        raise RuntimeError("Download returned no files. Check network access and Lightkurve setup.")

    downloaded_paths = _extract_downloaded_paths(collection)
    product_light_curves = tuple(collection)
    return KeplerLightCurveBundle(
        light_curve=collection.stitch(corrector_func=_normalize_before_stitch),
        target=target,
        mission=mission,
        author=author,
        cadence=cadence,
        source_flux_column=flux_column,
        quality_bitmask=quality_bitmask,
        stitching_policy=STITCHING_POLICY,
        downloaded_paths=downloaded_paths,
        n_products=len(collection),
        product_light_curves=product_light_curves,
    )


def _extract_downloaded_paths(collection) -> tuple[Path, ...]:
    paths: list[Path] = []
    for light_curve in collection:
        for key in ("FILENAME", "filename", "FILE", "path"):
            value = light_curve.meta.get(key)
            if value:
                path = Path(str(value))
                if path.exists():
                    paths.append(path)
                break
    return tuple(paths)


def _normalize_before_stitch(light_curve):
    """Normalize each product before stitching to preserve the explicit policy."""
    return light_curve.normalize()
