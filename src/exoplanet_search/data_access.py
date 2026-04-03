"""Helpers for downloading Kepler light-curve data."""

from pathlib import Path

import lightkurve as lk

from .config import DEFAULT_AUTHOR, DEFAULT_CADENCE, DEFAULT_MISSION, DEFAULT_TARGET


def download_kepler_light_curve(
    target: str = DEFAULT_TARGET,
    mission: str = DEFAULT_MISSION,
    author: str = DEFAULT_AUTHOR,
    cadence: str = DEFAULT_CADENCE,
    download_dir: Path | None = None,
):
    """Download and stitch a Kepler light curve for a target.

    This intentionally applies only minimal preparation needed for inspection:
    stitching multi-quarter downloads and dropping NaNs.
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

    collection = search_result.download_all(download_dir=str(download_dir) if download_dir else None)
    if collection is None or len(collection) == 0:
        raise RuntimeError("Download returned no files. Check network access and Lightkurve setup.")

    return collection.stitch().remove_nans()
