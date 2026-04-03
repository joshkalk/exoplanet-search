"""Light-touch light-curve inspection helpers."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def lightly_preprocess_light_curve(light_curve, sigma: float = 5.0, normalize: bool = True):
    """Apply conservative preprocessing suitable for initial data-quality inspection.

    Steps are intentionally minimal to avoid suppressing astrophysical transit signals:
    - remove NaNs
    - remove only high-sigma outliers
    - optional flux normalization
    """
    processed = light_curve.remove_nans()
    if sigma > 0:
        processed = processed.remove_outliers(sigma=sigma)
    if normalize:
        processed = processed.normalize()
    return processed


def summarize_light_curve(light_curve) -> dict[str, float]:
    """Return a small numeric summary for quick validation checks."""
    time_values = np.asarray(light_curve.time.value)
    flux_values = np.asarray(light_curve.flux.value)
    return {
        "n_cadences": int(len(light_curve)),
        "time_min": float(np.nanmin(time_values)),
        "time_max": float(np.nanmax(time_values)),
        "flux_median": float(np.nanmedian(flux_values)),
        "flux_std": float(np.nanstd(flux_values)),
    }


def save_light_curve_plot(light_curve, output_path: Path, target_name: str) -> None:
    """Save a simple scatter plot for visual confirmation the data loaded."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(10, 4))
    light_curve.scatter(ax=axis, color="black", s=0.5)
    axis.set_title(f"{target_name} light curve (inspection view)")
    axis.set_xlabel("Time [BTJD]")
    axis.set_ylabel("Relative Flux")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
