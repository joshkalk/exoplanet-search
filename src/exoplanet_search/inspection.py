"""Light-touch light-curve inspection helpers."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .config import DEFAULT_TIME_SYSTEM
from .preprocessing import PreprocessingConfig, preprocess_light_curve


def lightly_preprocess_light_curve(light_curve, sigma: float = 5.0, normalize: bool = True):
    """Backward-compatible default preprocessing wrapper.

    The scientific default is no flux-amplitude clipping; ``sigma`` is retained
    for API compatibility with older calls.
    """
    result = preprocess_light_curve(
        light_curve,
        PreprocessingConfig(mode="none", sigma=sigma, normalize=normalize),
    )
    return result.light_curve


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


def save_light_curve_plot(
    light_curve,
    output_path: Path,
    target_name: str,
    time_system: str = DEFAULT_TIME_SYSTEM,
) -> None:
    """Save a simple scatter plot for visual confirmation the data loaded."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(10, 4))
    light_curve.scatter(ax=axis, color="black", s=0.5)
    axis.set_title(f"{target_name} light curve (inspection view)")
    axis.set_xlabel(f"Time [{time_system}]")
    axis.set_ylabel("Relative Flux")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
