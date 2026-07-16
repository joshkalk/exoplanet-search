"""Phase 1C output writers and lightweight diagnostic plots."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .phase1c_types import PARAMETER_ORDER


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write indented UTF-8 JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(_json_safe(payload), output_file, indent=2, allow_nan=False)


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_trace_plot(path: Path, chain: np.ndarray, *, label: str | None = None) -> None:
    """Write a compact transformed-parameter trace plot."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(len(PARAMETER_ORDER), 1, figsize=(10, 12), sharex=True)
    for index, name in enumerate(PARAMETER_ORDER):
        axes[index].plot(chain[:, :, index].T, color="black", alpha=0.08, linewidth=0.5)
        axes[index].set_ylabel(name)
    if label:
        figure.suptitle(label, fontsize=13, fontweight="bold")
    axes[-1].set_xlabel("step")
    figure.tight_layout(rect=(0, 0, 1, 0.98) if label else None)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def write_marginal_plot(path: Path, flat_chain: np.ndarray, *, label: str | None = None) -> None:
    """Write marginal histograms for transformed parameters."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 4, figsize=(12, 6))
    for index, axis in enumerate(axes.ravel()):
        axis.hist(flat_chain[:, index], bins=30, color="#4c78a8", alpha=0.85)
        axis.set_title(PARAMETER_ORDER[index])
    if label:
        figure.suptitle(label, fontsize=13, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.94) if label else None)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def write_correlation_plot(path: Path, flat_chain: np.ndarray, *, label: str | None = None) -> None:
    """Write a transformed-parameter correlation image."""
    path.parent.mkdir(parents=True, exist_ok=True)
    corr = np.corrcoef(flat_chain.T)
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    axis.set_xticks(range(len(PARAMETER_ORDER)), PARAMETER_ORDER, rotation=45, ha="right")
    axis.set_yticks(range(len(PARAMETER_ORDER)), PARAMETER_ORDER)
    if label:
        axis.set_title(label, fontsize=13, fontweight="bold")
    figure.colorbar(image, ax=axis, label="correlation")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
