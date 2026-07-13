import json
from inspect import signature

import numpy as np
import pytest
from lightkurve import LightCurve

from exoplanet_search.config import DEFAULT_TIME_SYSTEM
from exoplanet_search.inspection import save_light_curve_plot
from exoplanet_search.preprocessing import (
    PreprocessingConfig,
    preprocess_light_curve,
    removal_diagnostics,
    transit_window_mask,
    validate_preprocessing_mode,
)
from exoplanet_search.provenance import build_provenance_manifest, write_json
from exoplanet_search.recovery import estimate_known_transit_signal


PERIOD_DAYS = 3.0
EPOCH_BKJD = 0.5
DURATION_HOURS = 2.4
DEPTH = 0.02


def make_synthetic_transit_light_curve():
    time = np.arange(0.0, 30.0, 0.02)
    rng = np.random.default_rng(42)
    flux = 1.0 + rng.normal(0.0, 0.0005, len(time))
    transit_mask = _transit_mask(time)
    flux[transit_mask] -= DEPTH
    flux[[100, 500, 900]] += 0.2
    return LightCurve(time=time, flux=flux), transit_mask


def test_none_mode_preserves_finite_transits_and_recovers_depth():
    lc, injected_transit_mask = make_synthetic_transit_light_curve()

    result = preprocess_light_curve(lc, PreprocessingConfig(mode="none"))
    recovery = estimate_known_transit_signal(
        result.light_curve,
        period_days=PERIOD_DAYS,
        epoch_bkjd=EPOCH_BKJD,
        duration_hours=DURATION_HOURS,
    )

    assert np.count_nonzero(result.retained_mask & injected_transit_mask) == np.count_nonzero(
        injected_transit_mask
    )
    assert result.clipped_count == 0
    assert recovery["transit_depth"] == pytest.approx(DEPTH, abs=0.001)


def test_positive_only_removes_positive_artifacts_and_preserves_transits():
    lc, injected_transit_mask = make_synthetic_transit_light_curve()

    result = preprocess_light_curve(lc, PreprocessingConfig(mode="positive_only"))
    recovery = estimate_known_transit_signal(
        result.light_curve,
        period_days=PERIOD_DAYS,
        epoch_bkjd=EPOCH_BKJD,
        duration_hours=DURATION_HOURS,
    )

    assert result.clipped_count == 3
    assert np.count_nonzero(result.clipped_mask & injected_transit_mask) == 0
    assert recovery["transit_depth"] == pytest.approx(DEPTH, abs=0.001)


def test_symmetric_mode_can_remove_genuine_injected_transits():
    lc, injected_transit_mask = make_synthetic_transit_light_curve()

    result = preprocess_light_curve(lc, PreprocessingConfig(mode="symmetric"))

    assert np.count_nonzero(result.clipped_mask & injected_transit_mask) > 0
    assert result.output_count < len(lc)


def test_transit_protected_symmetric_preserves_protected_transits_and_recovers_depth():
    lc, injected_transit_mask = make_synthetic_transit_light_curve()

    result = preprocess_light_curve(
        lc,
        PreprocessingConfig(
            mode="transit_protected_symmetric",
            period_days=PERIOD_DAYS,
            epoch_bkjd=EPOCH_BKJD,
            duration_hours=DURATION_HOURS,
        ),
    )
    recovery = estimate_known_transit_signal(
        result.light_curve,
        period_days=PERIOD_DAYS,
        epoch_bkjd=EPOCH_BKJD,
        duration_hours=DURATION_HOURS,
    )

    assert np.count_nonzero(result.clipped_mask & injected_transit_mask) == 0
    assert recovery["transit_depth"] == pytest.approx(DEPTH, abs=0.001)


def test_preprocessing_mode_validation_rejects_unknown_mode():
    with pytest.raises(ValueError, match="Unknown preprocessing mode"):
        validate_preprocessing_mode("mystery")


def test_cadence_removal_accounting_counts_transit_and_phase_bins():
    lc, _ = make_synthetic_transit_light_curve()
    result = preprocess_light_curve(lc, PreprocessingConfig(mode="symmetric"))

    counts, phase_rows = removal_diagnostics(
        lc,
        result,
        period_days=PERIOD_DAYS,
        epoch_bkjd=EPOCH_BKJD,
        duration_hours=DURATION_HOURS,
        phase_bins=5,
    )

    assert counts["removed_inside_known_transit_window"] > 0
    assert sum(row["removed_count"] for row in phase_rows) == result.removed_count


def test_provenance_manifest_serializes(tmp_path):
    manifest = build_provenance_manifest(
        target="Synthetic",
        mission="UnitTest",
        author="UnitTest",
        cadence="long",
        flux_product="pdcsap_flux",
        time_system="BKJD",
        quality_bitmask="default",
        preprocessing={"mode": "none"},
        cadence_counts={"output_count": 10},
    )
    path = tmp_path / "manifest.json"

    write_json(path, manifest)
    loaded = json.loads(path.read_text(encoding="utf-8"))

    assert loaded["target_query"] == "Synthetic"
    assert loaded["time_system"] == "BKJD"
    assert "numpy" in loaded["packages"]


def test_inspection_plot_defaults_to_bkjd_label():
    assert signature(save_light_curve_plot).parameters["time_system"].default == DEFAULT_TIME_SYSTEM


def _transit_mask(time):
    lc = LightCurve(time=time, flux=np.ones_like(time))
    return transit_window_mask(
        lc,
        period_days=PERIOD_DAYS,
        epoch_bkjd=EPOCH_BKJD,
        duration_hours=DURATION_HOURS,
        transit_mask_scale=1.0,
    )
