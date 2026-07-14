import csv
import importlib
import inspect
import json

import numpy as np
import pytest
from lightkurve import LightCurve

from exoplanet_search.phase1a import (
    BLSSearchConfig,
    alias_diagnostics,
    assign_harmonic_families,
    chronological_split,
    distinct_candidates,
    evaluate_holdout,
    odd_even_diagnostic,
    odd_even_warning_trigger,
    published_comparison,
    refine_single_candidate,
    run_bls_search,
    run_phase1a_search,
    write_csv,
)


SYNTHETIC_PERIOD = 9.75
SYNTHETIC_EPOCH = 2.2
SYNTHETIC_DURATION_DAYS = 0.22
SYNTHETIC_DEPTH = 0.015


def make_synthetic_phase1a_light_curve():
    rng = np.random.default_rng(123)
    time = np.arange(0.0, 180.0, 0.08)
    gap_mask = ~(
        ((time > 44.0) & (time < 49.0))
        | ((time > 101.0) & (time < 108.0))
        | ((time > 150.0) & (time < 153.0))
    )
    time = time[gap_mask]
    flux = 1.0 + rng.normal(0.0, 0.0008, len(time))
    phase = ((time - SYNTHETIC_EPOCH + 0.5 * SYNTHETIC_PERIOD) % SYNTHETIC_PERIOD) - (
        0.5 * SYNTHETIC_PERIOD
    )
    flux[np.abs(phase) <= SYNTHETIC_DURATION_DAYS / 2.0] -= SYNTHETIC_DEPTH
    return LightCurve(time=time, flux=flux)


def synthetic_config(**overrides):
    values = {
        "minimum_period_days": 2.0,
        "maximum_period_days": 30.0,
        "minimum_duration_hours": 2.0,
        "maximum_duration_hours": 10.0,
        "n_durations": 5,
        "n_periods": 450,
        "frequency_factor": 1.0,
        "top_n_peaks": 4,
        "training_fraction": 0.70,
        "local_max_period_samples": 1200,
        "local_duration_step_hours": 0.5,
    }
    values.update(overrides)
    return BLSSearchConfig(**values)


def time_flux(light_curve):
    return np.asarray(light_curve.time.value, dtype=float), np.asarray(light_curve.flux.value, dtype=float)


def test_broad_training_search_finds_transit_like_candidate():
    lc = make_synthetic_phase1a_light_curve()
    time, flux = time_flux(lc)
    train_mask, _, _ = chronological_split(time, 0.70)

    result = run_bls_search(time[train_mask], flux[train_mask], synthetic_config())
    candidate = distinct_candidates(result, time[train_mask], synthetic_config())[0]

    recovered_ratio = candidate["period_days"] / SYNTHETIC_PERIOD
    assert recovered_ratio == pytest.approx(1.0, rel=0.03) or recovered_ratio == pytest.approx(0.5, rel=0.03)
    assert candidate["duration_days"] == pytest.approx(SYNTHETIC_DURATION_DAYS, abs=0.12)
    assert candidate["depth"] > 0.006
    assert candidate["rank"] == 1


def test_local_refinement_improves_coarse_period_and_records_precision():
    lc = make_synthetic_phase1a_light_curve()
    time, flux = time_flux(lc)
    train_mask, _, _ = chronological_split(time, 0.70)
    config = synthetic_config(allowed_phase_drift_fraction=0.12)
    coarse_candidate = {
        "rank": 1,
        "period_days": 9.62,
        "duration_days": 0.25,
        "transit_time": SYNTHETIC_EPOCH,
        "bls_power": 1.0,
    }
    broad_result = {"periods": np.linspace(9.0, 10.3, 60)}

    refined = refine_single_candidate(
        time[train_mask],
        flux[train_mask],
        coarse_candidate,
        broad_result,
        config,
        coarse_rank=1,
        refined_rank=1,
        local_window=(9.3, 10.1),
    )

    assert abs(refined["refined_period_days"] - SYNTHETIC_PERIOD) < abs(
        coarse_candidate["period_days"] - SYNTHETIC_PERIOD
    )
    assert refined["refined_duration_days"] == pytest.approx(SYNTHETIC_DURATION_DAYS, abs=0.04)
    assert (
        refined["implied_training_baseline_timing_drift_fraction_of_duration"]
        <= config.allowed_phase_drift_fraction + 0.02
    )


def test_holdout_prediction_uses_locked_refined_training_ephemeris_without_recenter():
    lc = make_synthetic_phase1a_light_curve()
    time, flux = time_flux(lc)
    train_mask, holdout_mask, _ = chronological_split(time, 0.70)
    config = synthetic_config()
    result = run_bls_search(time[train_mask], flux[train_mask], config)
    broad = distinct_candidates(result, time[train_mask], config)[0]
    locked = refine_single_candidate(time[train_mask], flux[train_mask], broad, result, config, 1, 1)
    original_period = locked["refined_period_days"]
    original_epoch = locked["refined_transit_time"]

    summary, events = evaluate_holdout(time[holdout_mask], flux[holdout_mask], locked, config)

    assert summary["usable_event_count"] > 0
    assert summary["aggregate_depth"] > 0.006
    assert locked["refined_period_days"] == original_period
    assert locked["refined_transit_time"] == original_epoch
    assert all("predicted_transit_center" in event for event in events)


def test_leading_peak_deduplication_orders_distinct_candidates():
    result = {
        "periods": np.array([10.0, 10.03, 20.0, 30.0]),
        "power": np.array([5.0, 4.9, 4.0, 3.0]),
        "duration": np.array([0.2, 0.2, 0.3, 0.4]),
        "transit_time": np.array([1.0, 1.1, 2.0, 3.0]),
        "depth": np.array([0.01, 0.01, 0.005, 0.004]),
        "depth_snr": np.array([10.0, 9.0, 5.0, 4.0]),
    }
    config = BLSSearchConfig(top_n_peaks=3, peak_period_separation_fraction=0.01)

    candidates = distinct_candidates(result, np.arange(0.0, 100.0, 1.0), config)

    assert [candidate["period_days"] for candidate in candidates] == [10.0, 20.0, 30.0]
    assert [candidate["rank"] for candidate in candidates] == [1, 2, 3]


def test_harmonic_family_grouping_labels_small_integer_aliases():
    candidates = [
        {"refined_period_days": 10.0},
        {"refined_period_days": 5.0},
        {"refined_period_days": 20.0},
        {"refined_period_days": 30.0},
        {"refined_period_days": 10.0 / 3.0},
    ]

    grouped = assign_harmonic_families(candidates, BLSSearchConfig())

    assert [row["harmonic_family_id"] for row in grouped] == [1, 1, 1, 1, 1]
    assert [row["relationship_to_family_reference"] for row in grouped] == ["1", "1/2", "2", "3", "1/3"]


def test_odd_even_diagnostic_flags_half_period_alias():
    lc = make_synthetic_phase1a_light_curve()
    time, flux = time_flux(lc)
    alias_candidate = {
        "period_days": SYNTHETIC_PERIOD / 2.0,
        "duration_days": SYNTHETIC_DURATION_DAYS,
        "transit_time": SYNTHETIC_EPOCH,
    }

    diagnostic = odd_even_diagnostic(time, flux, alias_candidate, synthetic_config())

    assert diagnostic["usable_odd_event_count"] > 0
    assert diagnostic["usable_even_event_count"] > 0
    assert diagnostic["strong_odd_even_inconsistency"] is True
    assert diagnostic["odd_even_warning_trigger"] == "ratio_and_absolute_difference"


def test_odd_even_ratio_only_near_zero_depths_are_not_flagged():
    config = synthetic_config()

    trigger = odd_even_warning_trigger(
        ratio=45.0 / 7.0,
        absolute_difference=38.0e-6,
        local_depth_scatter=2563.0e-6,
        config=config,
    )

    assert trigger == "none"


def test_odd_even_positive_negative_near_zero_depths_are_not_ratio_flagged():
    config = synthetic_config()

    trigger = odd_even_warning_trigger(
        ratio=4.0,
        absolute_difference=4.6e-6,
        local_depth_scatter=2397.0e-6,
        config=config,
    )

    assert trigger == "none"


def test_odd_even_large_absolute_difference_can_flag_without_ratio():
    config = synthetic_config()

    trigger = odd_even_warning_trigger(
        ratio=1.5,
        absolute_difference=3000.0e-6,
        local_depth_scatter=900.0e-6,
        config=config,
    )

    assert trigger == "absolute_difference_only"


def test_alias_diagnostics_produce_half_nominal_and_double_rows():
    lc = make_synthetic_phase1a_light_curve()
    time, flux = time_flux(lc)
    train_mask, _, _ = chronological_split(time, 0.70)
    config = synthetic_config()
    result = run_bls_search(time[train_mask], flux[train_mask], config)
    broad = distinct_candidates(result, time[train_mask], config)[0]
    candidate = refine_single_candidate(time[train_mask], flux[train_mask], broad, result, config, 1, 1)

    aliases = alias_diagnostics(time[train_mask], flux[train_mask], candidate, config)

    assert [row["alias"] for row in aliases] == ["P/2", "P", "2P"]
    assert all(row["valid"] == 1 for row in aliases)
    nominal = next(row for row in aliases if row["alias"] == "P")
    assert nominal["period_days"] == pytest.approx(candidate["refined_period_days"])
    half_period = next(row for row in aliases if row["alias"] == "P/2")
    assert half_period["odd_even_warning_trigger"] == "ratio_and_absolute_difference"
    assert "Ratio alone never triggers" in half_period["odd_even_rule"]


def test_generic_phase1a_module_has_no_kepler5_constant_imports():
    module = importlib.import_module("exoplanet_search.phase1a")

    assert "KEPLER5B" not in inspect.getsource(module)


def test_published_comparison_is_post_search_separate_step():
    candidate = {
        "refined_period_days": SYNTHETIC_PERIOD,
        "refined_transit_time": SYNTHETIC_EPOCH,
        "refined_duration_days": SYNTHETIC_DURATION_DAYS,
    }

    comparison = published_comparison(
        candidate,
        candidate,
        {
            "period_days": SYNTHETIC_PERIOD,
            "epoch_bkjd": SYNTHETIC_EPOCH,
            "duration_hours": SYNTHETIC_DURATION_DAYS * 24.0,
        },
    )

    assert comparison["comparison_stage"] == "post_search_validation_only"
    assert comparison["known_values_used_during_search"] is False


def test_phase1a_outputs_are_serializable_and_separate_global_from_local(tmp_path):
    lc = make_synthetic_phase1a_light_curve()

    summary = run_phase1a_search(
        lc,
        output_dir=tmp_path,
        target="Synthetic",
        config=synthetic_config(),
        provenance={"synthetic": True},
        published_ephemeris=None,
    )

    expected_outputs = {
        "search_summary.json",
        "top_period_candidates.csv",
        "alias_diagnostics.csv",
        "holdout_event_diagnostics.csv",
        "odd_even_diagnostics.csv",
        "harmonic_family_diagnostics.csv",
        "periodogram.png",
        "recovered_folded_light_curve.png",
        "holdout_folded_light_curve.png",
        "provenance_manifest.json",
    }
    assert expected_outputs.issubset({path.name for path in tmp_path.iterdir()})
    loaded = json.loads((tmp_path / "search_summary.json").read_text(encoding="utf-8"))
    assert loaded["known_ephemeris_used_during_search"] is False
    assert (
        loaded["broad_training_search"]["settings"]["odd_even_ratio_min_scatter_multiple"]
        == synthetic_config().odd_even_ratio_min_scatter_multiple
    )
    odd_even_row = loaded["odd_even_diagnostics"][0]
    assert "Ratio alone never triggers" in odd_even_row["rule"]
    assert odd_even_row["odd_even_warning_trigger"] in {
        "ratio_and_absolute_difference",
        "absolute_difference_only",
        "none",
    }
    assert "full_mission_global_search_diagnostic" in summary
    assert "full_mission_local_refinement" in summary
    assert summary["full_mission_global_search_diagnostic"]["label"].endswith("not_refinement")
    locked_period = summary["locked_refined_training_candidate"]["refined_period_days"]
    local_period = summary["full_mission_local_refinement"]["refined_period_days"]
    assert abs(local_period - locked_period) < 0.25
    with (tmp_path / "top_period_candidates.csv").open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert rows[0]["refined_rank"] == "1"
    assert summary["holdout_summary"]["ephemeris_locked_from_training"] is True


def test_write_csv_handles_empty_rows(tmp_path):
    path = tmp_path / "empty.csv"

    write_csv(path, [])

    assert path.read_text(encoding="utf-8").strip() == ""
