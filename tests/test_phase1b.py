import importlib
import inspect
import json

import numpy as np
import pytest

from exoplanet_search.phase1b import load_limb_darkening_inputs
from exoplanet_search.phase1b_fit import run_fit_stage
from exoplanet_search.phase1b_model import (
    PhysicalParameters,
    batman_flux,
    q_to_u,
    solve_local_baselines,
    u_to_q,
    validate_physical,
)
from exoplanet_search.phase1b_observations import build_transit_windows
from exoplanet_search.phase1b_types import FitData, LimbDarkeningInputs, ObservationSet, Phase1BConfig


PERIOD = 5.0
T0 = 1.0
DURATION = 0.19
EXPOSURE = 0.020433
TRUE = PhysicalParameters(rp=0.1, a=9.0, b=0.35, q1=0.36, q2=1.0 / 3.0, jitter=0.0, period=PERIOD, t0=T0)


def make_observations(noise=0.0, baselines=False, seed=123):
    rng = np.random.default_rng(seed)
    centers = T0 + np.arange(0, 7) * PERIOD
    time_parts = [np.arange(center - 0.55, center + 0.55, EXPOSURE) for center in centers]
    time = np.concatenate(time_parts)
    event_numbers = np.concatenate([np.full(len(part), index) for index, part in enumerate(time_parts)])
    transit = batman_flux(
        time,
        np.full(len(time), EXPOSURE),
        rp=TRUE.rp,
        a=TRUE.a,
        b=TRUE.b,
        q1=TRUE.q1,
        q2=TRUE.q2,
        period=PERIOD,
        t0=T0,
        supersample_factor=15,
    )
    baseline = np.ones_like(time)
    if baselines:
        for event in np.unique(event_numbers):
            mask = event_numbers == event
            baseline[mask] = 1.0 + 0.0005 * event + (event - 3) * 0.0004 * (
                time[mask] - centers[event]
            )
    flux = baseline * transit + rng.normal(0.0, noise, len(time))
    return ObservationSet(
        time=time,
        flux=flux,
        flux_err=np.full(len(time), max(noise, 2.0e-5)),
        product_id=np.full(len(time), "synthetic_product", dtype=object),
        quarter=np.full(len(time), "synthetic_quarter", dtype=object),
        cadence=np.full(len(time), "long", dtype=object),
        exposure_days=np.full(len(time), EXPOSURE),
        mission=np.full(len(time), "Synthetic", dtype=object),
        flux_product=np.full(len(time), "relative_flux", dtype=object),
    )


def fit_data_from_observations(observations):
    config = Phase1BConfig(n_starts=4, supersample_factor=9, high_supersample_factor=15)
    windows = build_transit_windows(
        observations,
        period_days=PERIOD,
        transit_time=T0,
        duration_days=DURATION,
        config=config,
    )
    assert np.any(windows.accepted_mask)
    return FitData(
        observations=observations,
        time=observations.time[windows.accepted_mask],
        flux=observations.flux[windows.accepted_mask],
        flux_err=observations.flux_err[windows.accepted_mask],
        event_number=windows.event_number[windows.accepted_mask],
        predicted_center=windows.predicted_center[windows.accepted_mask],
        exposure_days=observations.exposure_days[windows.accepted_mask],
        product_id=observations.product_id[windows.accepted_mask],
        quarter=observations.quarter[windows.accepted_mask],
        phase1a_period_days=PERIOD,
        phase1a_transit_time=T0,
        phase1a_duration_days=DURATION,
    )


def limb_inputs():
    u1, u2 = q_to_u(TRUE.q1, TRUE.q2)
    return LimbDarkeningInputs(
        q1=TRUE.q1,
        q2=TRUE.q2,
        q1_sigma=0.05,
        q2_sigma=0.05,
        u1=u1,
        u2=u2,
        metadata={"synthetic": True},
    )


def test_noise_free_batman_recovery_with_exposure_integration():
    data = fit_data_from_observations(make_observations(noise=0.0))

    result = run_fit_stage(
        data,
        limb_inputs(),
        Phase1BConfig(n_starts=4, supersample_factor=9),
        stage="synthetic_noise_free",
        fit_timing=False,
    )

    assert result.parameters["rp_over_rstar"] == pytest.approx(TRUE.rp, abs=0.01)
    assert result.parameters["a_over_rstar"] == pytest.approx(TRUE.a, abs=0.8)
    assert result.parameters["impact_parameter"] == pytest.approx(TRUE.b, abs=0.15)


def test_noisy_recovery_is_seeded_and_reasonably_close():
    data = fit_data_from_observations(make_observations(noise=3.0e-4, seed=222))
    config = Phase1BConfig(n_starts=4, random_seed=999, supersample_factor=9)

    first = run_fit_stage(data, limb_inputs(), config, stage="noisy", fit_timing=False)
    second = run_fit_stage(data, limb_inputs(), config, stage="noisy", fit_timing=False)

    assert first.parameters["rp_over_rstar"] == pytest.approx(second.parameters["rp_over_rstar"])
    assert first.parameters["rp_over_rstar"] == pytest.approx(TRUE.rp, abs=0.025)


def test_analytical_local_baseline_recovery_limits_transit_bias():
    data = fit_data_from_observations(make_observations(noise=1.0e-4, baselines=True))

    result = run_fit_stage(
        data,
        limb_inputs(),
        Phase1BConfig(n_starts=4, supersample_factor=9),
        stage="baseline",
        fit_timing=False,
    )

    assert result.parameters["rp_over_rstar"] == pytest.approx(TRUE.rp, abs=0.02)
    assert all("baseline_intercept" in row for row in result.baseline_rows)


def test_baseline_solver_recovers_injected_linear_terms():
    time = np.array([0.0, 0.5, 1.0, 1.5])
    event = np.zeros(4, dtype=int)
    center = np.full(4, 0.75)
    transit = np.ones(4)
    flux = 1.02 - 0.01 * (time - 0.75)

    baseline, combined, rows = solve_local_baselines(
        time,
        flux,
        np.full(4, 0.001),
        event,
        center,
        transit,
    )

    assert baseline == pytest.approx(flux)
    assert combined == pytest.approx(flux)
    assert rows[0]["baseline_intercept"] == pytest.approx(1.02)
    assert rows[0]["baseline_slope_per_day"] == pytest.approx(-0.01)


def test_exposure_integration_beats_instantaneous_model_for_integrated_data():
    observations = make_observations(noise=0.0)
    integrated = batman_flux(
        observations.time,
        observations.exposure_days,
        rp=TRUE.rp,
        a=TRUE.a,
        b=TRUE.b,
        q1=TRUE.q1,
        q2=TRUE.q2,
        period=PERIOD,
        t0=T0,
        supersample_factor=15,
    )
    instantaneous = batman_flux(
        observations.time,
        np.full(len(observations.time), 1.0e-9),
        rp=TRUE.rp,
        a=TRUE.a,
        b=TRUE.b,
        q1=TRUE.q1,
        q2=TRUE.q2,
        period=PERIOD,
        t0=T0,
        supersample_factor=1,
    )
    converged = batman_flux(
        observations.time,
        observations.exposure_days,
        rp=TRUE.rp,
        a=TRUE.a,
        b=TRUE.b,
        q1=TRUE.q1,
        q2=TRUE.q2,
        period=PERIOD,
        t0=T0,
        supersample_factor=25,
    )

    assert np.std(integrated - observations.flux) < np.std(instantaneous - observations.flux)
    assert np.max(np.abs(converged - integrated)) < 2.0e-5


def test_window_audit_records_all_events_and_objective_reasons():
    observations = make_observations()
    mask = ~(
        ((observations.time > T0 + PERIOD - 0.08) & (observations.time < T0 + PERIOD + 0.08))
        | ((observations.time > T0 + 2 * PERIOD - 0.55) & (observations.time < T0 + 2 * PERIOD - 0.2))
        | ((observations.time > T0 + 3 * PERIOD + 0.2) & (observations.time < T0 + 3 * PERIOD + 0.55))
    )
    trimmed = ObservationSet(
        time=observations.time[mask],
        flux=observations.flux[mask],
        flux_err=observations.flux_err[mask],
        product_id=observations.product_id[mask],
        quarter=observations.quarter[mask],
        cadence=observations.cadence[mask],
        exposure_days=observations.exposure_days[mask],
        mission=observations.mission[mask],
        flux_product=observations.flux_product[mask],
    )

    windows = build_transit_windows(
        trimmed,
        period_days=PERIOD,
        transit_time=T0,
        duration_days=DURATION,
        config=Phase1BConfig(),
    )

    assert len(windows.audit_rows) == 7
    reasons = ";".join(row["exclusion_reasons"] for row in windows.audit_rows)
    assert "too_few_expected_in_transit_points" in reasons
    assert "too_few_pre_transit_baseline_points" in reasons
    assert "too_few_post_transit_baseline_points" in reasons


def test_physical_constraints_and_limb_darkening_conversion():
    assert validate_physical(PhysicalParameters(rp=-0.1, a=4.0, b=0.2, q1=0.5, q2=0.5))
    u1, u2 = q_to_u(0.36, 1.0 / 3.0)
    q1, q2 = u_to_q(u1, u2)
    assert q1 == pytest.approx(0.36)
    assert q2 == pytest.approx(1.0 / 3.0)
    assert 0.0 <= q1 <= 1.0
    assert 0.0 <= q2 <= 1.0


def test_limb_darkening_input_file_records_source_metadata(tmp_path):
    path = tmp_path / "stellar_inputs.json"
    path.write_text(
        json.dumps(
            {
                "stellar_inputs": {"teff": 6000, "logg": 4.2, "metallicity": 0.0},
                "source_catalog": "synthetic test fixture",
                "retrieval_date": "2026-07-15",
                "limb_darkening_package": "synthetic fixture",
                "atmosphere_grid": "synthetic",
                "kepler_bandpass_identifier": "Kepler",
                "quadratic_coefficients": {"u1": 0.4, "u2": 0.2},
                "coefficient_uncertainties": {"q1_sigma": 0.1, "q2_sigma": 0.1},
            }
        ),
        encoding="utf-8",
    )

    limb = load_limb_darkening_inputs(path)

    assert limb.metadata["source_catalog"] == "synthetic test fixture"
    assert limb.q1 == pytest.approx(0.36)


def test_generic_phase1b_modules_have_no_kepler5_planet_constant_imports():
    for module_name in [
        "exoplanet_search.phase1b",
        "exoplanet_search.phase1b_fit",
        "exoplanet_search.phase1b_model",
        "exoplanet_search.phase1b_observations",
    ]:
        module = importlib.import_module(module_name)
        assert "KEPLER5B_" not in inspect.getsource(module)

