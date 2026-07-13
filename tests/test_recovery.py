import numpy as np
from lightkurve import LightCurve

from exoplanet_search.recovery import (
    compute_phase_days,
    estimate_known_transit_signal,
    estimate_windowed_known_transit_signal,
)


def test_compute_phase_days_centers_expected_mid_transit():
    lc = LightCurve(time=np.array([10.0, 12.0, 14.0]), flux=np.ones(3))

    phase_days = compute_phase_days(lc, period_days=2.0, epoch_bkjd=10.0)

    assert np.allclose(phase_days, np.array([0.0, 0.0, 0.0]))


def test_estimate_known_transit_signal_recovers_injected_depth():
    time = np.arange(0.0, 20.0, 0.02)
    flux = np.ones_like(time)
    period_days = 3.5
    epoch_bkjd = 0.5
    duration_hours = 4.8
    half_duration_days = duration_hours / 24.0 / 2.0

    phase_days = ((time - epoch_bkjd + 0.5 * period_days) % period_days) - 0.5 * period_days
    flux[np.abs(phase_days) <= half_duration_days] -= 0.0072

    lc = LightCurve(time=time, flux=flux)
    recovery = estimate_known_transit_signal(
        lc,
        period_days=period_days,
        epoch_bkjd=epoch_bkjd,
        duration_hours=duration_hours,
    )

    assert recovery["n_in_transit"] > 0
    assert recovery["transit_depth"] > 0.006
    assert recovery["transit_depth_ppm"] > 6000
    assert recovery["depth_snr_proxy"] > 10


def test_estimate_windowed_known_transit_signal_handles_slow_baseline_trend():
    time = np.arange(0.0, 40.0, 0.02)
    period_days = 3.5
    epoch_bkjd = 0.5
    duration_hours = 4.8
    half_duration_days = duration_hours / 24.0 / 2.0

    flux = 1.0 + 0.0015 * np.sin(2.0 * np.pi * time / 18.0)
    phase_days = ((time - epoch_bkjd + 0.5 * period_days) % period_days) - 0.5 * period_days
    flux[np.abs(phase_days) <= half_duration_days] -= 0.0072

    lc = LightCurve(time=time, flux=flux)
    recovery = estimate_windowed_known_transit_signal(
        lc,
        period_days=period_days,
        epoch_bkjd=epoch_bkjd,
        duration_hours=duration_hours,
        window_half_width_days=0.35,
        transit_mask_scale=1.25,
        baseline_mask_scale=2.0,
    )

    assert recovery["n_transit_windows"] >= 5
    assert recovery["windowed_transit_depth"] > 0.006
    assert recovery["windowed_transit_depth_ppm"] > 6000
    assert recovery["windowed_depth_snr_proxy"] > 10
