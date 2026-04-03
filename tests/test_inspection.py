import numpy as np
from lightkurve import LightCurve

from exoplanet_search.inspection import lightly_preprocess_light_curve, summarize_light_curve


def test_summarize_light_curve_returns_expected_fields():
    lc = LightCurve(time=np.array([1.0, 2.0, 3.0]), flux=np.array([1.0, 0.99, 1.01]))

    summary = summarize_light_curve(lc)

    assert summary["n_cadences"] == 3
    assert summary["time_min"] == 1.0
    assert summary["time_max"] == 3.0


def test_lightly_preprocess_removes_nans():
    lc = LightCurve(time=np.array([1.0, 2.0, 3.0]), flux=np.array([1.0, np.nan, 0.99]))

    processed = lightly_preprocess_light_curve(lc, sigma=0, normalize=False)

    assert len(processed) == 2
    assert np.all(np.isfinite(processed.flux.value))
