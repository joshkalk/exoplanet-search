from exoplanet_search import __version__
from exoplanet_search.cli import build_parser
from exoplanet_search.data_access import download_kepler_light_curve
from exoplanet_search.inspection import lightly_preprocess_light_curve, summarize_light_curve


def test_version_exists():
    assert __version__ == "0.1.0"


def test_imports_smoke():
    assert build_parser is not None
    assert download_kepler_light_curve is not None
    assert lightly_preprocess_light_curve is not None
    assert summarize_light_curve is not None
