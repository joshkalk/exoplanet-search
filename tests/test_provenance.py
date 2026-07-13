import json

from astropy.io import fits

from exoplanet_search.data_access import STITCHING_POLICY, _extract_downloaded_paths
from exoplanet_search.provenance import build_provenance_manifest, write_json


def test_provenance_records_fits_header_metadata_and_stitching_policy(tmp_path):
    fits_path = tmp_path / "synthetic_timestamp_name.fits"
    primary = fits.PrimaryHDU()
    science = fits.BinTableHDU()
    science.header["QUARTER"] = 7
    science.header["DATA_REL"] = 25
    science.header["PROCVER"] = "unit-test-pipeline"
    fits.HDUList([primary, science]).writeto(fits_path)

    manifest = build_provenance_manifest(
        target="Kepler-5",
        mission="Kepler",
        author="Kepler",
        cadence="long",
        flux_product="pdcsap_flux",
        time_system="BKJD",
        quality_bitmask="default",
        preprocessing={"mode": "none"},
        stitching_policy=STITCHING_POLICY,
        downloaded_paths=(fits_path,),
    )

    raw_input = manifest["raw_inputs"][0]
    assert raw_input["kepler_quarter"] == 7
    assert raw_input["fits_header_metadata"]["data_release"] == 25
    assert raw_input["fits_header_metadata"]["pipeline_version"] == "unit-test-pipeline"
    assert manifest["stitching_policy"]["corrector_func"] == "per_product_light_curve_normalize"
    assert manifest["source_fits_flux_column"] == "pdcsap_flux"


def test_provenance_records_limitation_when_exact_paths_are_unavailable(tmp_path):
    manifest = build_provenance_manifest(
        target="Kepler-5",
        mission="Kepler",
        author="Kepler",
        cadence="long",
        flux_product="pdcsap_flux",
        time_system="BKJD",
        quality_bitmask="default",
        preprocessing={"mode": "none"},
        downloaded_paths=(),
    )
    path = tmp_path / "manifest.json"
    write_json(path, manifest)
    loaded = json.loads(path.read_text(encoding="utf-8"))

    assert loaded["raw_inputs"] == []
    assert any("Exact downloaded FITS paths" in item for item in loaded["limitations"])


def test_downloaded_path_extraction_does_not_scan_download_directory(tmp_path):
    class LightCurveWithoutPath:
        meta = {}

    assert _extract_downloaded_paths([LightCurveWithoutPath()]) == ()
