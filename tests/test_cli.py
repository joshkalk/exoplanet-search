import pytest

from exoplanet_search.cli import build_parser, _validate_target_specific_options


def test_plain_inspection_allows_non_kepler5_target():
    args = build_parser().parse_args(["--target", "Kepler-10"])

    _validate_target_specific_options(args)


def test_physical_transit_fit_parser_accepts_configurable_paths(tmp_path):
    args = build_parser().parse_args(
        [
            "--physical-transit-fit",
            "--phase1a-summary-path",
            str(tmp_path / "summary.json"),
            "--phase1a-provenance-path",
            str(tmp_path / "provenance.json"),
            "--stellar-inputs-path",
            str(tmp_path / "stellar.json"),
            "--phase1b-output-dir",
            str(tmp_path / "phase1b"),
        ]
    )

    assert args.physical_transit_fit is True
    assert args.phase1a_summary_path == tmp_path / "summary.json"
    assert args.stellar_inputs_path == tmp_path / "stellar.json"


@pytest.mark.parametrize(
    "flag",
    [
        "--recover",
        "--windowed-recovery",
        "--compare-preprocessing",
    ],
)
def test_known_ephemeris_flags_reject_non_kepler5_target(flag):
    args = build_parser().parse_args(["--target", "Kepler-10", flag])

    with pytest.raises(SystemExit, match="implemented only for Kepler-5"):
        _validate_target_specific_options(args)


def test_transit_protected_mode_rejects_non_kepler5_target():
    args = build_parser().parse_args(
        ["--target", "Kepler-10", "--preprocessing-mode", "transit_protected_symmetric"]
    )

    with pytest.raises(SystemExit, match="implemented only for Kepler-5"):
        _validate_target_specific_options(args)
