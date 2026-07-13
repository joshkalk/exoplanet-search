"""Project configuration constants for early Kepler validation tasks."""

from pathlib import Path

DEFAULT_TARGET = "Kepler-5"
DEFAULT_MISSION = "Kepler"
DEFAULT_AUTHOR = "Kepler"
DEFAULT_CADENCE = "long"
DEFAULT_FLUX_COLUMN = "pdcsap_flux"
DEFAULT_QUALITY_BITMASK = "default"
DEFAULT_TIME_SYSTEM = "BKJD"

DATA_RAW_DIR = Path("data/raw")
DATA_INTERIM_DIR = Path("data/interim")
DEFAULT_INSPECTION_DIR = DATA_INTERIM_DIR / "kepler5_inspection"
DEFAULT_RECOVERY_DIR = DATA_INTERIM_DIR / "kepler5_recovery"
DEFAULT_WINDOWED_RECOVERY_DIR = DATA_INTERIM_DIR / "kepler5_windowed_recovery"
DEFAULT_COMPARISON_DIR = DATA_INTERIM_DIR / "kepler5_preprocessing_comparison"

# Kepler-5 b ephemeris from the NASA Exoplanet Archive DR25 KOI solution.
# Kepler light-curve times are handled in BKJD for this recovery step.
KEPLER5B_PERIOD_DAYS = 3.548465446
KEPLER5B_EPOCH_BKJD = 122.9014315
KEPLER5B_DURATION_HOURS = 4.57705
KEPLER5B_WINDOW_HALF_WIDTH_DAYS = 0.35
KEPLER5B_TRANSIT_MASK_SCALE = 1.25
KEPLER5B_BASELINE_MASK_SCALE = 2.0
