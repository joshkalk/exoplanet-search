"""Project configuration constants for early Kepler validation tasks."""

from pathlib import Path

DEFAULT_TARGET = "Kepler-5"
DEFAULT_MISSION = "Kepler"
DEFAULT_AUTHOR = "Kepler"
DEFAULT_CADENCE = "long"

DATA_RAW_DIR = Path("data/raw")
DATA_INTERIM_DIR = Path("data/interim")
DEFAULT_INSPECTION_DIR = DATA_INTERIM_DIR / "kepler5_inspection"
DEFAULT_RECOVERY_DIR = DATA_INTERIM_DIR / "kepler5_recovery"

# Kepler-5 b ephemeris from the NASA Exoplanet Archive DR25 KOI solution.
# Kepler light-curve times are handled in BKJD for this recovery step.
KEPLER5B_PERIOD_DAYS = 3.548465446
KEPLER5B_EPOCH_BKJD = 122.9014315
KEPLER5B_DURATION_HOURS = 4.57705
