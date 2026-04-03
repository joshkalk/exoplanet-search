"""Project configuration constants for early Kepler validation tasks."""

from pathlib import Path

DEFAULT_TARGET = "Kepler-5"
DEFAULT_MISSION = "Kepler"
DEFAULT_AUTHOR = "Kepler"
DEFAULT_CADENCE = "long"

DATA_RAW_DIR = Path("data/raw")
DATA_INTERIM_DIR = Path("data/interim")
DEFAULT_INSPECTION_DIR = DATA_INTERIM_DIR / "kepler5_inspection"
