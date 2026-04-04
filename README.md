# exoplanet-search

Validate an exoplanet transit-search workflow by reloading known Kepler systems in a clean, testable Python package.

## Getting started (Task 1: Kepler-5 inspection)

### 1) Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

On Windows Git Bash, activate the venv with `source .venv/Scripts/activate`.
If `pip install -e ".[dev]"` fails because build isolation cannot reach the package index,
install local build tools first and retry without build isolation:

```bash
pip install "setuptools>=68" wheel
pip install --no-build-isolation -e ".[dev]"
```

### 2) Run tests

```bash
pytest
```

### 3) Run first validation step (Kepler-5)

This command downloads Kepler-5 light-curve products via `lightkurve`, applies only light preprocessing
(remove NaNs, remove high-sigma outliers, normalize), and writes a quick summary + plot.

```bash
kepler5-inspect
```

Equivalent module invocation:

```bash
python -m exoplanet_search.cli
```

### 4) Run minimal Kepler-5 recovery

This keeps preprocessing conservative, then phase-folds the light curve on the
known Kepler-5 b ephemeris to confirm the expected transit signal is present.

```bash
python -m exoplanet_search.cli --recover
```

### Outputs

- Download cache/files: `data/raw/`
- Inspection outputs: `data/interim/kepler5_inspection/`
  - `summary.json`
  - `light_curve.png`
- Recovery outputs: `data/interim/kepler5_recovery/`
  - `recovery_summary.json`
  - `folded_light_curve.png`

These directories are already git-ignored for generated data products.
