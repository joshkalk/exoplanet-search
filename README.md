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

### Outputs

- Download cache/files: `data/raw/`
- Inspection outputs: `data/interim/kepler5_inspection/`
  - `summary.json`
  - `light_curve.png`

These directories are already git-ignored for generated data products.
