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

This command downloads Kepler-5 light-curve products via `lightkurve`, stitches
the selected FITS flux column after explicit per-product normalization, removes
non-finite cadences, applies an additional global median normalization, and
writes a quick summary + plot. The default scientific preprocessing mode is
`none`, which means no generic flux-amplitude clipping is applied. This
preserves downward transit-like excursions, asymmetric shoulders,
ingress/egress structure, and unusual short cadence sequences for later
analysis.

```bash
kepler5-inspect
```

Equivalent module invocation:

```bash
python -m exoplanet_search.cli
```

To run a different preprocessing mode for the ordinary inspection/recovery
outputs:

```bash
python -m exoplanet_search.cli --preprocessing-mode positive_only
```

### 4) Run minimal Kepler-5 recovery

This keeps preprocessing conservative, then phase-folds the light curve on the
known Kepler-5 b ephemeris to confirm the expected transit signal is present.

```bash
python -m exoplanet_search.cli --recover
```

### 5) Run windowed known-period recovery

This cuts out windows around each expected Kepler-5 b transit, normalizes each
window using only the local out-of-transit wings, and stacks them for a more
robust known-period recovery check.

```bash
python -m exoplanet_search.cli --windowed-recovery
```

### 6) Compare preprocessing modes

Phase 0 includes a diagnostic comparison that runs the same downloaded Kepler-5
data through four preprocessing modes:

- `none`: remove non-finite cadences, apply the Lightkurve download quality
  policy, normalize, and perform no flux-amplitude clipping. Non-finite
  removals are accounted for separately from clipping. This is the default and
  the appropriate baseline for blind searches.
- `positive_only`: remove only sufficiently extreme positive flux excursions
  using the five-sigma convention, while preserving downward excursions.
- `symmetric`: preserve the old symmetric five-sigma clipping behavior as a
  comparison mode. This can erase real transit cadences and is not the blind
  search default.
- `transit_protected_symmetric`: protect the published Kepler-5 b transit
  windows, then apply symmetric clipping outside them. This is target-specific
  and diagnostic only; it is not suitable for blind searches.

```bash
python -m exoplanet_search.cli --compare-preprocessing
```

Known Kepler-5 b ephemeris values are used only after preprocessing for
diagnostics: recovery plots, cadence-removal phase counts, and comparison
summaries. Current SNR values are diagnostic proxies, not formal false-alarm
probabilities or detection claims.

The recovery and comparison flags are Kepler-5-specific because they use the
published Kepler-5 b ephemeris. For other targets, use the ordinary inspection
command without `--recover`, `--windowed-recovery`, `--compare-preprocessing`,
or `--preprocessing-mode transit_protected_symmetric`.

### Outputs

- Download cache/files: `data/raw/`
- Inspection outputs: `data/interim/kepler5_inspection/`
  - `summary.json`
  - `light_curve.png`
- Recovery outputs: `data/interim/kepler5_recovery/`
  - `recovery_summary.json`
  - `folded_light_curve.png`
- Windowed recovery outputs: `data/interim/kepler5_windowed_recovery/`
  - `windowed_recovery_summary.json`
  - `windowed_folded_light_curve.png`
- Preprocessing comparison outputs: `data/interim/kepler5_preprocessing_comparison/`
  - `preprocessing_comparison_summary.json`
  - `phase_binned_removed_cadences.csv`
  - `provenance_manifest.json`
  - `preprocessing_comparison.png`

These directories are already git-ignored for generated data products.

Each run records provenance where available, including UTC timestamp, Git commit
and dirty status, Python and package versions, target query, mission, cadence,
source FITS flux column, stitched-flux normalization policy, time system,
Lightkurve quality bitmask policy, preprocessing parameters, cadence counts,
raw input filenames, FITS header metadata, and SHA-256 checksums. If exact FITS
paths are not exposed by Lightkurve, the manifest records that limitation
instead of assigning unrelated files under `data/raw/` to the run.

## Phase 1A: blind period search

Phase 1A asks whether the Phase 0 Kepler-5 light curve can independently recover
a repeating transit using generic Box Least Squares settings. The search uses
Astropy's `BoxLeastSquares`, the Phase 0 default no-clipping preprocessing, and
a chronological split: the first 70% of the observed time baseline selects the
candidate, while the final 30% is held out for fixed-ephemeris validation.

The blind search does not use the published Kepler-5 b period, epoch, duration,
or depth for preprocessing, period bounds, duration choices, peak selection,
alias checks, or holdout evaluation. Published values are used only afterward in
`published_comparison.json`.

Default BLS settings are broad and generic: periods from about 0.5 to 100 days
with the upper bound limited by the training baseline to require at least three
possible transits, a 5000-sample uniform-frequency period grid, and broad trial
durations from 1 to 12 hours. Broad peaks are discovery locations only. Phase 1A
then performs a local, training-only refinement around each leading period
family using an explicit period-step rule:
`delta_P <= allowed_phase_drift_fraction * duration * period / training_baseline`.
This keeps the accumulated timing drift across the training baseline small
relative to the transit duration before any holdout cadences are evaluated.

The selected candidate is the highest-power locally refined training candidate
after period-neighborhood deduplication. The holdout period, transit time, and
duration are locked from that refined training result; the holdout is not
recentered or retuned. Phase 1A also records small-integer harmonic families and
odd/even depth diagnostics so half-period, double-period, and related aliases are
visible instead of hidden inside a single "best period" field.

A full-mission global BLS search is still written as a stability and alias
diagnostic, but it is not labeled as refinement and it does not replace the
locked training family. The full-mission local refinement is stored separately
and is centered on the locked refined training candidate.

```bash
python -m exoplanet_search.cli --blind-period-search
```

Outputs are written to `data/interim/kepler5_phase1a_search/`:

- `search_summary.json`
- `top_period_candidates.csv`
- `alias_diagnostics.csv`
- `holdout_event_diagnostics.csv`
- `odd_even_diagnostics.csv`
- `harmonic_family_diagnostics.csv`
- `periodogram.png`
- `recovered_folded_light_curve.png`
- `holdout_folded_light_curve.png`
- `published_comparison.json`
- `provenance_manifest.json`

The BLS power and SNR-like values are diagnostic statistics only. Phase 1A does
not perform BATMAN fitting, physical parameter inference, false-alarm
probabilities, or planet validation.

## Phase 1B: deterministic physical transit fit

Phase 1B asks whether the signal independently recovered in Phase 1A can be
represented by a physically valid, exposure-integrated BATMAN transit model. It
uses the Phase 0 Kepler long-cadence Pre-search Data Conditioning Simple
Aperture Photometry (PDCSAP) path with preprocessing mode `none`: the existing
Kepler quality mask, explicit per-product normalization before stitching,
non-finite removal, conservative normalization, and no generic flux-amplitude
clipping.

The Phase 1B starting ephemeris comes from
`data/interim/kepler5_phase1a_search/search_summary.json`, specifically
`full_mission_local_refinement`. The locked training-only Phase 1A ephemeris is
preserved in the Phase 1B provenance so the original predictive recovery remains
auditable.

```bash
python -m exoplanet_search.cli --physical-transit-fit
```

Configurable inputs:

```bash
python -m exoplanet_search.cli \
  --physical-transit-fit \
  --phase1a-summary-path data/interim/kepler5_phase1a_search/search_summary.json \
  --phase1a-provenance-path data/interim/kepler5_phase1a_search/provenance_manifest.json \
  --stellar-inputs-path data/interim/kepler5_phase1b_stellar_inputs.json \
  --phase1b-output-dir data/interim/kepler5_phase1b_fit
```

The stellar input file must contain independently sourced stellar-atmosphere
inputs and reproducible Kepler-band quadratic limb-darkening coefficients or
physical `q1`/`q2` values with source metadata. Phase 1B stops clearly if that
file is absent; it does not substitute arbitrary limb-darkening coefficients.

The fit is unbinned. Transit windows are selected using objective coverage rules
around every predicted Phase 1A full-mission transit center, and every predicted
event receives an audit row whether accepted or excluded. The physical model is
a circular, one-planet, quadratic-limb-darkened BATMAN model with finite
long-cadence exposure integration. Each accepted transit window has its own
multiplicative linear local baseline, solved analytically inside the
deterministic objective rather than pre-normalized and treated as exact.

Outputs are written to `data/interim/kepler5_phase1b_fit/`, including
`phase1b_summary.json`, `phase1b_configuration.json`,
`phase1a_input_record.json`, `observation_product_metadata.csv`,
`transit_window_audit.csv`, `accepted_fit_cadences.csv`,
`event_baseline_parameters.csv`, `multistart_diagnostics.csv`,
`deterministic_fit_parameters.csv`, `timing_refinement_comparison.json`,
`limb_darkening_inputs.json`, `limb_darkening_comparison.json`,
`exposure_integration_convergence.csv`, `stability_diagnostics.csv`,
`residual_summary.json`, `residuals.csv`, `residual_acf.csv`,
`residual_rms_binning.csv`, diagnostic PNG plots, and
`provenance_manifest.json`.

Phase 1B is deterministic and diagnostic. It records multi-start optimizer
behavior, residuals, exposure-integration convergence, fixed versus fitted
limb-darkening results, timing-refinement shifts, and stability checks, but it
does not claim final parameter uncertainties. Posterior sampling and convergence
analysis are deferred to Phase 1B-B.

Published physical planet values for Kepler-5 b are not used for Phase 1B
initialization, bounds, fitting, tests, diagnostics, or comparison in this
branch.
