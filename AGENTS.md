# AGENTS.md

## Project goal
This repository is for validating an exoplanet transit-search and transit-fitting workflow by rediscovering known exoplanets in Kepler data.

## Scope
Focus on:
- downloading and organizing Kepler light-curve data
- cleaning and detrending light curves
- identifying known transit signals
- fitting known planetary transits with standard astronomy tools
- building clean, testable Python modules

Do not work on exomoon modeling yet unless explicitly asked.

## Code rules
- Use Python only.
- Put production code in `src/exoplanet_search/`.
- Keep notebooks exploratory only, not as the main implementation.
- Prefer small, modular functions over large scripts.
- Add or update tests when behavior changes.
- Do not rewrite unrelated files.
- Keep changes minimal and easy to review.

## Science/tooling preferences
- Prefer established libraries before writing custom implementations.
- Use `lightkurve`, `astropy`, and `batman-package` where appropriate.
- Keep units and time systems explicit.
- Be cautious with detrending so real transit signals are not removed.
- Keep the default preprocessing transit-preserving: remove non-finite values,
  use a documented Kepler quality-mask policy, normalize conservatively, and do
  not apply generic flux-amplitude clipping by default.
- For blind period-search work, keep generic search and holdout code independent
  of published Kepler-5 b ephemeris values; use literature values only in
  clearly separated post-search diagnostics.
- Record assumptions clearly in docstrings or comments.

## Data handling
- Do not commit downloaded Kepler data or large generated outputs.
- Treat `data/raw/` as immutable input storage.
- Put derived products in `data/interim/` or `data/processed/`.

## Validation order
1. Confirm the package installs.
2. Confirm tests run.
3. Confirm a known Kepler target can be downloaded.
4. Confirm a known planet transit can be recovered.
5. Only then extend the pipeline.
