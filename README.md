# Southwest China vortex climatology

Research code for joint analyses of Southwest China vortex (SWCV) frequency,
intensity and tracks, with a focus on winter local genesis and its diurnal
environment. CMIP6/DAMIP scripts examine monthly background circulation,
not direct identification of individual model vortices.

This repository is a sanitized, source-only research snapshot. It is not a
published paper, an operational forecasting product or a packaged application.

## Contents

- `scripts/`: catalogue, environmental-field and model analyses and acquisition tools.
- `docs/`: selected scientific methods and diagnostic records (historical checkpoints).
- `refs/`: bibliographic records, not third-party full-text PDFs.
- `data/zenodo_record_15073815.json`: minimal public catalogue download metadata.
- `tests/test_public_smoke.py`: small synthetic tests requiring no external data.

No raw meteorological data, private correspondence, manuscripts, reviewer
packages, presentation files, workstation logs, credentials or original Git
history are included. Some historical method notes mention local output files
that must be regenerated. Historical notes are not a current completion claim.

## Setup

Use Python 3.13 and a virtual environment. The dependency snapshot records the
research environment; large-scale reproduction also requires substantial
storage, computing time and access to the source data services.

```sh
python -m venv .venv
# Activate .venv using your shell's normal activation command.
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

## Starting with the public track catalogue

```sh
python scripts/download_zenodo_tracks.py
python scripts/analyze_tracks.py
```

The downloader uses the archived public release metadata and checks file MD5
checksums. Analyses write local products to `data/processed/` and `figures/`.
Subsequent ERA5/CMIP6 stages depend on these generated products and additional
inputs; scripts are research stages rather than a single automatic pipeline.
Review the selected method notes before running a stage.

The public snapshot includes a project-local, reduced plot-rectangle QA helper
instead of depending on private desktop tools. It checks subplot geometry and
panel labels, not text collisions or the full original layout audit. Numerical
analysis code and source-value checks are unchanged by this replacement.

## Data access and credentials

- Track catalogue: He & Cao, Zenodo, doi:10.5281/zenodo.15073815.
- ERA5: Copernicus Climate Data Store; configure CDS credentials locally and
  accept the relevant dataset terms in your own account.
- Climate indices: official NOAA/NASA sources used by the acquisition scripts.
- CMIP6/DAMIP: ESGF and the public access endpoints selected by the scripts.
- CPC daily precipitation: NOAA PSL.

Keep `.cdsapirc`, tokens and `.env` files outside version control. Never place
credentials in Python code, Git remotes, command-line examples or issue reports.
Original data remain subject to provider licenses and acknowledgements.

## Interpretation

The principal analysis period is 1979–2023. December is assigned to the following
DJF season year. Winter night genesis uses Beijing time, 20:00–08:00.
The public CNN/ERA5 catalogue is a third-party product; clustering and subsequent
statistics are analysis steps. Circulation association does not itself identify
the cause of long-term change. Numerical claims should be regenerated from the
documented inputs, rather than inferred from a slide or a historical checkpoint.

## Rights and publication status

No blanket software license has been selected for this snapshot. Public hosting
does not waive copyright. Refer to each original provider's terms for data and
references; contact the repository owner about reuse permissions for project code.
Author metadata, funding and manuscript publication details are intentionally
omitted from this technical snapshot.
