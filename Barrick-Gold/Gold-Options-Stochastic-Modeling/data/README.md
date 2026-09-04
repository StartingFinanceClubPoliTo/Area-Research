# Data assets 🗃️

This folder separates redistributable source inputs and publication-safe aggregates from local-only processed research surfaces.

- `raw/` contains manually supplied public inputs.
- `processed/full_surfaces/` is the local destination for dense option surfaces; Git tracks only its documentation and public Treasury curve-fit metadata.
- `processed/sparse_historical_surfaces/` is the local destination for lower-coverage dates; Git tracks only an aggregate coverage summary.

Do not commit credentials, account metadata, unrestricted API dumps, contract identifiers, or row-level market records. Every calibration should identify its exact local source surface and curve date in a run manifest.
