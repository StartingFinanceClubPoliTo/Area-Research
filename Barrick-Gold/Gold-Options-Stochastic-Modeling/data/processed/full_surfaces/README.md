# Dense option surfaces

Locally generated dated GLD option surfaces are the primary inputs for structured sampling and model calibration. A date is admitted to the official dense rolling exercise only when it satisfies the configured quality gates, including DTE, uniqueness, row-count, and expiry coverage.

Use `src/audit_market_data.py` before calibration. Duplicate representations of a date are resolved by selecting the richest valid surface after the official filters, and the chosen path must be recorded in the run manifest.

The surface CSV files are intentionally excluded from Git because they contain row-level market observations and contract identifiers. Only this documentation and the public Treasury curve-fit metadata used by the final dense-date exercise are versioned.
