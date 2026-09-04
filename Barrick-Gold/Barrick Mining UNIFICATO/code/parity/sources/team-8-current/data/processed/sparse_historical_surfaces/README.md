# Sparse historical surfaces

Locally generated historical cross-sections do not necessarily satisfy the dense-date calibration gates. They support coverage analysis and robustness checks but are excluded from the official dense rolling exercise whenever they contain fewer than the required unique observations or expiries.

Sparse dates must never be padded, duplicated, or silently converted into a synthetic 8×8 calibration sample.

Row-level cross-sections are intentionally excluded from Git. The versioned `historical_surface_summary.csv` contains only date-level coverage ranges and counts.

## Versioned file

| File | Role |
| --- | --- |
| `historical_surface_summary.csv` | Reports curve date, spot, row/expiry/strike counts, moneyness/DTE/IV ranges, status, and the corresponding local surface name for each date. |
