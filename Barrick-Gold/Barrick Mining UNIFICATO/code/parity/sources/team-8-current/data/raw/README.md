# Raw public inputs

This folder contains small, redistributable inputs supplied independently of the executable pipeline.

`daily-treasury-rates2026.csv` is the source table used to construct dated Treasury curves. Preserve the original file and document any transformation in the processed output or run manifest. Account-linked option records and API credentials must never be stored here or committed.

## Versioned file

| File | Role |
| --- | --- |
| `daily-treasury-rates2026.csv` | U.S. Treasury daily par-yield observations supplied for the 2026 curve history. |
