# Hawkes Calibration Workflow

## Architecture

Hawkes point-process calibration follows the same repository convention as the
other models:

- `Hawkes.py` contains all reusable model, likelihood, simulation, and
  calibration logic.
- `Hawkes Calibration.ipynb` imports those classes and is limited to dataset
  preparation, calibration calls, tables, and plots.
- No separate Hawkes package or model-specific folder is required.

The notebook must not duplicate likelihood or kernel formulas. Changes to the
models belong in `Hawkes.py` and must be covered by tests.

## Input Contract

Both point-process calibrators accept a one-dimensional sequence of distinct event times and
an observation horizon in the same time unit. For example, if event times are
measured in trading days, all decay and cutoff parameters are also expressed in
trading-day units.

The event proxy must be fixed before calibration. Typical choices are absolute
returns above a documented threshold, negative returns below a documented
quantile, or externally identified stress events. Do not choose the threshold
separately for each kernel, because that invalidates model comparison.

## Models

### Exponential kernel

Use `ExponentialHawkesCalibration` for

```text
phi(t) = alpha exp(-beta t)
```

with branching ratio `alpha / beta < 1`. This is the Markovian specification
used by the affine Bates-Hawkes option-pricing engine.

### Rough power-law kernel

Use `RoughHawkesCalibration` for

```text
phi(t) = alpha / (t + cutoff)^(1 + tail_index)
```

with `0 < tail_index < 1` and branching ratio
`alpha / (tail_index * cutoff^tail_index) < 1`. The cutoff regularizes the
kernel at the origin while the power-law tail captures persistent excitation.

The rough point-process fit is a diagnostic model. It cannot be inserted into
`BatesHawkesExact.py` without replacing the current finite-dimensional affine
pricing construction.

### Exact affine option calibration

Use `ExactHawkesCalibration` for option-surface objectives and optimization.
The class is defined in `Hawkes.py` and calls the pricing-only engine in
`BatesHawkesExact.py`. It exposes `calibrate_constvol` for the constant-volatility
benchmark and `calibrate_heston` for the full affine Heston-Hawkes model.

## Calibration Sequence

1. Build and document one event-time series.
2. Express the event times and horizon in a consistent unit.
3. Fit both classes with `fit(event_times, horizon)`.
4. Check optimizer success and the estimated branching ratio.
5. Compare log likelihood, AIC, and BIC from `HawkesCalibrationResult`.
6. Inspect fitted intensities and kernel decay in the notebook.
7. Inspect time-rescaling residuals against an `Exp(1)` distribution.
8. Run sensitivity checks for the event threshold and observation window.

A lower AIC/BIC is not sufficient on its own. Reject a specification if the
residual diagnostics show systematic clustering or if estimates sit on a
parameter boundary.

## Minimal API

```python
from Hawkes import (
    ExactHawkesCalibration,
    ExponentialHawkesCalibration,
    RoughHawkesCalibration,
)

exp_fit = ExponentialHawkesCalibration.fit(event_times, horizon)
rough_fit = RoughHawkesCalibration.fit(event_times, horizon)

print(exp_fit.params, exp_fit.aic, exp_fit.bic)
print(rough_fit.params, rough_fit.aic, rough_fit.bic)

# Option-surface calibration, using the pricing engine internally.
exact_fit = ExactHawkesCalibration.calibrate_constvol(option_data, spot)
```

Use `result.as_dict()` when a calibration result must be written to JSON.
