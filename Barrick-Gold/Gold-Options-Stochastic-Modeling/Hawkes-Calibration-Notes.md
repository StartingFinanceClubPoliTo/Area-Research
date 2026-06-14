# Hawkes Calibration Notes

These notes reserve the Hawkes-process discussion for the calibration-focused option-modelling article that already treats Black-Scholes, Heston, and Bates.

## Role

The current Bates implementation extends Heston by adding compound Poisson jumps with a constant jump intensity. A Hawkes extension would instead let the jump-arrival intensity become self-exciting: after a jump or shock proxy, the model intensity can rise temporarily and then decay back toward its baseline level.

This is a better fit for a calibration discussion than for the Article 6 synthetic examples, because the relevant question is not only how the process is defined, but how its intensity parameters could be estimated and compared against Heston and Bates residuals.

## Calibration Link

A practical calibration workflow would need:

- a jump-event proxy, such as threshold returns or option-implied stress episodes;
- intensity parameters for baseline arrival rate, excitation size, and decay speed;
- a likelihood, filtering, or simulation-based calibration step;
- diagnostics comparing Heston, Bates, and Hawkes-extended residuals on the same implied-volatility surface.

## Current Status

This repository publishes Heston and Bates calibration assets, a Bates-Hawkes
stationary-intensity proxy, **and** a full event-dependent Hawkes
option-pricing engine in `BatesHawkesExact.py`.

The exact engine (`BatesHawkesExact`) prices European options under a
self-exciting jump intensity

```text
dlambda_t = beta (lambda_bar - lambda_t) dt + alpha dN_t
```

through the affine characteristic function of the compensated jump term (a
Riccati-type ODE for `A` and `B`), composed with a Black-Scholes or Heston
diffusion characteristic function and inverted with the same Carr-Madan routine
used elsewhere. It reduces to Bates (constant intensity, `alpha = 0` with
`lambda0 = lambda_bar`) and to Black-Scholes (no jumps), and a Monte-Carlo
simulation (Ogata thinning) reproduces the Fourier price within confidence
intervals.

### Calibrating the exact engine

`ExactHawkesCalibration.calibrate_constvol` in `Hawkes.py` fits the
constant-volatility model `[sigma, lambda_bar, alpha, beta, mu_J, sigma_J]`
with vega weighting,
mirroring the other calibration wrappers. Practical guidance:

- tie the initial intensity to the baseline (`lambda0 = lambda_bar`) in the
  first pass to avoid the `lambda0` / `lambda_bar` identifiability issue;
- bound the branching ratio `alpha / beta` below one (stationarity); the
  objective rejects `alpha >= beta` outright;
- seed `lambda_bar`, `mu_J`, `sigma_J` from the calibrated Bates parameters and
  start `alpha` small (a few tens of percent of `beta`);
- exact pricing is semi-analytic (one ODE solve per Fourier node), so the exact
  calibration is materially heavier than the proxy and should be multi-started
  only once the pricer is confirmed stable on the target surface.

The proxy remains a valid, fast baseline and must continue to be labelled as a
proxy, not a true Hawkes model.
