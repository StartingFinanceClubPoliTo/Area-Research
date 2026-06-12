# Hawkes Calibration Notes

These notes reserve the Hawkes-process discussion for the calibration-focused option-modelling article that already treats Heston and Bates.

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

This repository currently publishes Heston and Bates calibration assets. It does not yet include a fitted Hawkes model, a Hawkes pricing engine, or Hawkes residual diagnostics. Any future implementation should be added only when the article includes reproducible calibration code or clearly labelled conceptual material.
