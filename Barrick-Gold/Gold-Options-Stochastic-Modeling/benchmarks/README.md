# Pricing benchmarks

`benchmark_pricing.py` compares the frozen scalar Fourier reference with the
vectorised COS batch used during Bates calibration. It also checks that the
stationary Bates--Hawkes proxy reduces to Bates at the same effective jump
intensity.

Run from the project root:

```powershell
python benchmarks/benchmark_pricing.py
```

The benchmark writes no files and prints prices, maximum absolute error,
timings, and speed-up. Results depend on the local machine; numerical tolerances
are enforced separately by the regression tests.
