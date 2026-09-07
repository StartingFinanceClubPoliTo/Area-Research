# Barrick Mining PAPER 📄

[Operating-Value Sensitivity for Barrick Mining](paper/Articolo.pdf) is an eight-page working draft dated September 7, 2026. It compares option-implied gold-price laws within an unreconciled operating-proxy experiment.

## Current evidence

- Snapshot: September 2, 2026; 64 actual GLD call contracts, with curated Team 8 option and Treasury inputs included.
- Current-snapshot IV RMSE after local refinement: Heston 49.4881 bp, Bates 49.4077 bp, Bates–Hawkes 49.3258 bp. These are local improvements, not certified global optima.
- Historical conditional next-date repricing: 30 origins, 4,667 common forecasts; Hawkes has the lowest mean daily RMSE point estimate, 65.0507 bp. Root-mean-MSE is a different metric (70.3008 bp). No model beats persistence on its common support.
- Signed aggregate operating-proxy medians are approximately USD 59–60 billion. Numerical changes with path count, time resolution and seed exceed some inter-model differences. No equity values or market-relative verdicts are reported.

## Files and execution

- [paper/Articolo.pdf](paper/Articolo.pdf): eight-page Paper; LaTeX source and figures in the same directory.
- [code/README.md](code/README.md): standalone offline experiment, data provenance, parameters, tests and output tables.
- [Technical audit](code/technical_audit.pdf): all 27 review points, exact OOS denominators, parameter table and remaining limitations.
- [Full thesis draft](../Barrick%20Mining%20UNIFICATO/thesis/Articolo.pdf): broader historical project; not scientifically updated by this Paper revision.

```sh
python -m pip install -r code/requirements.txt
python main.py
python -m pytest code/tests -q
```

Use `python main.py --recalibrate` to repeat the Hawkes local refinement. Build the Paper by running `pdflatex Articolo.tex` twice from `paper/`.

## Interpretation

Q-law plus assumed WACC is a sensitivity operator. Sales versus production, corporate accounting, copper, ownership, finite reserves and risk-premium reconciliation remain unresolved. Signed terminal values and a five-year truncation comparison make the terminal convention explicit; neither is a calibrated mine-life model. Historical OOS is retained separately from the current refit.

## Authors

Stefano Falcione, Marco Fracca, Filippo Triassi, Giorgio Zoccatelli, Andrea Rostagno, Francesco Florio, Giacomo Scali, Jacopo Foralosso, Federico Vesco, Lorenzo Pietra, Bader Moussaif, Davide Sisto, Matteo Armando, Davide D'Amico, Pietro Weisz, Salvatore Gabriele Messina and Alessandro Coco.

## Citation

Barrick Gold Research Teams (2026), *Operating-Value Sensitivity for Barrick Mining*, Starting Finance Club PoliTo Research, working draft, September 7, 2026.
