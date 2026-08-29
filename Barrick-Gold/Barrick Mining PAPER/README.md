# Barrick Mining PAPER 📄

📄 Article PDFs are published on the Starting Finance Club PoliTo website: https://sfclubpolito.it/pubblicazioni. This folder contains the complete source and final PDF of the compact Research paper.

The paper condenses the integrated Barrick stochastic valuation architecture
into a nine-page, two-column format while retaining the core models, evidence,
limitations and cross-model valuation comparison.

## 👥 Authors

Stefano Falcione, Marco Fracca, Filippo Triassi, Giorgio Zoccatelli, Andrea
Rostagno, Francesco Florio, Giacomo Scali, Jacopo Foralosso, Federico Vesco,
Lorenzo Pietra, Bader Moussaif, Davide Sisto, Matteo Armando, Davide D'Amico,
Pietro Weisz, Salvatore Gabriele Messina and Alessandro Coco.

## 🎯 Purpose

This publication presents the implemented stochastic gold-price models,
operating assumptions, DCF mapping, calibration evidence and interpretation
boundaries in a concise scientific-paper layout.

## 🗂️ Structure

| Path | Role |
| --- | --- |
| `paper/Articolo.tex` | Main LaTeX source. |
| `paper/Articolo.pdf` | Final nine-page paper. |
| `paper/sections/` | Modular article sections. |
| `paper/figures/` and `paper/img/` | Analytical and template assets. |

The executable companion is maintained in
[`../Barrick Mining UNIFICATO/code/`](../Barrick%20Mining%20UNIFICATO/code/).

## ⚙️ Setup

A recent TeX distribution with pdfLaTeX is required.

## ▶️ Build

```bash
cd paper
pdflatex Articolo.tex
pdflatex Articolo.tex
```

The bibliography is frozen in `main.bbl`; `references.bib` remains the source
ledger. Build by-products should not be committed.

## 📊 Outputs

Figure 6 is arranged as a readable 2×2 comparison, with two residual maps per
row. The final PDF includes the cover and References within nine A4 pages.

## 📚 Citation

Barrick Gold Research Teams (2026), *Stochastic Valuation of Barrick Mining*,
Starting Finance Club PoliTo Research.
