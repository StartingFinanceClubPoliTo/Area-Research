"""Deterministically rebuild the modifiable Bates/Hawkes notebooks."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTECTED = {
    "Black and Scholes Calibration.ipynb",
    "Heston Calibration.ipynb",
}


def markdown(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().splitlines()],
    }


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.strip().splitlines()],
    }


def notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


BATES = notebook(
    [
        markdown("""
# Bates calibration

Reproducible calibration of the eight-parameter Bates model on the local-only
current LSE GLD Chebyshev sample. Pricing is maturity-batched with COS; scalar Fourier
inversion remains the regression reference.
"""),
        code("""
import json
from pathlib import Path

from Bates import Bates
from calibration_workflow import (
    load_calibration_surface, option_diagnostics, plot_residuals,
    plot_smiles, price_surface,
)

DATA = Path("Data/lse_local")
SEED = 20260811
DIVIDEND_YIELD = 0.0
surface, spot = load_calibration_surface(DATA, DIVIDEND_YIELD)
surface.head()
"""),
        code("""
report = Bates.calibrate_bates(
    surface,
    spot,
    q=DIVIDEND_YIELD,
    seed=SEED,
    pricing="cos",
    return_report=True,
)
report.as_dict()
"""),
        code("""
parameters = report.x
model_prices = price_surface(
    surface,
    lambda strikes, maturity, rate: Bates.bates_prices_cos(
        spot, strikes, maturity, *parameters, rate, DIVIDEND_YIELD
    ),
)
diagnostics = option_diagnostics(
    surface, spot, model_prices, DIVIDEND_YIELD
)
diagnostics.to_csv(DATA / "bates_option_diagnostics.csv", index=False)
(DATA / "bates_calibration_report.json").write_text(
    json.dumps(report.as_dict(), indent=2), encoding="utf-8"
)
diagnostics.head()
"""),
        code("""
plot_smiles(diagnostics, "Bates", DATA / "bates_volatility_smile.png")
plot_residuals(diagnostics, "Bates", DATA / "bates_residual_heatmap.png")
"""),
    ]
)


PROXY = notebook(
    [
        markdown("""
# Bates--Hawkes stationary proxy calibration

This notebook calibrates the documented stationary-intensity proxy on the
current local-only LSE sample. It is a
benchmark distinct from the exact event-dependent affine model.
"""),
        code("""
import json
from pathlib import Path

from BatesHawkes import BatesHawkes
from calibration_workflow import (
    load_calibration_surface, option_diagnostics, plot_residuals,
    plot_smiles, price_surface,
)

DATA = Path("Data/lse_local")
SEED = 20260811
DIVIDEND_YIELD = 0.0
surface, spot = load_calibration_surface(DATA, DIVIDEND_YIELD)
surface.head()
"""),
        code("""
report = BatesHawkes.calibrate_bates_hawkes_proxy(
    surface,
    spot,
    q=DIVIDEND_YIELD,
    seed=SEED,
    pricing="cos",
    return_report=True,
)
report.as_dict()
"""),
        code("""
parameters = report.x
model_prices = price_surface(
    surface,
    lambda strikes, maturity, rate: BatesHawkes.prices_proxy_cos(
        spot, strikes, maturity, *parameters, rate, DIVIDEND_YIELD
    ),
)
diagnostics = option_diagnostics(
    surface, spot, model_prices, DIVIDEND_YIELD
)
diagnostics.to_csv(DATA / "bates_hawkes_proxy_diagnostics.csv", index=False)
(DATA / "bates_hawkes_proxy_report.json").write_text(
    json.dumps(report.as_dict(), indent=2), encoding="utf-8"
)
diagnostics.head()
"""),
        code("""
plot_smiles(
    diagnostics,
    "Bates--Hawkes proxy",
    DATA / "bates_hawkes_proxy_volatility_smile.png",
)
plot_residuals(
    diagnostics,
    "Bates--Hawkes proxy",
    DATA / "bates_hawkes_proxy_residual_heatmap.png",
)
"""),
    ]
)


HAWKES = notebook(
    [
        markdown("""
# Hawkes calibration and exact option-model entry point

The first section validates exponential and rough Hawkes event calibration on
a deterministic synthetic sample. The optional second section runs the exact
affine option calibration on the immutable GLD surface.
"""),
        code("""
import numpy as np
import pandas as pd

from Hawkes import (
    ExactHawkesCalibration, ExponentialHawkesCalibration,
    RoughHawkesCalibration,
)
from calibration_workflow import load_calibration_surface

SEED = 20260811
horizon = 250.0
tail_index, cutoff, branching = 0.45, 0.25, 0.55
alpha = branching * tail_index * cutoff ** tail_index
events = RoughHawkesCalibration.simulate(
    lambda0=0.35,
    alpha=alpha,
    tail_index=tail_index,
    cutoff=cutoff,
    horizon=horizon,
    seed=SEED,
)
"""),
        code("""
exponential_fit = ExponentialHawkesCalibration.fit(events, horizon)
rough_fit = RoughHawkesCalibration.fit(events, horizon)
pd.DataFrame([
    {"model": exponential_fit.model, "success": exponential_fit.success,
     "aic": exponential_fit.aic, "bic": exponential_fit.bic,
     **exponential_fit.params},
    {"model": rough_fit.model, "success": rough_fit.success,
     "aic": rough_fit.aic, "bic": rough_fit.bic,
     **rough_fit.params},
])
"""),
        code("""
RUN_EXACT_OPTION_CALIBRATION = False

if RUN_EXACT_OPTION_CALIBRATION:
    surface, spot = load_calibration_surface("Data/lse_local")
    exact_result = ExactHawkesCalibration.calibrate_heston(
        surface,
        spot,
        maxiter=25,
        popsize=6,
        seed=SEED,
    )
    exact_parameters = ExactHawkesCalibration.unpack_heston_params(exact_result.x)
    display(exact_parameters)
"""),
    ]
)


def main():
    targets = {
        "Bates Calibration.ipynb": BATES,
        "Bates-Hawkes Proxy Calibration.ipynb": PROXY,
        "Hawkes Calibration.ipynb": HAWKES,
    }
    if PROTECTED.intersection(targets):
        raise RuntimeError("Protected notebooks must never be rebuilt.")
    for name, payload in targets.items():
        (ROOT / name).write_text(
            json.dumps(payload, indent=1, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"rebuilt {name}")


if __name__ == "__main__":
    main()
