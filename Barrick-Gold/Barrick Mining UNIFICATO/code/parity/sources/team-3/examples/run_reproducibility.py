"""Generate compact reproducibility outputs for the Monte Carlo risk article."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from monte_carlo_risk.option_pricing import (  # noqa: E402
    black_scholes_call,
    monte_carlo_european_call,
    price_asian_call_mc,
    price_asian_call_qmc,
)
from monte_carlo_risk.random_generators import (  # noqa: E402
    box_muller_from_uniforms,
    chi_square_uniform_test,
    lcg_uniform,
    qq_data_standard_normal,
)
from monte_carlo_risk.returns import (  # noqa: E402
    conditional_value_at_risk,
    simulate_gaussian_mixture_returns,
    simulate_terminal_wealth,
    value_at_risk,
)
from monte_carlo_risk.variance_reduction import (  # noqa: E402
    antithetic_variates,
    importance_sampling_normal_shift,
)


FIGURES = ROOT / "figures"
OUTPUTS = ROOT / "outputs"


def ensure_output_folders() -> None:
    FIGURES.mkdir(exist_ok=True)
    OUTPUTS.mkdir(exist_ok=True)


def plot_lcg_diagnostics() -> dict[str, float]:
    uniforms = lcg_uniform(10_000, seed=42)
    statistic, dof, p_value = chi_square_uniform_test(uniforms, bins=30)

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.hist(uniforms, bins=30, density=True, edgecolor="black")
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1.2)
    ax.set_title("LCG Uniform Diagnostics")
    ax.set_xlabel("u")
    ax.set_ylabel("density")
    fig.tight_layout()
    fig.savefig(FIGURES / "lcg_uniform_histogram.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    z1, z2 = box_muller_from_uniforms(uniforms[:5_000], uniforms[5_000:])
    theoretical, empirical, rmse = qq_data_standard_normal(np.r_[z1, z2])
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.scatter(theoretical, empirical, s=10, alpha=0.6)
    ax.plot(theoretical, theoretical, color="red", linestyle="--", linewidth=1.1)
    ax.set_title("Box-Muller Gaussian QQ Plot")
    ax.set_xlabel("theoretical quantiles")
    ax.set_ylabel("empirical quantiles")
    fig.tight_layout()
    fig.savefig(FIGURES / "box_muller_qq_plot.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {
        "lcg_chi_square": statistic,
        "lcg_chi_square_dof": float(dof),
        "lcg_chi_square_p_value": p_value,
        "box_muller_qq_rmse": rmse,
    }


def plot_call_convergence() -> dict[str, float]:
    s0, strike, rate, volatility, maturity = 100.0, 100.0, 0.05, 0.20, 1.0
    analytic = black_scholes_call(s0, strike, rate, volatility, maturity)
    n_grid = np.unique(np.round(np.logspace(2, 5, 16)).astype(int))
    estimates = []
    standard_errors = []
    for n_paths in n_grid:
        price, se = monte_carlo_european_call(s0, strike, rate, volatility, maturity, int(n_paths), seed=100 + int(n_paths))
        estimates.append(price)
        standard_errors.append(se)
    estimates = np.asarray(estimates)
    standard_errors = np.asarray(standard_errors)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(n_grid, estimates, marker="o", label="Monte Carlo")
    ax.fill_between(n_grid, estimates - 1.96 * standard_errors, estimates + 1.96 * standard_errors, alpha=0.18)
    ax.axhline(analytic, color="black", linestyle="--", linewidth=1.1, label="Black-Scholes")
    ax.set_xscale("log")
    ax.set_title("European Call Price Convergence")
    ax.set_xlabel("paths")
    ax.set_ylabel("price")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "european_call_convergence.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {
        "black_scholes_call": analytic,
        "mc_call_final": float(estimates[-1]),
        "mc_call_final_standard_error": float(standard_errors[-1]),
        "mc_call_final_absolute_error": float(abs(estimates[-1] - analytic)),
    }


def compare_asian_pricing() -> dict[str, float]:
    s0, strike, rate, volatility, maturity, n_steps = 100.0, 100.0, 0.04, 0.22, 1.0, 32
    mc_price, mc_se = price_asian_call_mc(s0, strike, rate, volatility, maturity, n_steps, 4_096, seed=7)
    qmc_price, qmc_se = price_asian_call_qmc(s0, strike, rate, volatility, maturity, n_steps, n_paths_power=12, seed=7)

    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    labels = ["MC", "Sobol QMC"]
    prices = [mc_price, qmc_price]
    errors = [1.96 * mc_se, 1.96 * qmc_se]
    ax.bar(labels, prices, yerr=errors, capsize=4, color=["#4C78A8", "#F58518"])
    ax.set_title("Asian Call Pricing Comparison")
    ax.set_ylabel("price")
    fig.tight_layout()
    fig.savefig(FIGURES / "asian_call_mc_vs_qmc.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {
        "asian_mc_price": mc_price,
        "asian_mc_standard_error": mc_se,
        "asian_qmc_price": qmc_price,
        "asian_qmc_standard_error": qmc_se,
    }


def run_variance_reduction() -> dict[str, float]:
    s0, strike, rate, volatility, maturity = 100.0, 120.0, 0.05, 0.20, 1.0

    def call_payoff(z: np.ndarray) -> np.ndarray:
        terminal = s0 * np.exp((rate - 0.5 * volatility**2) * maturity + volatility * np.sqrt(maturity) * z)
        return np.exp(-rate * maturity) * np.maximum(terminal - strike, 0.0)

    anti = antithetic_variates(call_payoff, n_paths=20_000, seed=11)

    def digital_payoff(z: np.ndarray) -> np.ndarray:
        terminal = s0 * np.exp((rate - 0.5 * volatility**2) * maturity + volatility * np.sqrt(maturity) * z)
        return np.exp(-rate * maturity) * (terminal > 140.0)

    importance = importance_sampling_normal_shift(digital_payoff, n_paths=30_000, shift=1.0, seed=12)
    return {
        "antithetic_call_estimate": anti.estimate,
        "antithetic_variance_ratio": anti.variance_reduced / anti.variance_plain,
        "importance_plain_estimate": importance.estimate_plain,
        "importance_shifted_estimate": importance.estimate_importance,
        "importance_variance_ratio": importance.variance_importance / importance.variance_plain,
    }


def run_return_risk_examples() -> dict[str, float]:
    returns, moments = simulate_gaussian_mixture_returns(20_000, seed=21)
    shifted_returns = 0.65 * returns + 0.35 * np.random.default_rng(22).normal(0.0002, 0.014, size=returns.size)
    terminal = simulate_terminal_wealth(
        returns,
        shifted_returns,
        weights=np.array([0.55, 0.45]),
        dependence=0.45,
        initial_value=100.0,
        n_paths=2_000,
        n_days=252,
        seed=23,
    )

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.hist(terminal, bins=45, edgecolor="black", alpha=0.82)
    ax.axvline(np.percentile(terminal, 5), color="red", linestyle="--", linewidth=1.2, label="5% quantile")
    ax.set_title("Terminal Wealth Distribution")
    ax.set_xlabel("terminal value")
    ax.set_ylabel("frequency")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "terminal_wealth_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {
        "mixture_mean": moments["mean"],
        "mixture_variance": moments["variance"],
        "mixture_skewness": moments["skewness"],
        "mixture_excess_kurtosis": moments["excess_kurtosis"],
        "terminal_wealth_var_5": value_at_risk(terminal, alpha=0.05),
        "terminal_wealth_cvar_5": conditional_value_at_risk(terminal, alpha=0.05),
        "terminal_wealth_loss_probability": float((terminal < 100.0).mean()),
    }


def write_summary(summary: dict[str, float]) -> None:
    with (OUTPUTS / "monte_carlo_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for key in sorted(summary):
            writer.writerow([key, f"{summary[key]:.10g}"])


def main() -> None:
    ensure_output_folders()
    summary = {}
    summary.update(plot_lcg_diagnostics())
    summary.update(plot_call_convergence())
    summary.update(compare_asian_pricing())
    summary.update(run_variance_reduction())
    summary.update(run_return_risk_examples())
    write_summary(summary)
    print(f"Wrote figures to {FIGURES}")
    print(f"Wrote summary table to {OUTPUTS / 'monte_carlo_summary.csv'}")


if __name__ == "__main__":
    main()
