"""Generate five-year gold-price path diagnostics for the article.

Run from the repository root:

    python calibrations/08_generate_path_simulations.py

By default the script writes to ``Data/``. Use ``--output-dir`` to write the
same assets into an Overleaf ``img/diagnostics`` folder.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm as scipy_norm, poisson as scipy_poisson, qmc, skew

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from BatesHawkesExact import BatesHawkesExact  # noqa: E402
from Hawkes import Hawkes  # noqa: E402
from common import DATA_DIR, load_chebyshev_dataset  # noqa: E402


HESTON_PARAMS = (0.13269, 4.72633, 0.08679, 1.16834, -0.17996)
BATES_PARAMS = (0.07997, 1.71643, 0.04410, 0.71955, 0.22059, 0.89639, -0.19071, 0.10083)


def normal_draws(n_paths, n_factors, seed, method="pseudo", antithetic=False, moment_match=True):
    """Return standard-normal draws with optional Sobol and moment matching."""
    if method == "sobol":
        power = int(math.ceil(math.log2(max(n_paths, 2))))
        sampler = qmc.Sobol(d=n_factors, scramble=True, seed=seed)
        uniforms = sampler.random_base2(power)[:n_paths]
        draws = scipy_norm.ppf(np.clip(uniforms, 1e-10, 1.0 - 1e-10))
    else:
        rng = np.random.default_rng(seed)
        if antithetic:
            half = int(math.ceil(n_paths / 2))
            base = rng.standard_normal((half, n_factors))
            draws = np.vstack([base, -base])[:n_paths]
        else:
            draws = rng.standard_normal((n_paths, n_factors))

    if moment_match:
        means = draws.mean(axis=0)
        stds = np.where(draws.std(axis=0) < 1e-12, 1.0, draws.std(axis=0))
        draws = (draws - means) / stds
    return draws


def uniform_draws(n_paths, n_factors, seed, method="pseudo"):
    """Return uniforms for non-Gaussian blocks such as jump counts."""
    if method == "sobol":
        power = int(math.ceil(math.log2(max(n_paths, 2))))
        sampler = qmc.Sobol(d=n_factors, scramble=True, seed=seed)
        uniforms = sampler.random_base2(power)[:n_paths]
    else:
        rng = np.random.default_rng(seed)
        uniforms = rng.random((n_paths, n_factors))
    return np.clip(uniforms, 1e-10, 1.0 - 1e-10)


def simulate_gbm_paths(s0, rates, sigmas, dt, n_paths, seed, q=0.0):
    rates = np.asarray(rates, dtype=float)
    sigmas = np.asarray(sigmas, dtype=float)
    shocks = normal_draws(n_paths, len(rates), seed=seed, method="sobol", moment_match=True)
    paths = np.empty((n_paths, len(rates) + 1), dtype=float)
    paths[:, 0] = s0
    for step in range(len(rates)):
        drift = (rates[step] - q - 0.5 * sigmas[step] ** 2) * dt
        diffusion = sigmas[step] * math.sqrt(dt) * shocks[:, step]
        paths[:, step + 1] = paths[:, step] * np.exp(drift + diffusion)
    return paths


def simulate_heston_paths(s0, rates, params, dt, n_paths, seed, q=0.0):
    v0, kappa, theta, xi, rho = params
    rates = np.asarray(rates, dtype=float)
    n_steps = len(rates)
    shocks = normal_draws(n_paths, 2 * n_steps, seed=seed, method="sobol", moment_match=True)
    z_v = shocks[:, :n_steps]
    z_ind = shocks[:, n_steps:]
    z_s = rho * z_v + math.sqrt(max(1.0 - rho**2, 0.0)) * z_ind

    prices = np.empty((n_paths, n_steps + 1), dtype=float)
    variances = np.empty((n_paths, n_steps + 1), dtype=float)
    prices[:, 0] = s0
    variances[:, 0] = v0
    for step in range(n_steps):
        v_pos = np.maximum(variances[:, step], 0.0)
        variances[:, step + 1] = np.maximum(
            variances[:, step]
            + kappa * (theta - v_pos) * dt
            + xi * np.sqrt(v_pos) * math.sqrt(dt) * z_v[:, step],
            0.0,
        )
        prices[:, step + 1] = prices[:, step] * np.exp(
            (rates[step] - q - 0.5 * v_pos) * dt + np.sqrt(v_pos) * math.sqrt(dt) * z_s[:, step]
        )
    return prices, variances


def simulate_bates_paths(s0, rates, params, dt, n_paths, seed, q=0.0):
    v0, kappa, theta, xi, rho, lambd, mu_j, sigma_j = params
    rates = np.asarray(rates, dtype=float)
    n_steps = len(rates)
    shocks = normal_draws(n_paths, 2 * n_steps, seed=seed, method="sobol", moment_match=True)
    z_v = shocks[:, :n_steps]
    z_ind = shocks[:, n_steps:]
    z_s = rho * z_v + math.sqrt(max(1.0 - rho**2, 0.0)) * z_ind
    jump_uniforms = uniform_draws(n_paths, n_steps, seed=seed + 101, method="sobol")
    jump_mark_shocks = normal_draws(n_paths, n_steps, seed=seed + 202, method="sobol", moment_match=True)

    prices = np.empty((n_paths, n_steps + 1), dtype=float)
    variances = np.empty((n_paths, n_steps + 1), dtype=float)
    jump_counts = np.zeros((n_paths, n_steps + 1), dtype=int)
    prices[:, 0] = s0
    variances[:, 0] = v0
    jump_compensator = math.exp(mu_j + 0.5 * sigma_j**2) - 1.0

    for step in range(n_steps):
        v_pos = np.maximum(variances[:, step], 0.0)
        variances[:, step + 1] = np.maximum(
            variances[:, step]
            + kappa * (theta - v_pos) * dt
            + xi * np.sqrt(v_pos) * math.sqrt(dt) * z_v[:, step],
            0.0,
        )
        n_jumps = scipy_poisson.ppf(jump_uniforms[:, step], lambd * dt).astype(int)
        jump_counts[:, step + 1] = jump_counts[:, step] + n_jumps
        jump_sizes = n_jumps * mu_j + np.sqrt(n_jumps) * sigma_j * jump_mark_shocks[:, step]
        prices[:, step + 1] = prices[:, step] * np.exp(
            (rates[step] - q - lambd * jump_compensator - 0.5 * v_pos) * dt
            + np.sqrt(v_pos) * math.sqrt(dt) * z_s[:, step]
            + jump_sizes
        )
    return prices, variances, jump_counts


def simulate_exact_hawkes_paths(s0, rates, params, dt, n_paths, seed, q=0.0):
    """Simulate the constant-volatility exact Bates-Hawkes path layer."""
    sigma = params["sigma"]
    lambda_bar = params["lambda_bar"]
    alpha = params["alpha"]
    beta = params["beta"]
    mu_j = params["mu_J"]
    sigma_j = params["sigma_J"]
    rng = np.random.default_rng(seed)
    n_steps = len(rates)
    horizon = n_steps * dt
    times = np.linspace(0.0, horizon, n_steps + 1)
    paths = np.empty((n_paths, n_steps + 1), dtype=float)
    intensities = np.empty((n_paths, n_steps + 1), dtype=float)
    jump_counts = np.zeros((n_paths, n_steps + 1), dtype=int)
    shocks = rng.standard_normal((n_paths, n_steps))
    paths[:, 0] = s0
    kappa_j = BatesHawkesExact.kappa_J(mu_j, sigma_j)

    for path_idx in range(n_paths):
        event_seed = int(rng.integers(1, 2**31 - 1))
        events = Hawkes.simulate_exponential(lambda_bar, alpha, beta, horizon, seed=event_seed)
        marks = rng.normal(mu_j, sigma_j, size=len(events)) if len(events) else np.array([])
        intensities[path_idx] = Hawkes.intensity_on_grid(times, events, lambda_bar, alpha, beta)

        for step in range(n_steps):
            start = times[step]
            end = times[step + 1]
            past = events[events <= start]
            new_mask = (events > start) & (events <= end)
            new_events = events[new_mask]
            integrated_intensity = lambda_bar * dt
            if past.size:
                integrated_intensity += (alpha / beta) * float(
                    np.sum(np.exp(-beta * (start - past)) - np.exp(-beta * (end - past)))
                )
            if new_events.size:
                integrated_intensity += (alpha / beta) * float(
                    np.sum(1.0 - np.exp(-beta * (end - new_events)))
                )
            jump_sum = float(np.sum(marks[new_mask])) if new_events.size else 0.0
            paths[path_idx, step + 1] = paths[path_idx, step] * math.exp(
                (rates[step] - q - 0.5 * sigma**2) * dt
                + sigma * math.sqrt(dt) * shocks[path_idx, step]
                - kappa_j * integrated_intensity
                + jump_sum
            )
            jump_counts[path_idx, step + 1] = jump_counts[path_idx, step] + int(new_events.size)
    return paths, intensities, jump_counts


def terminal_stats(model_name, s0, price_paths, jump_counts=None, status="calibrated"):
    terminal = np.asarray(price_paths, dtype=float)[:, -1]
    log_returns = np.log(np.maximum(terminal, 1e-12) / s0)
    return {
        "model": model_name,
        "terminal_mean": float(np.mean(terminal)),
        "terminal_std": float(np.std(terminal, ddof=1)),
        "p05": float(np.percentile(terminal, 5)),
        "p50": float(np.percentile(terminal, 50)),
        "p95": float(np.percentile(terminal, 95)),
        "logret_skew": float(skew(log_returns)),
        "logret_excess_kurtosis": float(kurtosis(log_returns, fisher=True)),
        "mean_jump_count": 0.0 if jump_counts is None else float(np.mean(jump_counts[:, -1])),
        "calibration_status": status,
        "missing_work": "",
    }


def summarize_price_paths(paths):
    paths = np.asarray(paths, dtype=float)
    return {
        "mean": np.mean(paths, axis=0),
        "p05": np.percentile(paths, 5, axis=0),
        "p50": np.percentile(paths, 50, axis=0),
        "p95": np.percentile(paths, 95, axis=0),
    }


def build_simulation_set(n_paths, n_steps, horizon, seed):
    df_market, s0, q = load_chebyshev_dataset()
    dt = horizon / n_steps
    times = np.linspace(0.0, horizon, n_steps + 1)
    rate = float(np.median(df_market["rate"].to_numpy(dtype=float)))
    sigma_bs = float(np.median(df_market["implied_vol"].to_numpy(dtype=float)))
    rates = np.repeat(rate, n_steps)
    sigmas = np.repeat(sigma_bs, n_steps)
    gbm_paths = simulate_gbm_paths(s0, rates, sigmas, dt, n_paths, seed, q=q)
    heston_paths, heston_var = simulate_heston_paths(s0, rates, HESTON_PARAMS, dt, n_paths, seed + 10, q=q)
    bates_paths, bates_var, bates_jumps = simulate_bates_paths(s0, rates, BATES_PARAMS, dt, n_paths, seed + 20, q=q)
    params = json.loads((DATA_DIR / "hawkes_exact_constvol_params.json").read_text())["parameters"]
    hawkes_paths, hawkes_intensities, hawkes_jumps = simulate_exact_hawkes_paths(
        s0, rates, params, dt, n_paths, seed + 30, q=q
    )
    return {
        "s0": s0,
        "q": q,
        "rate": rate,
        "sigma_bs": sigma_bs,
        "times": times,
        "dt": dt,
        "gbm_paths": gbm_paths,
        "heston_paths": heston_paths,
        "heston_var": heston_var,
        "bates_paths": bates_paths,
        "bates_var": bates_var,
        "bates_jumps": bates_jumps,
        "hawkes_params": params,
        "hawkes_paths": hawkes_paths,
        "hawkes_intensities": hawkes_intensities,
        "hawkes_jumps": hawkes_jumps,
    }


def figure_style():
    plt.rcParams.update(
        {
            "axes.grid": True,
            "grid.alpha": 0.22,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 140,
        }
    )


def save_five_path_panel(times, paths, title, output_path, subtitle):
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    terminal = paths[:, -1]
    order = np.argsort(terminal)
    selected = [order[int(q * (len(order) - 1))] for q in (0.05, 0.25, 0.50, 0.75, 0.95)]
    for label_idx, idx in enumerate(selected, start=1):
        ax.plot(times, paths[idx], linewidth=1.3, label=f"path {label_idx}")
    ax.set_title(title)
    ax.text(0.01, 0.02, subtitle, transform=ax.transAxes, fontsize=8, va="bottom")
    ax.set_xlabel("Years")
    ax.set_ylabel("GLD price")
    ax.legend(ncol=3, loc="upper left", frameon=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_stats_panel(simulation, output_dir):
    times = simulation["times"]
    fig, ax = plt.subplots(figsize=(10.8, 4.8))
    for label, key, color in [
        ("GBM / BS", "gbm_paths", "#1f77b4"),
        ("Heston", "heston_paths", "#2ca02c"),
        ("Bates", "bates_paths", "#d62728"),
        ("Exact Bates-Hawkes", "hawkes_paths", "#6f42c1"),
    ]:
        summary = summarize_price_paths(simulation[key])
        ax.plot(times, summary["mean"], color=color, linewidth=1.8, label=f"{label} mean")
        ax.fill_between(times, summary["p05"], summary["p95"], color=color, alpha=0.10)
    ax.set_title("Simulated gold-price paths: mean and 5-95 percent band")
    ax.set_xlabel("Years")
    ax.set_ylabel("GLD price")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_dir / "gold_path_stats_by_model.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_volatility_intensity_panel(simulation, output_dir):
    times = simulation["times"]
    params = simulation["hawkes_params"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.0))
    ax1.plot(times, np.repeat(simulation["sigma_bs"], len(times)), color="#1f77b4", linewidth=2.0, label="GBM / BS constant vol")
    ax1.plot(times, np.repeat(params["sigma"], len(times)), color="#6f42c1", linestyle="--", linewidth=2.0, label="Exact Hawkes constant vol")
    for idx in range(4):
        ax1.plot(times, np.sqrt(np.maximum(simulation["heston_var"][idx], 0.0)), color="#2ca02c", alpha=0.50)
        ax1.plot(times, np.sqrt(np.maximum(simulation["bates_var"][idx], 0.0)), color="#d62728", alpha=0.40)
    ax1.set_title("Volatility state: constant vs stochastic")
    ax1.set_xlabel("Years")
    ax1.set_ylabel("annualized volatility")
    ax1.legend(loc="upper left")

    ax2.plot(times, np.repeat(BATES_PARAMS[5], len(times)), color="#d62728", linewidth=2.0, label="Bates constant lambda")
    for idx in range(4):
        ax2.plot(times, simulation["hawkes_intensities"][idx], color="#6f42c1", alpha=0.55, linewidth=1.3)
    ax2.axhline(params["lambda_bar"], color="#2f2f2f", linestyle=":", linewidth=1.4, label="Hawkes baseline")
    ax2.set_title("Jump intensity: constant vs exact self-exciting")
    ax2.set_xlabel("Years")
    ax2.set_ylabel("annual jump intensity")
    ax2.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_dir / "volatility_and_jump_intensity_paths.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_outputs(simulation, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    times = simulation["times"]
    s0 = simulation["s0"]
    save_five_path_panel(times, simulation["gbm_paths"], "Black-Scholes / GBM: five gold-price paths", output_dir / "paths_black_scholes_5.png", "constant volatility, continuous lognormal paths")
    save_five_path_panel(times, simulation["heston_paths"], "Heston: five gold-price paths", output_dir / "paths_heston_5.png", "stochastic variance, continuous paths")
    save_five_path_panel(times, simulation["bates_paths"], "Bates: five gold-price paths", output_dir / "paths_bates_5.png", "stochastic variance plus Poisson jumps")
    save_five_path_panel(times, simulation["hawkes_paths"], "Exact Bates-Hawkes: five gold-price paths", output_dir / "paths_bates_hawkes_5.png", "constant-vol exact Hawkes; calibrated branching below 1")
    save_stats_panel(simulation, output_dir)
    save_volatility_intensity_panel(simulation, output_dir)
    rows = [
        terminal_stats("GBM / BS", s0, simulation["gbm_paths"]),
        terminal_stats("Heston", s0, simulation["heston_paths"]),
        terminal_stats("Bates", s0, simulation["bates_paths"], simulation["bates_jumps"]),
        terminal_stats("Exact Bates-Hawkes", s0, simulation["hawkes_paths"], simulation["hawkes_jumps"], "exact constant-vol calibrated"),
    ]
    stats_df = pd.DataFrame(rows)
    stats_df.to_csv(output_dir / "path_simulation_summary.csv", index=False)
    return stats_df


def main():
    parser = argparse.ArgumentParser(description="Generate path simulation diagnostics.")
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--n-paths", type=int, default=4096)
    parser.add_argument("--n-steps", type=int, default=260)
    parser.add_argument("--horizon", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=18062026)
    args = parser.parse_args()

    figure_style()
    simulation = build_simulation_set(args.n_paths, args.n_steps, args.horizon, args.seed)
    stats_df = write_outputs(simulation, args.output_dir)
    print(stats_df.to_string(index=False))
    print(f"[INFO] Wrote diagnostics to {args.output_dir}")


if __name__ == "__main__":
    main()
