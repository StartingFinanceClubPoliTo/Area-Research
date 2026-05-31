"""Generate static article-style figures for HFT Article 2.

By default this writes to output/generated_figures. Pass --output img/2 to
refresh the figures used directly by the LaTeX article.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rough_processes import (
    covariance_error,
    ensure_dir,
    fbm_covariance,
    simulate_fbm_cholesky,
    simulate_markov_lift,
    simulate_volterra,
)


def save_fbm(out: Path, hurst: float) -> None:
    X, t, cov = simulate_fbm_cholesky(300, hurst, 10, seed=100 + int(hurst * 100))
    diff, rmse = covariance_error(X, cov)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for row in X:
        axes[0, 0].plot(t, row, lw=1)
    axes[0, 0].set_title("Sample Paths")
    im = axes[0, 1].imshow(np.cov(X, rowvar=False), origin="lower", aspect="auto")
    fig.colorbar(im, ax=axes[0, 1])
    axes[0, 1].set_title(f"Covariance\nRMSE: {rmse:.2e}")
    axes[1, 0].plot(np.sqrt(np.mean(diff ** 2, axis=0)), color="#e74c3c", lw=2)
    axes[1, 0].set_title("Covariance Error")
    im2 = axes[1, 1].imshow(cov, origin="lower", aspect="auto")
    fig.colorbar(im2, ax=axes[1, 1])
    axes[1, 1].set_title("Theoretical Covariance")
    fig.suptitle(f"Time Steps: 300    Sample Paths: 10    Hurst H: {hurst:.1f}")
    fig.tight_layout()
    fig.savefig(out / f"fgm{hurst:.1f}.png", dpi=160)
    plt.close(fig)


def save_volterra(out: Path, hurst: float) -> None:
    X, W, t, K = simulate_volterra(300, hurst, 10, seed=200 + int(hurst * 100))
    cov_target = fbm_covariance(t, hurst)
    fig, axes = plt.subplots(3, 2, figsize=(10, 12))
    for row in X:
        axes[0, 0].plot(t, row, lw=1)
    axes[0, 0].set_title("Sample Paths")
    im = axes[0, 1].imshow(K, origin="lower", aspect="auto")
    fig.colorbar(im, ax=axes[0, 1])
    axes[0, 1].set_title("Volterra Kernel")
    for row in W[:3]:
        axes[1, 0].plot(t, row, lw=1)
    axes[1, 0].set_title("Gaussian Noise")
    for row in np.diff(W[:3], axis=1):
        axes[1, 1].plot(t[1:], row, lw=0.8)
    axes[1, 1].set_title("Gaussian Noise")
    for frac in (0.25, 0.50, 0.75):
        idx = int(frac * 299)
        axes[2, 0].plot(t[:idx], K[idx, :idx], lw=2, label=f"t={t[idx]:.2f}")
    axes[2, 0].legend()
    axes[2, 0].set_title("Causal Structure")
    im2 = axes[2, 1].imshow(cov_target, origin="lower", aspect="auto")
    fig.colorbar(im2, ax=axes[2, 1])
    axes[2, 1].set_title("Covariance vs fBM")
    fig.suptitle(f"Time Steps: 300    Sample Paths: 10    Hurst H: {hurst:.1f}")
    fig.tight_layout()
    fig.savefig(out / f"volterra{hurst:.1f}.png", dpi=160)
    plt.close(fig)


def save_iterative(out: Path, hurst: float) -> None:
    X, _, t, _ = simulate_volterra(300, hurst, 3, seed=222 + int(hurst * 100))
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, window in zip(axes.ravel(), (0.25, 0.50, 0.75, 1.00)):
        stop = int(window * (len(t) - 1))
        for i, row in enumerate(X):
            ax.plot(t[:stop], row[:stop], lw=1.5, label=f"Path {i+1}")
        ax.axvline(t[stop], color="red", ls="--", lw=1)
        ax.set_title(f"Filtration Window s: {window:g}")
        ax.set_xlabel("t")
        ax.set_ylabel("X(t)")
    fig.tight_layout()
    fig.savefig(out / f"iterative{hurst:.1f}.png", dpi=160)
    plt.close(fig)


def save_lift(out: Path, hurst: float) -> None:
    Y, t, _, _ = simulate_markov_lift(300, hurst, 25, 10, seed=300 + int(hurst * 100))
    target = fbm_covariance(t, hurst)
    emp = np.cov(Y, rowvar=False)
    diff = emp - target
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for row in Y[:10]:
        axes[0, 0].plot(t, row, lw=1)
    axes[0, 0].set_title("Sample Paths")
    axes[0, 1].plot(np.sqrt(np.mean(diff ** 2, axis=0)), color="#e74c3c", lw=2)
    axes[0, 1].set_title("Covariance Error")
    im = axes[1, 0].imshow(target, origin="lower", aspect="auto")
    fig.colorbar(im, ax=axes[1, 0])
    axes[1, 0].set_title("Theoretical Covariance")
    im2 = axes[1, 1].imshow(emp, origin="lower", aspect="auto")
    fig.colorbar(im2, ax=axes[1, 1])
    axes[1, 1].set_title(f"Covariance\nRMSE: {rmse:.2e}")
    fig.suptitle(f"Time Steps: 300    Sample Paths: 25    Hurst H: {hurst:.1f}    OU factors M: 10")
    fig.tight_layout()
    fig.savefig(out / f"ML{hurst:.1f}.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="output/generated_figures")
    args = parser.parse_args()
    out = ensure_dir(args.output)
    for h in (0.2, 0.5, 0.8):
        save_fbm(out, h)
        save_volterra(out, h)
        save_lift(out, h)
        save_iterative(out, h)


if __name__ == "__main__":
    main()
