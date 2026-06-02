import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.fft import fft

from config import DATA_PROCESSED_DIR, OUTPUT_DIR
from utils import save_csv, save_figure, write_text_file

def gph_long_memory_estimate(x, m=None):
    """
    Geweke-Porter-Hudak (GPH) estimator of the fractional differencing
    parameter d for a covariance-stationary long-memory process.
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    x = x - np.mean(x)
    T = len(x)

    if m is None:
        m = int(np.floor(T ** 0.5))

    if m <= 2:
        raise ValueError("m is too small for GPH estimation.")

    freqs = 2 * np.pi * np.arange(1, m + 1) / T
    fft_vals = fft(x)
    periodogram = (1 / (2 * np.pi * T)) * np.abs(fft_vals) ** 2
    I_lambda = periodogram[1:m + 1]

    y = np.log(I_lambda)
    X = -2 * np.log(2 * np.sin(freqs / 2))

    X_mat = np.column_stack((np.ones(m), X))
    beta = np.linalg.lstsq(X_mat, y, rcond=None)[0]
    residuals = y - X_mat @ beta
    sigma2 = np.sum(residuals ** 2) / (m - 2)
    XtX_inv = np.linalg.inv(X_mat.T @ X_mat)
    var_beta = sigma2 * XtX_inv

    d_hat = beta[1]
    se_d = np.sqrt(var_beta[1, 1])

    return d_hat, se_d


def compute_abs_returns(returns: pd.Series) -> pd.Series:
    return returns.abs()


def plot_long_acf_two_panel(
    series_gold: pd.Series,
    series_silver: pd.Series,
    max_lag: int,
    filename: str,
    ylim=(0, 0.30),
):
    """
    Plot ACF of gold and silver absolute returns in a single 2x1 figure.
    Lag 0 is excluded to avoid the trivial ACF=1 dominating the scale.
    """
    from statsmodels.tsa.stattools import acf

    gold_clean = pd.Series(series_gold).dropna()
    silver_clean = pd.Series(series_silver).dropna()

    acf_gold = acf(gold_clean, nlags=max_lag, fft=True)
    acf_silver = acf(silver_clean, nlags=max_lag, fft=True)

    lags = np.arange(1, max_lag + 1)
    acf_gold = acf_gold[1:max_lag + 1]
    acf_silver = acf_silver[1:max_lag + 1]

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # Gold
    axes[0].vlines(lags, 0, acf_gold, color="tab:orange", linewidth=1.2)
    axes[0].scatter(lags, acf_gold, color="tab:orange", s=18)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_ylabel("ACF")
    axes[0].set_title("ACF of gold absolute returns (lags 1–100)")
    axes[0].set_ylim(*ylim)

    # Silver
    axes[1].vlines(lags, 0, acf_silver, color="tab:green", linewidth=1.2)
    axes[1].scatter(lags, acf_silver, color="tab:green", s=18)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Lag")
    axes[1].set_ylabel("ACF")
    axes[1].set_title("ACF of silver absolute returns (lags 1–100)")
    axes[1].set_ylim(*ylim)

    fig.tight_layout()
    save_figure(fig, filename)


def fractional_difference(series, d, thresh=1e-4):
    """
    Approximate fractional differencing weights and apply to a series.
    Returns a pandas Series aligned on the tail of the original index.
    """
    if isinstance(series, pd.Series):
        clean = series.dropna()
        values = clean.values
        index = clean.index
    else:
        values = np.asarray(series, dtype=float)
        values = values[~np.isnan(values)]
        index = None

    w = [1.0]
    k = 1
    while True:
        w_k = -w[-1] * (d - k + 1) / k
        if abs(w_k) < thresh:
            break
        w.append(w_k)
        k += 1

    w = np.array(w[::-1])
    fdiff = np.convolve(values, w, mode="valid")

    if index is not None:
        fd_index = index[len(index) - len(fdiff):]
        return pd.Series(fdiff, index=fd_index)
    return pd.Series(fdiff)


def build_latex_table(df: pd.DataFrame) -> str:
    latex = (
        "\\begin{table}[H]\n"
        "  \\centering\n"
        "  \\caption{GPH estimates of the long-memory parameter $d$ for absolute returns of gold and silver}\n"
        "  \\label{tab:arfima-gph}\n"
        "  \\begin{tabular}{lccc}\n"
        "    \\toprule\n"
        "    Asset & $\\hat{d}$ & s.e.($\\hat{d}$) & $t$-stat \\\\\n"
        "    \\midrule\n"
    )

    for _, row in df.iterrows():
        latex += (
            f"    {row['asset']} & "
            f"{row['d_hat']:.3f} & "
            f"{row['se_d']:.3f} & "
            f"{row['t_d']:.2f} \\\\\n"
        )

    latex += (
        "    \\bottomrule\n"
        "  \\end{tabular}\n"
        "  \\vspace{0.2cm}\n"
        "  \\begin{minipage}{0.9\\textwidth}\n"
        "  \\footnotesize\n"
        "  \\textit{Notes}: The table reports Geweke--Porter--Hudak semiparametric estimates "
        "  of the fractional differencing parameter $d$ for the series of absolute daily log returns "
        "  on GLD and SLV (in percentage points). Values of $0 < \\hat{d} < 0.5$ indicate "
        "  covariance-stationary long-memory behaviour.\n"
        "  \\end{minipage}\n"
        "\\end{table}\n"
    )
    return latex


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    returns = pd.read_csv(
        DATA_PROCESSED_DIR / "asset_returns.csv",
        index_col=0,
        parse_dates=True
    )

    gold_ret = returns["gold"].dropna()
    silver_ret = returns["silver"].dropna()

    abs_gold = compute_abs_returns(gold_ret) * 100
    abs_silver = compute_abs_returns(silver_ret) * 100

    abs_desc = pd.concat(
        [
            abs_gold.describe().to_frame("gold_abs_ret"),
            abs_silver.describe().to_frame("silver_abs_ret"),
        ],
        axis=1
    )
    save_csv(abs_desc, "arfima_abs_returns_describe.csv")

    d_gold, se_gold = gph_long_memory_estimate(abs_gold.values)
    d_silver, se_silver = gph_long_memory_estimate(abs_silver.values)

    arfima_df = pd.DataFrame(
        {
            "asset": ["Gold", "Silver"],
            "d_hat": [d_gold, d_silver],
            "se_d": [se_gold, se_silver],
        }
    )
    arfima_df["t_d"] = arfima_df["d_hat"] / arfima_df["se_d"]

    save_csv(arfima_df, "arfima_gph_long_memory.csv", index=False)

    latex = build_latex_table(arfima_df)
    write_text_file(latex, "arfima_gph_table.tex", subdir="tables")

    plot_long_acf_two_panel(
        abs_gold,
        abs_silver,
        max_lag=100,
        filename="acf_abs_returns_gold_silver.png",
        ylim=(0, 0.30),
    )

    # Fractionally differenced series
    fd_gold = fractional_difference(abs_gold, d_gold)
    fd_silver = fractional_difference(abs_silver, d_silver)

    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    abs_gold.tail(1000).plot(ax=ax[0], color="tab:blue", linewidth=0.8)
    ax[0].set_title("Gold absolute returns (in %)")
    ax[0].set_ylabel("|r_t|")

    fd_gold.tail(1000).plot(ax=ax[1], color="tab:orange", linewidth=0.8)
    ax[1].set_title(f"Fractionally differenced gold series with d = {d_gold:.3f}")
    ax[1].set_ylabel("FD(|r_t|)")

    fig.tight_layout()
    save_figure(fig, "arfima_fd_gold.png")

    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    abs_silver.tail(1000).plot(ax=ax[0], color="tab:blue", linewidth=0.8)
    ax[0].set_title("Silver absolute returns (in %)")
    ax[0].set_ylabel("|r_t|")

    fd_silver.tail(1000).plot(ax=ax[1], color="tab:green", linewidth=0.8)
    ax[1].set_title(f"Fractionally differenced silver series with d = {d_silver:.3f}")
    ax[1].set_ylabel("FD(|r_t|)")

    fig.tight_layout()
    save_figure(fig, "arfima_fd_silver.png")

    print("\nARFIMA / GPH results:")
    print(arfima_df)


if __name__ == "__main__":
    main()