import pandas as pd
import matplotlib.pyplot as plt

from config import DATA_PROCESSED_DIR, ROLLING_WINDOW
from utils import run_ols_formula, run_ols_matrix, save_figure


def main():
    returns = pd.read_csv(DATA_PROCESSED_DIR / "asset_returns.csv", index_col=0, parse_dates=True)
    data = returns.dropna().copy()

    model1 = run_ols_formula("gold ~ ust10y_change", data)
    model2 = run_ols_formula("gold ~ ust10y_change + sp500 + dxy", data)

    print(model1.summary())
    print(model2.summary())

    rolling_corr_g_ust = returns["gold"].rolling(ROLLING_WINDOW).corr(returns["ust10y_change"])
    fig, ax = plt.subplots(figsize=(10, 4))
    rolling_corr_g_ust.plot(ax=ax, color="tab:blue", linewidth=1)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title(f"Rolling {ROLLING_WINDOW}-day correlation: gold vs 10Y yield change")
    ax.set_ylabel("Correlation")
    ax.set_xlabel("Date")
    save_figure(fig, "rolling_corr_gold_ust10y.png")

    rolling_corr_g_sp = returns["gold"].rolling(ROLLING_WINDOW).corr(returns["sp500"])
    fig, ax = plt.subplots(figsize=(10, 4))
    rolling_corr_g_sp.plot(ax=ax, color="tab:orange", linewidth=1)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title(f"Rolling {ROLLING_WINDOW}-day correlation: gold vs S&P 500")
    ax.set_ylabel("Correlation")
    ax.set_xlabel("Date")
    save_figure(fig, "rolling_corr_gold_sp500.png")

    y = returns["gold"]

    model1_eq = run_ols_matrix(y, returns[["sp500"]])
    model2_eq = run_ols_matrix(y, returns[["sp500", "ust10y_change", "dxy"]])

    print(model1_eq.summary())
    print(model2_eq.summary())

    rolling_corr_g_dxy = returns["gold"].rolling(ROLLING_WINDOW).corr(returns["dxy"])
    fig, ax = plt.subplots(figsize=(10, 4))
    rolling_corr_g_dxy.plot(ax=ax, color="tab:green", linewidth=1)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title(f"Rolling {ROLLING_WINDOW}-day correlation: gold vs U.S. dollar index")
    ax.set_ylabel("Correlation")
    ax.set_xlabel("Date")
    save_figure(fig, "rolling_corr_gold_dxy.png")

    model1_usd = run_ols_matrix(y, returns[["dxy"]])
    model2_usd = run_ols_matrix(y, returns[["dxy", "ust10y_change", "sp500"]])

    print(model1_usd.summary())
    print(model2_usd.summary())


if __name__ == "__main__":
    main()