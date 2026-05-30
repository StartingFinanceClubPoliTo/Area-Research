import pandas as pd
import matplotlib.pyplot as plt

from config import DATA_PROCESSED_DIR, ROLLING_WINDOW
from utils import run_ols_matrix, save_figure


def main():
    returns = pd.read_csv(DATA_PROCESSED_DIR / "asset_returns.csv", index_col=0, parse_dates=True)
    y_silver = returns["silver"]

    rolling_corr_s_ust = returns["silver"].rolling(ROLLING_WINDOW).corr(returns["ust10y_change"])
    fig, ax = plt.subplots(figsize=(10, 4))
    rolling_corr_s_ust.plot(ax=ax, color="tab:blue", linewidth=1)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title(f"Rolling {ROLLING_WINDOW}-day correlation: silver vs 10Y yield change")
    ax.set_ylabel("Correlation")
    ax.set_xlabel("Date")
    save_figure(fig, "rolling_corr_silver_ust10y.png")

    rolling_corr_s_sp = returns["silver"].rolling(ROLLING_WINDOW).corr(returns["sp500"])
    fig, ax = plt.subplots(figsize=(10, 4))
    rolling_corr_s_sp.plot(ax=ax, color="tab:orange", linewidth=1)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title(f"Rolling {ROLLING_WINDOW}-day correlation: silver vs S&P 500")
    ax.set_ylabel("Correlation")
    ax.set_xlabel("Date")
    save_figure(fig, "rolling_corr_silver_sp500.png")

    rolling_corr_s_dxy = returns["silver"].rolling(ROLLING_WINDOW).corr(returns["dxy"])
    fig, ax = plt.subplots(figsize=(10, 4))
    rolling_corr_s_dxy.plot(ax=ax, color="tab:green", linewidth=1)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title(f"Rolling {ROLLING_WINDOW}-day correlation: silver vs U.S. dollar index")
    ax.set_ylabel("Correlation")
    ax.set_xlabel("Date")
    save_figure(fig, "rolling_corr_silver_dxy.png")

    model_s_b1 = run_ols_matrix(y_silver, returns[["ust10y_change"]])
    model_s_b2 = run_ols_matrix(y_silver, returns[["ust10y_change", "sp500", "dxy"]])

    print(model_s_b1.summary())
    print(model_s_b2.summary())

    model_s_e1 = run_ols_matrix(y_silver, returns[["sp500"]])
    model_s_e2 = run_ols_matrix(y_silver, returns[["sp500", "ust10y_change", "dxy"]])

    print(model_s_e1.summary())
    print(model_s_e2.summary())

    model_s_d1 = run_ols_matrix(y_silver, returns[["dxy"]])
    model_s_d2 = run_ols_matrix(y_silver, returns[["dxy", "ust10y_change", "sp500"]])

    print(model_s_d1.summary())
    print(model_s_d2.summary())


if __name__ == "__main__":
    main()