import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import DATA_PROCESSED_DIR, FORMAL_NAMES
from utils import export_latex_table, save_csv, save_figure

def main():
    df_def = pd.read_csv(DATA_PROCESSED_DIR / "asset_levels.csv", index_col=0, parse_dates=True)
    returns = pd.read_csv(DATA_PROCESSED_DIR / "asset_returns.csv", index_col=0, parse_dates=True)

    desc = returns.describe().T
    desc = desc.rename(columns={
        "count": "N",
        "mean": "Mean",
        "std": "Std",
        "min": "Min",
        "25%": "Q1",
        "50%": "Median",
        "75%": "Q3",
        "max": "Max",
    })

    desc = desc.rename(index={
        "gold": "Gold (GLD)",
        "silver": "Silver (SLV)",
        "sp500": "S\\&P 500",
        "dxy": "U.S. dollar index",
        "ust10y_change": "10Y yield change",
    })

    export_latex_table(desc, "descriptive_stats.tex", "%.4f", "lrrrrrrrr")

    corr = returns.corr()
    corr = corr.rename(index=FORMAL_NAMES, columns=FORMAL_NAMES)
    export_latex_table(corr, "correlation_matrix.tex", "%.3f", "lrrrrr")

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        cbar_kws={"label": "Correlation"},
        ax=ax
    )
    ax.set_title("Correlation matrix heatmap")
    save_figure(fig, "correlation_heatmap.png")

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    df_def[["gold", "silver", "sp500", "dxy"]].plot(ax=axes[0], title="Price / Index Levels")
    axes[0].set_ylabel("Level")

    returns[["gold", "silver", "sp500", "dxy"]].plot(ax=axes[1], title="Log Returns")
    axes[1].set_ylabel("Log return")

    save_figure(fig, "quality_check_levels_returns.png")

    fig, ax = plt.subplots(figsize=(14, 4))
    df_def["ust10y_yield"].plot(ax=ax, title="10Y U.S. Treasury Yield")
    ax.set_ylabel("Yield level")
    save_figure(fig, "quality_check_ust10y_level.png")

    fig, ax = plt.subplots(figsize=(14, 4))
    returns["ust10y_change"].plot(ax=ax, title="Daily Change in 10Y U.S. Treasury Yield")
    ax.set_ylabel("Yield change")
    save_figure(fig, "quality_check_ust10y_change.png")


if __name__ == "__main__":
    main()