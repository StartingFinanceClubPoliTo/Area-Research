"""
Supporting code for the article:
"Portfolio Optimization: From Markowitz Theory to Practical Implementation"

This script reproduces the empirical analysis, figures and tables used in the article:
- classical multi-asset Markowitz optimization;
- common-sample comparison with and without Bitcoin;
- fixed out-of-sample test;
- rolling out-of-sample Markowitz test;
- portfolio weight and turnover plots.

Data are downloaded from Yahoo Finance through yfinance.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from datetime import date
from scipy.optimize import minimize


TRADING_DAYS = 252


def download_prices(
    tickers: list[str],
    start: str,
    end: str | None = None,
) -> pd.DataFrame:
    """
    Download adjusted closing prices from Yahoo Finance.
    Returns a DataFrame indexed by date with one column per ticker.
    """
    data = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
    )

    if "Adj Close" not in data:
        raise ValueError("Adjusted close prices not found in downloaded data.")

    prices = data["Adj Close"].copy()

    if isinstance(prices, pd.Series):
        prices = prices.to_frame()

    # Keep only dates with at least one observation, then align later in returns.
    prices = prices.dropna(how="all")

    return prices

def download_risk_free_rate(
    start: str,
    end: str | None = None,
    ticker: str = "^IRX",
) -> pd.Series:
    """
    Download a time-varying annualized risk-free rate proxy.

    ^IRX is the 13-week Treasury Bill yield from Yahoo Finance.
    Values are quoted in percentage points, so they are converted to decimals.
    """
    data = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
    )

    if "Adj Close" in data:
        rf = data["Adj Close"].copy()
    elif "Close" in data:
        rf = data["Close"].copy()
    else:
        raise ValueError("Risk-free rate data not found.")

    if isinstance(rf, pd.DataFrame):
        rf = rf.iloc[:, 0]

    rf = rf.dropna()
    rf = rf / 100.0
    rf.name = "risk_free_rate"

    return rf


def get_latest_risk_free_rate(
    rf_series: pd.Series,
    decision_date: pd.Timestamp,
    default_rf: float = 0.015,
) -> float:
    """
    Get the latest available annualized risk-free rate strictly before a decision date.
    """
    available = rf_series.loc[rf_series.index < decision_date]

    if available.empty:
        return default_rf

    return float(available.iloc[-1])


def compute_returns(prices: pd.DataFrame, method: str = "simple") -> pd.DataFrame:
    """
    Compute asset returns from price data.
    method: 'simple' or 'log'
    """
    # Use only dates where all assets have available prices.
    # This avoids artificial returns created by forward-filling non-trading days.
    aligned_prices = prices.dropna(how="any")

    if method == "simple":
        returns = aligned_prices.pct_change(fill_method=None)
    elif method == "log":
        returns = np.log(aligned_prices / aligned_prices.shift(1))
    else:
        raise ValueError("method must be either 'simple' or 'log'")

    returns = returns.dropna(how="any")
    return returns


def annualized_statistics(returns: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """
    Compute annualized expected returns and covariance matrix.
    """
    mu = returns.mean() * TRADING_DAYS
    sigma = returns.cov() * TRADING_DAYS
    return mu, sigma


def portfolio_performance(
    weights: np.ndarray,
    mu: pd.Series,
    sigma: pd.DataFrame,
    rf: float = 0.0,
) -> tuple[float, float, float]:
    """
    Compute annualized portfolio return, volatility, and Sharpe ratio.
    """
    port_return = float(weights @ mu.values)
    port_var = float(weights @ sigma.values @ weights)
    port_vol = np.sqrt(max(port_var, 0.0))

    sharpe = np.nan if port_vol <= 0 else (port_return - rf) / port_vol
    return port_return, port_vol, sharpe


def realized_portfolio_returns(
    returns: pd.DataFrame,
    weights: pd.Series,
) -> pd.Series:
    """
    Compute realized portfolio returns from asset returns and fixed portfolio weights.
    """
    aligned_weights = weights.reindex(returns.columns)

    if aligned_weights.isna().any():
        missing = aligned_weights[aligned_weights.isna()].index.tolist()
        raise ValueError(f"Missing weights for assets: {missing}")

    return returns @ aligned_weights.values

def backtest_fixed_weight_portfolio(
    returns: pd.DataFrame,
    target_weights: pd.Series,
    rebalance_frequency: str = "M",
) -> pd.Series:
    """
    Backtest a portfolio with fixed target weights and periodic rebalancing.

    rebalance_frequency:
        "M" = monthly rebalancing
        "Q" = quarterly rebalancing
        "A" = annual rebalancing
        "D" = daily rebalancing
        "BH" = buy and hold
    """
    weights = target_weights.reindex(returns.columns)

    if weights.isna().any():
        missing = weights[weights.isna()].index.tolist()
        raise ValueError(f"Missing weights for assets: {missing}")

    if not np.isclose(weights.sum(), 1.0):
        raise ValueError(f"Weights must sum to 1. Current sum: {weights.sum():.4f}")

    freq = rebalance_frequency.upper()

    if freq == "D":
        return returns @ weights.values

    if freq in ["BH", "BUY_AND_HOLD", "NONE"]:
        period_index = None
    elif freq in ["M", "Q", "A"]:
        period_index = returns.index.to_period(freq)
    else:
        raise ValueError("rebalance_frequency must be one of: 'D', 'M', 'Q', 'A', 'BH'")

    target_weights_array = weights.values.astype(float)

    portfolio_value = 1.0
    asset_values = portfolio_value * target_weights_array

    portfolio_returns = []
    previous_period = None

    for i, (_, asset_returns) in enumerate(returns.iterrows()):
        if period_index is not None:
            current_period = period_index[i]

            if previous_period is None:
                previous_period = current_period
            elif current_period != previous_period:
                asset_values = portfolio_value * target_weights_array
                previous_period = current_period

        new_asset_values = asset_values * (1 + asset_returns.values)
        new_portfolio_value = new_asset_values.sum()

        period_return = new_portfolio_value / portfolio_value - 1
        portfolio_returns.append(period_return)

        asset_values = new_asset_values
        portfolio_value = new_portfolio_value

    return pd.Series(
        portfolio_returns,
        index=returns.index,
        name=target_weights.name,
    )

def performance_summary(
    portfolio_returns: pd.Series,
    rf: float = 0.0,
    rf_series: pd.Series | None = None,
) -> dict[str, float]:
    """
    Compute annualized realized performance statistics.

    If rf_series is provided, it must contain annualized risk-free rates
    indexed by date. These are converted to daily rates and aligned with
    portfolio returns. Otherwise, rf is interpreted as a constant annual
    risk-free rate.
    """
    annual_return = portfolio_returns.mean() * TRADING_DAYS
    annual_volatility = portfolio_returns.std() * np.sqrt(TRADING_DAYS)

    cumulative_return = (1 + portfolio_returns).prod() - 1

    n_years = len(portfolio_returns) / TRADING_DAYS
    cagr = (1 + cumulative_return) ** (1 / n_years) - 1

    if rf_series is not None:
        aligned_rf = (
            rf_series
            .sort_index()
            .reindex(portfolio_returns.index, method="ffill")
        )
        aligned_rf = aligned_rf.fillna(rf)

        daily_rf = (1 + aligned_rf) ** (1 / TRADING_DAYS) - 1
        excess_returns = portfolio_returns - daily_rf
    else:
        daily_rf = (1 + rf) ** (1 / TRADING_DAYS) - 1
        excess_returns = portfolio_returns - daily_rf

    sharpe = np.nan
    if portfolio_returns.std() > 0:
        sharpe = (
            excess_returns.mean()
            / portfolio_returns.std()
            * np.sqrt(TRADING_DAYS)
        )

    wealth_index = (1 + portfolio_returns).cumprod()
    running_max = wealth_index.cummax()
    drawdown = wealth_index / running_max - 1
    max_drawdown = drawdown.min()

    return {
        "annual_return": annual_return,
        "cagr": cagr,
        "annual_volatility": annual_volatility,
        "sharpe": sharpe,
        "cumulative_return": cumulative_return,
        "max_drawdown": max_drawdown,
    }


def custom_weight_portfolio(
    assets: list[str],
    allocations: dict[str, float],
    name: str,
) -> pd.Series:
    """
    Build a custom benchmark portfolio using fixed weights.
    """
    weights = pd.Series(0.0, index=assets, name=name)

    for asset, weight in allocations.items():
        if asset not in weights.index:
            raise ValueError(f"{asset} is not available in the asset universe.")
        weights[asset] = weight

    if not np.isclose(weights.sum(), 1.0):
        raise ValueError(f"Portfolio weights must sum to 1. Current sum: {weights.sum():.4f}")

    return weights

def simulate_random_portfolios(
    n_portfolios: int,
    mu: pd.Series,
    sigma: pd.DataFrame,
    rf: float = 0.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Simulate random long-only, fully-invested portfolios using Dirichlet-distributed weights.
    """
    rng = np.random.default_rng(seed)
    n_assets = len(mu)

    weights_matrix = rng.dirichlet(np.ones(n_assets), size=n_portfolios)

    port_returns = weights_matrix @ mu.values
    port_vars = np.einsum("ij,jk,ik->i", weights_matrix, sigma.values, weights_matrix)
    port_vols = np.sqrt(np.maximum(port_vars, 0.0))
    sharpes = np.where(port_vols > 0, (port_returns - rf) / port_vols, np.nan)

    portfolios = pd.DataFrame({
        "return": port_returns,
        "volatility": port_vols,
        "sharpe": sharpes,
    })

    weights_df = pd.DataFrame(weights_matrix, columns=mu.index)
    portfolios = pd.concat([portfolios, weights_df], axis=1)

    return portfolios


def single_asset_portfolios(
    mu: pd.Series,
    sigma: pd.DataFrame,
    rf: float = 0.0,
) -> pd.DataFrame:
    """
    Build 100% single-asset portfolios for comparison.
    """
    records = []

    for i, asset in enumerate(mu.index):
        weights = np.zeros(len(mu))
        weights[i] = 1.0
        port_return, port_vol, sharpe = portfolio_performance(weights, mu, sigma, rf)

        records.append({
            "asset": asset,
            "return": port_return,
            "volatility": port_vol,
            "sharpe": sharpe,
        })

    return pd.DataFrame(records)


def summarize_simulated_portfolios(portfolios: pd.DataFrame) -> dict[str, pd.Series]:
    """
    Identify key portfolios from the simulated set.
    These are the best portfolios within the simulation, not exact analytical optima.
    """
    return {
        "min_vol_simulated": portfolios.loc[portfolios["volatility"].idxmin()],
        "max_sharpe_simulated": portfolios.loc[portfolios["sharpe"].idxmax()],
        "max_return_simulated": portfolios.loc[portfolios["return"].idxmax()],
    }


def _weight_constraints(n_assets: int) -> list[dict]:
    """
    Fully invested: sum of weights = 1
    """
    return [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]


def _weight_bounds(n_assets: int) -> list[tuple[float, float]]:
    """
    Long-only, no leverage, no short selling.
    """
    return [(0.0, 1.0)] * n_assets


def optimize_min_variance(
    mu: pd.Series,
    sigma: pd.DataFrame,
) -> pd.Series:
    """
    Numerically solve the long-only global minimum variance portfolio.
    """
    n_assets = len(mu)
    x0 = np.repeat(1 / n_assets, n_assets)

    def objective(w: np.ndarray) -> float:
        return float(w @ sigma.values @ w)

    result = minimize(
        objective,
        x0=x0,
        method="SLSQP",
        bounds=_weight_bounds(n_assets),
        constraints=_weight_constraints(n_assets),
    )

    if not result.success:
        raise RuntimeError(f"Min-variance optimization failed: {result.message}")

    return pd.Series(result.x, index=mu.index, name="min_variance_weights")


def optimize_max_sharpe(
    mu: pd.Series,
    sigma: pd.DataFrame,
    rf: float = 0.0,
    n_starts: int = 20,
    seed: int = 42,
) -> pd.Series:
    n_assets = len(mu)
    rng = np.random.default_rng(seed)

    starts = [np.repeat(1 / n_assets, n_assets)]
    starts.extend(np.eye(n_assets))
    starts.extend(rng.dirichlet(np.ones(n_assets), size=n_starts))

    def objective(w: np.ndarray) -> float:
        _, _, sharpe = portfolio_performance(w, mu, sigma, rf)
        if np.isnan(sharpe):
            return 1e6
        return -sharpe

    best_result = None

    for x0 in starts:
        result = minimize(
            objective,
            x0=x0,
            method="SLSQP",
            bounds=_weight_bounds(n_assets),
            constraints=_weight_constraints(n_assets),
        )

        if result.success and (best_result is None or result.fun < best_result.fun):
            best_result = result

    if best_result is None:
        raise RuntimeError("Max-Sharpe optimization failed for all starting points.")

    return pd.Series(best_result.x, index=mu.index, name="max_sharpe_weights")


def compute_efficient_frontier(
    mu: pd.Series,
    sigma: pd.DataFrame,
    n_points: int = 100,
    rf: float = 0.0,
) -> pd.DataFrame:
    """
    Compute the long-only efficient frontier by minimizing variance
    for a grid of target returns.
    """
    n_assets = len(mu)
    x0 = np.repeat(1 / n_assets, n_assets)

    min_var_w = optimize_min_variance(mu, sigma)
    mu_gmv = float(min_var_w.values @ mu.values)
    mu_max = float(mu.max())

    target_returns = np.linspace(mu_gmv, mu_max, n_points)

    bounds = _weight_bounds(n_assets)

    frontier_records = []

    for target in target_returns:
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "eq", "fun": lambda w, target=target: float(w @ mu.values) - target},
        ]

        def objective(w: np.ndarray) -> float:
            return float(w @ sigma.values @ w)

        result = minimize(
            objective,
            x0=x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        if result.success:
            w = result.x
            port_return, port_vol, sharpe = portfolio_performance(w, mu, sigma, rf=rf)

            record = {
                "target_return": target,
                "return": port_return,
                "volatility": port_vol,
                "sharpe": sharpe,
            }
            for i, asset in enumerate(mu.index):
                record[asset] = w[i]

            frontier_records.append(record)

    if not frontier_records:
        raise RuntimeError("Efficient frontier optimization failed for all target returns.")

    frontier = pd.DataFrame(frontier_records)
    frontier = frontier.sort_values("volatility").reset_index(drop=True)

    return frontier


def build_optimal_portfolio_table(
    mu: pd.Series,
    sigma: pd.DataFrame,
    rf: float = 0.0,
) -> pd.DataFrame:
    """
    Build a compact table with optimized portfolios.
    """
    min_var_w = optimize_min_variance(mu, sigma)
    max_sharpe_w = optimize_max_sharpe(mu, sigma, rf=rf)

    portfolios = {
        "Minimum Variance": min_var_w,
        "Maximum Sharpe": max_sharpe_w,
    }

    rows = []

    for label, weights in portfolios.items():
        port_return, port_vol, sharpe = portfolio_performance(weights.values, mu, sigma, rf)

        row = {
            "portfolio": label,
            "return": port_return,
            "volatility": port_vol,
            "sharpe": sharpe,
        }

        for asset in mu.index:
            row[asset] = weights[asset]

        rows.append(row)

    return pd.DataFrame(rows)


def print_statistics(mu: pd.Series, sigma: pd.DataFrame) -> None:
    """
    Print annualized asset statistics.
    """
    print("\nAnnualized expected returns:")
    print((mu * 100).round(2).astype(str) + "%")

    vols = pd.Series(np.sqrt(np.diag(sigma)), index=sigma.index)
    print("\nAnnualized volatilities:")
    print((vols * 100).round(2).astype(str) + "%")


def print_portfolio_details(
    name: str,
    weights: pd.Series,
    mu: pd.Series,
    sigma: pd.DataFrame,
    rf: float = 0.0,
    weight_threshold: float = 0.01,
) -> None:
    """
    Print a clean summary of an optimized portfolio.
    """
    port_return, port_vol, sharpe = portfolio_performance(weights.values, mu, sigma, rf)

    print(f"\n{name}")
    print("-" * len(name))
    print(f"Return:     {port_return:.2%}")
    print(f"Volatility: {port_vol:.2%}")
    print(f"Sharpe:     {sharpe:.3f}")

    filtered = weights[weights >= weight_threshold].sort_values(ascending=False)
    print("\nWeights:")
    for asset, w in filtered.items():
        print(f"  {asset:<10} {w:.2%}")


def plot_price_series(
    prices: pd.DataFrame,
    title: str,
    save_path: str | Path | None = None,
) -> None:
    """
    Plot normalized price series.
    """
    plot_prices = prices.dropna(how="any")
    normalized = plot_prices / plot_prices.iloc[0]

    plt.figure(figsize=(12, 7))
    for col in normalized.columns:
        plt.plot(normalized.index, normalized[col], label=col)

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Normalized Price (Base = 1)")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_correlation_matrix(
    returns: pd.DataFrame,
    title: str,
    save_path: str | Path | None = None,
) -> None:
    """
    Plot correlation matrix heatmap.
    """
    corr = returns.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr,
        cmap="Reds",
        center=0,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        annot_kws={"size": 10},
    )

    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_portfolios(
    simulated_portfolios: pd.DataFrame,
    single_assets: pd.DataFrame,
    frontier: pd.DataFrame,
    optimal_table: pd.DataFrame,
    title: str,
    save_path: str | Path | None = None,
) -> None:
    """
    Plot random portfolios, the optimized efficient frontier, single assets,
    and optimized portfolios.
    """
    fig, ax = plt.subplots(figsize=(11, 7))

    scatter = ax.scatter(
        simulated_portfolios["volatility"],
        simulated_portfolios["return"],
        c=simulated_portfolios["sharpe"],
        s=8,
        alpha=0.5,
    )
    plt.colorbar(scatter, ax=ax, label="Sharpe ratio")

    ax.plot(
        frontier["volatility"],
        frontier["return"],
        linewidth=2.5,
        label="Efficient Frontier",
    )

    ax.scatter(
        single_assets["volatility"],
        single_assets["return"],
        s=90,
        marker="X",
        label="Single Assets",
    )

    for _, row in single_assets.iterrows():
        ax.annotate(
            row["asset"],
            (row["volatility"], row["return"]),
            fontsize=9,
            xytext=(5, 5),
            textcoords="offset points",
        )

    for _, row in optimal_table.iterrows():
        ax.scatter(
            row["volatility"],
            row["return"],
            s=180,
            marker="*",
            label=row["portfolio"],
        )

    ax.set_title(title)
    ax.set_xlabel("Annualized Volatility")
    ax.set_ylabel("Annualized Return")
    ax.legend()
    plt.tight_layout()

    if save_path is not None:
      plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_cumulative_wealth(
    wealth: pd.DataFrame,
    title: str,
    save_path: str | Path | None = None,
) -> None:
    """
    Plot cumulative wealth index for one or more portfolios.
    """
    plt.figure(figsize=(12, 7))

    for col in wealth.columns:
        plt.plot(wealth.index, wealth[col], label=col)

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Cumulative Wealth (Initial Value = 1)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_weight_history(
    weights: pd.DataFrame,
    title: str,
    save_path: str | Path | None = None,
) -> None:
    """
    Plot rolling portfolio weights over time as a stacked area chart.
    """
    ax = weights.plot.area(figsize=(12, 7), linewidth=0)

    ax.set_title(title)
    ax.set_xlabel("Rebalance Date")
    ax.set_ylabel("Portfolio Weight")
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=len(weights.columns))

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_turnover_history(
    turnover: pd.DataFrame,
    title: str,
    save_path: str | Path | None = None,
) -> None:
    """
    Plot monthly turnover for rolling strategies.
    """
    plt.figure(figsize=(12, 7))

    for col in turnover.columns:
        plt.plot(turnover.index, turnover[col], label=col)

    plt.title(title)
    plt.xlabel("Rebalance Date")
    plt.ylabel("One-Way Monthly Turnover")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def build_fixed_vs_rolling_wealth(
    fixed_results: dict[str, object],
    rolling_results: dict[str, object],
) -> pd.DataFrame:
    """
    Build a cumulative wealth table comparing the fixed optimized portfolios
    with the rolling optimized portfolios.
    """
    fixed_wealth = fixed_results["wealth"]
    rolling_wealth = rolling_results["wealth"]

    comparison = pd.DataFrame(index=fixed_wealth.index)

    fixed_map = {
        "Fixed Minimum Variance": "Minimum Variance",
        "Fixed Maximum Sharpe": "Maximum Sharpe",
    }

    for new_name, old_name in fixed_map.items():
        if old_name in fixed_wealth.columns:
            comparison[new_name] = fixed_wealth[old_name]

    rolling_map = {
        "Rolling Minimum Variance": "Rolling Minimum Variance",
        "Rolling Maximum Sharpe": "Rolling Maximum Sharpe",
    }

    for new_name, old_name in rolling_map.items():
        if old_name in rolling_wealth.columns:
            comparison[new_name] = rolling_wealth[old_name].reindex(comparison.index).ffill()

    comparison = comparison.dropna(how="all")

    return comparison


def compare_rolling_with_benchmarks(
    rolling_results: dict[str, object],
    tickers: list[str],
    rf: float,
    figures_dir: str | Path,
    title: str,
) -> dict[str, object]:
    """
    Compare rolling optimized portfolios with the same fixed benchmark portfolios:
    equal weight, 60/40, 100% SPY, 100% IEF.
    """
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    returns = rolling_results["returns"]
    rolling_wealth = rolling_results["wealth"]
    common_index = rolling_wealth.index

    rf_series = rolling_results.get("rf_series", None)

    test_returns = returns.loc[common_index]

    assets = list(test_returns.columns)

    benchmark_portfolios = {}

    benchmark_portfolios["Benchmark: Equal Weight"] = custom_weight_portfolio(
        assets=assets,
        allocations={asset: 1 / len(assets) for asset in assets},
        name="equal_weight_weights",
    )

    if {"SPY", "IEF"}.issubset(set(assets)):
        benchmark_portfolios["Benchmark: 60/40 SPY-IEF"] = custom_weight_portfolio(
            assets=assets,
            allocations={"SPY": 0.60, "IEF": 0.40},
            name="sixty_forty_weights",
        )

    if "SPY" in assets:
        benchmark_portfolios["Benchmark: 100% SPY"] = custom_weight_portfolio(
            assets=assets,
            allocations={"SPY": 1.00},
            name="spy_weights",
        )

    if "IEF" in assets:
        benchmark_portfolios["Benchmark: 100% IEF"] = custom_weight_portfolio(
            assets=assets,
            allocations={"IEF": 1.00},
            name="ief_weights",
        )

    benchmark_returns = {}
    benchmark_summary_rows = []

    for name, weights in benchmark_portfolios.items():
        rets = backtest_fixed_weight_portfolio(
            returns=test_returns,
            target_weights=weights,
            rebalance_frequency="M",
        )

        benchmark_returns[name] = rets

        stats = performance_summary(
            rets,
            rf=rf,
            rf_series=rf_series,
        )
        row = {"portfolio": name}
        row.update(stats)
        row["average_monthly_turnover"] = 0.0
        row["max_monthly_turnover"] = 0.0

        for asset in assets:
            row[asset] = weights[asset]

        benchmark_summary_rows.append(row)

    benchmark_wealth = pd.DataFrame({
        name: (1 + rets).cumprod()
        for name, rets in benchmark_returns.items()
    })

    combined_wealth = pd.concat(
        [rolling_wealth, benchmark_wealth],
        axis=1,
    )

    combined_summary = pd.concat(
        [
            rolling_results["summary"],
            pd.DataFrame(benchmark_summary_rows),
        ],
        ignore_index=True,
    )

    scenario_name = title.lower().replace(" ", "_").replace("-", "_")

    combined_summary.to_csv(
        figures_dir / f"{scenario_name}_summary.csv",
        index=False,
    )

    plot_cumulative_wealth(
        combined_wealth,
        title=title,
        save_path=figures_dir / f"{scenario_name}_cumulative_wealth.png",
    )

    return {
        "wealth": combined_wealth,
        "summary": combined_summary,
        "benchmark_returns": benchmark_returns,
        "benchmark_wealth": benchmark_wealth,
    }


def run_analysis(
    tickers: list[str],
    start_date: str,
    title: str,
    end_date: str | None = None,
    rf: float = 0.02,
    n_portfolios: int = 50000,
    returns_method: str = "simple",
    frontier_points: int = 100,
    figures_dir: str | Path = "figures",
) -> dict[str, object]:
    """
    Full analysis pipeline for a given asset universe.
    """
    print(f"\n{'=' * 80}")
    print(title)
    print(f"{'=' * 80}")
    print(f"Tickers: {tickers}")
    print(f"Start date: {start_date}")
    print(f"End date: {end_date or date.today()}")
    print("Constraints: long-only, fully invested, no short selling")

    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    scenario_name = title.lower().replace(" ", "_").replace("-", "_")

    prices = download_prices(tickers, start=start_date, end=end_date)
    returns = compute_returns(prices, method=returns_method)
    mu, sigma = annualized_statistics(returns)
    print(f"Actual return sample: {returns.index.min().date()} to {returns.index.max().date()}")

    rf_series = download_risk_free_rate(
        start=start_date,
        end=end_date,
    )
    calibration_end_date = (
        pd.Timestamp(end_date)
        if end_date is not None
        else returns.index.max() + pd.Timedelta(days=1)
    )
    effective_rf = get_latest_risk_free_rate(
        rf_series=rf_series,
        decision_date=calibration_end_date,
        default_rf=rf,
    )
    print(f"Risk-free rate used for optimization and Sharpe ratios: {effective_rf:.2%} (latest IRX at calibration end)")

    plot_price_series(
        prices,
        f"Asset Prices (Normalized) - {title}",
        save_path=figures_dir / f"{scenario_name}_prices.png",
    )

    print_statistics(mu, sigma)

    plot_correlation_matrix(
        returns,
        f"Correlation Matrix - {title}",
        save_path=figures_dir / f"{scenario_name}_correlation_matrix.png",
    )

    simulated_portfolios = simulate_random_portfolios(
        n_portfolios=n_portfolios,
        mu=mu,
        sigma=sigma,
        rf=effective_rf,
        seed=42,
    )

    single_assets = single_asset_portfolios(mu, sigma, rf=effective_rf)
    simulated_summary = summarize_simulated_portfolios(simulated_portfolios)

    frontier = compute_efficient_frontier(
        mu,
        sigma,
        n_points=frontier_points,
        rf=effective_rf,
    )

    optimal_table = build_optimal_portfolio_table(
        mu,
        sigma,
        rf=effective_rf,
    )

    optimal_table.to_csv(
        figures_dir / f"{scenario_name}_optimal_portfolios.csv",
        index=False,
    )

    plot_portfolios(
        simulated_portfolios=simulated_portfolios,
        single_assets=single_assets,
        frontier=frontier,
        optimal_table=optimal_table,
        title=title,
        save_path=figures_dir / f"{scenario_name}_efficient_frontier.png",
    )

    min_var_weights = optimize_min_variance(mu, sigma)
    max_sharpe_weights = optimize_max_sharpe(mu, sigma, rf=effective_rf)

    print_portfolio_details(
        "Optimized Minimum Variance Portfolio",
        min_var_weights,
        mu,
        sigma,
        effective_rf,
    )

    print_portfolio_details(
        "Optimized Maximum Sharpe Portfolio",
        max_sharpe_weights,
        mu,
        sigma,
        effective_rf,
    )

    print("\nBest simulated portfolios (for reference only):")
    print(simulated_summary["min_vol_simulated"][["return", "volatility", "sharpe"]].round(4))
    print(simulated_summary["max_sharpe_simulated"][["return", "volatility", "sharpe"]].round(4))
    print(simulated_summary["max_return_simulated"][["return", "volatility", "sharpe"]].round(4))

    return {
        "prices": prices,
        "returns": returns,
        "rf_series": rf_series,
        "rf_used": effective_rf,
        "mu": mu,
        "sigma": sigma,
        "simulated_portfolios": simulated_portfolios,
        "single_assets": single_assets,
        "frontier": frontier,
        "optimal_table": optimal_table,
        "simulated_summary": simulated_summary,
    }

def run_out_of_sample_test(
    tickers: list[str],
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str | None = None,
    rf: float = 0.02,
    returns_method: str = "simple",
    figures_dir: str | Path = "figures",
    title: str = "Out-of-Sample Test",
    rebalance_frequency: str = "M",
) -> dict[str, object]:
    """
    Estimate optimized portfolios on a training period and evaluate them out-of-sample.

    The portfolio weights are estimated using only training-period data.
    They are then held fixed and applied to test-period realized returns.
    """
    print(f"\n{'=' * 80}")
    print(title)
    print(f"{'=' * 80}")
    print(f"Tickers: {tickers}")
    print(f"Training period: {train_start} to {train_end}")
    print(f"Test period: {test_start} to {test_end or date.today()}")
    print(
        "Weights are estimated in-sample and then rebalanced "
        f"out-of-sample with frequency: {rebalance_frequency}."
    )

    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    scenario_name = title.lower().replace(" ", "_").replace("-", "_")

    # Download full price history covering both train and test
    prices = download_prices(
        tickers=tickers,
        start=train_start,
        end=test_end,
    )

    returns = compute_returns(prices, method=returns_method)

    train_returns = returns.loc[
        (returns.index >= train_start) & (returns.index < train_end)
    ]

    test_returns = returns.loc[
        returns.index >= test_start
    ]

    if test_end is not None:
        test_returns = test_returns.loc[test_returns.index < test_end]

    if train_returns.empty:
        raise ValueError("Training returns are empty. Check train_start/train_end.")

    if test_returns.empty:
        raise ValueError("Test returns are empty. Check test_start/test_end.")

    print(f"Actual training sample: {train_returns.index.min().date()} to {train_returns.index.max().date()}")
    print(f"Actual test sample: {test_returns.index.min().date()} to {test_returns.index.max().date()}")

    # Estimate parameters only on training data
    train_mu, train_sigma = annualized_statistics(train_returns)

    rf_series = download_risk_free_rate(
        start=train_start,
        end=test_end,
    )

    decision_rf = get_latest_risk_free_rate(
        rf_series=rf_series,
        decision_date=pd.Timestamp(test_start),
        default_rf=rf,
    )

    print(f"Risk-free rate used for initial optimization: {decision_rf:.2%}")

    # Optimized portfolios estimated in-sample
    min_var_weights = optimize_min_variance(train_mu, train_sigma)
    max_sharpe_weights = optimize_max_sharpe(train_mu, train_sigma, rf=decision_rf)

    assets = list(train_returns.columns)

    # Optimized portfolios
    portfolios = {
        "Minimum Variance": min_var_weights,
        "Maximum Sharpe": max_sharpe_weights,
    }

    # Benchmarks
    portfolios["Benchmark: Equal Weight"] = custom_weight_portfolio(
        assets=assets,
        allocations={asset: 1 / len(assets) for asset in assets},
        name="equal_weight_weights",
    )

    if {"SPY", "IEF"}.issubset(set(assets)):
        portfolios["Benchmark: 60/40 SPY-IEF"] = custom_weight_portfolio(
            assets=assets,
            allocations={"SPY": 0.60, "IEF": 0.40},
            name="sixty_forty_weights",
        )

    if "SPY" in assets:
        portfolios["Benchmark: 100% SPY"] = custom_weight_portfolio(
            assets=assets,
            allocations={"SPY": 1.00},
            name="spy_weights",
        )

    if "IEF" in assets:
        portfolios["Benchmark: 100% IEF"] = custom_weight_portfolio(
            assets=assets,
            allocations={"IEF": 1.00},
            name="ief_weights",
        )


    # Evaluate realized out-of-sample returns
    oos_returns = {}
    summary_rows = []

    for name, weights in portfolios.items():
        port_rets = backtest_fixed_weight_portfolio(
            returns=test_returns,
            target_weights=weights,
            rebalance_frequency=rebalance_frequency,
        )

        oos_returns[name] = port_rets

        stats = performance_summary(
            port_rets,
            rf=rf,
            rf_series=rf_series,
        )

        row = {"portfolio": name}
        row.update(stats)

        for asset in test_returns.columns:
            row[asset] = weights[asset]

        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(
        figures_dir / f"{scenario_name}_oos_summary.csv",
        index=False,
    )

    # Build cumulative wealth index
    wealth = pd.DataFrame({
        name: (1 + rets).cumprod()
        for name, rets in oos_returns.items()
    })

    # Plot out-of-sample cumulative performance
    plot_cumulative_wealth(
        wealth,
        title=f"Cumulative Out-of-Sample Performance - {title}",
        save_path=figures_dir / f"{scenario_name}_oos_cumulative_wealth.png",
    )

    print("\nOut-of-sample performance:")
    display_columns = [
        "portfolio",
        "annual_return",
        "cagr",
        "annual_volatility",
        "sharpe",
        "cumulative_return",
        "max_drawdown",
    ]

    print(
        summary[display_columns]
        .assign(
            annual_return=lambda x: x["annual_return"].map(lambda v: f"{v:.2%}"),
            cagr=lambda x: x["cagr"].map(lambda v: f"{v:.2%}"),
            annual_volatility=lambda x: x["annual_volatility"].map(lambda v: f"{v:.2%}"),
            sharpe=lambda x: x["sharpe"].map(lambda v: f"{v:.3f}"),
            cumulative_return=lambda x: x["cumulative_return"].map(lambda v: f"{v:.2%}"),
            max_drawdown=lambda x: x["max_drawdown"].map(lambda v: f"{v:.2%}"),
        )
    )

    print("\nIn-sample optimized weights:")
    for name, weights in portfolios.items():
        print(f"\n{name}")
        print((weights[weights > 0.001] * 100).round(2).astype(str) + "%")

    return {
        "prices": prices,
        "returns": returns,
        "rf_series": rf_series,
        "train_returns": train_returns,
        "test_returns": test_returns,
        "train_mu": train_mu,
        "train_sigma": train_sigma,
        "weights": portfolios,
        "oos_returns": oos_returns,
        "wealth": wealth,
        "summary": summary,
    }


def clean_weights(weights: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    """
    Clean numerical optimization output:
    - set tiny weights to zero
    - clip small numerical violations
    - renormalize to sum to one
    """
    cleaned = np.asarray(weights, dtype=float).copy()
    cleaned[np.abs(cleaned) < tol] = 0.0
    cleaned = np.clip(cleaned, 0.0, 1.0)

    total = cleaned.sum()
    if total <= 0:
        raise ValueError("Optimization returned zero or invalid weights.")

    return cleaned / total


def optimize_portfolios_from_returns(
    train_returns: pd.DataFrame,
    rf: float,
) -> dict[str, pd.Series]:
    """
    Estimate mean-variance inputs from a training window and compute
    the optimized portfolios used in the out-of-sample analysis.
    """
    mu, sigma = annualized_statistics(train_returns)

    min_var_weights = optimize_min_variance(mu, sigma)
    max_sharpe_weights = optimize_max_sharpe(mu, sigma, rf=rf)

    return {
        "Rolling Minimum Variance": min_var_weights,
        "Rolling Maximum Sharpe": max_sharpe_weights,
    }


def get_monthly_rebalance_dates(
    returns: pd.DataFrame,
    test_start: str,
    test_end: str | None = None,
) -> pd.DatetimeIndex:
    """
    Return the first trading day of each month in the test period.
    These dates are used as portfolio formation/rebalancing dates.
    """
    test_returns = returns.loc[returns.index >= test_start].copy()

    if test_end is not None:
        test_returns = test_returns.loc[test_returns.index < test_end]

    if test_returns.empty:
        raise ValueError("No returns available in the test period.")

    first_days = (
        pd.Series(test_returns.index, index=test_returns.index)
        .groupby(test_returns.index.to_period("M"))
        .min()
    )

    return pd.DatetimeIndex(first_days.values)


def compute_turnover(
    current_weights: pd.Series,
    previous_weights: pd.Series | None,
) -> float:
    """
    One-way turnover between two consecutive target allocations.

    Turnover_t = 0.5 * sum_i |w_{i,t} - w_{i,t-1}|
    """
    if previous_weights is None:
        return np.nan

    current = current_weights.reindex(previous_weights.index)
    return float(0.5 * np.abs(current.values - previous_weights.values).sum())


def run_rolling_out_of_sample_test(
    tickers: list[str],
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str | None = None,
    lookback_years: int = 10,
    rf: float = 0.015,
    returns_method: str = "simple",
    figures_dir: str | Path = "figures",
    title: str = "Rolling Out-of-Sample Test",
) -> dict[str, object]:
    """
    Rolling Markowitz out-of-sample test.

    Initial calibration window:
        train_start to train_end

    Out-of-sample period:
        test_start to test_end

    At each monthly rebalancing date, the optimizer uses the most recent
    `lookback_years` of daily returns available up to the previous trading day.
    The optimized weights are then applied until the next monthly rebalancing date.
    """
    print(f"\n{'=' * 80}")
    print(title)
    print(f"{'=' * 80}")
    print(f"Tickers: {tickers}")
    print(f"Initial training period: {train_start} to {train_end}")
    print(f"Test period: {test_start} to {test_end or date.today()}")
    print(f"Rolling lookback window: {lookback_years} years")
    print("Re-estimation frequency: monthly")

    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    scenario_name = title.lower().replace(" ", "_").replace("-", "_")

    prices = download_prices(
        tickers=tickers,
        start=train_start,
        end=test_end,
    )

    rf_series = download_risk_free_rate(
        start=train_start,
        end=test_end,
    )

    returns = compute_returns(prices, method=returns_method)

    if returns.empty:
        raise ValueError("Returns are empty. Check dates and tickers.")

    rebalance_dates = get_monthly_rebalance_dates(
        returns=returns,
        test_start=test_start,
        test_end=test_end,
    )

    if len(rebalance_dates) < 2:
        raise ValueError("Not enough monthly rebalance dates in the test period.")

    strategy_returns: dict[str, list[pd.Series]] = {
        "Rolling Minimum Variance": [],
        "Rolling Maximum Sharpe": [],
    }

    weight_history: dict[str, list[pd.Series]] = {
        name: [] for name in strategy_returns
    }

    turnover_history: dict[str, list[tuple[pd.Timestamp, float]]] = {
        name: [] for name in strategy_returns
    }

    previous_weights: dict[str, pd.Series | None] = {
        name: None for name in strategy_returns
    }

    for i, rebalance_date in enumerate(rebalance_dates):
        # The portfolio formed at rebalance_date is held until the next rebalance date.
        if i + 1 < len(rebalance_dates):
            next_rebalance_date = rebalance_dates[i + 1]
            hold_returns = returns.loc[
                (returns.index >= rebalance_date) &
                (returns.index < next_rebalance_date)
            ]
        else:
            hold_returns = returns.loc[returns.index >= rebalance_date]
            if test_end is not None:
                hold_returns = hold_returns.loc[hold_returns.index < test_end]

        if hold_returns.empty:
            continue

        # Use only information strictly before the rebalance date.
        lookback_start = rebalance_date - pd.DateOffset(years=lookback_years)
        train_window = returns.loc[
            (returns.index >= lookback_start) &
            (returns.index < rebalance_date)
        ]

        min_required_obs = int(TRADING_DAYS * lookback_years * 0.8)
        if len(train_window) < min_required_obs:
            print(
                f"Skipping {rebalance_date.date()} - insufficient training observations "
                f"({len(train_window)})."
            )
            continue

        
        current_rf = get_latest_risk_free_rate(
            rf_series=rf_series,
            decision_date=rebalance_date,
            default_rf=rf,
        )

        monthly_weights = optimize_portfolios_from_returns(
            train_returns=train_window,
            rf=current_rf,
        )

        for strategy_name, weights in monthly_weights.items():
            weights = weights.reindex(returns.columns)

            if weights.isna().any():
                raise ValueError(f"Missing weights for {strategy_name}.")

            weights = pd.Series(
                clean_weights(weights.values),
                index=returns.columns,
                name=rebalance_date,
            )

            # During each month, we assume the portfolio is rebalanced to target weights
            # at the start of the holding period and then held until the next rebalance.
            period_returns = backtest_fixed_weight_portfolio(
                returns=hold_returns,
                target_weights=weights,
                rebalance_frequency="BH",
            )

            strategy_returns[strategy_name].append(period_returns)

            weight_history[strategy_name].append(weights)

            turnover = compute_turnover(
                current_weights=weights,
                previous_weights=previous_weights[strategy_name],
            )
            turnover_history[strategy_name].append((rebalance_date, turnover))
            previous_weights[strategy_name] = weights

    oos_returns = {}
    wealth = {}
    weights_history_df = {}
    turnover_df = {}

    for strategy_name, parts in strategy_returns.items():
        if not parts:
            raise RuntimeError(f"No returns generated for {strategy_name}.")

        full_returns = pd.concat(parts).sort_index()
        full_returns = full_returns[~full_returns.index.duplicated(keep="last")]

        oos_returns[strategy_name] = full_returns
        wealth[strategy_name] = (1 + full_returns).cumprod()

        weights_df = pd.DataFrame(weight_history[strategy_name])
        weights_df.index.name = "rebalance_date"
        weights_history_df[strategy_name] = weights_df

        turnover_series = pd.Series(
            data=[value for _, value in turnover_history[strategy_name]],
            index=[dt for dt, _ in turnover_history[strategy_name]],
            name=strategy_name,
        )
        turnover_df[strategy_name] = turnover_series

    wealth_df = pd.DataFrame(wealth)

    summary_rows = []

    for strategy_name, rets in oos_returns.items():
        stats = performance_summary(
            rets,
            rf=rf,
            rf_series=rf_series,
        )

        row = {"portfolio": strategy_name}
        row.update(stats)

        avg_turnover = turnover_df[strategy_name].dropna().mean()
        max_turnover = turnover_df[strategy_name].dropna().max()

        row["average_monthly_turnover"] = avg_turnover
        row["max_monthly_turnover"] = max_turnover

        last_weights = weights_history_df[strategy_name].iloc[-1]
        for asset in returns.columns:
            row[asset] = last_weights[asset]

        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)

    summary.to_csv(
        figures_dir / f"{scenario_name}_summary.csv",
        index=False,
    )

    for strategy_name, weights_df in weights_history_df.items():
        clean_name = strategy_name.lower().replace(" ", "_").replace(":", "").replace("-", "_")
        weights_df.to_csv(
            figures_dir / f"{scenario_name}_{clean_name}_weights.csv"
        )

    turnover_table = pd.DataFrame(turnover_df)
    turnover_table.to_csv(
        figures_dir / f"{scenario_name}_turnover.csv"
    )

    print("\nRolling out-of-sample performance:")
    display_columns = [
        "portfolio",
        "annual_return",
        "cagr",
        "annual_volatility",
        "sharpe",
        "cumulative_return",
        "max_drawdown",
        "average_monthly_turnover",
        "max_monthly_turnover",
    ]

    print(
        summary[display_columns]
        .assign(
            annual_return=lambda x: x["annual_return"].map(lambda v: f"{v:.2%}"),
            cagr=lambda x: x["cagr"].map(lambda v: f"{v:.2%}"),
            annual_volatility=lambda x: x["annual_volatility"].map(lambda v: f"{v:.2%}"),
            sharpe=lambda x: x["sharpe"].map(lambda v: f"{v:.3f}"),
            cumulative_return=lambda x: x["cumulative_return"].map(lambda v: f"{v:.2%}"),
            max_drawdown=lambda x: x["max_drawdown"].map(lambda v: f"{v:.2%}"),
            average_monthly_turnover=lambda x: x["average_monthly_turnover"].map(lambda v: f"{v:.2%}"),
            max_monthly_turnover=lambda x: x["max_monthly_turnover"].map(lambda v: f"{v:.2%}"),
        )
    )

    return {
        "prices": prices,
        "returns": returns,
        "rf_series": rf_series,
        "rebalance_dates": rebalance_dates,
        "oos_returns": oos_returns,
        "wealth": wealth_df,
        "weights_history": weights_history_df,
        "turnover": turnover_table,
        "summary": summary,
    }





if __name__ == "__main__":
    # Scenario 1: classical multi-asset portfolio over the longest available sample
    classical_tickers = ["SPY", "VEA", "EEM", "IEF", "GLD"]

    # Scenario 2: same classical portfolio over the Bitcoin sample period

    # Scenario 3: same portfolio plus Bitcoin
    bitcoin_tickers = ["SPY", "VEA", "EEM", "IEF", "GLD", "BTC-USD"]

    rf_rate = 0.02
    rf_oos_rate = 0.015
    analysis_end_date = "2026-04-26"

    try:
        BASE_DIR = Path(__file__).resolve().parent
    except NameError:
        BASE_DIR = Path.cwd()

    FIGURES_DIR = BASE_DIR / "figures"
    CLASSICAL_LONG_DIR = FIGURES_DIR / "01_classical_2005_2026"
    COMMON_SAMPLE_DIR = FIGURES_DIR / "02_common_2017_2026"
    OOS_DIR = FIGURES_DIR / "03_out_of_sample"
    OOS_FIXED_DIR = OOS_DIR / "fixed"
    OOS_ROLLING_DIR = OOS_DIR / "rolling"

    for directory in [
        CLASSICAL_LONG_DIR,
        COMMON_SAMPLE_DIR,
        OOS_DIR,
        OOS_FIXED_DIR,
        OOS_ROLLING_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    print("Base directory:", BASE_DIR)
    print("Figures directory:", FIGURES_DIR)
    print("Classical long directory:", CLASSICAL_LONG_DIR)
    print("Common sample directory:", COMMON_SAMPLE_DIR)
    print("Out-of-sample fixed directory:", OOS_FIXED_DIR)
    print("Out-of-sample rolling directory:", OOS_ROLLING_DIR)

    results_classical_long = run_analysis(
        tickers=classical_tickers,
        start_date="2005-01-01",
        end_date=analysis_end_date,
        title="Classical Portfolio - Long Sample",
        rf=rf_rate,
        n_portfolios=50000,
        returns_method="simple",
        frontier_points=120,
        figures_dir=CLASSICAL_LONG_DIR,
    )

    results_classical_common = run_analysis(
        tickers=classical_tickers,
        start_date="2017-01-01",
        end_date=analysis_end_date,
        title="Classical Portfolio - Common Sample",
        rf=rf_rate,
        n_portfolios=50000,
        returns_method="simple",
        frontier_points=120,
        figures_dir=COMMON_SAMPLE_DIR,
    )

    print(
        results_classical_common["optimal_table"]
        [["portfolio", "return", "volatility", "sharpe", "SPY", "GLD", "IEF", "VEA", "EEM"]]
        .round(4)
    )

    results_bitcoin = run_analysis(
        tickers=bitcoin_tickers,
        start_date="2017-01-01",
        end_date=analysis_end_date,
        title="Portfolio Including Bitcoin",
        rf=rf_rate,
        n_portfolios=50000,
        returns_method="simple",
        frontier_points=120,
        figures_dir=COMMON_SAMPLE_DIR,
    )

    results_classical_training = run_analysis(
        tickers=classical_tickers,
        start_date="2010-01-01",
        end_date="2020-01-01",
        title="Classical Portfolio - Training Sample 2010-2019",
        rf=rf_oos_rate,
        n_portfolios=50000,
        returns_method="simple",
        frontier_points=120,
        figures_dir=OOS_FIXED_DIR,
    )

    oos_classical_fixed = run_out_of_sample_test(
        tickers=classical_tickers,
        train_start="2010-01-01",
        train_end="2020-01-01",
        test_start="2020-01-01",
        test_end=analysis_end_date,
        rf=rf_oos_rate,
        returns_method="simple",
        figures_dir=OOS_FIXED_DIR,
        title="Fixed Markowitz OOS 2010-2019 to 2020-2026",
        rebalance_frequency="M",
    )

    oos_classical_rolling = run_rolling_out_of_sample_test(
        tickers=classical_tickers,
        train_start="2010-01-01",
        train_end="2020-01-01",
        test_start="2020-01-01",
        test_end=analysis_end_date,
        lookback_years=10,
        rf=rf_oos_rate,
        returns_method="simple",
        figures_dir=OOS_ROLLING_DIR,
        title="Rolling 10-Year Markowitz OOS 2020-2026",
    )

    rolling_vs_benchmarks = compare_rolling_with_benchmarks(
        rolling_results=oos_classical_rolling,
        tickers=classical_tickers,
        rf=rf_oos_rate,
        figures_dir=OOS_ROLLING_DIR,
        title="Rolling Markowitz vs Benchmarks OOS 2020-2026",
    )

    fixed_vs_rolling_wealth = build_fixed_vs_rolling_wealth(
        fixed_results=oos_classical_fixed,
        rolling_results=oos_classical_rolling,
    )

    plot_cumulative_wealth(
        fixed_vs_rolling_wealth,
        title="Fixed vs Rolling Markowitz Portfolios OOS 2020-2026",
        save_path=OOS_DIR / "fixed_vs_rolling_markowitz_oos_2020_2026.png",
    )

    # Rolling allocation plots
    plot_weight_history(
        oos_classical_rolling["weights_history"]["Rolling Minimum Variance"],
        title="Rolling Minimum-Variance Portfolio Weights",
        save_path=OOS_ROLLING_DIR / "rolling_min_variance_weights.png",
    )

    plot_weight_history(
        oos_classical_rolling["weights_history"]["Rolling Maximum Sharpe"],
        title="Rolling Maximum-Sharpe Portfolio Weights",
        save_path=OOS_ROLLING_DIR / "rolling_max_sharpe_weights.png",
    )

    plot_turnover_history(
        oos_classical_rolling["turnover"],
        title="Monthly Turnover from Rolling Re-Optimization",
        save_path=OOS_ROLLING_DIR / "rolling_markowitz_turnover.png",
    )

    print("\nDate checks:")
    print("Classical long prices start    :", results_classical_long["prices"].index.min())
    print("Classical long returns start   :", results_classical_long["returns"].index.min())
    print("Classical common prices start  :", results_classical_common["prices"].index.min())
    print("Classical common returns start :", results_classical_common["returns"].index.min())
    print("Bitcoin prices start           :", results_bitcoin["prices"].index.min())
    print("Bitcoin returns start          :", results_bitcoin["returns"].index.min())
