import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from arch import arch_model

from config import DATA_PROCESSED_DIR, OUTPUT_DIR
from utils import save_csv, save_figure, write_text_file

def fit_garch_11(series: pd.Series):
    model = arch_model(
        series,
        mean="constant",
        vol="GARCH",
        p=1,
        q=1,
        dist="normal"
    )
    result = model.fit(update_freq=50, disp="off")
    return result


def garch_params_summary(res, name: str):
    params = res.params
    se = res.std_err

    mu = params["mu"]
    omega = params["omega"]
    alpha = params["alpha[1]"]
    beta = params["beta[1]"]
    ab_sum = alpha + beta

    sigma2_inf = np.nan
    if ab_sum < 1:
        sigma2_inf = omega / (1 - ab_sum)

    return {
        "asset": name,
        "mu": mu,
        "se_mu": se["mu"],
        "omega": omega,
        "se_omega": se["omega"],
        "alpha": alpha,
        "se_alpha": se["alpha[1]"],
        "beta": beta,
        "se_beta": se["beta[1]"],
        "alpha_plus_beta": ab_sum,
        "sigma2_inf": sigma2_inf,
        "aic": res.aic,
        "bic": res.bic,
        "loglikelihood": res.loglikelihood,
        "nobs": int(res.nobs),
    }


def build_latex_garch_table(df: pd.DataFrame) -> str:
    display = df.copy()

    for col in ["mu", "omega", "alpha", "beta", "alpha_plus_beta", "sigma2_inf"]:
        display[col] = display[col].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")

    latex = (
        "\\begin{table}[H]\n"
        "\\centering\n"
        "\\caption{GARCH(1,1) estimates for gold and silver daily returns}\n"
        "\\label{tab:garch-params}\n"
        "\\begin{tabular}{lcccccc}\n"
        "\\toprule\n"
        "Asset & $\\hat{\\mu}$ & $\\hat{\\omega}$ & $\\hat{\\alpha}$ & $\\hat{\\beta}$ & "
        "$\\hat{\\alpha}+\\hat{\\beta}$ & $\\hat{\\sigma}^2_{\\infty}$ \\\\\n"
        "\\midrule\n"
    )

    for _, row in display.iterrows():
        latex += (
            f"{row['asset']} & {row['mu']} & {row['omega']} & {row['alpha']} & "
            f"{row['beta']} & {row['alpha_plus_beta']} & {row['sigma2_inf']} \\\\\n"
        )

    latex += (
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\vspace{0.2cm}\n"
        "\\begin{minipage}{0.9\\textwidth}\n"
        "\\footnotesize\n"
        "\\textit{Notes}: Returns are expressed in percentage points. "
        "$\\hat{\\sigma}^2_{\\infty} = \\hat{\\omega}/(1-\\hat{\\alpha}-\\hat{\\beta})$ "
        "when $\\hat{\\alpha}+\\hat{\\beta}<1$.\n"
        "\\end{minipage}\n"
        "\\end{table}\n"
    )

    return latex


def save_returns_and_vol_plot(
    returns_series,
    cond_vol_series,
    asset_name: str,
    filename: str,
    n_obs: int = 1000
):
    color_map = {
        "Gold": "tab:orange",
        "Silver": "tab:green",
    }

    color = color_map.get(asset_name, "tab:blue")

    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    returns_series.tail(n_obs).plot(ax=ax[0], color="tab:blue", linewidth=0.8)
    ax[0].set_title(f"{asset_name} daily returns (in %)")
    ax[0].set_ylabel("Return")

    cond_vol_series.tail(n_obs).plot(ax=ax[1], color=color, linewidth=1.0)
    ax[1].set_title(f"{asset_name} GARCH(1,1) conditional volatility")
    ax[1].set_ylabel("Volatility (std. dev., %)")

    fig.tight_layout()
    save_figure(fig, filename)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    returns = pd.read_csv(DATA_PROCESSED_DIR / "asset_returns.csv", index_col=0, parse_dates=True)

    ret_gold = returns["gold"].dropna() * 100
    ret_silver = returns["silver"].dropna() * 100

    gold_desc = ret_gold.describe().to_frame(name="gold")
    silver_desc = ret_silver.describe().to_frame(name="silver")
    desc = pd.concat([gold_desc, silver_desc], axis=1)
    save_csv(desc, "garch_input_returns_describe.csv")

    res_gold = fit_garch_11(ret_gold)
    res_silver = fit_garch_11(ret_silver)

    print(res_gold.summary())
    print(res_silver.summary())

    summary_rows = [
        garch_params_summary(res_gold, "Gold"),
        garch_params_summary(res_silver, "Silver"),
    ]
    params_df = pd.DataFrame(summary_rows)

    save_csv(params_df, "garch_params.csv", index=False)

    latex_table = build_latex_garch_table(params_df[[
        "asset", "mu", "omega", "alpha", "beta", "alpha_plus_beta", "sigma2_inf"
    ]])

    write_text_file(latex_table, "garch_params.tex", subdir="tables")

    cond_vol_gold = res_gold.conditional_volatility
    cond_vol_silver = res_silver.conditional_volatility

    cond_vol_df = pd.DataFrame({
        "gold_cond_vol": cond_vol_gold,
        "silver_cond_vol": cond_vol_silver,
    })
    save_csv(cond_vol_df, "garch_conditional_volatility.csv")

    horizon = 10
    forecasts_gold = res_gold.forecast(horizon=horizon)
    forecasts_silver = res_silver.forecast(horizon=horizon)

    fvar_gold = forecasts_gold.variance.iloc[-1].values
    fvar_silver = forecasts_silver.variance.iloc[-1].values

    forecast_df = pd.DataFrame({
        "horizon": np.arange(1, horizon + 1),
        "gold_variance_forecast": fvar_gold,
        "silver_variance_forecast": fvar_silver,
    })
    save_csv(forecast_df, "garch_variance_forecasts.csv", index=False)

    print("1- to 10-step ahead variance forecasts for gold:", fvar_gold)
    print("1- to 10-step ahead variance forecasts for silver:", fvar_silver)

    save_returns_and_vol_plot(
        ret_gold,
        cond_vol_gold,
        asset_name="Gold",
        filename="garch_condvol_gold.png",
        n_obs=1000
    )

    save_returns_and_vol_plot(
        ret_silver,
        cond_vol_silver,
        asset_name="Silver",
        filename="garch_condvol_silver.png",
        n_obs=1000
    )


if __name__ == "__main__":
    main()