import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from config import DATA_PROCESSED_DIR
from utils import save_csv, write_text_file


P_MAX = 3
Q_MAX = 3


def fit_arima_grid(series: pd.Series, p_max: int = P_MAX, q_max: int = Q_MAX, d: int = 1) -> pd.DataFrame:
    rows = []

    for p in range(p_max + 1):
        for q in range(q_max + 1):
            if p == 0 and q == 0:
                continue

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = ARIMA(series, order=(p, d, q), trend="t")
                    res = model.fit()

                rows.append(
                    {
                        "p": p,
                        "d": d,
                        "q": q,
                        "AIC": res.aic,
                        "BIC": res.bic,
                        "loglik": res.llf,
                        "nobs": int(res.nobs),
                    }
                )
            except Exception:
                continue

    if not rows:
        raise RuntimeError("No ARIMA model could be estimated.")

    return pd.DataFrame(rows).sort_values(["AIC", "BIC"]).reset_index(drop=True)


def build_latex_arima_table(df: pd.DataFrame) -> str:
    latex = df.to_latex(
        index=False,
        float_format="%.4f",
        column_format="lrrrrrrr",
    )

    wrapped = (
        "\\begin{table}[H]\n"
        "\\centering\n"
        "\\caption{Selected ARIMA($p,1,q$) models for daily log prices of gold and silver, 2006--2024}\n"
        "\\label{tab:arima-selection}\n"
        f"{latex}\n"
        "\\vspace{0.2cm}\n"
        "\\begin{minipage}{0.9\\textwidth}\n"
        "\\footnotesize\n"
        "\\textit{Notes}: The table reports selected ARIMA($p,1,q$) specifications for daily log prices "
        "of GLD and SLV. A single unit difference is imposed to reflect the near-unit-root behaviour "
        "of log-price levels documented by the AR(1) benchmark. Selection is performed over a grid with "
        "$p,q \\in \\{0,1,2,3\\}$ using AIC and BIC.\n"
        "\\end{minipage}\n"
        "\\end{table}\n"
    )
    return wrapped


def main():
    levels = pd.read_csv(
        DATA_PROCESSED_DIR / "asset_levels.csv",
        index_col=0,
        parse_dates=True,
    )

    log_levels = pd.DataFrame(
        {
            "gold": np.log(levels["gold"]),
            "silver": np.log(levels["silver"]),
        }
    ).dropna()

    summary_rows = []

    for asset in ["gold", "silver"]:
        series = log_levels[asset]

        grid = fit_arima_grid(series)
        save_csv(grid, f"arima_grid_{asset}.csv", index=False)

        best_aic = grid.loc[grid["AIC"].idxmin()]
        best_bic = grid.loc[grid["BIC"].idxmin()]

        summary_rows.append(
            {
                "Asset": asset.capitalize(),
                "p_AIC": int(best_aic["p"]),
                "d_AIC": int(best_aic["d"]),
                "q_AIC": int(best_aic["q"]),
                "AIC_min": float(best_aic["AIC"]),
                "p_BIC": int(best_bic["p"]),
                "d_BIC": int(best_bic["d"]),
                "q_BIC": int(best_bic["q"]),
                "BIC_min": float(best_bic["BIC"]),
            }
        )

    summary = pd.DataFrame(summary_rows)
    save_csv(summary, "arima_summary_logprices.csv", index=False)

    latex_table = build_latex_arima_table(summary)
    write_text_file(latex_table, "arima_selection_logprices.tex", subdir="tables")

    print("\n=== ARIMA(p,1,q) selection summary ===")
    print(summary)


if __name__ == "__main__":
    main()