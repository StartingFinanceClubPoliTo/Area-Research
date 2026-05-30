import pandas as pd

from config import DATA_PROCESSED_DIR, OUTPUT_DIR
from utils import run_ols_formula


DISPLAY_NAMES = {
    "Intercept": "Constant",
    "ust10y_change": r"$\Delta y^{10Y}_t$",
    "sp500": r"$r^{equity}_t$",
    "dxy": r"$r^{usd}_t$",
}

REGRESSOR_ORDER = ["dxy", "ust10y_change", "sp500", "Intercept"]


def significance_stars(pval: float) -> str:
    if pval < 0.01:
        return "***"
    if pval < 0.05:
        return "**"
    if pval < 0.10:
        return "*"
    return ""


def format_coef(model, var_name: str) -> str:
    if var_name not in model.params.index:
        return ""
    coef = model.params[var_name]
    pval = model.pvalues[var_name]
    return f"{coef:.4f}{significance_stars(pval)}"


def format_se(model, var_name: str) -> str:
    if var_name not in model.bse.index:
        return ""
    se = model.bse[var_name]
    return f"({se:.4f})"


def build_regression_table(models, model_labels):
    rows = []

    for var in REGRESSOR_ORDER:
        label = DISPLAY_NAMES[var]
        coef_row = [label]
        se_row = [""]

        statsmodels_name = "const" if var == "Intercept" else var

        for model in models:
            coef_row.append(format_coef(model, statsmodels_name))
            se_row.append(format_se(model, statsmodels_name))

        rows.append(coef_row)
        rows.append(se_row)

    rows.append(["Observations"] + [f"{int(model.nobs)}" for model in models])
    rows.append([r"$R^2$"] + [f"{model.rsquared:.3f}" for model in models])
    rows.append(["Adjusted $R^2$"] + [f"{model.rsquared_adj:.3f}" for model in models])

    table = pd.DataFrame(rows, columns=[""] + model_labels)
    return table


def export_table_latex(table, filename, caption, label):
    latex_str = table.to_latex(
        index=False,
        escape=False,
        column_format="l" + "c" * (table.shape[1] - 1)
    )

    wrapped = (
        "\\begin{table}[H]\n"
        "\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        f"{latex_str}\n"
        "\\vspace{0.2cm}\n"
        "\\begin{minipage}{0.9\\textwidth}\n"
        "\\footnotesize\n"
        "\\textit{Notes}: Daily data. Heteroskedasticity--robust (HC1) standard errors in parentheses.\\\\\n"
        "$^*$ $p<0.10$, $^{**}$ $p<0.05$, $^{***}$ $p<0.01$.\n"
        "\\end{minipage}\n"
        "\\end{table}\n"
    )

    with open(OUTPUT_DIR / filename, "w", encoding="utf-8") as f:
        f.write(wrapped)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    returns = pd.read_csv(DATA_PROCESSED_DIR / "asset_returns.csv", index_col=0, parse_dates=True)
    data = returns.dropna().copy()

    gold_bond_models = [
        run_ols_formula("gold ~ ust10y_change", data),
        run_ols_formula("gold ~ ust10y_change + sp500 + dxy", data),
    ]
    gold_equity_models = [
        run_ols_formula("gold ~ sp500", data),
        run_ols_formula("gold ~ sp500 + ust10y_change + dxy", data),
    ]
    gold_dollar_models = [
        run_ols_formula("gold ~ dxy", data),
        run_ols_formula("gold ~ dxy + ust10y_change + sp500", data),
    ]

    silver_bond_models = [
        run_ols_formula("silver ~ ust10y_change", data),
        run_ols_formula("silver ~ ust10y_change + sp500 + dxy", data),
    ]
    silver_equity_models = [
        run_ols_formula("silver ~ sp500", data),
        run_ols_formula("silver ~ sp500 + ust10y_change + dxy", data),
    ]
    silver_dollar_models = [
        run_ols_formula("silver ~ dxy", data),
        run_ols_formula("silver ~ dxy + ust10y_change + sp500", data),
    ]

    gold_bond_table = build_regression_table(gold_bond_models, ["(1)", "(2)"])
    gold_equity_table = build_regression_table(gold_equity_models, ["(1)", "(2)"])
    gold_dollar_table = build_regression_table(gold_dollar_models, ["(1)", "(2)"])

    silver_bond_table = build_regression_table(silver_bond_models, ["(1)", "(2)"])
    silver_equity_table = build_regression_table(silver_equity_models, ["(1)", "(2)"])
    silver_dollar_table = build_regression_table(silver_dollar_models, ["(1)", "(2)"])

    gold_bond_table.to_csv(OUTPUT_DIR / "gold_bond_regressions.csv", index=False)
    gold_equity_table.to_csv(OUTPUT_DIR / "gold_equity_regressions.csv", index=False)
    gold_dollar_table.to_csv(OUTPUT_DIR / "gold_dollar_regressions.csv", index=False)

    silver_bond_table.to_csv(OUTPUT_DIR / "silver_bond_regressions.csv", index=False)
    silver_equity_table.to_csv(OUTPUT_DIR / "silver_equity_regressions.csv", index=False)
    silver_dollar_table.to_csv(OUTPUT_DIR / "silver_dollar_regressions.csv", index=False)

    export_table_latex(
        gold_bond_table,
        "gold_bond_regressions.tex",
        "OLS regressions of daily gold returns on bond, dollar, and equity variables",
        "tab:ols_gold_bond",
    )
    export_table_latex(
        gold_equity_table,
        "gold_equity_regressions.tex",
        "OLS regressions of daily gold returns on equity, bond, and dollar variables",
        "tab:ols_gold_equity",
    )
    export_table_latex(
        gold_dollar_table,
        "gold_dollar_regressions.tex",
        "OLS regressions of daily gold returns on dollar, bond, and equity variables",
        "tab:ols_gold_dollar",
    )
    export_table_latex(
        silver_bond_table,
        "silver_bond_regressions.tex",
        "OLS regressions of daily silver returns on bond, dollar, and equity variables",
        "tab:ols_silver_bond",
    )
    export_table_latex(
        silver_equity_table,
        "silver_equity_regressions.tex",
        "OLS regressions of daily silver returns on equity, bond, and dollar variables",
        "tab:ols_silver_equity",
    )
    export_table_latex(
        silver_dollar_table,
        "silver_dollar_regressions.tex",
        "OLS regressions of daily silver returns on dollar, bond, and equity variables",
        "tab:ols_silver_dollar",
    )


if __name__ == "__main__":
    main()