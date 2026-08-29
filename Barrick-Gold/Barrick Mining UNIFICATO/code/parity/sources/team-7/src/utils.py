from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from config import (
    FORMAL_NAMES,
    OUTPUT_DIR,
    OUTPUT_FIGURES_DIR,
    OUTPUT_CSV_DIR,
    OUTPUT_TABLES_DIR,
)

plt.style.use("seaborn-v0_8-whitegrid")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)


def ensure_directories():
    for path in [OUTPUT_DIR, OUTPUT_FIGURES_DIR, OUTPUT_CSV_DIR, OUTPUT_TABLES_DIR]:
        Path(path).mkdir(parents=True, exist_ok=True)


def save_figure(fig, filename: str, dpi: int = 300):
    ensure_directories()
    fig.tight_layout()
    fig.savefig(OUTPUT_FIGURES_DIR / filename, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_csv(df: pd.DataFrame, filename: str, index: bool = True):
    ensure_directories()
    df.to_csv(OUTPUT_CSV_DIR / filename, index=index)


def export_latex_table(df: pd.DataFrame, filename: str, float_format: str, column_format: str):
    ensure_directories()
    df.to_latex(OUTPUT_TABLES_DIR / filename, float_format=float_format, column_format=column_format)


def write_text_file(content: str, filename: str, subdir: str = "tables"):
    ensure_directories()
    if subdir == "tables":
        path = OUTPUT_TABLES_DIR / filename
    elif subdir == "csv":
        path = OUTPUT_CSV_DIR / filename
    elif subdir == "figures":
        path = OUTPUT_FIGURES_DIR / filename
    else:
        path = OUTPUT_DIR / filename

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def add_formal_labels(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(index=FORMAL_NAMES, columns=FORMAL_NAMES)


def run_ols_formula(formula: str, data: pd.DataFrame):
    return smf.ols(formula=formula, data=data).fit(cov_type="HC1")


def run_ols_matrix(y: pd.Series, X: pd.DataFrame):
    X = sm.add_constant(X)
    return sm.OLS(y, X).fit(cov_type="HC1")