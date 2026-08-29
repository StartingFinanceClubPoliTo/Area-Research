import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import skew, kurtosis
from statsmodels.tsa.stattools import acf

from config import DATA_PROCESSED_DIR, OUTPUT_DIR
from utils import save_csv

def ar1_results(series: pd.Series):
    y = series.dropna()
    y_lag = y.shift(1).dropna()
    y = y.loc[y_lag.index]

    X = sm.add_constant(y_lag)
    model = sm.OLS(y, X).fit(cov_type="HC1")

    phi = model.params.iloc[1]
    se_phi = model.bse.iloc[1]
    t_phi = model.tvalues.iloc[1]
    r2 = model.rsquared

    return phi, se_phi, t_phi, r2


def main():
    levels = pd.read_csv(DATA_PROCESSED_DIR / "asset_levels.csv", index_col=0, parse_dates=True)
    returns = pd.read_csv(DATA_PROCESSED_DIR / "asset_returns.csv", index_col=0, parse_dates=True)

    log_gold = np.log(levels["gold"])
    log_silver = np.log(levels["silver"])
    gold_ret = returns["gold"]
    silver_ret = returns["silver"]

    gold_price_res = ar1_results(log_gold)
    silver_price_res = ar1_results(log_silver)
    gold_ret_res = ar1_results(gold_ret)
    silver_ret_res = ar1_results(silver_ret)

    ar_price = pd.DataFrame(
        {
            "phi": [gold_price_res[0], silver_price_res[0]],
            "se_phi": [gold_price_res[1], silver_price_res[1]],
            "t_phi": [gold_price_res[2], silver_price_res[2]],
            "R2": [gold_price_res[3], silver_price_res[3]],
        },
        index=["Gold (log price)", "Silver (log price)"],
    )

    ar_return = pd.DataFrame(
        {
            "phi": [gold_ret_res[0], silver_ret_res[0]],
            "se_phi": [gold_ret_res[1], silver_ret_res[1]],
            "t_phi": [gold_ret_res[2], silver_ret_res[2]],
            "R2": [gold_ret_res[3], silver_ret_res[3]],
        },
        index=["Gold (return)", "Silver (return)"],
    )

    print("AR(1) on log prices:")
    print(ar_price)

    print("\nAR(1) on log returns:")
    print(ar_return)

    stylized = pd.DataFrame(
        {
            "mean": returns[["gold", "silver"]].mean(),
            "std": returns[["gold", "silver"]].std(),
            "skewness": returns[["gold", "silver"]].apply(skew),
            "excess_kurtosis": returns[["gold", "silver"]].apply(lambda x: kurtosis(x, fisher=True)),
            "acf_abs_ret_1": returns[["gold", "silver"]].apply(lambda x: acf(np.abs(x), nlags=5, fft=False)[1]),
            "acf_sq_ret_1": returns[["gold", "silver"]].apply(lambda x: acf((x ** 2), nlags=5, fft=False)[1]),
        }
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_csv(stylized, "stylized_facts_summary.csv")
    save_csv(ar_price, "ar1_log_prices.csv")
    save_csv(ar_return, "ar1_returns.csv")


if __name__ == "__main__":
    main()