import numpy as np
import pandas as pd

from config import DATA_PROCESSED_DIR, DATA_RAW_DIR, FINAL_SAMPLE_END, FINAL_SAMPLE_START, TICKERS


def load_raw():
    return pd.read_csv(
        DATA_RAW_DIR / "yahoo_raw_download.csv",
        header=[0, 1],
        index_col=0,
        parse_dates=True
    )


def main():
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_raw()

    adj_close = raw["Adj Close"].copy()
    close = raw["Close"].copy()

    df = pd.DataFrame(index=raw.index)

    df["gold"] = adj_close[TICKERS["gold"]]
    df["silver"] = adj_close[TICKERS["silver"]]
    df["sp500"] = adj_close[TICKERS["sp500"]]
    df["dxy"] = adj_close[TICKERS["dxy"]]
    df["ust10y_yield"] = close[TICKERS["ust10y"]]

    print("\nDataFrame unificato (con NaN):")
    print(df.head(10))
    print("\nMissing values per colonna:")
    print(df.isna().sum())

    df.to_csv(DATA_RAW_DIR / "asset_levels_with_nans.csv")

    df_full = df.dropna().copy()

    print("\nPrima data con tutti i dati disponibili:", df_full.index.min())
    print("Ultima data:", df_full.index.max())
    print("Shape prima dropna totale:", df.shape)
    print("Shape dopo dropna totale:", df_full.shape)

    df_def = df_full[FINAL_SAMPLE_START:FINAL_SAMPLE_END].copy()

    print("\nPeriodo effettivo usato:")
    print("Start:", df_def.index.min())
    print("End: ", df_def.index.max())
    print("Shape df_def:", df_def.shape)

    price_cols = ["gold", "silver", "sp500", "dxy"]
    log_returns = np.log(df_def[price_cols] / df_def[price_cols].shift(1))

    yield_change = df_def["ust10y_yield"].diff()

    returns = log_returns.copy()
    returns["ust10y_change"] = yield_change
    returns = returns.dropna()

    print("\nReturns head:")
    print(returns.head())
    print("\nReturns describe:")
    print(returns.describe())

    df_def.to_csv(DATA_PROCESSED_DIR / "asset_levels.csv")
    returns.to_csv(DATA_PROCESSED_DIR / "asset_returns.csv")


if __name__ == "__main__":
    main()