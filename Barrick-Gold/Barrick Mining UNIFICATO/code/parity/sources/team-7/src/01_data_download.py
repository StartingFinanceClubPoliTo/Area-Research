import yfinance as yf

from config import DATA_RAW_DIR, END_DATE, START_DATE, TICKERS


def main():
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)

    raw = yf.download(
        list(TICKERS.values()),
        start=START_DATE,
        end=END_DATE,
        auto_adjust=False,
        progress=False,
    )

    print("Raw head:\n", raw.head())
    print("\nRaw columns:\n", raw.columns)

    raw.to_csv(DATA_RAW_DIR / "yahoo_raw_download.csv")


if __name__ == "__main__":
    main()