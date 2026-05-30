from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "output"

TICKERS = {
    "gold": "GLD",
    "silver": "SLV",
    "sp500": "^GSPC",
    "ust10y": "^TNX",
    "dxy": "DX-Y.NYB",
}

START_DATE = "2000-01-01"
END_DATE = "2024-12-31"
FINAL_SAMPLE_START = "2006-05-01"
FINAL_SAMPLE_END = "2024-12-31"
ROLLING_WINDOW = 252

FORMAL_NAMES = {
    "gold": "Gold (GLD)",
    "silver": "Silver (SLV)",
    "sp500": "S\\&P 500",
    "dxy": "U.S. dollar index",
    "ust10y_change": "10Y yield change",
    "ust10y_yield": "10Y Treasury yield",
}