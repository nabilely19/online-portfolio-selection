"""Rebuild the price files used by the notebooks (daily adjusted close, Yahoo Finance via yfinance).

Usage: python scripts/fetch_data.py
Writes vb_ftrl_restart/data/{normalized_prices,covid_data,SP500_normalized_prices}.csv
"""
from pathlib import Path

import pandas as pd
import yfinance as yf

DATA = Path(__file__).resolve().parent.parent / "vb_ftrl_restart" / "data"
START, END = "2010-01-01", "2025-01-01"


def fetch(name: str) -> None:
    tickers = (DATA / f"tickers_{name}.txt").read_text().split()
    px = yf.download(tickers, start=START, end=END, auto_adjust=True, progress=False)["Close"]
    px = px[tickers]
    px.index.name = "Date"
    px.to_csv(DATA / f"{name}.csv")
    print(f"{name}: {px.shape[0]} days x {px.shape[1]} assets")


if __name__ == "__main__":
    for name in ["normalized_prices", "covid_data", "SP500_normalized_prices"]:
        fetch(name)
