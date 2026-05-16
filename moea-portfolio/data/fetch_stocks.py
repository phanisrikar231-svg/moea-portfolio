import yfinance as yf
import pandas as pd
import numpy as np
from realtime.live_feed import NIFTY50_TICKERS

def fetch_and_save(tickers=None, start="2018-01-01", end="2024-12-31"):
    if tickers is None:
        tickers = NIFTY50_TICKERS
    print(f"Downloading {len(tickers)} stocks {start} → {end} ...")
    raw = yf.download(tickers, start=start, end=end,
                      auto_adjust=True, progress=True)
    prices = raw["Close"].dropna(how="all")
    prices.to_csv("data/stocks_data.csv")
    print(f"Saved → data/stocks_data.csv  {prices.shape}")
    return prices

def load_data(path="data/stocks_data.csv"):
    return pd.read_csv(path, index_col=0, parse_dates=True)

def compute_stats(prices):
    returns = prices.pct_change().dropna()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    return returns, mean_returns, cov_matrix

if __name__ == "__main__":
    fetch_and_save()
