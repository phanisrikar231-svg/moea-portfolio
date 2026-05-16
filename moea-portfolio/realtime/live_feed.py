import yfinance as yf
import pandas as pd

NIFTY50_TICKERS = [
    "RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS",
    "HINDUNILVR.NS","WIPRO.NS","AXISBANK.NS","KOTAKBANK.NS","LT.NS",
    "SBIN.NS","BHARTIARTL.NS","BAJFINANCE.NS","ASIANPAINT.NS","MARUTI.NS",
    "TITAN.NS","ULTRACEMCO.NS","NESTLEIND.NS","TECHM.NS","POWERGRID.NS"
]

def fetch_live_prices(tickers=None):
    if tickers is None:
        tickers = NIFTY50_TICKERS[:5]
    data = {}
    for t in tickers:
        try:
            ticker = yf.Ticker(t)
            info = ticker.fast_info
            prev = info.previous_close or 1
            last = info.last_price or prev
            data[t] = {
                "price": round(last, 2),
                "change_pct": round((last - prev) / prev * 100, 2),
            }
        except Exception:
            data[t] = {"price": 0.0, "change_pct": 0.0}
    return data

def fetch_historical(tickers, start, end, interval="1d"):
    raw = yf.download(tickers, start=start, end=end,
                      interval=interval, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].dropna(how="all")
    else:
        close = raw[["Close"]].dropna()
    return close

def get_crisis_periods():
    return {
        "Full Period (2018–2024)": ("2018-01-01", "2024-12-31"),
        "COVID Crash (2020)":      ("2020-01-01", "2020-06-30"),
        "Rate Hike (2022)":        ("2022-01-01", "2022-12-31"),
        "Custom":                   None,
    }
