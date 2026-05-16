import numpy as np
import pandas as pd

def compute_all_metrics(weights, mean_returns, cov_matrix, returns_df, rf=0.06):
    w = np.array(weights)
    w = w / w.sum()
    ann_ret  = float(np.dot(w, mean_returns) * 252)
    vol      = float(np.sqrt(np.dot(w.T, np.dot(cov_matrix * 252, w))))
    sharpe   = (ann_ret - rf) / vol if vol > 0 else 0.0
    pr       = returns_df.dot(w)
    cum      = (1 + pr).cumprod()
    rm       = cum.cummax()
    max_dd   = float(abs(((cum - rm) / rm).min()))
    calmar   = ann_ret / max_dd if max_dd > 0 else 0.0
    return {
        "Annual Return (%)": round(ann_ret * 100, 2),
        "Volatility (%)":    round(vol * 100, 2),
        "Sharpe Ratio":      round(sharpe, 3),
        "Max Drawdown (%)":  round(max_dd * 100, 2),
        "Calmar Ratio":      round(calmar, 3),
    }

def backtest_portfolio(weights, prices_df):
    w = np.array(weights)
    w = w / w.sum()
    returns = prices_df.pct_change().dropna()
    port_ret = returns.dot(w)
    return (1 + port_ret).cumprod() * 100
