import numpy as np

def portfolio_return(weights, mean_returns):
    return float(np.dot(weights, mean_returns) * 252)

def portfolio_volatility(weights, cov_matrix):
    return float(np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights))))

def max_drawdown(weights, returns_df):
    port_ret = returns_df.dot(weights)
    cumulative = (1 + port_ret).cumprod()
    rolling_max = cumulative.cummax()
    dd = (cumulative - rolling_max) / rolling_max
    return float(abs(dd.min()))

def sharpe_ratio(weights, mean_returns, cov_matrix, rf=0.06):
    r = portfolio_return(weights, mean_returns)
    v = portfolio_volatility(weights, cov_matrix)
    return (r - rf) / v if v > 0 else 0.0

def evaluate(individual, mean_returns, cov_matrix, returns_df):
    w = np.array(individual)
    s = w.sum()
    if s == 0:
        w = np.ones(len(w)) / len(w)
    else:
        w = w / s
    f1 = -portfolio_return(w, mean_returns)
    f2 =  portfolio_volatility(w, cov_matrix)
    f3 =  max_drawdown(w, returns_df)
    return f1, f2, f3
