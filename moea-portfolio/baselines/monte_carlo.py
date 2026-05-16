import numpy as np
from engine.objectives import portfolio_return, portfolio_volatility

def monte_carlo_best(mean_returns, cov_matrix, n=10000, rf=0.06):
    n_assets = len(mean_returns)
    best_sharpe, best_w = -np.inf, None
    for _ in range(n):
        w = np.random.dirichlet(np.ones(n_assets))
        r = portfolio_return(w, mean_returns)
        v = portfolio_volatility(w, cov_matrix)
        s = (r - rf) / v if v > 0 else 0
        if s > best_sharpe:
            best_sharpe, best_w = s, w
    return best_w
