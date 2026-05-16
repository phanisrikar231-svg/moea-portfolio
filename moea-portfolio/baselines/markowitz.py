import numpy as np
from scipy.optimize import minimize
from engine.objectives import portfolio_volatility

def markowitz_min_variance(mean_returns, cov_matrix):
    n = len(mean_returns)
    w0 = np.ones(n) / n
    result = minimize(
        portfolio_volatility, w0,
        args=(cov_matrix,),
        method="SLSQP",
        bounds=[(0, 1)] * n,
        constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1},
    )
    return result.x if result.success else w0
