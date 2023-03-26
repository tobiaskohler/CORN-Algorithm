import numpy as np
import pandas as pd
from scipy.optimize import minimize
from numba import jit



def find_optimal_portfolio(returns: np.array, period: int, list_order: int):

    
    '''
    Long only portfolio optimization using the Sharpe ratio as the objective function.
    '''

    # Set up the optimization problem
    n_assets = returns.shape[1]
    bounds = tuple((0,1) for _ in range(n_assets))
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    x0 = np.ones(n_assets) / n_assets # start with equal weights


    # Define the objective function
    def objective(x, returns):
        portfolio_return = np.sum(returns.mean(axis=0) * x)
        portfolio_risk = np.sqrt(np.dot(x.T, np.dot(np.cov(returns.T), x)))
        sharpe_ratio = portfolio_return / portfolio_risk
        return -sharpe_ratio

    # Solve the optimization problem
    opt = minimize(objective, x0, args=returns, bounds=bounds, constraints=constraints)
    
    # Calculate the Sharpe ratio
    sharpe_ratio = np.sum(returns.mean(axis=0) * opt.x) / np.sqrt(np.dot(opt.x.T, np.dot(np.cov(returns.T), opt.x)))

    print(f"Period: {period}, List order: {list_order}, Sharpe ratio: {sharpe_ratio}")
    # Return the optimal weights and the Sharpe ratio as a tuple
    return (sharpe_ratio, opt.x)


if __name__ == '__main__':
    
    # make seed in order to get the same random numbers

    periods = 20
    weights_per_period = {}
    
    for i in range(0, periods):
        np.random.seed(i)
        returns = np.random.normal(0, 0.01, (20, 20))
    
        weights_per_period[i] = find_optimal_portfolio(returns)