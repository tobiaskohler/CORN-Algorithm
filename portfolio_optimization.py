import numpy as np
from scipy.optimize import minimize
import pandas as pd
import plotly.graph_objs as go




def find_optimal_portfolio(returns: pd.DataFrame):
    
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

    return opt.x


if __name__ == '__main__':
    
    # make seed in order to get the same random numbers

    periods = 20
    weights_per_period = {}
    
    for i in range(0, periods):
        np.random.seed(i)
        returns = np.random.normal(0, 0.01, (20, 20))
    
        weights_per_period[i] = find_optimal_portfolio(returns)
        
    df = pd.DataFrame(weights_per_period)
    df = df.T

    # plot using Plotly
    fig = go.Figure(data=[go.Bar(x=df.index, y=df[col], name=col) for col in df.columns])
    fig.update_layout(title='Weights per period', xaxis_title='Period', yaxis_title='Weights', barmode='stack', template='plotly_dark')
    fig.show()