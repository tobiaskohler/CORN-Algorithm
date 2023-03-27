import numpy as np
import pandas as pd
import yfinance as yf
import time

import numba as nb
from scipy.optimize import minimize
import time

import matplotlib.pyplot as plt
import seaborn as sns


def get_data(investment_universe):

    df = yf.download(investment_universe, ignore_tz=True)['Adj Close']

    # Check, that if at least 15x252 trading days are available (15 years * 252 trading days) 
    trading_days = 15*252
    non_null_counts = df.count()

    for symbol in non_null_counts.index:
        if non_null_counts[symbol] < trading_days:
            print(f"Symbol {symbol} has less than {trading_days} non-null values: {non_null_counts[symbol]}")
        else:
            print(f"Ok! {symbol} has at least 15 years of data history ({non_null_counts[symbol]} days in total).")
    
    min_counts = non_null_counts.min()
    min_symbol = non_null_counts[non_null_counts == min_counts].index[0]
    min_start_date = df[df[min_symbol].notnull()].index.min()
    min_end_date = df[df[min_symbol].notnull()].index.max()
    df = df.loc[(df.index >= min_start_date) & (df.index <= min_end_date)]
    df = df.dropna()

    print(f"Shape of DataFrame after pruning: {df.shape}")
    log_returns = np.log(df / df.shift(1)).dropna()
    log_returns.to_csv(f'input/{investment_universe}_log_returns.csv')
   
   
def csv_to_numpy(investment_universe):

    log_returns = pd.read_csv(f'input/{investment_universe}_log_returns.csv')
    log_returns = log_returns.drop(columns=['Date'])
    log_returns = log_returns.to_numpy()
    
    return log_returns


@nb.njit()
def calc_corr_coeff(x, y):
    
    n = len(x)
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    cov = np.sum((x - mean_x) * (y - mean_y))
    std_x = np.sqrt(np.sum((x - mean_x) ** 2))
    std_y = np.sqrt(np.sum((y - mean_y) ** 2))
    corr_coeff = cov / (std_x * std_y)
    return corr_coeff


def csv_to_pd(investment_universe: list):
    
    log_returns_pd = pd.read_csv(f'input/{investment_universe}_log_returns.csv')
    log_returns_pd = log_returns_pd.drop(columns=['Date'])

    return log_returns_pd


def plot_weights(investment_universe: list):
    
    weights = pd.read_csv(f'output/{investment_universe}_weights.csv')
    
    _asset_names = csv_to_pd(investment_universe)
    assets = _asset_names.columns

    weights.columns = assets
    
    plt.style.use('dark_background')
    plt.figure(figsize=(20, 10))
    plt.stackplot(weights.index, weights.T, labels=assets, colors=plt.cm.tab20.colors)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=5, fancybox=True, shadow=True)
    plt.title('Portfolio weights in hindsight')
    plt.xlabel('Trading day')
    plt.ylabel('Weight')
    plt.savefig(f'output/{investment_universe}_weights{len(weights)}.png')


@nb.njit()
def calc_equal_weights(num_assets):

    weights = np.ones(num_assets) / num_assets
    return weights


def benchmarking(weights, log_returns, window_size):
    
    universal_returns = np.sum(log_returns * calc_equal_weights(log_returns.shape[1]), axis=1)
    universal_returns = universal_returns[:-window_size]
    universal_cum_returns = np.exp(np.cumsum(universal_returns, axis=0))
    
    spy_returns = log_returns[:, 0]
    spy_cum_returns = np.exp(np.cumsum(spy_returns, axis=0))
    
    corn_returns = np.sum(log_returns * weights, axis=1)
    corn_cum_returns = np.exp(np.cumsum(corn_returns, axis=0))
    
    # calculate annualized return of corn_returns
    corn_annualized_return = np.mean(corn_returns) * 252
    # #calculate annualized volatility of corn_returns
    corn_vol = np.std(corn_returns) * np.sqrt(252)

    plt.style.use('dark_background')
    plt.figure(figsize=(20, 10))
    plt.plot(universal_cum_returns, label='Benchmark')
    plt.plot(corn_cum_returns, label='Portfolio')
    plt.plot(spy_cum_returns, label='SPY')
    
    textstr = '\n'.join((
        r'Annualized return: %.2f' % (corn_annualized_return, ),
        r'Annualized volatility: %.2f' % (corn_vol, )))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.05, 0.1, textstr, transform=plt.gca().transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
    
    plt.legend(loc='upper left')
    plt.title('Cumulative returns')
    plt.xlabel('Trading day')
    plt.ylabel('Cumulative returns')
    
    plt.savefig(f'output/{investment_universe}_cum_returns{len(weights)}.png')


def optimize_portfolio(returns: np.array, return_target: float = None):
    
    '''
    Long only portfolio optimization that maximizes returns and minimizes risk.
    '''

    n_assets = returns.shape[1]
    bounds = tuple((0,1) for _ in range(n_assets))
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    x0 = np.ones(n_assets) / n_assets # start with equal weights
    
    def objective(x, returns: np.array):
        portfolio_return = np.sum(returns.mean(axis=0) * x)
        portfolio_risk = np.sqrt(np.dot(x.T, np.dot(np.cov(returns.T), x)))
        return -portfolio_return + portfolio_risk

    opt = minimize(objective, x0, args=returns, bounds=bounds, constraints=constraints)
    
    sharpe_ratio = np.sum(returns.mean(axis=0) * opt.x) / np.sqrt(np.dot(opt.x.T, np.dot(np.cov(returns.T), opt.x)))

    return sharpe_ratio, opt.x


def optimize_portfolio_predefined(returns: np.array, return_target: float = None):
    '''
    Long only portfolio optimization that maximizes returns and minimizes risk.
    '''

    # Set up the optimization problem
    n_assets = returns.shape[1]
    bounds = tuple((0,1) for _ in range(n_assets))
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    x0 = np.ones(n_assets) / n_assets # start with equal weights
    
    ## Maximize Return / Minimize Risk ###
    def objective(x, returns: np.array):
        portfolio_return = np.sum(returns.mean(axis=0) * x)
        portfolio_risk = np.sqrt(np.dot(x.T, np.dot(np.cov(returns.T), x)))
        return -portfolio_return + portfolio_risk

    # Solve the optimization problem
    opt = minimize(objective, x0, args=returns, bounds=bounds, constraints=constraints)

    # # Calculate the Sharpe ratio
    # sharpe_ratio = np.sum(returns.mean(axis=0) * opt.x) / np.sqrt(np.dot(opt.x.T, np.dot(np.cov(returns.T), opt.x)))

    return (returns, opt.x)


def expert_portfolio_weight(data: np.array, rolling_windows: np.array, window: int, rho: float, predefined_weights_list: np.array) -> np.array:

    ts_length = len(data)
    num_assets = len(data[0])
    correlation_similiar_set_list = []
    weights_array = np.zeros((ts_length, num_assets))

    
    for i in range(0, (2*window)):
        weights = calc_equal_weights(num_assets)

        weights_array[i] = weights

    for i in range(2*window, len(data)-window): 
        most_recent_window = rolling_windows[i-window]
        most_recent_window_flattened = most_recent_window.reshape(-1, num_assets)
        
        correlation_similiar_set_filled = False

        for j in range(i-window):
            
            previous_window = rolling_windows[j]
            previous_window_flattened = previous_window.reshape(-1, num_assets)

            optim_window = rolling_windows[j+window] # used for calculating weights
            optim_window_flattened = optim_window.reshape(-1, num_assets)
            
            corr_coeff = calc_corr_coeff(most_recent_window_flattened.flatten(), previous_window_flattened.flatten())

            if abs(corr_coeff) > rho: 
                
                correlation_similiar_set_filled = True
                correlation_similiar_set_list.append(optim_window_flattened)

        if correlation_similiar_set_filled:
            
            _weights_list = []

            for elem in correlation_similiar_set_list:

                if any(np.array_equal(elem, tpl[0]) for tpl in predefined_weights_list):

                    # print the found predefined weights
                    for tpl in predefined_weights_list:
                        if np.array_equal(elem, tpl[0]):
                            _weights = tpl[1]
                            _weights_list.append(_weights)

            weights = np.mean(_weights_list, axis=0)
            weights_array[i] = weights

        else:

            weights = calc_equal_weights(num_assets)
            weights_array[i] = weights
            
        print(f'Current iteration: {i}')

    return weights_array


if __name__ == '__main__':
    
    investment_universe = ['SPY', 'VTI', 'QQQ', 'EFA', 'AGG', 'VWO', 'IJR', 'IJH', 'IWF', 'GLD', 'LQD', 'TLT', 'VNQ', 'IEF', 'SHY', 'DIA', 'VGK', 'VB', 'EXS1.DE', 'CAC.PA']

    #get_data(investment_universe)
    
    log_returns_array = csv_to_numpy(investment_universe)

    start = time.perf_counter()
    
    window = 20
    rho = 0.5
    window_shape = (window, len(log_returns_array[0]))
    rolling_windows = np.lib.stride_tricks.sliding_window_view(log_returns_array, window_shape)
    

    predefined_weights_list = []
    
    '''
    predefined_weights_list is necessary, otherwise the algorithm would take too long to run (exponential problem)
    '''
    
    for elem in rolling_windows:
        optim_window = elem.reshape(-1, len(investment_universe))
        _optimal_weight = optimize_portfolio_predefined(optim_window)

        predefined_weights_list.append(_optimal_weight)

    portfolio_weights = expert_portfolio_weight(data=log_returns_array, rolling_windows=rolling_windows, window=window, rho=rho, predefined_weights_list=predefined_weights_list)
    
    end = time.perf_counter()
    print("Elapsed = {}s".format((end - start)))
    
    np.savetxt(f'output/{investment_universe}_weights.csv', portfolio_weights, delimiter=",")
    
    plot_weights(investment_universe=investment_universe)
    benchmarking(weights=portfolio_weights, log_returns=log_returns_array, window_size=window)