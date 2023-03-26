import numpy as np
import pandas as pd
import yfinance as yf
import time

from utils import *
from portfolio_optimization import find_optimal_portfolio

import numba as nb
from scipy.optimize import minimize
import time

import matplotlib.pyplot as plt
import seaborn as sns



def get_data(investment_universe):
    '''
    This function downloads the historical data for the given tickers, cleans it, calculates log_returns and then saves as .csv file.
    '''
    num_assets = len(investment_universe)

    df = yf.download(investment_universe, ignore_tz=True)['Adj Close']

    # Check if at least 15x252 trading days are available (15 years * 252 trading days) 
    trading_days = 15*252
    non_null_counts = df.count()

    for symbol in non_null_counts.index:
        if non_null_counts[symbol] < trading_days:
            print(f"Symbol {symbol} has less than {trading_days} non-null values: {non_null_counts[symbol]}")
        else:
            print(f"Ok! {symbol} has at least 15 years of data history ({non_null_counts[symbol]} days in total).")
    
    # Prune whole df to match ETF with shortest history

    min_counts = non_null_counts.min()
    min_symbol = non_null_counts[non_null_counts == min_counts].index[0]
    min_start_date = df[df[min_symbol].notnull()].index.min()
    min_end_date = df[df[min_symbol].notnull()].index.max()
    df = df.loc[(df.index >= min_start_date) & (df.index <= min_end_date)]
    df = df.dropna()
    #df = df.reset_index()
    #df = df.drop(columns=['index'])
    print(f"Shape of DataFrame after pruning: {df.shape}")


    log_returns = np.log(df / df.shift(1)).dropna()
    
    #save log_returns to .csv file, subfolder input, concatenate with investment universe in filename
    log_returns.to_csv(f'input/{investment_universe}_log_returns.csv')
   
   
def csv_to_numpy(investment_universe):
    '''
    This function reads the log_returns from the .csv file and converts it to a numpy array.
    '''
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
    '''
    This function reads the log_returns from the .csv file and converts it to a pandas dataframe.
    '''
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
    
    # add legend below the plot
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=5, fancybox=True, shadow=True)
    plt.title('Portfolio weights in hindsight')
    plt.xlabel('Trading day')
    plt.ylabel('Weight')
    plt.show()
    
    #print rows for which the sum of weights over columns is not 1
    print(weights[weights.sum(axis=1) != 1])
    

@nb.njit()
def calc_equal_weights(num_assets):
    '''
    This function calculates equally weighted weights for a given number of assets.
    '''
    weights = np.ones(num_assets) / num_assets
    return weights

def find_optimal_portfolio(returns: np.array, return_target: float = None):
    '''
    Long only portfolio optimization that maximizes returns and minimizes risk.
    '''

    # Set up the optimization problem
    n_assets = returns.shape[1]
    bounds = tuple((0,1) for _ in range(n_assets))
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    x0 = np.ones(n_assets) / n_assets # start with equal weights

    # Define the objective function to maximize returns and minimize risk
    def objective(x, returns):
        portfolio_return = np.sum(returns.mean(axis=0) * x)
        portfolio_risk = np.sqrt(np.dot(x.T, np.dot(np.cov(returns.T), x)))
        return -portfolio_return + portfolio_risk

    # Solve the optimization problem
    opt = minimize(objective, x0, args=returns, bounds=bounds, constraints=constraints)
    
    # Calculate the Sharpe ratio
    sharpe_ratio = np.sum(returns.mean(axis=0) * opt.x) / np.sqrt(np.dot(opt.x.T, np.dot(np.cov(returns.T), opt.x)))

    # Return the optimal weights and the Sharpe ratio as a tuple
    return (sharpe_ratio, opt.x)


def expert_portfolio_weight(data: np.array, rolling_windows: np.array, window: int, rho: float) -> np.array:
    '''
    Clean and simple implementation of the CORN algorithm. Supported by numba. Returns weights for each period.
    '''
    ts_length = len(data)
    num_assets = len(data[0])
    correlation_similiar_set_list = [] #initialize correlation_similiar_set_array
    weights_array = np.zeros((ts_length, num_assets)) #initialize weights_array

    
    for i in range(0, (2*window)):
        weights = calc_equal_weights(num_assets)

        weights_array[i] = weights

    for i in range(2*window, len(data)): 
        most_recent_window = rolling_windows[i-window]
        most_recent_window_flattened = most_recent_window.reshape(-1, num_assets)
        
        correlation_similiar_set_filled = False

        for j in range(i-window):
            
            previous_window = rolling_windows[j]
            previous_window_flattened = previous_window.reshape(-1, num_assets)

            corr_coeff = calc_corr_coeff(most_recent_window_flattened.flatten(), previous_window_flattened.flatten())

            if abs(corr_coeff) > rho: 
                
                correlation_similiar_set_filled = True
                correlation_similiar_set_list.append(previous_window_flattened)

        if correlation_similiar_set_filled:
            _weights_list = []
            
            # 1. loop over correlation_similiar_set_list and pass over to portfolio_optim function
            # 2. function returns a dictionary with sharpe-ratio and weights
            # 3. select top k sharpe-ratios and calculate the average weights
            # 4. save weights in weights_array at index iq
            for elem in correlation_similiar_set_list:
                _weights = find_optimal_portfolio(elem)
                _weights_list.append(_weights)
                
            # calculate average weights
            weights = np.mean(_weights_list, axis=0)

            weights_array[i] = weights[1]

        else:
            # calculate weights for the most recent window by equally weighting all assets
            weights = calc_equal_weights(num_assets)
            weights_array[i] = weights

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

    portfolio_weights = expert_portfolio_weight(data=log_returns_array, rolling_windows=rolling_windows, window=window, rho=rho)
    
    end = time.perf_counter()
    print("Elapsed = {}s".format((end - start)))
    
    # save weights to .csv file, subfolder output, concatenate with investment universe in filename
    np.savetxt(f'output/{investment_universe}_weights.csv', portfolio_weights, delimiter=",")
    
    plot_weights(investment_universe=investment_universe)
    
    
    #Elapsed = 801.9361368920017s #without numba
    #Elapsed = 71.72372195400021s # Factor 11.2 faster :)   fun calc_corr_coeff() implemented with numba, rest not
    
    ###Implemented equal weight calculations
    #Elapsed = 72.47795482299989s #without numba
    #Elapsed = 50.82744755800013s # Factor 1.4 faster :)   fun calc_corr_coeff() implemented with numba
    
    
    