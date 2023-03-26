import numpy as np
import pandas as pd
import yfinance as yf
import time

from utils import *
from portfolio_optimization import find_optimal_portfolio

import numba as nb
import time



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


@nb.njit()
def calc_equal_weights(num_assets):
    '''
    This function calculates equally weighted weights for a given number of assets.
    '''
    weights = np.ones(num_assets) / num_assets
    return weights


def expert_portfolio_weight(data: np.array, rolling_windows: np.array, window: int, rho: float) -> np.array:
    '''
    Clean and simple implementation of the CORN algorithm. Supported by numba. Returns weights for each period.
    '''
    ts_length = len(data)
    num_assets = len(data[0])
    weights_array = np.zeros((ts_length, num_assets)) #initialize weights_array

    for i in range(2*window, len(data)): 
        most_recent_window = rolling_windows[i-window]
        most_recent_window_flattened = most_recent_window.reshape(-1, num_assets)

        for j in range(i-window):
            
            previous_window = rolling_windows[j]
            previous_window_flattened = previous_window.reshape(-1, num_assets)

            corr_coeff = calc_corr_coeff(most_recent_window_flattened.flatten(), previous_window_flattened.flatten())

            if corr_coeff > rho: 
                weights = np.ones(num_assets) / 1000
                weights_array[i] = weights
                
                # collect all timeframes in Ct, then loop over them and calculate the weights by passing over the optim function
                
            else:
                # calculate weights for the most recent window by equally weighting all assets
                weights = calc_equal_weights(num_assets)
                weights_array[i] = weights
             
    print(weights_array)
    print(len(weights_array))
    print(weights_array.shape)
    return weights_array
            
        # # print progress every 5%
        # if i % (len(data) // 20) == 0:
        #     print(f"{i / len(data) * 100:.2f}% done")
        
        
        
        
        
        # # Define previous_window_period as the windows before the most recent window
        # previous_window_period = rolling_windows[:i-window]
        # print(most_recent_window.shape, previous_window_period.shape)
        # # Compute the correlation between most_recent_window and previous_window_period
        # #corr_coef = np.corrcoef(most_recent_window, previous_window_period.reshape(-1, len(data[0])).T)
        # correlation = corr_coef[0, 1:]

        # # Add to correlation_array
        # correlation_array[i, :len(correlation)] = correlation

    return weights_array





if __name__ == '__main__':
    
    investment_universe = ['SPY', 'VTI', 'QQQ', 'EFA', 'AGG', 'VWO', 'IJR', 'IJH', 'IWF', 'GLD', 'LQD', 'TLT', 'VNQ', 'IEF', 'SHY', 'DIA', 'VGK', 'VB', 'EXS1.DE', 'CAC.PA']

    get_data(investment_universe)
    log_returns_array = csv_to_numpy(investment_universe)

    start = time.perf_counter()
    
    window = 20
    rho = 0.2
    window_shape = (window, len(log_returns_array[0]))
    rolling_windows = np.lib.stride_tricks.sliding_window_view(log_returns_array, window_shape)

    correlation_array = expert_portfolio_weight(data=log_returns_array, rolling_windows=rolling_windows, window=window, rho=rho)
    print("LOOP DONE")
    end = time.perf_counter()
    print("Elapsed = {}s".format((end - start)))
    
    #Elapsed = 801.9361368920017s #without numba
    #Elapsed = 71.72372195400021s # Factor 11.2 faster :)   fun calc_corr_coeff() implemented with numba, rest not
    
    ###Implemented euqal weight calculations
    #Elapsed = 72.47795482299989s #without numba
    #Elapsed = 50.82744755800013s # Factor 1.4 faster :)   fun calc_corr_coeff() implemented with numba
    
    
    