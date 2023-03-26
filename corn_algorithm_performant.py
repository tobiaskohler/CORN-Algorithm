import numpy as np
import pandas as pd
import yfinance as yf
import time

from utils import *
from portfolio_optimization import find_optimal_portfolio


from numba import jit
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

#@njit(nb.float64[:, :](nb.float64[:, :], nb.int64, nb.float64))
def expert_portfolio_weight(data: np.array, window: int, rho: float) -> np.array:
    '''
    Clean and simple implementation of the CORN algorithm. Supported by numba. Returns weights for each period.
    '''

    window_shape = (window, len(data[0]))
    print(window_shape)
    correlation_array = np.zeros((len(data), len(data[0]))) #initialize correlation array
    rolling_windows = np.lib.stride_tricks.sliding_window_view(data, window_shape)

    for i in range(2*window, len(data)): 
        most_recent_window = rolling_windows[i-window]
        most_recent_window_flattened = most_recent_window.reshape(-1, len(data[0]))

        for j in range(i-window):
            
            previous_window = rolling_windows[j]
            previous_window_flattened = previous_window.reshape(-1, len(data[0]))
            
            corr_coeff = np.corrcoef(most_recent_window_flattened.flatten(), previous_window_flattened.flatten())[0, 1]

            if abs(corr_coeff) > rho:
                #add to correlation-similiar-set called C_t
                
                
            
    # print progress every 5%
    if i % (len(data) // 20) == 0:
        print(f"{i / len(data) * 100:.2f}% done")
        
        
        
        
        
        # # Define previous_window_period as the windows before the most recent window
        # previous_window_period = rolling_windows[:i-window]
        # print(most_recent_window.shape, previous_window_period.shape)
        # # Compute the correlation between most_recent_window and previous_window_period
        # #corr_coef = np.corrcoef(most_recent_window, previous_window_period.reshape(-1, len(data[0])).T)
        # correlation = corr_coef[0, 1:]

        # # Add to correlation_array
        # correlation_array[i, :len(correlation)] = correlation

    return correlation_array





    # for i in range(2*window, len(data)):
    #     # define most_recent_window as the last window in the rolling windows
    #     most_recent_window = rolling_windows[i-window].reshape(window, len(data[0]))

    #     # define previous_window_period as the windows before the most recent window
    #     previous_window_period = rolling_windows[:i-window].reshape(i-window, window*len(data[0]))

    #     # compute the correlation between most_recent_window and previous_window_period
    #     correlation = np.corrcoef(most_recent_window, previous_window_period.T)[0,1:]
    #     print(correlation)
        
    #     # add to correlation_array
    #     #correlation_array[i, :len(correlation)] = correlation

    return correlation_array

        
        # Experiment with numba: it took forever...i dont know why. 
        
#         The CORrelation-driven Nonlocal means (CORN) algorithm involves calculating the correlation between a sliding window and a reference window at every time step in a time series. This process requires accessing overlapping subsets of the time series, which makes it challenging to vectorize in a straightforward way.

# However, it is possible to use numpy functions to efficiently compute the correlation between the sliding and reference windows without explicitly looping through the time series. The numpy.correlate() function can be used to calculate the cross-correlation between two sequences, which can be used to compute the correlation between the sliding and reference windows.

# Here's an example implementation of the CORN algorithm using numpy:

# python

# import numpy as np

# def corn_algorithm(time_series, window_size):
#     n = len(time_series)
#     # Compute the rolling window of size window_size for each time step
#     rolling_windows = np.lib.stride_tricks.sliding_window_view(time_series, window_size)

#     # Compute the correlation between the most recent window and all the other windows
#     reference_window = rolling_windows[-1]
#     correlation = np.zeros(n-window_size)
#     for i in range(n-window_size):
#         sliding_window = rolling_windows[i]
#         correlation[i] = np.correlate(sliding_window, reference_window)

#     return correlation

# This implementation uses numpy's sliding_window_view() function to compute the rolling windows for each time step. It then loops through the time series and uses numpy's correlate() function to compute the correlation between the most recent window and all the other windows.

# Note that this implementation may still be slower than a fully vectorized implementation due to the overhead of creating the rolling window views and the loop over the time series. However, it should be more efficient than a naive implementation that explicitly loops through all the windows at each time step.
        
    return correlation_array
        
        

##########

# implement CORN algorithm

class Expert():
    
    def __init__(self, w, rho):
        
        self.w = w # window size
        self.rho = rho # correlation threshold
        self.Ct = {} # correlation similarity set
        self.bt = {} # portfolio weights


    def estimate_portfolio_weights(self, history: pd.DataFrame) -> pd.DataFrame:
        '''
        This function estimates the portfolio weights in hindsight for a given time series of asset prices.
        It returns a pandas DataFrame with the portfolio weights for each time step.
        '''
        cprint(f'{history}', 'yellow')
        
        # historic_start_date = history.index[0]
        # historic_end_date = history.index[-1]
        # print(f'Received history of shape {history.shape}. Start date: {historic_start_date}, end date: {historic_end_date}')
        j = 0
        
        if len(history) <= (3*self.w):
            print("Not enough data to calculate 2 x window. Creating uniform weights for portfolio.")
            # bt = [1/len(history.columns)]*len(history.columns)
            bt = np.array([1/history.shape[1]]*history.shape[1]) #np specific code
            for i in range(0, len(history)):
                bt_dict = {}
                bt_dict[j+1] = bt
                self.bt.update(bt_dict)
                j = j + 1


        else:
            bt_dict = {}
            # move through time series with a window size of w and calculate correlatioon coefficient between most recent window and all previous windows
            
            for t in range(self.w*3, len(history)):
                most_recent_window = history[t-self.w:t] # np version
                previous_windows_period = history[:t-self.w] # np version

                # most_recent_window = history.iloc[t-self.w:t] #does not change
                # previous_windows_period = history.iloc[:t-self.w] #must be looped over again
                
                Ct_filled = False
                Ct = {}
                Ct_list = []

                for i in range(len(previous_windows_period) - self.w + 1):
                    '''
                    This loops searches for the most recent window in the previous windows period. It then calculates the correlation coefficient between the two windows and stores the resulting correlated window as a pandas dataframe in a list 'Ct_list'. The list is then used to calculate the portfolio weights. If no correlation coefficient is above the threshold (Ct_filled is still False), the portfolio weights are set to equal.
                    '''
                    cprint(f'Current time step: {t}, current index: {i}', 'green')
                    previous_window = previous_windows_period[i:i+self.w] # np version

                    # previous_window = previous_windows_period.iloc[i:i+self.w]
                    # start_previous_window = previous_window.index[0]
                    # end_previous_window = previous_window.index[-1]
                    
                    optim_window = history[i+self.w:i+self.w*2] # np version

                    # optim_window = history.iloc[i+self.w:i+self.w*2]

                    # calculate correlation coefficient between most_recent_window and previous_window by first flattening them into two vectors, each with a length of 400 (20x20) and then using the standard correlation coefficient formula to calculate the correlation between the two vectors

                    corr = np.corrcoef(most_recent_window.flatten(), previous_window.flatten())[0,1]

                    if abs(corr) >= self.rho:
                        # cprint(f'Threshold passed! Added period from {start_previous_window} till {end_previous_window} with {corr} to Ct', 'red')
                        
                        Ct = np.empty((0, optim_window.shape[1]))
                        Ct = np.concatenate((Ct, optim_window), axis=0) # np version
                        # Ct = pd.concat([Ct, optim_window], axis=0)
                        Ct_list.append(Ct)
                        print(Ct_list)
                        Ct_filled = True
                                                
                if Ct_filled: 
                    
                    '''
                    if at least one similar-correlated window is found, the portfolio weights are based on optimization. If more than one similar-correlated window is found, the portfolio weights are based on the average of the portfolio weights of all similar-correlated windows
                    '''
                    
                    cprint(f'Length of Ct_list: {len(Ct_list)}', 'yellow')

                    optimal_weights_list = []
                    for n in range(len(Ct_list)):
                        optimal_weights = find_optimal_portfolio(Ct_list[n], period=t, list_order=n)
                        optimal_weights_list.append(optimal_weights)
                    
                    # calculate average of optimal weights
                    bt = np.mean(optimal_weights_list, axis=0)
                    bt_dict[t+1] = bt
                    self.bt.update(bt_dict)

                else: # if no similar-correlated window is found, the portfolio weights are set to uniform
                    bt = [1/len(history.columns)]*len(history.columns)
                    bt_dict[t+1] = bt
                    
                    self.bt.update(bt_dict)


if __name__ == '__main__':
    
    investment_universe = ['SPY', 'VTI', 'QQQ', 'EFA', 'AGG', 'VWO', 'IJR', 'IJH', 'IWF', 'GLD', 'LQD', 'TLT', 'VNQ', 'IEF', 'SHY', 'DIA', 'VGK', 'VB', 'EXS1.DE', 'CAC.PA']

    get_data(investment_universe)
    log_returns_array = csv_to_numpy(investment_universe)

    start = time.perf_counter()
    correlation_array = expert_portfolio_weight(data=log_returns_array, window=20, rho=0.2)
    print("LOOP DONE")
    end = time.perf_counter()
    print("Elapsed = {}s".format((end - start)))
    
    #Elapsed = 801.9361368920017s
    #
    
    
    
    
    
    
    # investment_universe = ['SPY', 'VTI', 'QQQ', 'EFA', 'AGG', 'VWO', 'IJR', 'IJH', 'IWF', 'GLD', 'LQD', 'TLT', 'VNQ', 'IEF', 'SHY', 'DIA', 'VGK', 'VB', 'EXS1.DE', 'CAC.PA']
    # num_assets = len(investment_universe)

    # print(f'Assets in investment universe: {num_assets}')
    
    # df = yf.download(investment_universe, ignore_tz=True)['Adj Close']
    # trading_days = 15*252
    # non_null_counts = df.count()
    # min_counts = non_null_counts.min()
    # min_symbol = non_null_counts[non_null_counts == min_counts].index[0]
    # min_start_date = df[df[min_symbol].notnull()].index.min()
    # min_end_date = df[df[min_symbol].notnull()].index.max()
    # df = df.loc[(df.index >= min_start_date) & (df.index <= min_end_date)]
    # df = df.dropna()
    # log_returns = np.log(df / df.shift(1)).dropna()
    # print(log_returns.head(5))

    # # Convert to numpy array to speed up calculations
    # log_returns = log_returns.to_numpy()

    
    # window_size = 20 
    # steps = window_size # check only every n-th trading days for similar-correlated windows

    # rho_threshold = 0.2

    # expert = Expert(w=window_size, rho=rho_threshold)

    # start = time.time()
    
    # max_iterations = len(log_returns) # will be length of log_returns later
    
    # for t in range(1, 70):
        
    #     '''
    #     #before: 70 days: 2.4 mins
    #     #after cleaning up of code 70 days: 0.94 mins -> Performance boost: 2.5x
    #     #switching over to using only np.arrays instead of pd.Dataframes 70 days: 0.16 mins -> Performance boost: 5.8x
    #     #tried with 140 iterations: 24.764684696992237 minutes.
    #     '''
        
    #     # 1. Step: Identify all similar-correlated windows in hindsight
        
    #     expert.estimate_portfolio_weights(history=log_returns[:t])
    #     portfolio_weights = expert.bt # order of portfolio weights is the same as the order of the assets in the investment universe
        
    #     cprint(f'{portfolio_weights}', 'green')
        
    #     # 2. Step: Pass over portfolio weights to backtester to calculate portfolio returns
        
    #     #  This is where we can adjust that the rebalancing happens every 20 days or every 10 days or every 5 days etc.

    # end = time.time()
    # print(f'Time elapsed for loop: {(end-start)/60} minutes.')