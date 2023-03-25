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

@jit(nopython=True)
def expert_portfolio_weight(data: np.array, window: int, rho: float) -> np.array:
    
    '''
    clean and simple implementation of the CORN algorithm. Supported by numba.
    '''
    
    # loop over data
    
    correlation_array = np.zeros((len(data), len(data[0]))) #initialize correlation array
    print(correlation_array)
    
    for i in range(2*window, len(data)):
        
        #define most_recent_window which is the last window 
        most_recent_window = data[i-window:i]
        
        #define previos_window_period which is the rest of the data
        previous_window_period = data[:i-window]
        
        #calculate correlation between most_recent_window and previous_window_period
        correlation = np.corrcoef(most_recent_window, previous_window_period)[0,1]
        
        #add to correlation_array
        correlation_array[i] = correlation
        
        # Experiment with numba: it took forever...i dont know why. 
        
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
    expert_portfolio_weight(data=log_returns_array, window=20, rho=0.2)
    end = time.perf_counter()
    print("Elapsed = {}s".format((end - start)))
    
    #Elapsed = 196.5547334699986s ohne numba
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