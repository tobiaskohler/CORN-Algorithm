import numpy as np
import pandas as pd
import yfinance as yf
import time

from utils import *
from portfolio_optimization import find_optimal_portfolio





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
        
        historic_start_date = history.index[0]
        historic_end_date = history.index[-1]
        print(f'Received history of shape {history.shape}. Start date: {historic_start_date}, end date: {historic_end_date}')
        j = 0
        
        if len(history) <= (3*self.w):
            print("Not enough data to calculate 2 x window. Creating uniform weights for portfolio.")
            bt = [1/len(history.columns)]*len(history.columns)
            for i in range(0, len(history)):
                bt_dict = {}
                bt_dict[j+1] = bt
                self.bt.update(bt_dict)
                j = j + 1


        else:
            bt_dict = {}
            # move through time series with a window size of w and calculate correlatioon coefficient between most recent window and all previous windows
            
            for t in range(self.w*3, len(history)):
                most_recent_window = history.iloc[t-self.w:t] #does not change
                previous_windows_period = history.iloc[:t-self.w] #must be looped over again
                
                Ct_filled = False
                Ct = {}
                Ct_list = []

                for i in range(len(previous_windows_period) - self.w + 1):
                    '''
                    This loops searches for the most recent window in the previous windows period. It then calculates the correlation coefficient between the two windows and stores the resulting correlated window as a pandas dataframe in a list 'Ct_list'. The list is then used to calculate the portfolio weights. If no correlation coefficient is above the threshold (Ct_filled is still False), the portfolio weights are set to equal.
                    '''
                    
                    previous_window = previous_windows_period.iloc[i:i+self.w]
                    start_previous_window = previous_window.index[0]
                    end_previous_window = previous_window.index[-1]
                    
                    optim_window = history.iloc[i+self.w:i+self.w*2]

                    # calculate correlation coefficient between most_recent_window and previous_window by first flattening them into two vectors, each with a length of 400 (20x20) and then using the standard correlation coefficient formula to calculate the correlation between the two vectors

                    corr = np.corrcoef(most_recent_window.values.flatten(), previous_window.values.flatten())[0,1]

                    if abs(corr) >= self.rho:
                        cprint(f'Threshold passed! Added period from {start_previous_window} till {end_previous_window} with {corr} to Ct', 'red')
                        
                        Ct = pd.DataFrame()
                        Ct = pd.concat([Ct, optim_window], axis=0)
                        Ct_list.append(Ct)
                        
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
    num_assets = len(investment_universe)

    print(f'Assets in investment universe: {num_assets}')
    
    df = yf.download(investment_universe, ignore_tz=True)['Adj Close']
    trading_days = 15*252
    non_null_counts = df.count()
    min_counts = non_null_counts.min()
    min_symbol = non_null_counts[non_null_counts == min_counts].index[0]
    min_start_date = df[df[min_symbol].notnull()].index.min()
    min_end_date = df[df[min_symbol].notnull()].index.max()
    df = df.loc[(df.index >= min_start_date) & (df.index <= min_end_date)]
    df = df.dropna()
    log_returns = np.log(df / df.shift(1)).dropna()
    
    
    window_size = 20
    rho_threshold = 0.2

    expert = Expert(w=window_size, rho=rho_threshold)

    start = time.time()
    
    for t in range(1, 140):
        
        #before: 2.4 mins
        #after cleaning up of code: 0,94 mins
        #switching over to pypy now
        
        # 1. Step: Identify all similar-correlated windows in hindsight
        
        expert.estimate_portfolio_weights(history=log_returns.iloc[:t])
        portfolio_weights = pd.DataFrame.from_dict(expert.bt, orient='index', columns=log_returns.columns)
        
        cprint(f'{portfolio_weights}', 'green')
        cprint(f'Expert\'s correlation similarity set in hindsight for day {t}: {expert.Ct}\n', 'cyan')
        
        # 2. Step: Pass over portfolio weights to backtester to calculate portfolio returns
        
        #  This is where we can adjust that the rebalancing happens every 20 days or every 10 days or every 5 days etc.

    end = time.time()
    print(f'Time elapsed for loop: {(end-start)/60} minutes.')
    
