import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns

from utils import *
from portfolio_optimization import find_optimal_portfolio

import time

#1. Get data

investment_universe = ['SPY', 'VTI', 'QQQ', 'EFA', 'AGG', 'VWO', 'IJR', 'IJH', 'IWF', 'GLD', 'LQD', 'TLT', 'VNQ', 'IEF', 'SHY', 'DIA', 'VGK', 'VB', 'EXS1.DE', 'CAC.PA']
num_assets = len(investment_universe)

print(f'Assets in investment universe: {num_assets}')



#2. Data pre-processing

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

# Check if NaN values are in df

if df.isnull().values.any():
    print("NaN values present!")
else:
    print("DataFrame is clean.")
    
print(df.describe())



# 3. Show stock data

plt.style.use('dark_background')
plt.figure(figsize=(6, 9))
sns.lineplot(data=df, linewidth=0.8)
plt.ylabel('Price')
plt.xlabel('Trading Day')

#plt.show()

# 4. Calculate daily returns

log_returns = np.log(df / df.shift(1)).dropna()

#lets put both the returns and std into an ordered list and plot the values to get a first glance of our investment universe and how each individual asset performaned

return_ol = log_returns.mean().sort_values()
std_ol = log_returns.std().sort_values()
sr_ol = (return_ol/std_ol).sort_values()
risk_return_ol = pd.concat([return_ol, std_ol], axis=1)
risk_return_ol.columns = ['mean', 'std']


plt.style.use('dark_background')
plt.figure(figsize=(7, 8))
ax = sns.scatterplot(x=risk_return_ol['std'], y=risk_return_ol['mean'], hue=risk_return_ol.index, s=25, alpha=1, legend=False)
plt.xlabel('Standard Deviation')
plt.ylabel('Mean log Return')
plt.title('Mean vs.Std-Dev. (daily)')

for i, symbol in enumerate(risk_return_ol.index):
    ax.text(risk_return_ol.iloc[i]['std'], risk_return_ol.iloc[i]['mean'], symbol, fontsize=10)

#plt.show()

# Assuming return_ol and std_ol are your ordered lists
plt.style.use('dark_background')
fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, figsize=(16,8))

# Plot for return_ol
sns.barplot(x=return_ol.index, y=return_ol.values, ax=ax1)
ax1.set_xlabel('Symbol')
ax1.set_ylabel('Mean log Return')

# Plot for std_ol
sns.barplot(x=std_ol.index, y=std_ol.values, ax=ax2)
ax2.set_xlabel('Symbol')
ax2.set_ylabel('Standard Deviation')

# Plot for sr_ol
sns.barplot(x=sr_ol.index, y=sr_ol.values, ax=ax3)
ax3.set_xlabel('Symbol')
ax3.set_ylabel('Mean / Std')

#plt.show()




##########

# implement CORN algorithm

class Expert():
    
    def __init__(self, w, rho):
        
        self.w = w # window size
        self.rho = rho # correlation threshold
        self.Ct = {} # correlation similarity set
        self.bt = {} # portfolio weights

    @timeit
    def estimate_portfolio_weights(self, history: pd.DataFrame) -> pd.DataFrame:
        '''
        This function estimates the portfolio weights in hindsight for a given time series of asset prices.
        It returns a pandas DataFrame with the portfolio weights for each time step.
        '''
        
        historic_start_date = history.index[0]
        historic_end_date = history.index[-1]
        print(f'Received history of shape {history.shape}. Start date: {historic_start_date}, end date: {historic_end_date}')
        j = 0
        
        if len(history) <= (2*self.w):
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
            
            for t in range(self.w*2, len(history)):
                most_recent_window = history.iloc[t-self.w:t] #does not change
                start_most_recent_window = most_recent_window.index[0]
                end_most_recent_window = most_recent_window.index[-1]
                
                # cprint(f'length of most recent window: {len(most_recent_window)}\nstart of most recent window: {start_most_recent_window}\nend of most recent window: {end_most_recent_window}', 'green')
                
                previous_windows_period = history.iloc[:t-self.w] #must be looped over again
                start_previous_windows = previous_windows_period.index[0]
                end_previous_windows = previous_windows_period.index[-1]
                
                # cprint(f'length of previous windows period: {len(previous_windows_period)}\nstart of previous windows period: {start_previous_windows}\nend of previous windows period: {end_previous_windows}', 'blue')
                
                # loop over previous_windows_period and calculate correlation coefficient between most_recent_window and each previous window with size self.w. If at least one similiar-correlated window is found, the portfolio weights are no more calculated using uniform weights. If no similiar-correlated window is found, the portfolio weights are calculated using uniform weights.
                
                Ct_filled = False
                Ct = {}
                Ct_list = []

                for i in range(len(previous_windows_period) - self.w + 1):
                    '''
                    This loops searches for the most recent window in the previous windows period. It then calculates the correlation coefficient between the two windows and stores the resulting correlated window as a pandas dataframe in a list 'Ct_list'. The list is then used to calculate the portfolio weights. If no correlation coefficient is above the threshold (Ct_filled is still False), the portfolio weights are set to equal.
                    '''
                    
                    # cprint(f"   Within the loop of previos_windows_period: Trading day {i}\n   Total length of previos_windows_period: {len(previous_windows_period)}", "magenta")
                    
                    previous_window = previous_windows_period.iloc[i:i+self.w]
                    start_previous_window = previous_window.index[0]
                    end_previous_window = previous_window.index[-1]
                    
                    # cprint(f'   length of previous window: {len(previous_window)}\n   start of previous window: {start_previous_window}\n   end of previous window: {end_previous_window}', 'magenta')
                    
                    # calculate correlation coefficient between most_recent_window and previous_window by first flattening them into two vectors, each with a length of 400 (20x20) and then using the standard correlation coefficient formula to calculate the correlation between the two vectors

                    corr = np.corrcoef(most_recent_window.values.flatten(), previous_window.values.flatten())[0,1]
                    # print(f'correlation coefficient between most recent window and previous window: {corr}')

                    if abs(corr) >= self.rho:
                        cprint(f'Threshold passed! Added period from {start_previous_window} till {end_previous_window} with {corr} to Ct', 'red')
                        
                        
                        # OLD CODE, BEFORE USING DF
                        # Ct[2*self.w+(i+1)] = [start_previous_window, end_previous_window, corr]
                        # self.Ct[t] = Ct
                        
                        # NEW CODE, USING DF AND CONCATENATING TOGETHER ALL CORRELATION SIMILARITY SETS
                        
                        # create a pandas df called Ct, containing all returns for all n-days for each previous_window that is similar-correlated to the most_recent_window
                        Ct = pd.DataFrame()
                        Ct = pd.concat([Ct, previous_window], axis=0)
                        Ct_list.append(Ct)
                        
                        Ct_filled = True
                        
                        # # Calculate portfolio weights by passing over to optimization function (simplex)
                        # cprint(f'{self.Ct}', 'yellow')
                        
                        # bt = [1/len(history.columns)]*len(history.columns)
                        # bt_dict[2*self.w+(t+1)] = bt
                        
                if Ct_filled: 
                    
                    '''
                    if at least one similar-correlated window is found, the portfolio weights are based on optimization. If more than one similar-correlated window is found, the portfolio weights are based on the average of the portfolio weights of all similar-correlated windows
                    '''
                    
                    cprint(f'Length of Ct_list: {len(Ct_list)}', 'yellow')

                    optimal_weights_list = []
                    for i in range(len(Ct_list)):
                        optimal_weights = find_optimal_portfolio(Ct_list[i])
                        # cprint(f'Optimal weights based on similiar-correlated set: {optimal_weights}', 'green')
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

    
    window_size = 20
    rho_threshold = 0.2

    expert = Expert(w=window_size, rho=rho_threshold)

    #calculate time it takes to run the loop 
    start = time.time()

    for t in range(1, 51):
        
        # 1. Step: Identify all similar-correlated windows in hindsight
        
        expert.estimate_portfolio_weights(history=log_returns.iloc[:t])
        portfolio_weights = pd.DataFrame.from_dict(expert.bt, orient='index', columns=log_returns.columns)
        
        cprint(f'{portfolio_weights}', 'green')
        cprint(f'Expert\'s correlation similarity set in hindsight for day {t}: {expert.Ct}\n', 'cyan')

    end = time.time()
    runtime = end - start
    cprint(f'Runtime for loop: {runtime}', 'red')

        # 2. Step: Pass over portfolio weights to backtester to calculate portfolio returns
        
        #  This is where we can adjust that the rebalancing happens every 20 days or every 10 days or every 5 days etc.

