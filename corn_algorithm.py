import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns

from utils import *

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

    def estimate_portfolio_weights(self, history: pd.DataFrame):
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
                
                cprint(f'length of most recent window: {len(most_recent_window)}\nstart of most recent window: {start_most_recent_window}\nend of most recent window: {end_most_recent_window}', 'green')
                
                previous_windows_period = history.iloc[:t-self.w] #must be looped over again
                start_previous_windows = previous_windows_period.index[0]
                end_previous_windows = previous_windows_period.index[-1]
                
                cprint(f'length of previous windows period: {len(previous_windows_period)}\nstart of previous windows period: {start_previous_windows}\nend of previous windows period: {end_previous_windows}', 'blue')
                
                # loop over previous_windows_period and calculate correlation coefficient between most_recent_window and each previous window with size self.w
                
                Ct_filled = False
                for i in range(len(previous_windows_period) - self.w + 1):
                    cprint(f"   Within the loop of previos_windows_period: Trading day {i}\n   Total length of previos_windows_period: {len(previous_windows_period)}", "magenta")
                    
                    previous_window = previous_windows_period.iloc[i:i+self.w]
                    start_previous_window = previous_window.index[0]
                    end_previous_window = previous_window.index[-1]
                    
                    cprint(f'   length of previous window: {len(previous_window)}\n   start of previous window: {start_previous_window}\n   end of previous window: {end_previous_window}', 'magenta')
                    
                    # calculate correlation coefficient between most_recent_window and previous_window by first flattening them into two vectors, each with a length of 400 (20x20) and then using the standard correlation coefficient formula to calculate the correlation between the two vectors
                    # print(f'Shape of most recent window: {most_recent_window.shape} and previous window: {previous_window.shape}')
                    corr = np.corrcoef(most_recent_window.values.flatten(), previous_window.values.flatten())[0,1]
                    print(f'correlation coefficient between most recent window and previous window: {corr}')

                    if abs(corr) >= self.rho:
                        cprint(f'Threshold passed! Added period from {start_previous_window} till {end_previous_window} with {corr} to Ct', 'red')
                        
                        self.Ct[2*self.w+(i+1)] = (start_previous_window, end_previous_window, corr)
                        Ct_filled = True
                        # Calculate portfolio weights by passing over to optimization function (simplex)
                    
                    # only use equal weights if no similar-correlated window was found
                    if not Ct_filled:
                        
                        # If no correlation threshold is found, then equal portfolio weights for that day
                        bt = [1/len(history.columns)]*len(history.columns)
                        bt_dict[2*self.w+(i+1)] = bt
                        
                        self.bt.update(bt_dict)

        
if __name__ == '__main__':

    window_size = 20
    rho_threshold = 0.2

    expert = Expert(w=window_size, rho=rho_threshold)

    for t in range(1, 44):
        expert.estimate_portfolio_weights(history=log_returns.iloc[:t])
        
        cprint(f"Expert's portfolio weights in hindsight for day {t}: {expert.bt}\n", "cyan")
        cprint(f'Expert\'s correlation similarity set in hindsight for day {t}: {expert.Ct}\n', 'cyan')


