# CORN algorithm

This repo aims to implement the CORN algorithm in Python 3. CORN stands for **COR**relation-driven **N**onparametric and was first introduced by Bin Li, Steven C. H. Hoi and Vivek Gopalkrishnan in 2011.

The implementation here uses the same approach as to go optimize the weights with the *similiar-correlated-set* but instead of purely maximizing the return, paying attention also to minimizing risk. Also this implementation differs in a way, that it does not use the correlated window, but the subsequent window (as this is the one for which the optimization is done).

*(LI, Bin; HOI, Steven C. H.; and Gopalkrishnan, Vivek. CORN: Correlation-driven Nonparametric Learning Approach for Portfolio Selection. (2011). ACM Transactions on Intelligent Systems and Technology. 2, (3),. Research Collection School Of Information Systems. Available at: http://ink.library.smu.edu.sg/sis_research/2265
)*

The codebase provides functionality to download stock price data using Yahoo Finance API, perform portfolio optimization with the `minimize` function from Scipy, and benchmark the performance of the optimized portfolio against an equally-weighted pendant strategy and a simple SPY buy-and-hold strategy. The portfolio optimization problem is solved using quadratic programming and is subject to constraints on asset weights and a return target.

## Installation

To use this codebase, you need to install the following Python packages:

- numpy
- pandas
- yfinance
- numba
- scipy
- matplotlib
- seaborn

## Usage

1. Run the `get_data` function with a list of stock symbols to download and clean historical price data. The function checks that there is enough data for all symbols in the list, prunes the DataFrame to contain only the common time range of all symbols, and saves the log returns to a CSV file.

2. Run the `csv_to_pd` function to load the log returns into a pandas DataFrame.

3. Run the `optimize_portfolio` function with the log returns DataFrame to obtain optimal asset weights. You can optionally set a return target.

4. Run the `plot_weights` function to plot the evolution of the asset weights over time.

5. Run the `benchmarking` function with the optimal weights, the log returns DataFrame, and a window size to compare the performance of the optimized portfolio against a benchmark index (universal portfolio) and a simple buy-and-hold strategy (S&P 500 index). The function saves a plot of the cumulative returns to a PNG file.

## Be aware!

1. If you read papers regarding CORN or any derivatives of it, you really need to pay strong attention to the assumptions. Especially, transaction costs and rebalancing period. Most of the papers assume no tx-cost and a daily rebalancing.

2. 