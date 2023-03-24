import numpy as np
from scipy.optimize import minimize
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

np.seed=42
returns = np.random.normal(0, 0.01, (20, 20))
print(returns)


# Set up the optimization problem
n_assets = returns.shape[1]
bounds = tuple((0,1) for _ in range(n_assets))
constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
x0 = np.ones(n_assets) / n_assets

# Define the objective function
def objective(x, returns):
    portfolio_return = np.sum(returns.mean(axis=0) * x)
    portfolio_risk = np.sqrt(np.dot(x.T, np.dot(np.cov(returns.T), x)))
    sharpe_ratio = portfolio_return / portfolio_risk
    return -sharpe_ratio

# Solve the optimization problem
opt = minimize(objective, x0, args=returns, bounds=bounds, constraints=constraints)

# Create a DataFrame with the portfolio weights
returns_df = pd.DataFrame(returns)
weights_df = pd.DataFrame(opt.x, index=returns_df.columns, columns=['weights'])
#calculate weights in %
weights_df['weights'] = weights_df['weights'] * 100

# Plot the portfolio weights in percentage using stacked sns barplot
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(16,8))
sns.barplot(x=weights_df.index, y=weights_df['weights'], ax=ax)
ax.set_xlabel('Symbol')
ax.set_ylabel('Weight in %')
plt.show()


# hier gehts weiter, erstmal stacked bild fertig bekommen, danach für mehrere perioden. danach mit den echten daten. 