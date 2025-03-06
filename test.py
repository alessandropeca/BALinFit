import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt



from BAsymLinFit import *




df = pd.read_csv('test_emcee.csv', names=['names', 'x', 'xerrl', 'xerru', 'y', 'yerrl', 'yerru'], header=0)
df = df = df[~df['x'].isna() & (df['x'] != -np.inf)]



results = bayesian_regression_mcmc(
    x=df['x'].values, 
    y=df['y'].values, 
    y_err_lower=df['yerrl'].values/1.645, 
    y_err_upper=df['yerru'].values/1.645,
    x_err_lower=df['xerrl'].values/1.645, 
    x_err_upper=df['xerru'].values/1.645,
    x_min=-2, x_max=0.5
)

fig, ax = plt.subplots()
ax.errorbar(df['x'].values, df['y'].values, yerr=[df['yerrl'].values, df['yerru'].values], xerr=[df['xerrl'].values, df['xerru'].values], fmt='o', label='Data', color='black')
ax.plot(results['x_plot'], results['y_median'], color='tab:blue')#, label=f'Best Fit: y = {results['slope_median']:.2f} ± {results['slope_err_lower']:.2f}/{results['slope_err_upper']:.2f} x + {results['intercept_median']:.2f} ± {results['intercept_err_lower']:.2f}/{results['intercept_err_upper']:.2f}')
ax.fill_between(results['x_plot'], results['y_lower'], results['y_upper'], color='tab:blue', alpha=0.3, label='68% Confidence Interval')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.legend()
plt.show()