# BAsymLinFit


This tool performs Bayesian linear regression using MCMC (emcee), incorporating asymmetric uncertainties in both x and y through Monte Carlo resampling with Half-Gaussian sampling. It runs multiple realizations (num_realizations) where x and y values are perturbed according to their uncertainties, ensuring a robust treatment of measurement errors. For each realization, an MCMC fit is performed using n_walkers independent chains over n_steps iterations, producing a posterior distribution for the slope and intercept. The final regression parameters are derived from the full posterior distribution, ensuring accurate uncertainty estimates. 


Python libraries needed: pandas, numpy, matplotlib, emcee, corner, scipy, tqdm

See notebook example for a simple run.
