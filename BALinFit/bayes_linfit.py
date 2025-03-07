from scipy.stats import norm
from tqdm import tqdm
import emcee 
import corner 
import numpy as np

def bayesian_regression_mcmc(x, y, x_err_lower, x_err_upper, y_err_lower, y_err_upper, num_realizations=1000, n_walkers=10, n_steps=1000, x_min=None, x_max=None):
    """
    Performs Bayesian linear regression using emcee with asymmetric error bars.
    
    Parameters:
    x (array-like): Independent variable.
    y (array-like): Dependent variable.
    y_err_lower (array-like): Lower uncertainties on y (90% confidence level).
    y_err_upper (array-like): Upper uncertainties on y (90% confidence level).
    num_realizations (int): Number of Monte Carlo realizations.
    n_walkers (int): Number of MCMC walkers.
    n_steps (int): Number of MCMC steps.
    
    Returns:
    dict: Dictionary with slope, intercept, and confidence intervals for regression.
    """

    # Define log-likelihood function
    def log_likelihood(theta, x, y):
        """Gaussian likelihood for linear regression"""
        intercept, slope = theta
        model = intercept + slope * x
        sigma = np.std(y)  # Approximate scatter
        return -0.5 * np.sum(((y - model) / sigma) ** 2 + np.log(2 * np.pi * sigma ** 2))

    # Define log-prior function
    def log_prior(theta):
        """Uniform priors on intercept and slope"""
        intercept, slope = theta
        if -10 < intercept < 10 and -10 < slope < 10:
            return 0.0  # Flat prior
        return -np.inf  # Outside allowed range

    # Define full log-posterior function
    def log_probability(theta, x, y):
        """Combined log-posterior = log-prior + log-likelihood"""
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(theta, x, y)

    # Define function for Half-Gaussian sampling
    def sample_half_gaussian(y_best, y_err_low, y_err_high):
        """Samples y from an asymmetric Half-Gaussian distribution."""
        if np.random.rand() < 0.5:
            return y_best - abs(norm.rvs(scale=y_err_low))
        else:
            return y_best + abs(norm.rvs(scale=y_err_high))

    # Generate best-fit x range for plotting
    x_plot_min = x_min if x_min is not None else min(x)
    x_plot_max = x_max if x_max is not None else max(x)
    x_plot = np.linspace(x_plot_min, x_plot_max, 100)

    # Initialize arrays to store confidence intervals
    y_median = np.zeros_like(x_plot)
    y_lower = np.zeros_like(x_plot)
    y_upper = np.zeros_like(x_plot)

    # Store slope and intercept results
    slopes = []
    intercepts = []
    all_samples = []

    # Run Monte Carlo sampling
    for _ in tqdm(range(num_realizations), desc="Running MCMC realizations"):
        # Sample x and y values from asymmetric Half-Gaussian distribution
        x_sampled = np.array([
            sample_half_gaussian(x[i], x_err_lower[i], x_err_upper[i])
            for i in range(len(x))
        ])
        y_sampled = np.array([
            sample_half_gaussian(y[i], y_err_lower[i], y_err_upper[i])
            for i in range(len(y))
        ])
        
        # Initialize MCMC
        pos = np.random.randn(n_walkers, 2)  # Random initial positions

        # Run MCMC using emcee
        sampler = emcee.EnsembleSampler(n_walkers, 2, log_probability, args=(x_sampled, y_sampled))
        sampler.run_mcmc(pos, n_steps, progress=False)

        # Extract samples
        samples = sampler.get_chain(discard=100, thin=10, flat=True)
        all_samples.append(samples)

        # Store median slope and intercept from this iteration
        slopes.append(np.median(samples[:, 1]))  # Slope
        intercepts.append(np.median(samples[:, 0]))  # Intercept

        # Compute confidence intervals for each x_plot point
        for i, x_ in enumerate(x_plot):
            y_samples = np.array([sample[0] + sample[1] * x_ for sample in samples])
            y_median[i] += np.percentile(y_samples, 50)  # 50th percentile (median)
            y_lower[i] += np.percentile(y_samples, 16)   # 16th percentile (lower bound)
            y_upper[i] += np.percentile(y_samples, 84)   # 84th percentile (upper bound)

    # Normalize confidence interval values by the number of realizations
    y_median /= num_realizations
    y_lower /= num_realizations
    y_upper /= num_realizations

    # Convert all_samples list to a single NumPy array
    all_samples = np.vstack(all_samples)  # Shape: (total_samples, 2), where 2 = [intercept, slope]

    # Compute Bayesian posterior statistics
    intercept_median, slope_median = np.median(all_samples, axis=0)  # Median from MCMC posterior
    intercept_lower, slope_lower = np.percentile(all_samples, 16, axis=0)  # 16th percentile
    intercept_upper, slope_upper = np.percentile(all_samples, 84, axis=0)  # 84th percentile

    # Compute asymmetric uncertainties
    intercept_err_lower = intercept_median - intercept_lower
    intercept_err_upper = intercept_upper - intercept_median
    slope_err_lower = slope_median - slope_lower
    slope_err_upper = slope_upper - slope_median

    # Generate Corner Plot
    figure = corner.corner(all_samples, labels=["Intercept", "Slope"], 
                           truths=[intercept_median, slope_median],
                           quantiles=[0.16, 0.5, 0.84], show_titles=True)


    # Return final results
    return {
        'slope_median': slope_median,
        'slope_err_lower': slope_err_lower,
        'slope_err_upper': slope_err_upper,
        'intercept_median': intercept_median,
        'intercept_err_lower': intercept_err_lower,
        'intercept_err_upper': intercept_err_upper,
        'x_plot': x_plot,
        'y_median': y_median,
        'y_lower': y_lower,
        'y_upper': y_upper,
        'corner_figure': figure
    }