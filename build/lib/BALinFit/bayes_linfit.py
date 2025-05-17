from scipy.stats import norm
from tqdm import tqdm
import emcee 
import corner 
import numpy as np


# Define function for Half-Gaussian sampling with limit handling
def sample_with_limits(y_best=None, y_err_low=None, y_err_high=None, 
                       lower_limit=None, upper_limit=None, sigma_int=0.0):
    """
    Samples y from an asymmetric Half-Gaussian distribution if a best-fit value is known.
    If only an upper or lower limit is given, samples from a half-Gaussian distribution 
    with width set to half the limit value.

    Parameters:
    - y_best: Best-fit y value (None if using only a limit).
    - y_err_low: Lower uncertainty in y.
    - y_err_high: Upper uncertainty in y.
    - lower_limit: If set, ensures the sampled value does not go below this limit.
    - upper_limit: If set, ensures the sampled value does not exceed this limit.
    - sigma_int: Intrinsic scatter (default is 0, meaning no extra scatter).

    Returns:
    - A perturbed y value incorporating both measurement uncertainty and limits.
    """

    if y_best is not None:
        if np.random.rand() < 0.5:
            sampled_y = y_best - abs(norm.rvs(scale=y_err_low))  # Sample from lower half-Gaussian
        else:
            sampled_y = y_best + abs(norm.rvs(scale=y_err_high))  # Sample from upper half-Gaussian

    else:
        if upper_limit is not None:
            sampled_y = upper_limit - abs(norm.rvs(scale=upper_limit / 2))
        elif lower_limit is not None:
            sampled_y = lower_limit + abs(norm.rvs(scale=lower_limit / 2))
        else:
            raise ValueError("Either y_best or a limit (upper or lower) must be provided.")

    if sigma_int > 0:
        sampled_y += np.random.normal(0, sigma_int)

    return sampled_y

# Define the log-likelihood function
def log_likelihood(theta, x, y):
    """Log-likelihood function assuming Gaussian errors"""
    intercept, slope = theta
    y_model = intercept + slope * x
    sigma_y = np.std(y)  # Approximate scatter
    return -0.5 * np.sum(((y - y_model) / sigma_y) ** 2 + np.log(2 * np.pi * sigma_y**2))


# Define log-prior function
def log_prior(theta):
    """Uniform priors on intercept and slope"""
    intercept, slope = theta
    if -10 < intercept < 10 and -10 < slope < 10:
        return 0.0  # Flat prior
    return -np.inf  # Outside allowed range


# Define full log-posterior function
def log_probability(theta, x, y):
    """Compute the full posterior probability (prior + likelihood)."""
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y)


# Define Bayesian regression function
def bayesian_regression_mcmc(x, y, x_err_lower, x_err_upper, y_err_lower, y_err_upper,
                             x_upper_limits=None, x_lower_limits=None, y_upper_limits=None, y_lower_limits=None,
                             num_realizations=1000, n_walkers=10, n_steps=1000, sigma_intrinsic=0.1,
                             x_min=None, x_max=None):

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
            sample_with_limits(
                y_best=x[i], y_err_low=x_err_lower[i], y_err_high=x_err_upper[i], 
                lower_limit=x_lower_limits[i] if x_lower_limits is not None else None, 
                upper_limit=x_upper_limits[i] if x_upper_limits is not None else None,
                sigma_int=sigma_intrinsic
            )
            for i in range(len(x))
        ])
        y_sampled = np.array([
            sample_with_limits(
                y_best=y[i], y_err_low=y_err_lower[i], y_err_high=y_err_upper[i], 
                lower_limit=y_lower_limits[i] if y_lower_limits is not None else None, 
                upper_limit=y_upper_limits[i] if y_upper_limits is not None else None,
                sigma_int=sigma_intrinsic
            )
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