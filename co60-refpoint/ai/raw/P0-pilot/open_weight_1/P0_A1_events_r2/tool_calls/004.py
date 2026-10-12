import numpy as np
import math
import random

# Precompute Compton sampling lookup tables for a range of energies
# We'll create a function that returns a sampling function for a given alpha.

def compton_cdf_factory(alpha):
    """Return a function that samples cosθ from Klein-Nishina for given alpha."""
    # Create grid of x = 1 - cosθ from 0 to 2
    n_pts = 2000
    x_vals = np.linspace(0, 2, n_pts)
    # Compute unnormalized PDF at each x
    # dσ/dx ∝ (1+αx)^{-1} + (1+αx)^{-3} - (2x - x^2)/(1+αx)^2
    term1 = 1.0 / (1 + alpha * x_vals)
    term2 = 1.0 / (1 + alpha * x_vals)**3
    term3 = (2 * x_vals - x_vals**2) / (1 + alpha * x_vals)**2
    pdf = term1 + term2 - term3
    # Handle small negative values due to numerical issues
    pdf = np.maximum(pdf, 0.0)
    # Cumulative
    cdf = np.cumsum(pdf)
    cdf = cdf / cdf[-1]
    # Store
    return x_vals, cdf

def sample_compton(alpha, x_vals, cdf):
    """Sample x = 1 - cosθ from CDF."""
    r = random.random()
    idx = np.searchsorted(cdf, r)
    if idx == 0:
        x = x_vals[0]
    elif idx >= len(cdf):
        x = x_vals[-1]
    else:
        # Linear interpolation
        x0 = x_vals[idx-1]
        x1 = x_vals[idx]
        y0 = cdf[idx-1]
        y1 = cdf[idx]
        x = x0 + (x1 - x0) * (r - y0) / (y1 - y0)
    return x  # 1 - cosθ

# Test
alpha = 2.0 / 0.511  # for 2 MeV
x_vals, cdf = compton_cdf_factory(alpha)
print("Sample some cosθ values:")
for i in range(10):
    x = sample_compton(alpha, x_vals, cdf)
    print(f"  cosθ = {1 - x:.4f}")

# Performance
import time
t0 = time.time()
for i in range(10000):
    x = sample_compton(alpha, x_vals, cdf)
t1 = time.time()
print(f"10k samples: {t1-t0:.3f} s")