import numpy as np
import math

me = 0.511  # MeV
r_e = 2.817940e-13  # cm

def sample_compton(E):
    # E in MeV
    alpha = E / me
    # Rejection sampling for cosθ ∈ [-1, 1]
    # maximum of dσ/dΩ occurs at cosθ=1 (forward)
    # The differential cross section per unit u=cosθ is (dσ/dΩ)*2π? Actually we need probability density dσ/dΩ per solid angle.
    # For rejection, we can use the fact that the distribution is bounded.
    # We'll use the algorithm from EGS4: sample from (1+cos^2θ) and a correction.
    # But we can simply do rejection with envelope: (1+cos^2θ) * (1 + (alpha*(1-cosθ)))^(-2) maybe.
    # Let's implement a simple rejection using the maximum value of the PDF.
    # The PDF is proportional to (dσ/dΩ) integrated over φ. We'll use the function f(cosθ) = (dσ/dΩ) * 2π (since azimuthal symmetry).
    # The maximum of f is at cosθ=1.
    # Compute f_max:
    # f(1) = 2π * (r_e^2/2) * (1)^2 * (1+1-0) = 2π * r_e^2
    f_max = 2 * math.pi * r_e**2
    while True:
        u = np.random.uniform(-1, 1)
        # compute f(u)
        k = alpha
        kp = k / (1 + k * (1 - u))
        sin2 = 1 - u*u
        f = 2*math.pi * (r_e**2/2) * (kp/k)**2 * (kp/k + k/kp - sin2)
        if f < 0:
            continue
        # rejection
        y = np.random.uniform(0, f_max)
        if y <= f:
            cos_theta = u
            break
    # compute E'
    E_prime = E / (1 + alpha * (1 - cos_theta))
    T = E - E_prime
    # electron direction
    p = E  # in MeV/c (c=1)
    p_prime = E_prime
    p_e = math.sqrt(T*T + 2*me*T) if T > 0 else 0.0
    if p_e > 0:
        cos_electron = (p - p_prime * cos_theta) / p_e
        # clamp
        if cos_electron > 1: cos_electron = 1.0
        if cos_electron < -1: cos_electron = -1.0
    else:
        cos_electron = 1.0
    return cos_theta, E_prime, T, cos_electron