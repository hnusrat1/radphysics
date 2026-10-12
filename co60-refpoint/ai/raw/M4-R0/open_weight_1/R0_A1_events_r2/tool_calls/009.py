import numpy as np
import time

# Constants
r_e = 2.8179403227e-13  # cm, classical electron radius
me = 0.5109989461  # MeV/c^2

def sample_compton(E_in):
    """Sample Compton scattering: return (cos_theta, E_out, cos_electron)"""
    alpha = E_in / me
    # Sample cosθ using Kahn's method (rejection)
    while True:
        # Sample two random numbers
        xi1 = np.random.random()
        xi2 = np.random.random()
        # Kahn method:
        # r = 1 + 2*alpha
        # P = (1+alpha^2)/(1+2*alpha) + (alpha*(1+2*alpha))/(1+alpha^2) * something?
        # Actually, use the method from:
        # Sample cosθ using the formula:
        # cosθ = 1 - 2*xi1 / (1 + 2*alpha*(1-xi1))   ??
        # Let's use the direct inversion method for Klein-Nishina.
        # The cumulative distribution for cosθ is available analytically.
        pass

# Let's implement a simpler rejection method:
def sample_compton_angle(E_in):
    """Return cos(θ) for Compton scattering using rejection sampling"""
    alpha = E_in / me
    # The differential cross section (unnormalized) is:
    # dσ/dΩ ∝ (1/(1+α(1-cosθ))^2) * ((1/(1+α(1-cosθ))) + (1/(1+α(1-cosθ)))^(-1) - sin^2θ)
    # We can sample μ = cosθ uniformly in [-1, 1] and accept with probability proportional to the DCS.
    # Maximum of DCS is at μ=1 (forward), where DCS_max = (1)^2 * (1 + 1 - 0) = 2.
    # Actually, at μ=1, f=1, DCS = 1^2 * (1 + 1 - 0) = 2.
    # So we can use max DCS = 2.
    max_dcs = 2.0
    while True:
        mu = np.random.uniform(-1, 1)
        f = 1.0 / (1.0 + alpha*(1.0 - mu))
        dcs = f**2 * (f + 1.0/f - (1.0 - mu**2))
        if np.random.random() < dcs / max_dcs:
            return mu

def compton_outgoing_energy(E_in, cos_theta):
    """Return E_out for Compton scattering"""
    alpha = E_in / me
    return E_in / (1.0 + alpha*(1.0 - cos_theta))

def compton_electron_angle(E_in, cos_theta):
    """Return cos(electron angle) relative to incident photon direction"""
    # From momentum conservation:
    # tan(φ) = sinθ / ( (1+α) * tan(θ/2) )
    # Actually: cot φ = (1 + α) * tan(θ/2)
    # So tan(φ) = 1 / ((1+α) * tan(θ/2))
    # Then cos(φ) = 1 / sqrt(1 + tan^2(φ))
    sin_theta = np.sqrt(1 - cos_theta**2)
    tan_half = sin_theta / (1 + cos_theta)  # tan(θ/2) = sinθ/(1+cosθ)
    alpha = E_in / me
    cot_phi = (1 + alpha) * tan_half
    # Avoid division by zero
    if cot_phi == 0:
        return 0.0  # electron at 90 degrees
    tan_phi = 1.0 / cot_phi
    cos_phi = 1.0 / np.sqrt(1.0 + tan_phi**2)
    # The electron is always forward (φ < 90°), so cos_phi >= 0
    return cos_phi

# Test
import random
random.seed(123)
np.random.seed(123)
cos = sample_compton_angle(1.1732)
print("cos_theta:", cos)
print("E_out:", compton_outgoing_energy(1.1732, cos))
print("cos_electron:", compton_electron_angle(1.1732, cos))