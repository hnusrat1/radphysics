import numpy as np
import random
import time

# Constants
r_e = 2.8179403227e-13# cm
me = 0.5109989461# MeV
NA = 6.02214076e23
M_H2O = 18.015# g/mol
rho = 1.0# g/cm3

N_ e = 10# electrons per water molecule

# Source energy lines
E1 = 1.1732
E2 = 1.3325

# Geometry
# Phantom: z from 0 to 30 cm, x,y from -15 to 15
# Scoring volume: cylinder on z-axis, radius 1.0 cm, z from 4.9 to 5.1
# Source at (0,0,-100)

def sigma_KN(E):
    k = E / me
    term = ((1+k)/k**2 * (2*(1+k)/(1+2*k) - np.log(1+2*k)/k) 
            + np.log(1+2*k)/(2*k) 
            - (1+3*k)/(1+2*k)**2)
    return 2*np.pi * r_e**2 * term

def sigma_total_approx(E):
    """Return total and partial cross sections per molecule (cm^2)"""
    # Compton
    sig_comp = N_e * sigma_KN(E)
    
    # Rayleigh: approximate using NIST data at 1 MeV
    # μ_rayleigh/ρ at 1 MeV ≈ 0.0014 cm2/g
    # σ_Rayleigh = μ/ρ * M / NA
    sig_rayl = 0.0014 * M_H2O / NA * (1.0 / max(E, 0.01))**2  # approximate as ~1/E^2
    
    # Photoelectric: at 1 MeV, μ_PE/ρ ≈ 0.0016
    sig_pe = 0.0016 * M_H2O / NA * (1.0 / max(E, 0.01))**3.5
    
    # Pair production: only if E > 1.022 MeV
    if E > 1.022:
        # At 1.25 MeV, μ_pair/ρ ≈ 0.00087
        sig_pair = 0.00087 * M_H2O / NA * (E - 1.022) / (1.25 - 1.022)
    else:
        sig_pair = 0.0
    
    total = sig_comp + sig_rayl + sig_pe + sig_pair
    return total, sig_comp, sig_rayl, sig_pe, sig_pair

def sample_compton_angle(E_in):
    """Return cos(θ) for Compton scattering using rejection sampling"""
    alpha = E_in / me
    max_dcs = 2.0
    while True:
        mu = np.random.uniform(-1, 1)
        f = 1.0 / (1.0 + alpha*(1.0 - mu))
        dcs = f**2 * (f + 1.0/f - (1.0 - mu**2))
        if np.random.random() < dcs / max_dcs:
            return mu

def compton_electron_angle(E_in, cos_theta):
    """Return cos(electron angle) relative to incident photon direction"""
    sin_theta = np.sqrt(1 - cos_theta**2)
    tan_half = sin_theta / (1 + cos_theta)  # tan(θ/2)
    alpha = E_in / me
    cot_phi = (1 + alpha) * tan_half    if cot_phi == 0:
        return 0.0
    tan_phi = 1.0 / cot_phi
    cos_phi = 1.0 / np.sqrt(1.0 + tan_phi**2)
    return cos_phi

def sample_rayleigh_angle():
    """Sample cos(θ) for Rayleigh scattering (very forward)"""
    # Use a simple model: sample θ from distribution proportional to 
    # (1+cos^2θ) * |F(x)|^2, with F(x) approximated as Gaussian
    # For water, characteristic angle ~ lambda / (2*pi*R) where R ~ 0.1 nm
    # lambda = hc/E ~ 1.24e-2 nm / E(MeV) = 0.0124/1.1732 = 0.0106 nm
    # theta_char = lambda / (2*pi*R) ≈ 0.0106/(6.28*0.1) ≈ 0.017 rad
    # So cosθ ≈ 1 - θ^2/2 ≈ 1 - 0.00014
    # We'll sample cosθ from a distribution that is sharply peaked near 1.
    # Use a simple rejection: sample mu uniformly in [cos_theta_min, 1] where cos_theta_min such that
    # the Rayleigh DCS is small. 
    # For simplicity, use mu = 1 - 0.5 * (sigma * randn)^2 ? 
    # Let's just sample mu uniformly between 0.999 and 1.0.
    return np.random.uniform(0.999, 1.0)

def sample_photoelectron_angle(E_kin):
    """Return cos(electron angle) for photoelectric effect using Sauter distribution"""
    # Sauter: dP/dΩ ∝ sin^2θ / (1 - β cosθ)^4
    # where β = sqrt(1 - (mec2/E_total)^2)
    E_total = E_kin + me
    beta = np.sqrt(1 - (me/E_total)**2)
    # Rejection sampling
    max_val = 10.0  # approximate maximum
    while True:
        mu = np.random.uniform(-1, 1)
        # The distribution is forward peaked, but at high energies, 
        # the peak is very forward. 
        # sin^2θ = 1 - mu^2
        val = (1 - mu**2) / (1 - beta*mu)**4
        if np.random.random() < val / max_val:
            return mu

def sample_pair(K_e_available):
    """Return (t_e, t_pos, cos_theta_e, cos_theta_pos) for pair production.
       K_e_available = E_in - 1.022 (total kinetic energy available)"""
    # Sample electron kinetic energy uniformly
    t_e = np.random.uniform(0, K_e_available)
    t_p = K_e_available - t_e
    
    # Angular distributions: forward peaked, characteristic angle ~ me/E
    # For each particle, sample cosθ from distribution proportional to 1/(1 - β cosθ)^2
    # where β = sqrt(1 - (me/(E_kin+me))^2)
    
    def sample_pair_particle_angle(E_kin):
        if E_kin < 1e-6:
            return 1.0  # forward
        E_tot = E_kin + me
        beta = np.sqrt(1 - (me/E_tot)**2)
        # Sample using rejection: max at mu=1 is 1/(1-beta)^2
        max_val = 1.0 / (1 - beta)**2 if beta < 1 else 1e10
        while True:
            mu = np.random.uniform(-1, 1)
            val = 1.0 / (1 - beta*mu)**2
            if np.random.random() < val / max_val:
                return mu
    
    cos_e = sample_pair_particle_angle(t_e)
    cos_p = sample_pair_particle_angle(t_p)
    
    return t_e, t_p, cos_e, cos_p

# Test pair sampling
print("Testing pair sampling...")
for _ in range(5):
    t_e, t_p, cs_e, cs_p = sample_pair(1.1732 - 1.022)
    print(f"t_e={t_e:.4f}, t_p={t_p:.4f}, cos_e={cs_e:.4f}, cos_p={cs_p:.4f}")