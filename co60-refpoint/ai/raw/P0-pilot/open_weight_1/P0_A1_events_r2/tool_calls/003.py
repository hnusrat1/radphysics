import numpy as np
import math
import random
import time

# Constants
rho = 1.0  # g/cm^3
m_e = 0.51099895  # MeV
r_e = 1.0  # We'll use cm units; Klein-Nishina uses r_e^2 = 7.9408e-26 cm^2
r_e_sq = 7.9408e-26  # cm^2
# Actually classical electron radius = 2.81794e-13 cm, so r_e^2 = 7.9408e-26 cm^2

# Electron density for water
n_e = 3.3428e23  # electrons/cm^3

# Source geometry
SSD = 100.0  # cm
source_pos = np.array([0.0, 0.0, -SSD])  # z = -100
field_size_at_surface = 10.0  # cm (square)
# The field at z=0 is [-5,5] x [-5,5]

# Phantom boundaries
x_min, x_max = -15.0, 15.0
y_min, y_max = -15.0, 15.0
z_min, z_max = 0.0, 30.0

# Scoring volume: cylinder
scoring_center = np.array([0.0, 0.0, 10.0])  # z=10 cm depth
scoring_radius = 1.0  # cm
scoring_half_thick = 0.1  # cm, so thickness 0.2 cm

# Cross section functions
# Compton
def mu_Compton(E):
    alpha = E / m_e
    # Klein-Nishina cross section per electron
    t1 = (1+alpha)/alpha**2 * (2*(1+alpha)/(1+2*alpha) - math.log(1+2*alpha)/alpha)
    t2 = math.log(1+2*alpha) / (2*alpha)
    t3 = (1+3*alpha) / (1+2*alpha)**2
    sigma = 2*math.pi * r_e_sq * (t1 + t2 - t3)
    return n_e * sigma  # cm^-1

# Rayleigh: approximate using NIST data. We'll use a simple power law.
# mu_Rayleigh/rho at 2 MeV ~ 0.0002, at 1 MeV ~ 0.0008, at 0.5 MeV ~ 0.003
# We'll fit: mu_Rayleigh = a * E^b
# Using points: (0.5, 0.003), (1.0, 0.0008), (2.0, 0.0002)
# b = log(0.0002/0.0008)/log(2/1) = log(0.25)/log(2) = -2
# So mu ~ E^-2. Then a = 0.0008 * 1^2 = 0.0008? Actually 0.0008 * 1^2 = 0.0008, but at 2 MeV, 0.0008/4=0.0002, good.
# So mu_Rayleigh(E) = 0.0008 * (1/E)^2 cm^-1? Wait, these are mu/rho. So mu_Rayleigh(E) = rho * 0.0008 * (E/1)^(-2) cm^-1.
# But for E in MeV, use reference at 1 MeV: mu_Rayleigh(1 MeV) = 0.0008 cm^-1 (since rho=1)
def mu_Rayleigh(E):
    if E <= 0:
        return 0.0
    # Use power law: at 2 MeV, 0.0002; at 1 MeV, 0.0008
    # mu = 0.0008 * (1/E)**2
    # But at very low E, Rayleigh increases but eventually becomes comparable to Compton.
    # We'll cap at some value.
    mu = 0.0008 * (1.0 / max(E, 0.01))**2
    return mu

# Photoelectric: similar power law
# At 2 MeV: 1.3e-5, at 1 MeV: 1.9e-5, at 0.5 MeV: 0.0001? Actually at 0.5 MeV, photo for water is about 0.0002.
# Let's use: mu_photo = a * E^b
# Points: (1.0, 1.9e-5), (0.5, 0.0002)
# b = log(0.0002/1.9e-5)/log(0.5/1.0) = log(10.53)/log(0.5) = 2.354 / -0.693 = -3.4
# So ~ E^-3.4. At 2 MeV: 1.9e-5 * (1/2)^3.4 = 1.9e-5 * 0.093 = 1.8e-6, but we want 1.3e-5. So not perfect.
# Let's use a simple formula: mu_photo = 1.9e-5 * (E/1)^(-3.5)
def mu_Photoelectric(E):
    if E <= 0:
        return 0.0
    mu = 1.9e-5 * (1.0 / max(E, 0.01))**3.5
    return mu

# Pair production: threshold at 1.022 MeV
# At 2 MeV: 0.0004, at 3 MeV: 0.0007, at 5 MeV: 0.0020
# Approximate: mu_pair = 0.0004 * ((E-1.022)/0.978) for E>1.022, but more complex.
# Let's use a simple linear rise after threshold: mu_pair = 0.0004 * (E - 1.022) / (2.0 - 1.022) for E <= 2, then continue.
def mu_Pair(E):
    if E <= 1.022:
        return 0.0
    # Approx: at 2 MeV = 0.0004, at 3 MeV = 0.0007, at 5 MeV = 0.0020
    # Simple fit: mu_pair = 0.00058 * (E - 1.022)^1.2? Let's just use linear for simplicity.
    # We'll use: mu_pair = 0.0004 * (E - 1.022) / (2.0 - 1.022) for E < 2, else scale.
    if E < 2.0:
        return 0.0004 * (E - 1.022) / (2.0 - 1.022)
    elif E < 3.0:
        return 0.0004 + (0.0007 - 0.0004) * (E - 2.0)
    elif E < 5.0:
        return 0.0007 + (0.0020 - 0.0007) * (E - 3.0) / 2.0
    else:
        return 0.0020 * (E / 5.0)**0.5  # rough

def mu_Total(E):
    return mu_Compton(E) + mu_Rayleigh(E) + mu_Photoelectric(E) + mu_Pair(E)

# Test
print("Cross sections at 2 MeV:")
print(f"  Compton: {mu_Compton(2.0):.6f}")
print(f"  Rayleigh: {mu_Rayleigh(2.0):.6f}")
print(f"  Photoelectric: {mu_Photoelectric(2.0):.6f}")
print(f"  Pair: {mu_Pair(2.0):.6f}")
print(f"  Total: {mu_Total(2.0):.6f}")

print("\nCross sections at 1 MeV:")
print(f"  Compton: {mu_Compton(1.0):.6f}")
print(f"  Rayleigh: {mu_Rayleigh(1.0):.6f}")
print(f"  Photoelectric: {mu_Photoelectric(1.0):.6f}")
print(f"  Pair: {mu_Pair(1.0):.6f}")
print(f"  Total: {mu_Total(1.0):.6f}")

print("\nCross sections at 0.5 MeV:")
print(f"  Compton: {mu_Compton(0.5):.6f}")
print(f"  Rayleigh: {mu_Rayleigh(0.5):.6f}")
print(f"  Photoelectric: {mu_Photoelectric(0.5):.6f}")
print(f"  Pair: {mu_Pair(0.5):.6f}")
print(f"  Total: {mu_Total(0.5):.6f}")