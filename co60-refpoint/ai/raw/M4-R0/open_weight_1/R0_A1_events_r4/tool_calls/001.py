import numpy as np
import math
from scipy.special import ellipe, ellipk

# Constants
mec2 = 0.511  # MeV, electron rest energy
r0 = 2.8179403262e-13  # cm, classical electron radius
NA = 6.02214076e23  # particles/mol

# Water composition: H2O
# Z=8, A=16 for O; Z=1, A=1 for H
# Mass fractions: w_H = 2/18, w_O = 16/18
# Number of electrons per gram: NA * (w_H * 1 + w_O * 8) / (2+16) = NA * (2+8)/18 = NA*10/18
n_e_per_g = NA * 10.0 / 18.0  # electrons/g
print(f"Electrons per gram: {n_e_per_g:.4e}")
# At density 1 g/cm3, electrons per cm3:
n_e_per_cm3 = n_e_per_g  # since density=1
print(f"Electrons per cm3: {n_e_per_cm3:.4e}")

# Klein-Nishina formula for total cross section per electron
def sigma_kn(E):
    """Klein-Nishina total cross section in cm^2 per electron"""
    alpha = E / mec2
    if alpha <= 0:
        return 0.0
    x = 1.0 + 2.0*alpha
    term1 = (x*x - 1.0) / (2.0*alpha**2 * x) + 1.0/(alpha**2) * ( (alpha**2 - 1.0)/ (2.0*alpha) * math.log(x) + (1.0 - 2.0*alpha - 3.0*alpha**2)/ (6.0) )
    # Actually let's use the standard formula:
    # sigma_KN = 2*pi*r0^2 * [ (1+alpha)/alpha^2 * (2*(1+alpha)/(1+2*alpha) - ln(1+2*alpha)/alpha ] + ...
    # Better: use known expression
    # From: sigma_KN = 2*pi*r0^2 * { (1+alpha)/alpha^2 * [ 2*(1+alpha)/(1+2*alpha) - ln(1+2*alpha)/alpha ] + ln(1+2*alpha)/(2*alpha) - (1+3*alpha)/(1+2*alpha)^2 }
    a = alpha
    a2 = a*a
    x = 1.0 + 2.0*a
    lnx = math.log(x)
    term = (1.0+a)/a2 * (2.0*(1.0+a)/x - lnx/a)
    term2 = lnx/(2.0*a) - (1.0+3.0*a)/(x*x)
    sigma = 2.0*math.pi*r0*r0 * (term + term2)
    return sigma

# Test
for E in [1.1732, 1.3325]:
    print(f"E={E} MeV, sigma_KN={sigma_kn(E):.6e} cm^2/electron")
    print(f"  mu_inc = sigma_KN * n_e_per_cm3 = {sigma_kn(E)*n_e_per_cm3:.6f} cm^-1")