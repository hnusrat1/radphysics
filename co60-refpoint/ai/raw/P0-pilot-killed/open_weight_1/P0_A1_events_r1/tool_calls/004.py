import numpy as np
import math
from scipy.integrate import quad

# Klein-Nishina total cross section per electron (cm^2)
r_e = 2.817940e-13  # cm
sigma_T = 8.0/3.0 * math.pi * r_e**2  # Thomson cross section

def sigma_KN(E_MeV):
    # E in MeV
    alpha = E_MeV / 0.511
    # integrate numerically
    def integrand(u):
        # u = cosθ
        k = alpha
        kp = k / (1 + k * (1 - u))
        # dσ/dΩ = (r_e^2/2) * (kp/k)^2 * (kp/k + k/kp - (1-u^2))
        # (1-u^2) = sin^2θ
        dsd = (r_e**2/2) * (kp/k)**2 * (kp/k + k/kp - (1-u**2))
        # integrate over dΩ = 2π sinθ dθ = 2π * sqrt(1-u^2) * dθ? Actually dΩ = 2π sinθ dθ, but we have dΩ = 2π * d(cosθ) * (-1) but using u as cosθ, dΩ = 2π * (-du) * sinθ? Wait, sinθ dθ = -du, so dΩ = 2π * sinθ dθ = 2π * (-du). So integrating over u from -1 to 1, dΩ = 2π * du? Actually sinθ dθ = -du, so dΩ = 2π (-du) = -2π du. Since we integrate from -1 to 1 reversing, we can just integrate dΩ/du = 2π * sinθ * (dθ/du) = 2π. Not exactly.
        # Let's write dσ = (dσ/dΩ) * dΩ, with dΩ = 2π sinθ dθ. Since u = cosθ, du = -sinθ dθ, so dΩ = 2π sinθ dθ = 2π * (-du) * sinθ/ sinθ? Actually sinθ dθ = -du, so dΩ = 2π * (-du). So dσ = (dσ/dΩ) * 2π * (-du). When integrating from u=1 to u=-1, we have ∫ dΩ = ∫_{-1}^{1} 2π du (since we can drop sign). So σ = ∫ (dσ/dΩ) * 2π du.
        return dsd * 2*math.pi
    # integrate from u=-1 to 1
    result, err = quad(integrand, -1, 1, limit=200)
    return result

# Compute for 2 MeV
sigma_2 = sigma_KN(2.0)
print("Sigma_KN per electron (cm^2):", sigma_2)
# Number of electrons per gram for water
NA = 6.022e23
Z_H2O = 10  # sum of atomic numbers
A_H2O = 18.01528  # g/mol
electrons_per_gram = NA * Z_H2O / A_H2O
print("Electrons per gram:", electrons_per_gram)
mu_inc_calc = electrons_per_gram * sigma_2
print("μ_incoherent (cm^2/g) from KN:", mu_inc_calc)