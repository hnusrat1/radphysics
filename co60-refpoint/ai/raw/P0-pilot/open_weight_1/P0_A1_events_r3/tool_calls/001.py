import numpy as np
import math

# Water composition: H2O, density 1.0 g/cm^3
# Atomic numbers: H (Z=1), O (Z=8)
# Mass fractions: H 2/18 = 0.1119, O 16/18 = 0.8881
# Number of electrons per gram: ~3.34e23 for water

# Approximate cross-sections for 2 MeV photons in water
# Using NIST data from memory/approximations

# Mass attenuation coefficients (cm^2/g) for water at 2 MeV
# From NIST: 
# Total with coherent: 0.0494 cm^2/g? Let me compute more precisely.

# Actually, let's use known values:
# At 2 MeV, for water:
# μ/ρ (cm^2/g):
#   Compton: ~0.0470
#   Photoelectric: ~0.00025
#   Pair production: ~0.0007 (in nuclear field) + ~0.0001 (in electron field) = ~0.0008
#   Rayleigh: ~0.00001
# Total: ~0.0480

# Let me verify using approximate formulas
# Klein-Nishina: cross section per electron
# σ_KN = 2πr_e^2 * [ (1+γ)/γ^2 * (2(1+γ)/(1+2γ) - ln(1+2γ)/γ) + ln(1+2γ)/(2γ) - (1+3γ)/(1+2γ)^2 ]
# where r_e = 2.818e-13 cm, γ = E/m_e c^2 = E/0.511

γ = 2.0 / 0.511
r_e = 2.817940e-13  # cm
# Klein-Nishina cross section per electron
term1 = (1+γ)/γ**2 * (2*(1+γ)/(1+2*γ) - math.log(1+2*γ)/γ)
term2 = math.log(1+2*γ)/(2*γ) - (1+3*γ)/(1+2*γ)**2
σ_KN = 2 * math.pi * r_e**2 * (term1 + term2)
print(f"Klein-Nishina σ per electron = {σ_KN:.6e} cm^2")
# Number of electrons per cm^3 in water
# Water density 1 g/cm^3, molar mass 18.015 g/mol
# electrons per molecule: 10 (2 from H, 8 from O)
N_elec = 1.0 * (10 * 6.022e23 / 18.015)  # electrons/cm^3
print(f"Electron density = {N_elec:.3e} cm^-3")
μ_compton = σ_KN * N_elec
print(f"μ_compton = {μ_compton:.4f} cm^-1")

# Photoelectric: approximate using empirical fit
# For water at 2 MeV, photoelectric is very small
# Use μ_pe/ρ ~ (Z^4)/(E^3.5) roughly, but better to use known data
# From NIST: for water at 2 MeV, μ_pe/ρ ~ 2.5e-4 cm^2/g
μ_pe = 0.00025  # cm^-1 (approximate)

# Pair production: approximate using known data
# At 2 MeV, μ_pair/ρ ~ 7e-4 cm^2/g? 
# Actually from NIST data for water: at 2 MeV, μ_pair/ρ ≈ 0.0007 cm^2/g
μ_pair = 0.0007  # cm^-1

# Rayleigh: very small, ~ 1e-5 cm^-1
μ_rayleigh = 0.00001

μ_total = μ_compton + μ_pe + μ_pair + μ_rayleigh
print(f"μ_compton = {μ_compton:.6f} cm^-1")
print(f"μ_pe = {μ_pe:.6f} cm^-1")
print(f"μ_pair = {μ_pair:.6f} cm^-1")
print(f"μ_rayleigh = {μ_rayleigh:.6f} cm^-1")
print(f"μ_total = {μ_total:.6f} cm^-1")
print(f"Compton fraction = {μ_compton/μ_total:.4f}")
print(f"PE fraction = {μ_pe/μ_total:.4f}")
print(f"Pair fraction = {μ_pair/μ_total:.4f}")
print(f"Rayleigh fraction = {μ_rayleigh/μ_total:.4f}")