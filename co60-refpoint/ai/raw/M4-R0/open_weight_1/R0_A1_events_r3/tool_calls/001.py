import numpy as np
from scipy import constants

# Let's compute mass attenuation coefficients for water at 1.1732 and 1.3325 MeV
# We'll use the known NIST data approximations and also compute for interpolation

# Known mass attenuation coefficients for water (cm^2/g) from NIST:
# Energy (MeV): 1.0 -> 0.0706, 1.25 -> 0.0636, 1.5 -> 0.0576
# Let me use more precise values

# For water, I'll interpolate NIST data
energies_nist = np.array([1.0, 1.25, 1.5])
mu_rho_nist = np.array([0.0706, 0.0636, 0.0576])

# Let me get more detailed data for water from NIST
# Actually let me compute using basic formulas

# H: Z=1, A=1.008
# O: Z=8, A=16.00
# Water: H2O, molar mass = 2*1.008 + 16.00 = 18.016 g/mol

# Mass fractions
w_H = 2*1.008/18.016
w_O = 16.00/18.016
print(f"Mass fractions: H={w_H:.4f}, O={w_O:.4f}")

# Let me compute approximate cross sections using the Klein-Nishina formula for Compton,
# and known approximations for photoelectric, pair production, and Rayleigh.

# For a more accurate approach, let me use the NIST XCOM data which I'll approximate
# Water composition by weight: H 11.19%, O 88.81%

# At 1.1732 MeV:
# μ/ρ (total) ≈ 0.0633 cm^2/g (from NIST for water at 1.17 MeV)
# At 1.3325 MeV:
# μ/ρ (total) ≈ 0.0595 cm^2/g (from NIST for water at 1.33 MeV)

# Let me be more precise
# Using NIST XCOM data for water:

# Energy (MeV), μ_Compton/ρ, μ_photo/ρ, μ_pair/ρ, μ_Rayleigh/ρ, μ_total/ρ
# 1.0:    0.0633,  0.00377,  0.0000,  0.00303,  0.0701
# 1.17:   
# 1.25:   0.0575,  0.00258,  0.00071, 0.00242,  0.0632
# 1.33:   
# 1.5:   0.0522,  0.00179,  0.00146, 0.00186,  0.0573

# Let me look up better values. I'll use the NIST data I can recall.

# For water at 1.25 MeV (approximate):
# Compton: μ/ρ ≈ 0.0575 cm^2/g
# Photo: μ/ρ ≈ 0.0026 cm^2/g  
# Pair: μ/ρ ≈ 0.0007 cm^2/g
# Rayleigh: μ/ρ ≈ 0.0024 cm^2/g
# Total: ≈ 0.0632 cm^2/g

# At 1.0 MeV:
# Compton: 0.0633, Photo: 0.00377, Pair: 0.0000, Rayleigh: 0.00303

# Let me get more accurate values by interpolation from known data points
# I'll use python to compute using the Klein-Nishina formula for Compton

print("Let me compute more precisely using Klein-Nishina and other formulas")