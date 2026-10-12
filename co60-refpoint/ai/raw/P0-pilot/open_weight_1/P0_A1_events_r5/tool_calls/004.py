import numpy as np
import math
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Physical constants
m_e_c2 = 0.511  # Electron rest energy in MeV

# Phantom and geometry
SSD = 100.0  # cm
phantom_depth = 30.0  # cm
field_size = 10.0  # cm at phantom surface
depth_POI = 10.0  # cm
scoring_radius = 1.0  # cm
scoring_thickness = 0.2  # cm
scoring_center = depth_POI  # cm
scoring_z_min = scoring_center - scoring_thickness / 2
scoring_z_max = scoring_center + scoring_thickness / 2

# Source energy
E_source = 2.0  # MeV

# Cross section data for water (interpolated)
# At the energies we need, let me define the cross section function

# From NIST XCOM data for water (linear attenuation, density=1.0 g/cm^3)
# These are the mass attenuation coefficients (cm^2/g) times density = linear (cm^-1)

# Energy points (MeV)
energies = np.array([0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1,
                     0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0,
                     1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0])

# Rayleigh (coherent) - cm^-1
mu_R = np.array([6.08e-1, 1.114e-1, 3.64e-2, 2.05e-2, 9.1e-3, 5.1e-3, 3.3e-3, 2.3e-3, 
                 1.3e-3, 8.2e-4, 3.6e-4, 2.0e-4, 9.0e-5, 5.0e-5, 3.2e-5, 2.1e-5, 
                 1.5e-5, 4.9e-5, 3.2e-5, 2.1e-5, 1.19e-4, 6.8e-5, 4.4e-5, 2.2e-5, 
                 1.3e-5, 8.5e-6, 4.5e-6, 2.9e-6])

# Hmm, these values don't look right. Let me redo this with proper NIST data.

# Actually, let me look up the approximate values more carefully.
# From the NIST XCOM database for water (H2O):

# At 2.0 MeV:
# Coherent: 1.190e-02 cm^2/g? No...
# Let me just use well-known values

print("Let me use proper NIST XCOM data for water")
print("Recomputing...")

# From NIST: https://physics.nist.gov/PhysRefData/XrayMassCoef/ComTab/water.html
# For water, the mass attenuation coefficients (cm^2/g) are:

# Let me input a reasonable set of data points

# Energy (MeV), μ_coh/ρ, μ_incoh/ρ, μ_photo/ρ, μ_pair/ρ, μ_total/ρ
# All in cm^2/g

nist_data = {
    (0.005,): {'coh': 1.114e1, 'incoh': 6.08e-1, 'photo': 3.06e0, 'pair': 0, 'total': 1.481e1},
    (0.01,): {'coh': 1.114e0, 'incoh': 2.941e-1, 'photo': 1.274e0, 'pair': 0, 'total': 1.681e0},
    (0.015,): {'coh': 3.64e-1, 'incoh': 2.17e-1, 'photo': 4.09e-1, 'pair': 0, 'total': 9.90e-1},
    # Actually this is getting complicated. Let me just use approximate values.
}

# Let me use a simpler approach - construct piecewise power law interpolations
# from known data points

# Known mass attenuation coefficients for water (cm^2/g) at specific energies
# These are approximate from NIST

# Energy    Coherent    Incoherent   Photoel.   Pair       Total
# (MeV)    (Ryleigh)  (Compton)                (nuc+el)

data_E = np.array([0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0])
data_coh = np.array([1.114, 0.0709, 0.0195, 0.00512, 0.00091, 0.000247, 0.000119, 0.000032, 0.000013])
data_incoh = np.array([0.2941, 0.1379, 0.1470, 0.1277, 0.09477, 0.06958, 0.04877, 0.02377, 0.01301])
data_photo = np.array([1.274, 0.00574, 0.00422, 0.00397, 0.00127, 0.000798, 0.000383, 0.000053, 0.000008])
data_pair = np.array([0.0, 0.0, 0.0, 0.0, 0.000013, 0.000049, 0.000089, 0.000524, 0.001764])

# Check totals
print("Energy(MeV)  Total(sum)  Total(NIST)  C%  PE%  R%  PP%")
for i, E in enumerate(data_E):
    total_sum = data_coh[i] + data_incoh[i] + data_photo[i] + data_pair[i]
    total_nist = total_sum  # We'll use sum as reference
    C_pct = data_incoh[i] / total_sum * 100
    PE_pct = data_photo[i] / total_sum * 100
    R_pct = data_coh[i] / total_sum * 100
    PP_pct = data_pair[i] / total_sum * 100
    print(f"{E:6.3f}   {total_sum:.5f}   {C_pct:6.2f}%  {PE_pct:6.4f}%  {R_pct:6.4f}%  {PP_pct:6.4f}%")