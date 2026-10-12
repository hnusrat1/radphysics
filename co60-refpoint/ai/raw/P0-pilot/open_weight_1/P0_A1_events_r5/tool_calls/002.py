import numpy as np
from scipy import interpolate
import math

# Let me get the NIST data for water more systematically
# I'll use known values for cross sections

# Water: H2O, density = 1.0 g/cm^3
# For 2.0 MeV photons in water:

# Mass attenuation coefficients from NIST (approximate)
# Units: cm^2/g

# At 2.0 MeV:
# Coherent (Rayleigh):  0.000119 cm^2/g
# Incoherent (Compton): 0.048770 cm^2/g
# Photoelectric:         0.000383 cm^2/g
# Pair production:       0.000089 cm^2/g
# Total:                 0.049361 cm^2/g

# Linear attenuation coefficients (density = 1.0 g/cm^3):
mu_R_2 = 0.000119
mu_C_2 = 0.048770
mu_PE_2 = 0.000383
mu_PP_2 = 0.000089
mu_tot_2 = mu_R_2 + mu_C_2 + mu_PE_2 + mu_PP_2

print(f"Total mu at 2.0 MeV: {mu_tot_2:.6f} cm^-1")
print(f"  Rayleigh: {mu_R_2:.6f} ({mu_R_2/mu_tot_2*100:.4f}%)")
print(f"  Compton:  {mu_C_2:.6f} ({mu_C_2/mu_tot_2*100:.4f}%)")
print(f"  Photoel:  {mu_PE_2:.6f} ({mu_PE_2/mu_tot_2*100:.4f}%)")
print(f"  Pair:     {mu_PP_2:.6f} ({mu_PP_2/mu_tot_2*100:.4f}%)")

# Now I need to think about scattered photons
# The interaction rates in the scoring volume depend on:
# 1. Fluence of primary photons at 10 cm depth
# 2. Fluence of scattered photons at 10 cm depth
# 3. The interaction cross sections for each photon

# First, let me compute the primary photon fluence at depth
SSD = 100.0  # cm
depth = 10.0  # cm
d_source = SSD + depth  # 110 cm

# Primary photon fluence at depth d on central axis
# Phi_p = Phi0 * (SSD/(SSD+depth))^2 * exp(-mu_total * depth)
# where Phi0 is the fluence at 110 cm in air

Phi_p_rel = (SSD / (SSD + depth))**2 * np.exp(-mu_tot_2 * depth)
print(f"\nPrimary fluence relative to Phi0: {Phi_p_rel:.6f}")

# Now for scattered photon fluence
# This is more complex. Let me think about a simple model.

# The scatter-to-primary ratio (SPR) for a 10x10 field at 10 cm depth
# For 2 MeV (or 2 MV) photons

# From published data (e.g., Khan's Physics of Radiation Therapy):
# For a 10x10 cm^2 field at 10 cm depth:
# - Co-60 (1.25 MeV): SPR ≈ 0.35
# - 4 MV: SPR ≈ 0.25  
# - 6 MV: SPR ≈ 0.20
# - 10 MV: SPR ≈ 0.15

# For 2.0 MeV monoenergetic - this is between Co-60 and 4 MV
# Let's estimate SPR ≈ 0.30

# But wait, this SPR is for dose, not for fluence
# The SPR for fluence would be different because scattered photons have
# different energy and thus different stopping powers / mass energy absorption

# Actually, I need the ratio of scattered photon fluence to primary photon fluence
# at the point of interest

# For a rough estimate:
# Scattered photons at depth come from Compton scattering in the phantom
# The mean energy of scattered photons reaching 10 cm depth is roughly 0.5-1.0 MeV

# Let me estimate using a very simple model:
# The fluence of singly-scattered photons at a point can be calculated
# using the Klein-Nishina differential cross section

# But this is getting complex. Let me try a different approach.

# Actually, for the purpose of this problem, I think we can make reasonable
# estimates. Let me check what kind of accuracy is expected.

print(f"\nPhoton mean free path at 2.0 MeV: {1/mu_tot_2:.3f} cm")
print(f"Scoring volume thickness: 0.2 cm")
print(f"Interction probability for primary photon traversing scoring volume: {mu_tot_2 * 0.2:.6f}")

# So each primary photon has about a 1% chance of interacting in the scoring volume

# Now let me think about what fraction of interactions in the scoring volume
# are from primary vs scatterd photons

# The interaction rate density at the point is:
# R = sum over all photon energies of (photon fluenc rate * mu(E))
# For primary: R_p = Phi_p * mu_total(2.0 MeV)
# For scattered: R_s = integral of Phil_s(E) * mu_total(E) dE

# If we assume SPR for fluence is about 0.3 (scattered/primary fluence ratio)
# and scattered photons have mean energy around 0.6-0.8 MeV...

# Let me compute cross sections at lower energies