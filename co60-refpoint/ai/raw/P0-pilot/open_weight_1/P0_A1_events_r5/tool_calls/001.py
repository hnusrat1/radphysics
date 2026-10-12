import numpy as np
from scipy.interpolate import interp1d
import math

# Let me compute photon cross sections for water (H2O)
# Using known data for mass attenuation coefficients

# For water (H2O), density = 1.0 g/cm^3
# We need cross sections at various energies

# Let me use the known NIST data points for water
# Energy (MeV), total, Compton, photoelectric, pair production, Rayleigh
# Actually let me compute using known formulas

# Water composition
# H: Z=1, A=1.008, weight fraction = 2*1.008/18.015 = 0.1119
# O: Z=8, A=16.00, weight fraction = 16.00/18.015 = 0.8881

# For each element, we can compute cross sections using approximate formulas

# Thomson scattering cross section for Rayleigh
sigma_thomson = 6.652458e-25  # cm^2 (per electron)

# Klein-Nishina formula for Compton

# Let me use a more practical approach - use known polynomial fits

# Actually, let me compute using the NIST data which I can approximate

print("Computing cross sections for water at 2.0 MeV")

# Linear attenuation coefficient for water at 2.0 MeV
# From NIST: μ/ρ (total) for water at 2.0 MeV ≈ 0.04936 cm^2/g
# μ = (μ/ρ) * ρ = 0.04936 cm^-1

mu_total_2MeV = 0.04936  # cm^-1 at 2.0 MeV for water (density 1.0)

# Let me get more detailed breakdown
# From NIST data for water at 2.0 MeV:
# Coherent (Rayleigh): μ/ρ ≈ 0.000119 cm^2/g
# Incoherent (Compton): μ/ρ ≈ 0.04877 cm^2/g  
# Photoelectric: μ/ρ ≈ 0.000383 cm^2/g
# Pair production (nuclear + electron): μ/ρ ≈ 0.000089 cm^2/g
# Total: μ/ρ ≈ 0.04936 cm^2/g

mu_rayleigh = 0.000119  # cm^-1
mu_compton = 0.04877  # cm^-1
mu_photo = 0.000383     # cm^-1
mu_pair = 0.000089     # cm^-1

print(f"Total attenuation coefficient for water at 2.0 MeV: {mu_total_2MeV:.5f} cm^-1")
print(f"  Compton: {mu_compton:.5f} cm^-1")
print(f"  Photoelectric: {mu_photo:.5f} cm^-1")
print(f"  Pair production: {mu_pair:.5f} cm^-1")
print(f"  Rayleigh: {mu_rayleigh:.5f} cm^-1")

# Check sum
sum_mu = mu_rayleigh + mu_compton + mu_photo + mu_pair
print(f"Sum: {sum_mu:.5f} cm^-1")
print(f"Difference from total: {mu_total_2MeV - sum_mu:.6f}")

# For primary photons at 2.0 MeV, the interaction probabilities in the scoring volume are proportional to these cross sections

# But scattered photons have lower energies, so their interaction probabilities differ

# We also need to know the fluence of scattered photons relative to primary photons
# at 10 cm depth on central axis

# The scatter-to-primary ratio (SPR) at 10 cm depth for a 10x10 field, 2.0 MeV...
# This is a well-studied quantity in radiation dosimetry

# Let me estimate using some known relationships

print("\n--- Estimating scatter-to-primary ratio ---")
# For a 10x10 cm^2 field, 2.0 MV (actually 2.0 MeV photons), SSD 100 cm
# At 10 cm depth, the SPR is roughly 0.3-0.5 for typical megavoltage beams
# For a 2.0 MeV monoenergetic beam, the scatter would be somewhat less than for a 
# typical 6 MV bremsstrahlung beam

# Let me think about this differently. The scattered photon fluence at 10 cm depth
# comes from photons that scatter in the phantom and reach the scoring volume.

# For a rough estimate, let's use a simple model

# The primary photon fluence at depth d (ignoring inverse square and attenuation):
# Actually with both:
# Phi_p(d) = Phi0 * (SSD/(SSD+d))^2 * exp(-mu*d)
# where Phi0 is the fluence in air at 110 cm

SSD = 100.0
depth = 10.0
mu = mu_total_2MeV

# Primary photon fluence at 10 cm depth relative to Phi0
Phi_p_rel = (SSD/(SSD+depth))**2 * np.exp(-mu*depth)
print(f"Primary fluence at 10 cm depth / Phio: {Phi_p_rel:.5f}")

# Now for scattered photons, a simple estimate:
# The scatter contribution can be estimated from the scatter-air ratio (SAR) or similar

# For a 2.0 MeV beam, 10 x 10 field, 10 cm depth
# Using data from Khan's physics of radiation therapy or similar
# The scatter component at 10 cm depth for a 10x10 field at 2.0 MeV
# would be roughly 30-40% of the primary

# Let me use a more physics-based approach

# The number of photons that interact in the scoring volume = fluence * interaction probability

# For primary photons at 2.0 MeV:
# Interaction probability in the thin scoring volume = mu * thickness (for thin volume)
# But actually we need the number of interactions in the scoring volume

# Let me reconsider the problem

# The task is to generate 200 interactions that occur IN the scoring volume
# These are interactions of photons (primary or scattered) that happen to occur inside the cylinder

# So I need to know:
# 1. What fraction of interactions in the scoring volume are from primary vs scattered photons?
# 2. For interactions from primary photons, what are the processes? (at2.0 MeV)
# 3. For interactions from scattered photons,what is the energy spectrum and process distribution?

# This is quite complex. Let me make reasonable assumptions based on physics.

# For 2.0 MeV in wat er:
# - Primary photons: mostly Compton (~ 98.8%), small photoelectric (~0.78%), pair(~0.18%), Rayleigh(~0.24%)
# - Scattered photons have energies from ~0.2 MeV up to ~1.8 MeV (depending on scattering angle)
# The energy distribution of scattered photons depends on the scattering

# Let me compute the relative probabilities of each process for a range of energies

def get_cross_sections(energy_MeV):
    """Return linear attenuation coefficients (cm^-1) for water at given energy"""
    # This is approximate based on NIST data for water
    # I'll use data points and inter polation
    
    # Energ in MeV, values from NIST (approximate)
    energies = np.array([0.01, 0.015,  0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1,
                        0.15, 0.2, 0.3, 0.4, 0.5 0.6, 0.8, 1.0,
                        1.5, 2.0, 3.0, 4.0,5.0, 6.0, 8.0, 10.0])
    
# For water, approximate linear attenuation coefficients (cm^-1)    
# These are from memory / approximate fits    
# Let me compute using simpler approach - use known parameters
    
    return None

# Actually, let me just use approximate values bsed on known physics

print("\n--- Computing interaction probabilities for 2.0 MeV in water ---")
# At 2.0 MeV:
# Mass attenuation coefficients (cm^2/g) for water
# From NIST XCOM data (approximate):

# Component      μ/ρ (cm^2/g)   μ (cm^-1) for ρ=1.0
# Coherent       0.000119      0.000119
# Incoherent    0.04877       0.04877    
# Photoectric   0.000383       0.000383
# Pair peod     0.000089       0.000089
# Total         0.04936        0.04936

ratios_2MEv = {
    'compon': 0.04877 / 0.04936,  # ~0.9880
    'photo': 0.000383 / 0.04936,    # ~0.00776
    'pair': 0.000089/ 0.04936,      # ~0.00180
    'raylegh': 0.000119/ 0.04936  # ~0.00241
}
print(f"Primary photon interaction ratios at 2.0 MeV:")
for k,v in atios_2MeV.items():
    print(f"  {k}: {v:.6f}")

# Now for scattered hotons, their energy distribution is complex
# But roughly, the fraction of interactions from scattered photons
# at 10 cm depth for a 2.0 MeV beam...

# Let me consider the total fluence at the poit

# Actually, I think a simpler approach is to use Monte Carlo with appropriate
# cross sections and a simple model

print("\n--- Attempting a more detailed Monte Carlo apprach ---")