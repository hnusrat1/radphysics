import numpy as np

# For energy transfer fractions, I need mean energy transferred per interaction
# For Compton scattering at Co-60 energies, the average fractional energy transfer can be estimated
# Using the Klein-Nishina cross section integration or approximate formulas

# For Rayleigh scattering, essentially no energy is transferred (elastic scattering)
# For photoelectric, nearly all photon energy transfers (binding energy is small compared to 1.25 MeV)
# For pair production, about E - 2*m_e*c^2 = E - 1.022 MeV transfers to kinetic energy

# Let me use simplified estimates:
# Mean energy transfer fraction for Compton scattering ~ 0.2-0.3 for Co-60 energies
# This depends on scattering angle distribution

# From ESTAR, I can compute collision kerma more directly later
# For now, let's compute energy-weighted interaction fractions

# Rough estimate of mean energy transfer per interaction process:
# Rayleigh: f_energy_transfer ≈ 0 (coherent elastic)
# Compton: f_energy_transfer ≈ 0.25 (average for Co-60 energies in water)  
# Photoelectric: f_energy_transfer ≈ 1.0 (all energy to electrons)
# Pair: f_energy_transfer ≈ (E - 1.022)/E ≈ 0.18 for E~1.25 MeV

# Energy transfer fractions weighted by interaction probabilities and energy transfer efficiency:
f_transfer_rayleigh = 0.0
f_transfer_compton = 0.25  # approximate
f_transfer_photo = 1.0
f_transfer_pair = 0.18

# Weighted by interaction fractions calculated earlier:
fraction_rayleigh = 0.0005844
fraction_compton = 0.998919
fraction_photo = 3.784e-05
fraction_pair = 0.0004598

energy_transfer_weighted = (fraction_rayleigh * f_transfer_rayleigh + 
                           fraction_compton * f_transfer_compton + 
                           fraction_photo * f_transfer_photo + 
                           fraction_pair * f_transfer_pair)

print(f"Total energy transfer weight: {energy_transfer_weighted:.6f}")

# Fractional contribution to energy transfer from each process:
et_frac_rayleigh = fraction_rayleigh * f_transfer_rayleigh / energy_transfer_weighted if energy_transfer_weighted > 0 else 0
et_frac_compton = fraction_compton * f_transfer_compton / energy_transfer_weighted if energy_transfer_weighted > 0 else 0
et_frac_photo = fraction_photo * f_transfer_photo / energy_transfer_weighted if energy_transfer_weighted > 0 else 0
et_frac_pair = fraction_pair * f_transfer_pair / energy_transfer_weighted if energy_transfer_weighted > 0 else 0

print("\nEnergy transfer fractions:")
print(f"  Rayleigh: {et_frac_rayleigh:.6e}")
print(f"  Compton: {et_frac_compton:.6f}")
print(f"  Photoelectric: {et_frac_photo:.6e}")
print(f"  Pair: {et_frac_pair:.6e}")
