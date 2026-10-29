import numpy as np
from scipy.interpolate import interp1d

# Load data
xcom = np.loadtxt('/tmp/data/xcom_water.txt')
estar = np.loadtxt('/tmp/data/estar_water.txt')

# Define energy grids
e_photon = xcom[:, 0]
coherent = xcom[:, 1]  # Rayleigh
incoherent = xcom[:, 2]  # Compton
photoelectric = xcom[:, 3]
pair_nuclear = xcom[:, 4]
pair_electron = xcom[:, 5]
total_with_coherent = xcom[:, 6]
total_without_coherent = xcom[:, 7]

e_electron = estar[:, 0]
s_col = estar[:, 1]  # Collision stopping power
s_rad = estar[:, 2]  # Radiative stopping power
s_tot = estar[:, 3]  # Total stopping power
csda_range = estar[:, 4]  # CSDA range
rad_yield = estar[:, 5]  # Radiation yield

# Interpolating functions
def mass_attenuation(e):
    """Total mass attenuation coefficient (including coherent)"""
    f = interp1d(e_photon, total_with_coherent, kind='cubic', fill_value='extrapolate')
    return f(e)

def mass_energy_transfer(e):
    """Mass energy transfer coefficient (excluding coherent scattering)"""
    # Energy transfer = total without coherent
    # But we also need to account for pair production rest mass
    # For pair production, energy transferred = E_gamma - 2*m_e*c^2
    # Actually mass energy-transfer excludes rest mass of electron-positron pairs
    # Mass energy-absorption further subtracts radiative losses
    
    # Let's compute mass energy-transfer properly
    # = sum over all processes except coherent of (mu/rho)_i * (E_transferred/E_gamma)
    
    # For coherent: no energy transfer
    # For incoherent (Compton): fraction of energy transferred depends on angle
    # For photoelectric: full energy transferred (neglecting fluorescence)
    # For pair nuclear/electron: E_gamma - 2m_ec^2 transferred (rest mass not transferred)
    
    return interp1d(e_photon, total_without_coherent, kind='cubic', fill_value='extrapolate')(e)

def mass_energy_absorption(e):
    """Mass energy absorption coefficient"""
    # This requires accounting for radiative losses from secondary electrons
    # mu_en = mu_tr * (1 - g) where g is radiative yield
    # Approximate: mu_en ≈ mu_tr * (1 - average_radiative_yield)
    
    # Get mass energy transfer
    mu_tr = mass_energy_transfer(e)
    
    # Approximate radiative yield for electrons produced by photons
    # At ~1 MeV, g is small (~0.002-0.003 for water)
    # We'll use a rough approximation
    f_rad_yield = interp1d(e_electron, rad_yield, kind='linear', fill_value='extrapolate')
    
    # Average energy of secondary electrons ~ 0.5 * E_gamma for Compton
    avg_electron_energy = e * 0.5
    g_avg = f_rad_yield(avg_electron_energy)
    
    return mu_tr * (1 - g_avg)

# Co-60 energies
e_co60_1 = 1.1732  # MeV
e_co60_2 = 1.3325  # MeV

print("At Co-60 energies:")
for e in [e_co60_1, e_co60_2]:
    print(f"\nEnergy = {e} MeV:")
    print(f"  Coherent (Rayleigh): {interp1d(e_photon, coherent, kind='cubic')(e):.6e}")
    print(f"  Incoherent (Compton): {interp1d(e_photon, incoherent, kind='cubic')(e):.6e}")
    print(f"  Photoelectric: {interp1d(e_photon, photoelectric, kind='cubic')(e):.6e}")
    print(f"  Pair nuclear: {interp1d(e_photon, pair_nuclear, kind='cubic')(e):.6e}")
    print(f"  Pair electron: {interp1d(e_photon, pair_electron, kind='cubic')(e):.6e}")
    print(f"  Total (with coherent): {interp1d(e_photon, total_with_coherent, kind='cubic')(e):.6e}")
    print(f"  Total (without coherent): {interp1d(e_photon, total_without_coherent, kind='cubic')(e):.6e}")
