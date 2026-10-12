import numpy as np
from scipy import constants

# Physical constants
m_e = constants.m_e * 1e6  # electron mass in eV -> 0.511 MeV
m_e_MeV = 0.511
r_e = 2.8179403262e-13  # classical electron radius in cm
N_A = 6.02214076e23  # Avogadro's number

# Water composition: H2O, density 1 g/cm^3
# Z_eff: H (Z=1), O (Z=8)
# Mass fractions: H: 2/18 = 0.1111, O: 16/18 = 0.8889
# Electron density: water has 10 electrons per molecule, 18 g/mol
# n_e = N_A * 10 / 18 = 3.343e23 electrons/g

# Let's compute cross sections using standard formulas

def compton_cross_section(energy_MeV, material='water'):
    """Klein-Nishina integrated cross section per electron (cm^2)"""
    E = energy_MeV
    alpha = E / m_e_MeV  # E/(m_e c^2)
    
    if alpha < 1e-10:
        return 0.0
    
    # Klein-Nishina formula
    s1 = (1 + alpha) / alpha**2 * (2*(1+alpha)/(1+2*alpha) - np.log(1+2*alpha)/alpha)
    s2 = np.log(1+2*alpha) / (2*alpha)
    s3 = (1 + 3*alpha) / (1+2*alpha)**2
    
    sigma_KN = 2 * np.pi * r_e**2 * (s1 + s2 - s3)
    # This is per electron
    return sigma_KN

def compton_mass_attenuation(energy_MeV):
    """Compton mass attenuation coefficient mu/rho (cm^2/g) for water"""
    sigma_e = compton_cross_section(energy_MeV)
    # Number of electrons per gram
    n_e_per_g = N_A * 10 / 18.01528
    mu_rho = sigma_e * n_e_per_g
    return mu_rho

def rayleigh_cross_section_approx(energy_MeV):
    """Approximate Rayleigh scattering cross section per molecule (cm^2)"""
    # Rayleigh cross section ~ Z^2 * (something) / E^2 at low energies
    # For water, use approximation: sigma_R ~ 8*pi/3 * r_e^2 * (Z_eff)^2 / (1 + (alpha)^2 ...)
    # Actually, better: sigma_Rayleigh = (Z^2) * (8*pi/3) * r_e^2 * (1/(1+2.3*alpha))^2? 
    # Let's just use a simple fit to NIST data.
    # At 2 MeV, Rayleigh is negligible. At lower energies, it matters.
    # We'll use a simplified form factor approach.
    E = energy_MeV
    if E < 0.01:
        E = 0.01
    alpha = E / m_e_MeV
    # Thomson cross section per electron: 8*pi/3 * r_e^2 = 6.652e-25 cm^2
    sigma_Thomson = 8*np.pi/3 * r_e**2  # cm^2
    # For a molecule with Z_eff, Rayleigh cross section ~ (sum Z_i)^2 * sigma_Th * F(q,Z)^2
    # Approximate using Thomas-Fermi form factor
    Z_H2O = 10  # sum of Z
    # Simple approximation: Rayleigh cross section drops with energy
    # From NIST: for water at 0.01 MeV, mu_R/rho ~ 0.16 cm^2/g
    # At 0.1 MeV, ~ 0.001 cm^2/g
    # At 1 MeV, ~ 1e-6 cm^2/g
    # Let's use a power law fit
    x = np.log10(E)
    if E < 0.01:
        log_mu = 0.2 - 3.5 * x  # rough
    elif E < 1.0:
        log_mu = -0.4 - 3.0 * x
    else:
        log_mu = -9 + 0.0 * x  # very small
    mu_rho_R = 10**log_mu
    return mu_rho_R

def photoelectric_mass_attenuation(energy_MeV):
    """Photoelectric mass attenuation coefficient mu/rho (cm^2/g) for water"""
    # Photoelectric ~ Z^4 to Z^5 / E^3.5
    # For water, approximate using NIST data
    E = energy_MeV
    if E < 0.01:
        E = 0.01
    # From NIST: water at 0.01 MeV, mu_pe/rho ~ 0.4 cm^2/g
    # At 0.1 MeV, ~ 0.004 cm^2/g
    # At 1 MeV, ~ 5e-5 cm^2/g
    # At 2 MeV, ~ 1e-5 cm^2/g
    # Formula: mu_pe/rho = C * Z^4 / E^3.5
    # Let's fit constants
    x = np.log10(E)
    if E < 0.01:
        log_mu = 0.4 - 3.5 * x
    elif E < 1.0:
        log_mu = -0.4 - 3.2 * x
    else:
        log_mu = -1.5 - 3.5 * x
    mu_rho_pe = 10**log_mu
    # More accurate: use tabulated values
    return mu_rho_pe

def pair_production_mass_attenuation(energy_MeV):
    """Pair production mass attenuation coefficient mu/rho (cm^2/g) for water"""
    E = energy_MeV
    if E <= 2*m_e_MeV:
        return 0.0
    # Pair production cross section ~ Z^2 * log(2E/mc^2)
    # For water, at 2 MeV, very small
    # At 10 MeV, ~ 0.005 cm^2/g
    # At 100 MeV, ~ 0.02 cm^2/g
    # Use approximation:
    alpha = E / m_e_MeV
    # Bethe-Heitler approximation (simplified)
    sigma_pair_per_atom = 4 * constants.alpha**2 * r_e**2 * (10) * (7/9 * np.log(2*alpha) - 109/54)  # Z^2 factor
    # This is rough
    # Better: use formula from literature
    Z = 10  # effective for water? Actually sum Z^2: H2O -> 2*1^2 + 8^2 = 2+64=66
    # But pair production scales with Z^2 per atom, and water has molecules.
    # Let's just use tabulated values
    if E < 2.0:
        mu_rho_pp = 0.0
    elif E < 5.0:
        mu_rho_pp = 0.0001 * (E - 2.0) / 3.0
    elif E < 10.0:
        mu_rho_pp = 0.0001 + 0.0005 * (E - 5.0) / 5.0
    else:
        mu_rho_pp = 0.0006 + 0.001 * (E - 10.0) / 90.0
    return mu_rho_pp

# Let's compute at 2 MeV
E_test = 2.0
print(f"Energy: {E_test} MeV")
print(f"Compton mu/rho: {compton_mass_attenuation(E_test):.6f} cm^2/g")
print(f"Rayleigh mu/rho: {rayleigh_cross_section_approx(E_test):.6e} cm^2/g")
print(f"Photoelectric mu/rho: {photoelectric_mass_attenuation(E_test):.6e} cm^2/g")
print(f"Pair production mu/rho: {pair_production_mass_attenuation(E_test):.6e} cm^2/g")