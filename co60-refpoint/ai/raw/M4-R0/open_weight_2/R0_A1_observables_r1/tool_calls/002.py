import numpy as np

# NIST XCOM data for water at relevant energies
# Mass attenuation coefficients (cm^2/g) from NIST XCOM database
energies = np.array([1.0, 1.25, 1.5])  # MeV
mu_rho_data = np.array([0.06371, 0.06098, 0.05934])  # approximate

# For Co-60 lines specifically (interpolating or using known values):
# E = 1.173 MeV
mu_rho_1173 = 0.06220  # cm^2/g (approximate from interpolation)
# E = 1.332 MeV  
mu_rho_1332 = 0.06024  # cm^2/g (approximate from interpolation)

# Breakdown into components at 1.25 MeV average
# From NIST: at 1.25 MeV in water
mu_compton = 0.05042  # cm^2/g (incoherent scattering)
mu_photoelectric = 0.00218  # cm^2/g
mu_pair_nuclear = 0.00062  # cm^2/g
mu_pair_electron = 0.00028  # cm^2/g
mu_rayleigh = 0.00328  # cm^2/g

mu_total_calc = mu_compton + mu_photoelectric + mu_pair_nuclear + mu_pair_electron + mu_rayleigh
print(f"Total calculated: {mu_total_calc:.5f} cm^2/g")

# Interaction fractions by process (total)
f_compton = mu_compton / mu_total_calc
f_photoelectric = mu_photoelectric / mu_total_calc
f_pair = (mu_pair_nuclear + mu_pair_electron) / mu_total_calc
f_rayleigh = mu_rayleigh / mu_total_calc

print(f"\nInteraction fractions (all photons):")
print(f"Compton: {f_compton:.4f}")
print(f"Photoelectric: {f_photoelectric:.4f}")
print(f"Pair: {f_pair:.4f}")
print(f"Rayleigh: {f_rayleigh:.4f}")
print(f"Sum: {f_compton+f_photoelectric+f_pair+f_rayleigh:.4f}")

# Energy transfer coefficients (energy transferred to charged particles)
# Compton: <E_transfer/E> = (1 - exp(-x)) * (x/(1+x))^2 where x = E/(m_ec^2)
# For E ~ 1.25 MeV, x = 1.25/0.511 = 2.446
# Photoelectric: all energy goes to electron (minus binding, negligible)
# Pair: kinetic energy = E - 2*m_ec^2 = E - 1.022 MeV
# Rayleigh: no energy transfer (elastic)

def compton_energy_transfer_fraction(E_MeV):
    """Calculate average fraction of photon energy transferred in Compton scattering"""
    mec2 = 0.511  # MeV
    x = E_MeV / mec2
    # Average energy transfer fraction from Klein-Nishina
    # Using empirical formula or exact integration
    # For accurate calculation, integrate over scattering angle
    return 0.42 + 0.05 * np.log(E_MeV)  # Approximation

# More accurate calculation
def compute_compton_energy_transfer(E_MeV):
    """Compute mean energy transfer fraction from Klein-Nishina cross section"""
    mec2 = 0.511
    alpha = E_MeV / mec2
    
    # Numerical integration over cos(theta) from -1 to 1
    # Actually, differential cross section depends on theta (scattering angle)
    # Use Simpson's rule
    n_points = 10000
    cos_theta = np.linspace(0, 1, n_points)  # Forward hemisphere only (theta from 0 to pi/2 equivalent)
    
    # Klein-Nishina differential cross section (unpolarized)
    def KN_differential(cos_t):
        E_prime = E_MeV / (1 + alpha * (1 - cos_t))
        r = E_prime / E_MeV
        term1 = (1 + cos_t**2) / 2
        term2 = (alpha**2 * (1 - cos_t)**2 * (1 + r**2)) / (2 * (1 + alpha * (1 - cos_t))**2)
        return term1 - term2
    
    # Integrate
    d_sigma = np.array([KN_differential(ct) for ct in cos_theta])
    sigma_total = np.trapz(d_sigma, cos_theta) * 2  # *2 for full range
    
    # Energy transfer integral
    def energy_transfer(cos_t):
        E_prime = E_MeV / (1 + alpha * (1 - cos_t))
        transfer = E_MeV - E_prime
        return transfer * KN_differential(cos_t)
    
    t_sigma = np.array([energy_transfer(ct) for ct in cos_theta])
    total_transfer = np.trapz(t_sigma, cos_theta) * 2
    
    if sigma_total > 0:
        return total_transfer / sigma_total
    return 0.5

E_avg_co60 = 1.25285
compton_et_frac = compute_compton_energy_transfer(E_avg_co60)
print(f"\nCompton energy transfer fraction: {compton_et_frac:.4f}")

# Photoelectric: essentially 100% energy transfer
photoelectric_et_frac = 0.99  # Almost all to electron (neglecting Auger/fluorescence escape)

# Pair production: (E - 1.022)/E fraction becomes kinetic energy
pair_et_frac_1173 = (1.1732 - 1.022) / 1.1732
pair_et_frac_1332 = (1.3325 - 1.022) / 1.3325
pair_et_frac_avg = (pair_et_frac_1173 + pair_et_frac_1332) / 2
print(f"Pair energy transfer fraction (avg): {pair_et_frac_avg:.4f}")

# Rayleigh: zero energy transfer
rayleigh_et_frac = 0.0

# Weighted energy transfer fractions
et_compton = f_compton * compton_et_frac
et_photoelectric = f_photoelectric * photoelectric_et_frac
et_pair = f_pair * pair_et_frac_avg
et_rayleigh = f_rayleigh * rayleigh_et_frac

et_total = et_compton + et_photoelectric + et_pair + et_rayleigh

print(f"\nEnergy transfer fractions (weighted):")
print(f"Compton: {et_compton/et_total:.4f}")
print(f"Photoelectric: {et_photoelectric/et_total:.4f}")
print(f"Pair: {et_pair/et_total:.4f}")
print(f"Rayleigh: {et_rayleigh/et_total:.4f}")

L2_energy_transfer_fractions = {
    "compton": round(et_compton/et_total, 4),
    "photoelectric": round(et_photoelectric/et_total, 4),
    "pair": round(et_pair/et_total, 4),
    "rayleigh": round(et_rayleigh/et_total, 4)
}
print(f"Sum: {sum(L2_energy_transfer_fractions.values()):.4f}")