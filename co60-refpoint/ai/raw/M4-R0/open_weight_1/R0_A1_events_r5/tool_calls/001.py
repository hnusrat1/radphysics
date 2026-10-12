import numpy as np
from scipy import integrate, interpolate
import json

# Physical constants
m_e_c2 = 0.51099895  # electron rest mass energy in MeV
r_e = 2.8179403262e-13  # classical electron radius in cm

# ============================================================
# Cross section data for water (density 1.0 g/cm^3)
# From NIST XCOM and ESTAR data, approximate values
# ============================================================

# Energies in MeV
energies = np.array([1.0, 1.1, 1.1732, 1.2, 1.25, 1.3, 1.3325, 1.4, 1.5])

# Total mass attenuation coefficients (cm^2/g) for water
# Interpolated from NIST data
mu_rho_total = np.array([0.0707, 0.0678, 0.0659, 0.0652, 0.0640, 0.0628, 0.0622, 0.0608, 0.0578])

# Mass attenuation coefficients for each process (cm^2/g)
mu_rho_compton = np.array([0.0695, 0.0665, 0.0643, 0.0635, 0.0620, 0.0606, 0.0600, 0.0585, 0.0555])
mu_rho_photo = np.array([0.00040, 0.00030, 0.00025, 0.00022, 0.00020, 0.00017, 0.00015, 0.00013, 0.00010])
mu_rho_pair = np.array([0.0, 0.0, 0.00020, 0.00028, 0.00050, 0.00080, 0.00100, 0.00130, 0.00150])
mu_rho_rayleigh = np.array([0.00080, 0.00068, 0.00060, 0.00055, 0.00050, 0.00043, 0.00040, 0.00035, 0.00030])

# Create interpolation functions
f_total = interpolate.interp1d(energies, mu_rho_total, kind='cubic', bounds_error=False, fill_value='extrapolate')
f_compton = interpolate.interp1d(energies, mu_rho_compton, kind='cubic', bounds_error=False, fill_value='extrapolate')
f_photo = interpolate.interp1d(energies, mu_rho_photo, kind='cubic', bounds_error=False, fill_value='extrapolate')
f_pair = interpolate.interp1d(energies, mu_rho_pair, kind='cubic', bounds_error=False, fill_value='extrapolate')
f_rayleigh = interpolate.interp1d(energies, mu_rho_rayleigh, kind='cubic', bounds_error=False, fill_value='extrapolate')

def get_mu_total(E):
    return f_total(E) * 1.0  # multiply by density to get linear coefficient

def get_mu_comp(E):
    return f_compton(E) * 1.0

def get_mu_photo(E):
    return f_photo(E) * 1.0

def get_mu_pair(E):
    return f_pair(E) * 1.0

def get_mu_rayl(E):
    return f_rayleigh(E) * 1.0

# Verify sum is close to total
for E in [1.1732, 1.3325]:
    sum_p = get_mu_comp(E) + get_mu_photo(E) + get_mu_pair(E) + get_mu_rayl(E)
    print(f"E={E} MeV: total={get_mu_total(E):.6f}, sum={sum_p:.6f}")

# ============================================================
# Klein-Nishina differential cross section
# ============================================================

def compton_out_energy(E_in, cos_theta):
    """Outgoing photon energy after Compton scattering."""
    return E_in / (1 + (E_in / m_e_c2) * (1 - cos_theta))

def klein_nishina_dcs(E_in, cos_theta):
    """Klein-Nishina differential cross section dσ/dΩ (in cm^2/sr/electron)."""
    theta = np.arccos(cos_theta)
    E_out = compton_out_energy(E_in, cos_theta)
    # Klein-Nishina
    dcs = (r_e**2 / 2) * (E_out / E_in)**2 * (E_in/E_out + E_out/E_in - np.sin(theta)**2)
    return dcs

def sample_compton_photon(E_in):
    """Sample outgoing photon direction from Klein-Nishina distribution using rejection sampling."""
    # Maximum of KN dcs occurs at forward scattering (cos_theta=1)
    # As cos_theta -> 1, E_out -> E_in, sin^2(theta) -> 0
    # dσ/dΩ -> (r_e^2/2) * 1^2 * (1+1-0) = r_e^2
    max_dcs = r_e**2
    
    while True:
        cos_theta = np.random.uniform(-1, 1)
        dcs = klein_nishina_dcs(E_in, cos_theta)
        if np.random.random() * max_dcs < dcs:
            break
    
    E_out = compton_out_energy(E_in, cos_theta)
    return E_out, cos_theta

def sample_compton_electron(E_in, cos_theta):
    """Return kinetic energy and cos of electron from Compton scatter."""
    E_out = compton_out_energy(E_in, cos_theta)
    t_e = E_in - E_out
    # Electron direction from momentum conservation
    # cos(electron) = (E_in - E_out*cos_theta) / sqrt(E_in^2 + E_out^2 - 2*E_in*E_out*cos_theta)
    num = E_in - E_out * cos_theta
    den = np.sqrt(E_in**2 + E_out**2 - 2*E_in*E_out*cos_theta)
    cos_electron = num / den if den > 0 else 1.0
    return t_e, cos_electron

print("Klein-Nishina sampling test:")
for E_in in [1.1732, 1.3325]:
    E_out, cos_th = sample_compton_photon(E_in)
    t_e, cos_el = sample_compton_electron(E_in, cos_th)
    print(f"  E_in={E_in}, E_out={E_out:.4f}, cos_th={cos_th:.4f}, t_e={t_e:.4f}, cos_el={cos_el:.4f}")