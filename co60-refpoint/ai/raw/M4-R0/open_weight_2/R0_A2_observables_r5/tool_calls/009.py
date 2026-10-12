import numpy as np
from scipy.special import eval_legendre

# ==================== RE-READ DATA FILES ====================
with open('/tmp/data/xcom_water.txt', 'r') as f:
    xcom_lines = f.readlines()
    
with open('/tmp/data/estar_water.txt', 'r') as f:
    estar_lines = f.readlines()

# Parse XCOM
xcom_data = []
for line in xcom_lines[4:]:
    parts = line.split()
    if len(parts) >= 7:
        try:
            vals = [float(p) for p in parts]
            xcom_data.append(vals)
        except:
            pass
xcom_array = np.array(xcom_data)

# Parse ESTAR
estar_data = []
for line in estar_lines[4:]:
    parts = line.split()
    if len(parts) >= 7:
        try:
            vals = [float(p) for p in parts]
            estar_data.append(vals)
        except:
            pass
estar_array = np.array(estar_data)

# ==================== INTERPOLATION FUNCTIONS ====================
def interpolate_xlog(E, col_idx):
    energies = xcom_array[:, 0]
    idx = np.searchsorted(energies, E)
    if idx == 0 or idx >= len(energies):
        return None
    
    log_E = np.log10(max(E, 1e-20))
    log_E_low = np.log10(energies[idx-1])
    log_E_high = np.log10(energies[idx])
    
    t = (log_E - log_E_low) / (log_E_high - log_E_low)
    
    val_low = xcom_array[idx-1, col_idx]
    val_high = xcom_array[idx, col_idx]
    
    if val_low > 0 and val_high > 0:
        return val_low * (val_high/val_low)**t
    elif val_high > 0:
        return val_high
    else:
        return 0.0

def interpolate_estar_log(E):
    energies = estar_array[:, 0]
    idx = np.searchsorted(energies, E)
    if idx == 0 or idx >= len(energies):
        return None
    
    log_E = np.log10(max(E, 1e-20))
    log_E_low = np.log10(energies[idx-1])
    log_E_high = np.log10(energies[idx])
    
    t = (log_E - log_E_low) / (log_E_high - log_E_low)
    
    result = {}
    cols_names = ['collision', 'radiative', 'total']
    col_indices = [1, 2, 3]
    
    for name, idx_col in zip(cols_names, col_indices):
        val_low = estar_array[idx-1, idx_col]
        val_high = estar_array[idx, idx_col]
        
        if val_low > 0 and val_high > 0:
            result[name] = val_low * (val_high/val_low)**t
        elif val_high > 0:
            result[name] = val_high
        else:
            result[name] = 0.0
    
    return result

# ==================== CONSTANTS ====================
E_co1 = 1.1732  # MeV
E_co2 = 1.3325  # MeV
rho = 1.0  # g/cm^3
depth = 5.0  # cm

# Cross sections
xs_co1 = {
    'coherent': interpolate_xlog(E_co1, 1),
    'incoherent': interpolate_xlog(E_co1, 2),
    'photoelectric': interpolate_xlog(E_co1, 3),
    'pair_nuclear': interpolate_xlog(E_co1, 4),
    'pair_electron': interpolate_xlog(E_co1, 5),
    'total': interpolate_xlog(E_co1, 6)
}

xs_co2 = {
    'coherent': interpolate_xlog(E_co2, 1),
    'incoherent': interpolate_xlog(E_co2, 2),
    'photoelectric': interpolate_xlog(E_co2, 3),
    'pair_nuclear': interpolate_xlog(E_co2, 4),
    'pair_electron': interpolate_xlog(E_co2, 5),
    'total': interpolate_xlog(E_co2, 6)
}

print("Cross sections Co-60:")
print(f"E1={E_co1} MeV: mu/rho = {xs_co1['total']:.6e}")
print(f"E2={E_co2} MeV: mu/rho = {xs_co2['total']:.6e}")

# Attenuation through 5cm water
mu1 = xs_co1['total'] * rho
mu2 = xs_co2['total'] * rho
att1 = np.exp(-mu1 * depth)
att2 = np.exp(-mu2 * depth)
print(f"Attenuation at 5cm: E1={att1:.6f}, E2={att2:.6f}")

# Average primary fluence ratio
L1_primary_fluence_per_phi0 = 0.5 * att1 + 0.5 * att2
print(f"L1_primary_fluence_per_phi0 = {L1_primary_fluence_per_phi0:.6f}")

# ==================== KINEMATICS OF COMPTON SCATTERING ====================
# Klein-Nishina differential cross section
def klein_nishinan_dsigma_dcoss(theta, E_gamma):
    """
    Differential cross section per unit solid angle (Klein-Nishina formula)
    Returns relative value proportional to actual cross section
    theta: scattering angle in radians
    E_gamma: incident photon energy in MeV
    """
    alpha = E_gamma / 0.510998950  # In units of electron rest mass
    
    # Energy of scattered photon
    E_prime = E_gamma / (1 + alpha * (1 - np.cos(theta)))
    
    # Klein-Nishina formula (unpolarized)
    r0_sq_over_4pi = 0.066524e-24 / (4 * np.pi)  # Classical electron radius squared / 4pi (cm^2/sr)
    
    factor = (1 + np.cos(theta)**2) / 2
    eprime_over_E = E_prime / E_gamma
    d_sigma = factor * eprime_over_E**2 * r0_sq_over_4pi
    
    return d_sigma, E_prime

# Get Compton edge energy
def compton_edge(E_gamma):
    """Maximum energy transferred to electron in Compton scattering"""
    alpha = E_gamma / 0.510998950
    E_prime_min = E_gamma / (1 + 2 * alpha)  # Backscatter (theta=180 deg)
    T_max = E_gamma - E_prime_min
    return T_max

print(f"\nCompton edges:")
print(f"  E1={E_co1}: T_max = {compton_edge(E_co1):.4f} MeV")
print(f"  E2={E_co2}: T_max = {compton_edge(E_co2):.4f} MeV")

# Calculate average energy transfer for Compton
def mean_compton_energy_transfer(E_gamma):
    """Mean fractional energy transfer for Compton scattering"""
    alpha = E_gamma / 0.510998950
    
    # Integrate numerically over cosine of scattering angle
    costheta = np.linspace(-1, 1, 1000)
    dsig_dcoss = []
    fractions = []
    
    for cos_t in costheta:
        theta = np.arccos(np.clip(cos_t, -1, 1))
        sigma, E_prime = klein_nishinan_dsigma_dcoss(theta, E_gamma)
        dsig_dcoss.append(sigma)
        T = E_gamma - E_prime
        frac = T / E_gamma
        fractions.append(frac)
    
    dsig_dcoss = np.array(dsig_dcoss)
    fractions = np.array(fractions)
    
    # Average weighted by cross section
    avg_frac = np.sum(fractions * dsig_dcoss) / np.sum(dsig_dcoss)
    return avg_frac

avg_fracs = {}
for E, name in [(E_co1, 'E1'), (E_co2, 'E2')]:
    avg_f = mean_compton_energy_transfer(E)
    avg_fracs[name] = avg_f
    print(f"  {name} ({E} MeV): <T/E>_Compton = {avg_f:.4f}")
