import numpy as np
from scipy.integrate import quad

# Read data files
xcom_data = []
with open('/tmp/data/xcom_water.txt', 'r') as f:
    for line in f:
        if not line.startswith('#'):
            parts = [float(x) for x in line.split()]
            xcom_data.append(parts)
xcom = np.array(xcom_data)

estar_data = []
with open('/tmp/data/estar_water.txt', 'r') as f:
    for line in f:
        if not line.startswith('#'):
            parts = [float(x) for x in line.split()]
            estar_data.append(parts)
estar = np.array(estar_data)

# Parameters
E_photon = 2.0
rho = 1.0
depth = 10.0
E_idx = 28  # 2.0 MeV

mu_ray_mass = rho * xcom[E_idx, 1]
mu_compton_mass_val = rho * xcom[E_idx, 2]
mu_photo_mass_val = rho * xcom[E_idx, 3]
mu_pair_nuc_mass_val = rho * xcom[E_idx, 4]
mu_pair_el_mass_val = rho * xcom[E_idx, 5]
mu_total_mass_val = rho * xcom[E_idx, 7]

mu_total = mu_total_mass_val
mu_compton = mu_compton_mass_val
mu_pair = mu_pair_nuc_mass_val + mu_pair_el_mass_val
mu_rayleigh = mu_ray_mass
mu_photo = mu_photo_mass_val

alpha = E_photon / 0.511

def KN(ct):
    k = 1.0 + alpha*(1.0-ct)
    s = 1.0 - ct**2
    return (1.0/k**2)*(k + 1.0/k - s)

def e_KN(ct):
    k = 1.0 + alpha*(1.0-ct)
    return E_photon*(1 - 1.0/k)

n_ebins = 25
ebins = np.array([0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 
                  0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.05, 
                  1.1, 1.15, 1.2, 1.4])

# Compton electron spectrum (including low energies)
cos_vals = np.linspace(-1, 1, 10001)
w_KN = KN(cos_vals)
e_energies = e_KN(cos_vals)

total_w = np.sum(w_KN)
L3_comp_spec = np.zeros(n_ebins)
for i in range(n_ebins):
    mask = (e_energies >= ebins[i]) & (e_energies < ebins[i+1])
    L3_comp_spec[i] = np.sum(w_KN[mask]) / total_w

# Check sum includes full range
print(f"Compton spectrum sum: {np.sum(L3_comp_spec):.6f}")
print(f"First bin (<0.05 MeV): {np.sum(e_energies[e_energies < 0.05] > 0):} electrons")

# For low-energy electrons, we need to include bin starting at 0 (not shown in bins)
# The bins start at 0.01 MeV, so electrons with energy < 0.01 MeV are not counted
# But our integration covers full cos(theta) range, so let's check
print(f"Min e energy: {np.min(e_energies):.6f}, Max: {np.max(e_energies):.6f}")

# Fraction below 0.01 MeV
mask_low = e_energies < 0.01
frac_below = np.sum(w_KN[mask_low]) / total_w
print(f"Fraction below 0.01 MeV: {frac_below:.6f}")

# Adjusted spectrum (normalized over given bins)
adjusted_spec = L3_comp_spec / (1.0 - frac_below) if frac_below < 1 else L3_comp_spec
print(f"Adjusted spectrum sum: {np.sum(adjusted_spec):.6f}")