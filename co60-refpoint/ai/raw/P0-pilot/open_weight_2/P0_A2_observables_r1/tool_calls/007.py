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

# Define parameters
E_photon = 2.0  # MeV
rho = 1.0  # g/cm^3
depth = 10.0  # cm
r_source_surf = 100.0  # cm
r_source_PI = 110.0  # cm

# Cross-sections at 2.0 MeV (index 28)
E_idx = 28
mu_rayleigh_mass = xcom[E_idx, 1]
mu_compton_mass = xcom[E_idx, 2]
mu_photo_mass = xcom[E_idx, 3]
mu_pair_nuc_mass = xcom[E_idx, 4]
mu_pair_el_mass = xcom[E_idx, 5]
mu_total_mass = xcom[E_idx, 7]  # without coherent

mu_rayleigh = rho * mu_rayleigh_mass
mu_compton = rho * mu_compton_mass
mu_photo = rho * mu_photo_mass
mu_pair = rho * (mu_pair_nuc_mass + mu_pair_el_mass)
mu_total = rho * mu_total_mass

# Bin definitions
photon_energy_edges = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 
                                0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.05, 
                                1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4])
electron_energy_edges = np.array([0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 
                                  0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.05, 
                                  1.1, 1.15, 1.2, 1.4])
cos_theta_edges = np.array([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])

n_photon_bins = len(photon_energy_edges) - 1
n_electron_bins = len(electron_energy_edges) - 1
n_cos_bins = len(cos_theta_edges) - 1

print("Calculating Klein-Nishina distribution for 2.0 MeV photons")
alpha = E_photon / 0.511
print(f"alpha = {alpha:.4f}")

def klein_nishina_intensity(alpha, cos_theta):
    """Klein-Nishina differential cross-section intensity"""
    k = 1.0 + alpha * (1.0 - cos_theta)
    # Intensity is proportional to (1/k^2) * (k + 1/k - sin^2(theta))
    # where sin^2(theta) = 1 - cos^2(theta)
    sin_sq_theta = 1.0 - cos_theta**2
    intensity = (1.0/k**2) * (k + 1.0/k - sin_sq_theta)
    return intensity

def integrate_kn(alpha):
    """Integrate Klein-Nishina over all angles to get total cross-section normalization"""
    result, _ = quad(klein_nishina_intensity, -1, 1, args=(alpha,), weight='legendre', wvar=(0,))
    return result

# Actually, let's integrate numerically over cos(theta) from -1 to 1
result_int, error = quad(lambda ct: klein_nishina_intensity(alpha, ct), -1, 1)
print(f"Integrated Klein-Nishina: {result_int:.6f}")

# Now compute average scattered photon energy
def scattered_photon_energy(alpha, cos_theta):
    """Energy of scattered photon"""
    k = 1.0 + alpha * (1.0 - cos_theta)
    return E_photon / k

# Weight by intensity and integrate
def weighted_scattered_E(alpha, cos_theta):
    return scattered_photon_energy(alpha, cos_theta) * klein_nishina_intensity(alpha, cos_theta)

num_int, _ = quad(weighted_scattered_E, -1, 1)
denom_int, _ = quad(lambda ct: klein_nishina_intensity(alpha, ct), -1, 1)
mean_scattered_E = num_int / denom_int

print(f"Mean scattered photon energy: {mean_scattered_E:.4f} MeV")
print(f"Fractional energy transfer to electron: {(E_photon - mean_scattered_E)/E_photon:.4f}")

# Compton edge calculation
T_max = E_photon * (1 - 1/(1 + 2*alpha))
print(f"Compton edge (max electron energy): {T_max:.4f} MeV")
print(f"Minimum scattered photon energy: {E_photon - T_max:.4f} MeV")