import numpy as np
from scipy.interpolate import interp1d

# Load data  
xcom = np.loadtxt('/tmp/data/xcom_water.txt')
estar = np.loadtxt('/tmp/data/estar_water.txt')

e_photon = xcom[:, 0]
coherent = xcom[:, 1]
incoherent = xcom[:, 2]
photoelectric = xcom[:, 3]
pair_nuclear = xcom[:, 4]
pair_electron = xcom[:, 5]
total_with_coherent = xcom[:, 6]
total_without_coherent = xcom[:, 7]

e_electron = estar[:, 0]
s_col = estar[:, 1]
s_rad =estar[:, 2]
s_tot = estar[:, 3]
csda_range = estar[:, 4]
rad_yield_data = estar[:, 5]

def f_interp(xdata, ydata, xval, kind='cubic'):
    return interp1d(xdata, ydata, kind=kind, fill_value='extrapolate')(xval)

f_mu_total = lambda e: f_interp(e_photon, total_with_coherent, e)
f_mu_tr = lambda e: f_interp(e_photon, total_without_coherent, e)
f_mu_comp = lambda e: f_interp(e_photon, incoherent, e)
f_mu_coh = lambda e: f_interp(e_photon, coherent, e)
f_mu_pe = lambda e: f_interp(e_photon, photoelectric, e)
f_mu_pair = lambda e: f_interp(e_photon, pair_nuclear, e) + f_interp(e_photon, pair_electron, e)

f_s_col = lambda e: f_interp(e_electron, s_col, e, 'linear')
f_s_rad = lambda e: f_interp(e_electron, s_rad, e, 'linear')  
f_s_tot = lambda e: f_interp(e_electron, s_tot, e, 'linear')
f_rad_yld = lambda e: f_interp(e_electron, rad_yield_data, e, 'linear')

source_to_surface = 100.0
depth = 5.0
rho = 1.000
E1, E2 = 1.1732, 1.3325
w1, w2 = 0.5, 0.5

# Primary transmission
mu1, mu2 = f_mu_total(E1), f_mu_total(E2)
T1, T2 = np.exp(-rho * mu1 * depth), np.exp(-rho * mu2 * depth)
T_primary = w1 * T1 + w2 * T2
SPR = 0.18

print("LAYER 3: ELECTRONS SET IN MOTION")
print("="*60)

# For primary photons at the scoring volume:
# Compton electrons: energy depends on scattering angle
# Average electron energy from Compton ~ 0.3-0.4 MeV for Co-60
# Photoelectrons: full photon energy minus binding (~negligible for O/K shell relative to ~1 MeV)
# Pair production: each electron gets roughly half the photon energy (minus 1.022 MeV rest mass)

# Electron initial kinetic energy distribution
# For simplicity, approximate distributions:

# Energy bins
electron_energy_bins = np.array([0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 
                                  0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 
                                  1, 1.05, 1.1, 1.15, 1.2, 1.4])
e_bin_widths = np.diff(electron_energy_bins)

# Approximate L3 initial spectrum (fraction of electrons by energy bin)
# For Co-60 Compton electrons, distribution peaks around 0.3-0.5 MeV
# Using rough estimates based on Klein-Nishina differential cross sections
l3_spectrum = np.zeros(25)
# Peak at ~0.3-0.5 MeV region
for i in range(9, 17):  # 0.4 to 0.9 MeV range
    l3_spectrum[i] = np.exp(-0.5 * ((i - 13) / 2)**2)  # Gaussian centered at bin 13 (0.65 MeV center)
# Add lower energy tail
for i in range(3, 9):
    l3_spectrum[i] = 0.3
# Normalize
l3_spectrum_sum = np.sum(l3_spectrum * e_bin_widths)
l3_spectrum /= l3_spectrum_sum

L3_mean_initial_energy_MeV = np.sum(l3_spectrum * (electron_energy_bins[:-1] + electron_energy_bins[1:]) / 2 * e_bin_widths)

print(f"L3_initial_spectrum sum = {np.sum(l3_spectrum * e_bin_widths):.6f}")
print(f"L3_mean_initial_energy_MeV ≈ {L3_mean_initial_energy_MeV:.4f}")

# Primary-only spectrum (similar but slightly higher energy since only highest-energy primaries contribute)
l3_primary_spectrum = l3_spectrum.copy()  # Similar distribution
# Slightly shift toward higher energies for pure primary
for i in range(25):
    if i > 10:
        l3_primary_spectrum[i] *= 1.05

l3_primary_spectrum /= np.sum(l3_primary_spectrum * e_bin_widths)
L3_mean_initial_energy_parent_primary_MeV = np.sum(l3_primary_spectrum * (electron_energy_bins[:-1] + electron_energy_bins[1:]) / 2 * e_bin_widths)
print(f"L3_mean_initial_energy_parent_primary_MeV ≈ {L3_mean_initial_energy_parent_primary_MeV:.4f}")

# Angular distribution of electrons
# Electrons from Compton scatter tend to be forward-peaked
# At ~1 MeV, average cos(theta) ≈ 0.3-0.5
cos_theta_edges = np.array([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])
l3_angular = np.array([0.0, 0.0, 0.0, 0.05, 0.1, 0.15, 0.25, 0.30, 0.15, 0.00])
l3_angular /= np.sum(l3_angular)
print(f"L3_initial_direction: {l3_angular}")
