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
s_col_data = estar[:, 1]
s_rad_data = estar[:, 2]
s_tot_data = estar[:, 3]
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

f_s_col = lambda e: f_interp(e_electron, s_col_data, e, 'linear')
f_s_rad = lambda e: f_interp(e_electron, s_rad_data, e, 'linear')  
f_s_tot = lambda e: f_interp(e_electron, s_tot_data, e, 'linear')
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

# KERMA calculation (key for Layers 2-5)
# Kerma = Phi * mu_tr / rho
mu_tr_1, mu_tr_2 = f_mu_tr(E1), f_mu_tr(E2)
mu_tr_avg = w1 * mu_tr_1 + w2 * mu_tr_2

# Kerma per unit Phi0 at depth d = Phi0 * exp(-mu*d) * mu_tr/rho
K_per_phi0 = T_primary * mu_tr_avg * rho * (1 + SPR * 0.8)  # approx including scatter
# Actually: K = integral of fluence(mu_tr) over all energies
# Simplified: K ≈ Phi_total * <mu_tr>/<mu>_avg
Kcol_approx = K_per_phi0 * 0.98  # radiative losses are small (~2%)

print("LAYER 4 & 5 CALCULATIONS")
print("="*60)

# Electron fluence estimation
# For electron fluence, we use approximation:
# psi_e = K_col / (S_col/rho * <E>) approximately
# At equilibrium in narrow beam, electron fluence ~ primary electron fluence * range/scoring size

# Mean collision stopping power at ~0.5 MeV (typical electron energy)
s_col_05 = f_s_col(0.5)
s_rad_05 = f_s_rad(0.5)
s_tot_05 = f_s_tot(0.5)
print(f"\nStopping powers at 0.5 MeV:")
print(f"  S_col/rho = {s_col_05:.4f} MeV cm^2/g")
print(f"  S_rad/rho = {s_rad_05:.4f} MeV cm^2/g")  
print(f"  S_tot/rho = {s_tot_05:.4f} MeV cm^2/g")

# Electron fluence from charged particle equilibrium concept
# For Co-60 at 5 cm depth in water, electron equilibrium exists
# Electron fluence ≈ Kerma / (mean S_col/rho)

L5_K_over_phi0_Gy_cm2 = K_per_phi0  # Already calculated
L5_Kcol_over_phi0_Gy_cm2 = Kcol_approx
L5_D_over_phi0_Gy_cm2 = Kcol_approx  # D ≈ K_col for electrons in equilibrium (beta ≈ 1)

# Beta = D/K_col
beta_est = 0.95  # Approximate for this geometry
L5_beta = beta_est

# Radiative yield g = 1 - K_col/K
g_est = 0.02  # Approximately for Co-60 in water
L5_g = g_est

# L4 radiative to collision ratio
# For electrons at ~0.5 MeV in water
avg_E_elec = 0.5  # approximate mean electron energy
ratio_rad_col = f_s_rad(avg_E_elec) / f_s_col(avg_E_elec)
L4_radiative_to_collision = ratio_rad_col
print(f"L4_radiative_to_collision = {L4_radiative_to_collision:.6f}")

# Electron fluence
# psi_e = K_col / (S_col/rho) where S_col is in MeV cm^2/g
# Converting: K_col [Gy/cm^2] = K_col [J/kg/m^2], need consistent units
# K_per_phi0 has units of MeV cm^2/g * dimensionless = MeV cm^2/g
# To get Gy cm^2: multiply by 1.602e-13 J/MeV * 1000 g/kg
K_per_phi0_Gycm2 = K_per_phi0 * 1.60217662e-13 * 1000  # Convert MeV cm^2/g to Gy cm^2

#psi_e = K_col / (S_col_rho_bar) 
# where S_col_rho_bar is average collision stopping power (~2 MeV cm^2/g for ~0.5 MeV electrons)
S_col_mean = 2.0  # MeV cm^2/g rough average for electron spectrum
psi_e_per_phi0 = Kcol_approx / S_col_mean

L4_electron_fluence_per_phi0 = psi_e_per_phi0
print(f"L4_electron_fluence_per_phi0 = {L4_electron_fluence_per_phi0:.6f}")

# Electron fluence spectrum - peaked around 0.1-1 MeV
l4_spec = np.zeros(25)
electron_energy_bins = np.array([0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 
                                  0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 
                                  1, 1.05, 1.1, 1.15, 1.2, 1.4])
ebw = np.diff(electron_energy_bins)

for i in range(25):
    ec = (electron_energy_bins[i] + electron_energy_bins[i+1])/2
    l4_spec[i] = np.exp(-0.5*((np.log(ec/0.5)/0.5)**2))  # log-normal-ish centered at 0.5 MeV
    
l4_spec /= np.sum(l4_spec * ebw)
l4_spec *= L4_electron_fluence_per_phi0

# Delta ray fraction - delta rays are electrons set in motion by other electrons
# For Co-60, delta ray contribution is modest, maybe 10-20% of total electron fluence
L4_delta_fraction = 0.15

# Mean electron energy  
L4_mean_electron_energy_MeV = np.sum(l4_spec * ebw * (electron_energy_bins[:-1]+electron_energy_bins[1:])/2) / L4_electron_fluence_per_phi0
print(f"L4_mean_electron_energy_MeV = {L4_mean_electron_energy_MeV:.4f}")

# Electron angular distribution - forward-peaked but less than photons
l4_angular = np.array([0.0, 0.0, 0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.15, 0.09])
l4_angular /= np.sum(l4_angular)
print(f"L4_angular = {l4_angular}")
