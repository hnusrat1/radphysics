import numpy as np
from scipy.interpolate import interp1d, spline

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
s_rad = estar[:, 2]
s_tot = estar[:, 3]
csda_range = estar[:, 4]
rad_yield_data = estar[:, 5]

# Create interpolation functions
f_mu_coherent = interp1d(e_photon, coherent, kind='cubic', fill_value='extrapolate')
f_mu_incoherent = interp1d(e_photon, incoherent, kind='cubic', fill_value='extrapolate')
f_mu_photoelectric = interp1d(e_photon, photoelectric, kind='cubic', fill_value='extrapolate')
f_mu_pair_nuc = interp1d(e_photon, pair_nuclear, kind='cubic', fill_value='extrapolate')
f_mu_pair_elt = interp1d(e_photon, pair_electron, kind='cubic', fill_value='extrapolate')
f_mu_total = interp1d(e_photon, total_with_coherent, kind='cubic', fill_value='extrapolate')
f_mu_total_nocoherent = interp1d(e_photon, total_without_coherent, kind='cubic', fill_value='extrapolate')

f_s_col = interp1d(e_electron, s_col, kind='linear', fill_value='extrapolate')
f_s_rad = interp1d(e_electron, s_rad, kind='linear', fill_value='extrapolate')
f_s_tot = interp1d(e_electron, s_tot, kind='linear', fill_value='extrapolate')

# Co-60 energies (equal probability = 0.5 each)
E1 = 1.1732
E2 = 1.3325
w1 = 0.5
w2 = 0.5

print("Calculating Co-60 interaction coefficients at exact energies...")

# At E1 = 1.1732 MeV
mu_coherent_1 = f_mu_coherent(E1)
mu_incoherent_1 = f_mu_incoherent(E1)
mu_photoelectric_1 = f_mu_photoelectric(E1)
mu_pair_nuc_1 = f_mu_pair_nuc(E1)
mu_pair_elt_1 = f_mu_pair_elt(E1)
mu_total_1 = f_mu_total(E1)
mu_tr_1 = f_mu_total_nocoherent(E1)  # mass energy transfer

print(f"\nAt {E1} MeV:")
print(f"  mu/rho (total, with coherent) = {mu_total_1:.6e}")
print(f"  mu_tr/rho = {mu_tr_1:.6e}")

# Fractional contributions
sum_xsec_1 = mu_coherent_1 + mu_incoherent_1 + mu_photoelectric_1 + mu_pair_nuc_1 + mu_pair_elt_1
print(f"  sum = {sum_xsec_1:.6e}")

frac_coherent_1 = mu_coherent_1 / mu_total_1
frac_compton_1 = mu_incoherent_1 / mu_total_1
frac_pe_1 = mu_photoelectric_1 / mu_total_1
frac_pair_1 = (mu_pair_nuc_1 + mu_pair_elt_1) / mu_total_1

print(f"  Fractions: Rayleigh={frac_coherent_1:.4e}, Compton={frac_compton_1:.4e}, PE={frac_pe_1:.4e}, Pair={frac_pair_1:.4e}")

# Similar for E2
mu_coherent_2 = f_mu_coherent(E2)
mu_incoherent_2 = f_mu_incoherent(E2)
mu_photoelectric_2 = f_mu_photoelectric(E2)
mu_pair_nuc_2 = f_mu_pair_nuc(E2)
mu_pair_elt_2 = f_mu_pair_elt(E2)
mu_total_2 = f_mu_total(E2)
mu_tr_2 = f_mu_total_nocoherent(E2)

sum_xsec_2 = mu_coherent_2 + mu_incoherent_2 + mu_photoelectric_2 + mu_pair_nuc_2 + mu_pair_elt_2

frac_coherent_2 = mu_coherent_2 / mu_total_2
frac_compton_2 = mu_incoherent_2 / mu_total_2
frac_pe_2 = mu_photoelectric_2 / mu_total_2
frac_pair_2 = (mu_pair_nuc_2 + mu_pair_elt_2) / mu_total_2

print(f"\nAt {E2} MeV:")
print(f"  mu/rho (total, with coherent) = {mu_total_2:.6e}")
print(f"  mu_tr/rho = {mu_tr_2:.6e}")
print(f"  Fractions: Rayleigh={frac_coherent_2:.4e}, Compton={frac_compton_2:.4e}, PE={frac_pe_2:.4e}, Pair={frac_pair_2:.4e}")

# Weighted averages for Co-60 beam (equal probability)
mu_avg = w1 * mu_total_1 + w2 * mu_total_2
mu_tr_avg = w1 * mu_tr_1 + w2 * mu_tr_2
mu_en_approx = mu_tr_avg * 0.99  # Rough approximation for energy absorption (neglecting bremsstrahlung)

print(f"\nWeighted averages for Co-60:")
print(f"  mu/rho = {mu_avg:.6e}")
print(f"  mu_tr/rho = {mu_tr_avg:.6e}")
print(f"  Average Compton fraction ≈ {(w1*frac_compton_1 + w2*frac_compton_2):.4f}")
