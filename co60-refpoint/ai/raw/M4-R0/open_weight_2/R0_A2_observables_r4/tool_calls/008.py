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
f_mu_tr = interp1d(e_photon, total_without_coherent, kind='cubic', fill_value='extrapolate')

f_s_col = interp1d(e_electron, s_col, kind='linear', fill_value='extrapolate')
f_s_rad = interp1d(e_electron, s_rad, kind='linear', fill_value='extrapolate')
f_s_tot = interp1d(e_electron, s_tot, kind='linear', fill_value='extrapolate')
f_rad_yield = interp1d(e_electron, rad_yield_data, kind='linear', fill_value='extrapolate')

# Geometry
source_to_surface = 100.0
depth = 5.0
source_to_point = 105.0
scoring_r = 1.0
scoring_halfz = 0.1
rho_water = 1.000

E1, E2 = 1.1732, 1.3325
w1, w2 = 0.5, 0.5

# Interaction coefficients at Co-60 energies (averaged)
mu_total_1 = f_mu_total(E1)
mu_total_2 = f_mu_total(E2)
mu_tr_1 = f_mu_tr(E1)
mu_tr_2 = f_mu_tr(E2)

# For each energy, get fractions by process
def get_fractions(E):
    mu_coh = f_mu_coherent(E)
    mu_comp = f_mu_incoherent(E)
    mu_pe = f_mu_photoelectric(E)
    mu_pair = f_mu_pair_nuc(E) + f_mu_pair_elt(E)
    mu_tot = f_mu_total(E)
    return {
        'compton': mu_comp / mu_tot,
        'photoelectric': mu_pe / mu_tot,
        'pair': mu_pair / mu_tot,
        'rayleigh': mu_coh / mu_tot
    }

frac_E1 = get_fractions(E1)
frac_E2 = get_fractions(E2)
frac_avg = {k: w1*frac_E1[k] + w2*frac_E2[k] for k in frac_E1.keys()}

print("Interaction fractions (averaged for Co-60):")
for k, v in frac_avg.items():
    print(f"  {k}: {v:.6f}")

# Mass energy transfer - need to account for partial energy transfer in Compton
# For Compton scattering, fraction of energy transferred depends on scattering angle
# Average energy transfer fraction for Co-60 in water is ~0.3-0.4 per interaction
# We need to compute mass energy-transfer coefficients properly

# Actually, XCOM provides "total without coherent" which IS the mass energy-transfer coefficient
# (it accounts for the fact that Compton scatters only part of the energy)
# So mu_tr/rho = (W/E) * mu_total where W is mean energy transferred per interaction

print(f"\nMass energy-transfer coefficients:")
print(f"  mu_tr(E1)/rho = {mu_tr_1:.6e}")
print(f"  mu_tr(E2)/rho = {mu_tr_2:.6e}")

# Weighted average
mu_tr_avg = w1 * mu_tr_1 + w2 * mu_tr_2
mu_total_avg = w1 * mu_total_1 + w2 * mu_total_2
print(f"  mu_tr_avg/rho = {mu_tr_avg:.6e}")
print(f"  mu_total_avg/rho = {mu_total_avg:.6e}")

# Ratio shows average fraction of energy transferred
energy_transfer_fraction = mu_tr_avg / mu_total_avg
print(f"  Fraction of energy transferred per interaction = {energy_transfer_fraction:.4f}")
