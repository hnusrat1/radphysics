import numpy as np
from scipy.interpolate import interp1d

xcom = np.loadtxt('/tmp/data/xcom_water.txt')
x_e = xcom[:,0]
x_inc = xcom[:,2]
x_tot = xcom[:,6]
x_coh = xcom[:,1]
x_pho = xcom[:,3]
x_pn = xcom[:,4]
x_pe = xcom[:,5]

estar = np.loadtxt('/tmp/data/estar_water.txt')
es_e = estar[:,0]
es_coll = estar[:,1]
es_rad = estar[:,2]
es_csda = estar[:,3]
es_ry = estar[:,4]

f_inc = interp1d(x_e, x_inc, kind='linear', bounds_error=False, fill_value=0.0)
f_tot = interp1d(x_e, x_tot, kind='linear', bounds_error=False, fill_value=0.0)
f_coh = interp1d(x_e, x_coh, kind='linear', bounds_error=False, fill_value=0.0)
f_pho = interp1d(x_e, x_pho, kind='linear', bounds_error=False, fill_value=0.0)
f_pn = interp1d(x_e, x_pn, kind='linear', bounds_error=False, fill_value=0.0)
f_pe = interp1d(x_e, x_pe, kind='linear', bounds_error=False, fill_value=0.0)
f_coll = interp1d(es_e, es_coll, kind='linear', bounds_error=False, fill_value=0.0)
f_rad = interp1d(es_e, es_rad, kind='linear', bounds_error=False, fill_value=0.0)
f_csda = interp1d(es_e, es_csda, kind='linear', bounds_error=False, fill_value=0.0)
f_ry = interp1d(es_e, es_ry, kind='linear', bounds_error=False, fill_value=0.0)

mec2 = 0.511; rho = 1.0; E0 = 2.0; SSD = 100.0

mu_tot_0 = float(f_tot(E0))
mu_inc_0 = float(f_inc(E0))
mu_coh_0 = float(f_coh(E0))
mu_pho_0 = float(f_pho(E0))
mu_pair_0 = float(f_pn(E0)) + float(f_pe(E0))

print("Cross sections at 2.0 MeV (cm^2/g):")
print(f"  total={mu_tot_0:.6f}")
print(f"  incoherent={mu_inc_0:.6f}")
print(f"  coherent={mu_coh_0:.6e}")
print(f"  photo={mu_pho_0:.6e}")
print(f"  pair={mu_pair_0:.6e}")

# L1 primary
L1_pri = np.exp(-mu_tot_0 * 10.0)
print(f"\nL1_primary_fluence_per_phi0 = {L1_pri:.6f}")

# Interaction fractions for primary photons at 2.0 MeV
frac_inc_pri = mu_inc_0 / mu_tot_0
frac_coh_pri = mu_coh_0 / mu_tot_0
frac_pho_pri = mu_pho_0 / mu_tot_0
frac_pair_pri = mu_pair_0 / mu_tot_0
print(f"\nInteraction fractions for primary photons:")
print(f"  Compton: {frac_inc_pri:.6f}")
print(f"  Rayleigh: {frac_coh_pri:.6f}")
print(f"  Photo: {frac_pho_pri:.6f}")
print(f"  Pair: {frac_pair_pri:.6f}")