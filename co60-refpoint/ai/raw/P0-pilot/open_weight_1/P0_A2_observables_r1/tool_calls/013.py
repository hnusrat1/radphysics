import numpy as np
from scipy.interpolate import interp1d

# Load data
xcom_data = np.loadtxt('/tmp/data/xcom_water.txt')
xcom_en = xcom_data[:, 0]
xcom_coh = xcom_data[:, 1]
xcom_inc = xcom_data[:, 2]
xcom_photo = xcom_data[:, 3]
xcom_pair_n = xcom_data[:, 4]
xcom_pair_e = xcom_data[:, 5]
xcom_tot = xcom_data[:, 6]

estar_data = np.loadtxt('/tmp/data/estar_water.txt')
estar_en = estar_data[:, 0]
estar_coll = estar_data[:, 1]
estar_rad = estar_data[:, 2]
estar_tot = estar_data[:, 3]
estar_csda = estar_data[:, 4]
estar_rad_yield = estar_data[:, 5]
estar_delta = estar_data[:, 6]

# Interpolation functions
f_coh = interp1d(xcom_en, xcom_coh, kind='linear', bounds_error=False, fill_value=0.0)
f_inc = interp1d(xcom_en, xcom_inc, kind='linear', bounds_error=False, fill_value=0.0)
f_photo = interp1d(xcom_en, xcom_photo, kind='linear', bounds_error=False, fill_value=0.0)
f_pair_n = interp1d(xcom_en, xcom_pair_n, kind='linear', bounds_error=False, fill_value=0.0)
f_pair_e = interp1d(xcom_en, xcom_pair_e, kind='linear', bounds_error=False, fill_value=0.0)
f_tot = interp1d(xcom_en, xcom_tot, kind='linear', bounds_error=False, fill_value=0.0)

f_coll = interp1d(estar_en, estar_coll, kind='linear', bounds_error=False, fill_value=0.0)
f_rad = interp1d(estar_en, estar_rad, kind='linear', bounds_error=False, fill_value=0.0)
f_csda = interp1d(estar_en, estar_csda, kind='linear', bounds_error=False, fill_value=0.0)
f_rad_yield = interp1d(estar_en, estar_rad_yield, kind='linear', bounds_error=False, fill_value=0.0)

# Constants
mec2 = 0.511
rho = 1.0
E0 = 2.0

mu_tot_0 = float(f_tot(E0))
mu_inc_0 = float(f_inc(E0))
mu_coh_0 = float(f_coh(E0))
mu_photo_0 = float(f_photo(E0))
mu_pair_0 = float(f_pair_n(E0)) + float(f_pair_e(E0))

print(f"Cross sections at {E0} MeV:")
print(f"  Total: {mu_tot_0:.6f}")
print(f"  Incoherent: {mu_inc_0:.6f}")

# L1 primary fluence per phi0
L1_primary_fluence_per_phi0 = np.exp(-mu_tot_0 * rho * 10.0)
print(f"L1_primary_fluence_per_phi0 = {L1_primary_fluence_per_phi0:.6f}")

# Angular primary - all in last bin (cos theta near 1)
L1_angular_primary = [0.0]*9 + [1.0]

print("Basic setup done")