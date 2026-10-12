import numpy as np
from scipy.interpolate import interp1d

xcom = np.loadtxt('/tmp/data/xcom_water.txt')
xcom_e = xcom[:,0]
xcom_tot = xcom[:,6]
f_tot = interp1d(xcom_e, xcom_tot, kind='linear', bounds_error=False, fill_value=0.0)
mu_tot_0 = float(f_tot(2.0))
print(f"mu_tot at 2 MeV = {mu_tot_0:.6f}")
print(f"Primary atten = {np.exp(-mu_tot_0*10):.6f}")