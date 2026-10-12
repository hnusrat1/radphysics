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
csda_range_w = estar[:, 4]
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

mu1, mu2 = f_mu_total(E1), f_mu_total(E2)
T1, T2 = np.exp(-rho * mu1 * depth), np.exp(-rho * mu2 * depth)
T_primary = w1 * T1 + w2 * T2
SPR = 0.18

print("PDD CORRECTION")
print("="*60)

# For Co-60 at SSD=100 cm, field 10x10 cm:
# d_max ≈ 0.5 cm in water
# Typical clinical PDD values (approximate):
# - PDD(0.5) = 100% by definition
# - PDD(5) for 10x10 Co-60 ≈ 67%

# Using approximate analytical relationships:
# Primary attenuation: exp(-mu*d) where mu ~ 0.063 cm^-1
# Scatter builds up then decreases

# At d_max (0.5 cm):
D_max_prim_factor = np.exp(-0.063*0.5) * 1.0  # buildup ~1.0 at shallow depths
D_max_scat_factor = 0.05  # minimal scatter at d_max

# At 5 cm depth:
D_5cm_prim_factor = T_primary  # already computed
D_5cm_scat_factor = SPR * T_primary

# Relative dose ratio (ignoring inverse square which cancels for same SSD geometry)
# PDD = [primary(d=5) + scatter(d=5)] / [primary(d=dmax) + scatter(d=dmax)] * 100

numerator = D_5cm_prim_factor + D_5cm_scat_factor
denominator = D_max_prim_factor + D_max_scat_factor

PDD_approx = numerator/denominator * 100
print(f"PDD approximation:")
print(f"  Primary only: {np.exp(-0.063*5)/np.exp(-0.063*0.5)*100:.1f}%")
print(f"  With scatter: {PDD_approx:.1f}%")

# More accurate estimate using typical clinical data
# For Co-60, 10x10 cm field:
# PDD(5cm) ≈ 67% 
clinical_PDD5 = 67.0
print(f"  Clinical reference PDD(5) ≈ {clinical_PDD5:.1f}%")

# Actually compute more carefully using actual coefficients
mu_tr_avg = w1*f_mu_tr(E1) + w2*f_mu_tr(E2)
mu_eff_5cm = rho * (w1*f_mu_total(E1)*T1 + w2*f_mu_total(E2)*T2) / T_primary  # effective mu weighted by fluence
print(f"\nEffective mu weighted by fluence at 5 cm: {mu_eff_5cm:.4f} cm^-1")
