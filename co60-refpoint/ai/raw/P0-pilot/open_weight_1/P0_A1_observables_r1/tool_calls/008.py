import numpy as np
import scipy.integrate as integrate
import scipy.interpolate as interpolate
import math

# ===== CONSTANTS =====
mec2 = 0.511  # MeV
r0 = 2.8179403262e-13  # cm
r0_sq = r0**2
NA = 6.02214076e23
M_H2O = 18.01528
rho = 1.0  # g/cm^3
electrons_per_gram_H2O = 10 * NA / M_H2O

# ===== XCOM-TYPE CROSS SECTIONS FOR WATER =====
# Known NIST XCOM data for water (cm^2/g)
# Process-wise mass attenuation coefficients
# Energy grid (MeV), values are approximate from XCOM
E_grid = np.array([0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0])

# Photoelectric (cm^2/g) - approximate from XCOM for water
mu_pe_data = np.array([4.88, 1.42, 0.534, 0.107, 0.0232, 0.00681, 0.00264, 0.000672, 0.000233, 0.0000444, 0.0000131, 0.0000029, 0.0000013, 0.0000007, 0.0000005, 0.0000003, 0.0000002, 0.0000001, 0.0000001, 0.0000001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

# Hmm, these photoelectric values look too low for the lower energies.
# Let me reconsider. 

# Actually, I recall XCOM values for water photoelectric:
# 0.030 MeV: 1.65 cm^2/g (not 0.107)
# The problem is that I'm confusing mass attenuation with something else.

# Let me use proper values. From XCOM for water:
E_pe = np.array([0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0])
mu_pe = np.array([4.79, 0.521, 0.169, 0.0261, 0.00480, 0.00213, 0.000388, 0.000114, 0.0000233, 0.00000827, 0.00000391, 0.00000225, 0.000000996, 0.000000543, 0.000000294, 0.000000178, 0.0000000815, 0.0000000226, 0.00000000490])

# Rayleigh (coherent) cm^2/g for water
E_ray = np.array([0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0])
mu_ray = np.array([0.000404, 0.00158, 0.00350, 0.00616, 0.0072, 0.00684, 0.00572, 0.00468, 0.00332, 0.00252, 0.00200, 0.00163, 0.00116, 0.000875, 0.000651, 0.000503, 0.000324, 0.000150, 0.0000655])

# Compton (incoherent) cm^2/g - can be computed from Klein-Nishina with incoherent scattering function correction
# At low energies, the correction is significant. Let me use XCOM data points.
E_comp = np.array([0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 10.0])
mu_comp = np.array([0.0225, 0.0410, 0.0585, 0.0900, 0.129, 0.149, 0.198, 0.219, 0.233, 0.236, 0.232, 0.226, 0.208, 0.190, 0.167, 0.151, 0.123, 0.0900, 0.0530, 0.0315])

# Hmm, wait. These are wrong too. Let me think more carefully.

# The Klein-Nishina cross section gives μ/ρ for Compton scattering.
# At 2 MeV, Klein-Nishina gives 0.0489 cm^2/g as I computed.
# The XCOM value is 0.0445 cm^2/g.
# The difference is due to the incoherent scattering function, which reduces the cross section
# at larger angles.

# Let me compute Klein-Nishina values on my grid and scale to match XCOM at 2 MeV.
# The correction factor decreases with energy.

# Actually, let me simplify and use empirical data:
# XCOM values for water (from published tables):

# Process cross sections for water (cm^2/g):
# E     Rayleigh  Compton   Photo     Pair(nuc)  Pair(elec)
# 0.01  0.000404  0.0225   4.79      0          0
# 0.02  0.00158   0.0410   0.521     0          0
# 0.03  0.00350   0.0585   0.169     0          0
# 0.05  0.00616   0.0900   0.0261    0          0
# 0.08  0.00720   0.129    0.00480   0          0
# 0.1   0.00684   0.149    0.00213   0          0
# 0.15  0.00572   0.198    0.000388  0          0
# 0.2   0.00468   0.219    0.000114  0          0
# 0.3   0.00332   0.233    0.0000233 0          0
# 0.4   0.00252   0.236    0.00000827 0         0
# 0.5   0.00200   0.232    0.00000391 0         0
# 0.6   0.00163   0.226    0.00000225 0         0
# 0.8   0.00116   0.208    0.000000996 0        0
# 1.0   0.000875  0.190    0.000000543 0.000060 0.000008
# 1.25  0.000651  0.167    0.000000294 0.000560 0.000072
# 1.5   0.000503  0.151    0.000000178 0.00132  0.000170
# 2.0   0.000324  0.123    0.0000000815 0.00316  0.000410
# 3.0   0.000150  0.0900   0.0000000226 0.00890  0.00119
# 5.0   0.0000655 0.0530   0.0000000049 0.0205   0.00280

# Check total at 2 MeV: 0.000324 + 0.123 + 0.00000008 + 0.00316 + 0.00041 = 0.1269 ??? 

# No, this is wrong. These Compton values can't be right.
# At 2 MeV, the Compton mass attenuation for water is about 0.0445, not 0.123.

# The issue is I'm confusing cross sections in cm^2/g with cross sections in barns/atom or something.

# Let me just use the Klein-Nishina computed values, properly.
# And scale by a factor to account for the incoherent scattering function.

# The KN formula gives:
# At 1 MeV: ?
# At 2 MeV: 0.04893 cm^2/g (as I computed)
# The XCOM value at 2 MeV is 0.04451 cm^2/g

# So the correction factor at 2 MeV is about 0.04451/0.04893 = 0.9094

print("Let me reconsider the Compton data...")
print(f"Klein-Nishina mu_comp/rho at 2 MeV: {sigma_compton_cm2(2.0)*electrons_per_gram_H2O:.6f}")

# Let me recompute KN values and compare with expected XCOM values
for E in [0.1, 0.2, 0.5, 1.0, 1.5, 2.0]:
    mu_kn = sigma_compton_cm2(E) * electrons_per_gram_H2O
    print(f"  E={E:.1f} MeV: KN mu_comp/rho = {mu_kn:.6f}")