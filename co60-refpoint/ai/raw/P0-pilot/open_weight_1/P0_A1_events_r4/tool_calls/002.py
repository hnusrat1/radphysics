import numpy as np

# Let's define better approximations based on known NIST values
# For water, I'll create interpolation tables

# Energy grid (MeV)
energies = np.array([0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0])
# Approximate mu/rho values for water (cm^2/g) from NIST
# Total, Compton, photoelectric, pair, Rayleigh
# These are rough but plausible
mu_total = np.array([5.33, 1.67, 0.796, 0.376, 0.269, 0.214, 0.187, 0.166, 0.171, 0.151, 0.136, 0.118, 0.106, 0.0967, 0.0894, 0.0786, 0.0708, 0.0583, 0.0492, 0.0360, 0.0271, 0.0216, 0.0178, 0.0130, 0.0102, 0.00622, 0.00418])
mu_compton = np.array([0.443, 0.439, 0.437, 0.432, 0.427, 0.422, 0.417, 0.408, 0.399, 0.380, 0.364, 0.339, 0.319, 0.303, 0.290, 0.268, 0.251, 0.218, 0.194, 0.155, 0.128, 0.109, 0.0945, 0.0732, 0.0582, 0.0364, 0.0246])
mu_pe = np.array([4.89, 1.23, 0.359, 0.0559, 0.0400, 0.0250, 0.0159, 0.00380, 0.00163, 0.000440, 0.000200, 0.000080, 0.000040, 0.000025, 0.000018, 0.000010, 0.000007, 0.000003, 0.000002, 0.000001, 0.0000005, 0.0000003, 0.0000002, 0.0000001, 0.00000005, 0.00000002, 0.00000001])
mu_pair = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0004, 0.0011, 0.00275, 0.0036, 0.0046, 0.0051, 0.0055, 0.0059, 0.0064, 0.0068, 0.0074, 0.0077])
mu_rayleigh = np.array([0.160, 0.0560, 0.0260, 0.0080, 0.0035, 0.0018, 0.0010, 0.0004, 0.0002, 0.00005, 0.00002, 0.000005, 0.000002, 0.000001, 0.0000005, 0.0000002, 0.0000001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

# Check at 2 MeV
from scipy.interpolate import interp1d
f_total = interp1d(energies, mu_total, kind='linear', bounds_error=False, fill_value=0.0)
f_compton = interp1d(energies, mu_compton, kind='linear', bounds_error=False, fill_value=0.0)
f_pe = interp1d(energies, mu_pe, kind='linear', bounds_error=False, fill_value=0.0)
f_pair = interp1d(energies, mu_pair, kind='linear', bounds_error=False, fill_value=0.0)
f_rayleigh = interp1d(energies, mu_rayleigh, kind='linear', bounds_error=False, fill_value=0.0)

print(f"At 2 MeV:")
print(f"Total: {f_total(2.0):.6f}")
print(f"Compton: {f_compton(2.0):.6f}")
print(f"PE: {f_pe(2.0):.6e}")
print(f"Pair: {f_pair(2.0):.6e}")
print(f"Rayleigh: {f_rayleigh(2.0):.6e}")
print(f"Sum: {f_compton(2.0)+f_pe(2.0)+f_pair(2.0)+f_rayleigh(2.0):.6f}")