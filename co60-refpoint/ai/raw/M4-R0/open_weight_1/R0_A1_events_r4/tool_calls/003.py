import numpy as np
import math

# Let's create a more accurate set of cross sections for water
# We'll compute using known data points

# Energy grid (MeV)
energies = np.array([
    0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 
    0.8, 1.0, 1.1732, 1.3325, 1.5, 2.0, 3.0, 5.0, 10.0
])

# Mass attenuation coefficients (cm^2/g) for water
# Coherent (Rayleigh)
mu_coh = np.array([
    0.0000, 0.0000, 0.0000,  # approx zero at low E (actually not zero but small)
    0.0000, 0.0000, 0.0000,
    0.0000, 0.0000, 0.0000,
    0.0000, 0.0000, 0.0000,
    0.00253, 0.00213, 
    0.00180, 0.00120, 0.00070, 0.00030, 0.00008
])  
# Actually let me compute properly.

# Let me use data from NIST XCOM for water.
# I'll hard-code values based on my knowledge.

# For water (H2O), density 1 g/cm^3
# Mass attenuation coefficients (mu/rho in cm^2/g)
# Source: from memory of NIST data

# Energy (MeV):   total      coherent   incoherent  photoelectric  pair prod
data = {
    0.001:   [4080.,     0.0,       0.0,       4080.,       0.0],
    0.002:   [599.,      0.0,       0.0,       599.,        0.0],
    0.005:   [41.1,      0.0,       0.0,       41.1,        0.0],
    0.01:    [5.33,      0.0002,    0.0041,    5.33,        0.0],
    0.02:    [0.797,     0.0004,    0.055,     0.742,       0.0],
    0.05:    [0.214,     0.0010,    0.181,    0.0321,      0.0],
    0.1:     [0.169,     0.0016,    0.151,   0.0164,     0.0],
    0.2:     [0.136,     0.0021,    0.124,   0.0097,      0.0],
    0.3:     [0.118,     0.0023,    0.110,    0.0057,      0.0],
    0.5:     [0.0972,    0.0024,    0.0922,   0.0026,      0.0],
    0.8:     [0.0787,    0.0025,    0.0746,   0.0016,      0.0],
    1.0:     [0.0702,    0.0025,    0.0666,   0.0011,     0.0],
    1.1732:  [0.06407,   0.00253,   0.05360,   0.001735,    0.006205],
    1.3325:  [0.06067,   0.00213,   0.05073,   0.001524,    0.006286],
    1.5:     [0.0575,   0.0018,    0.0480,   0.0012,      0.0065],
    2.0:     [0.0492,   0.0012,    0.0390,    0.00078,      0.0082],
    3.0:     [0.0405,   0.00070,   0.0285,    0.00045,      0.0109],
    5.0:     [0.0331,   0.00030,   0.0200,    0.00022,      0.0126],
    10.0:    [0.0242,   0.00008,   0.0120,    0.00008,      0.0120],
}

# Let's just use the two energies and define approximate cross sections for scattered photons.
# Actually, for accurate simulation we need cross sections at all energies.

# Let me use a simpler approach: use analytical formulas for the cross sections.

# For Compton: Klein-Nishina (free electron approximation)
# For pair production: approximate formula
# For photoelectric: approximate power law
# For Rayleigh: approximate power law

# Let's define functions.

mec2 = 0.511
r0 = 2.8179403262e-13
n_e_per_cm3 = 3.3456e23  # electrons/cm^3 in water

def mu_compton(E):
    """Compton linear attenuation coefficient in cm^-1"""
    if E <= 0:
        return 0.0
    a = E / mec2
    a2 = a*a
    x = 1.0 + 2.0*a
    lnx = math.log(x)
    term1 = (1.0+a)/a2 * (2.0*(1.0+a)/x - lnx/a)
    term2 = lnx/(2.0*a) - (1.0+3.0*a)/(x*x)
    sigma = 2.0*math.pi*r0*r0 * (term1 + term2)
    return sigma * n_e_per_cm3

# For pair production, use approximate formula for water
# Z_eff for water: Z_eff = (w_H * Z_H^2 + w_O * Z_O^2) / (w_H * Z_H + w_O * Z_O)
# Actually for pair production cross section per atom ~ Z^2
# We can approximate using the cross section per molecule.
# H: Z=1, O: Z=8
# For molecule H2O, effective Z^2: 2*1^2 + 1*8^2 = 2 + 64 = 66
# Number of molecules per cm^3: NA/18 = 6.022e23/18 = 3.3456e22

# Bethe-Heitler approximate cross section per atom (for unscreened):
# sigma_pair = alpha * r0^2 * Z^2 * (28/9 * ln(2E/mec2) - 218/27)  (for high E)
# But near threshold, it's small.

# Let's use a simple interpolation based on known data.
# For water, the pair production mass attenuation coefficient (cm^2/g) at 1.1732 MeV:
# mu_pair/rho = 0.006205
# At 1.3325 MeV: 0.006286
# At 2 MeV: 0.0082
# At 5 MeV: 0.0126
# At 10 MeV: 0.0120 (actually it decreases at very high E due to screening)

# I'll use a simple function that fits these points.
def mu_pair(E):
    """Pair production linear attenuation coefficient in cm^-1"""
    if E <= 2*mec2:
        return 0.0
    # Use a simple fit
    # mu_pair/rho (cm^2/g) = a * (E - 2*mec2)^b
    # Fit to data points
    # Data points (E in MeV, mu_pair/rho in cm^2/g):
    # 1.1732: 0.006205
    # 1.3325: 0.006286
    # 2.0: 0.0082
    # 3.0: 0.0109
    # 5.0: 0.0126
    # 10.0: 0.0120
    # Let's use a linear interpolation in log-log
    e_vals = np.array([1.1732, 1.3325, 2.0, 3.0, 5.0, 10.0])
    mu_vals = np.array([0.006205, 0.006286, 0.0082, 0.0109, 0.0126, 0.0120])
    # Linear interpolation in log space
    logE = math.log(E)
    if E <= 1.1732:
        # extrapolate to threshold
        # Approximate: ~ (E-1.022)^2
        t = E - 2*mec2
        if t <= 0: return 0.
        # Scale from 1.1732
        t0 = 1.1732 - 1.022 = 0.1512
        mu0 = 0.006205
        return mu0 * (t/t0)**2 * 1.0  # density
    elif E >= 10.0:
        return 0.0120
    else:
        logE_vals = np.log(e_vals)
        logmu_vals = np.log(mu_vals)
        return math.exp(np.interp(logE, logE_vals, logmu_vals)) * 1.0  # density=1

# Photoelectric effect
# For water, photoelectric mass attenuation coefficient roughly ~ Z^4 per atom
# H: Z=1, O: Z=8
# At 1.1732 MeV: mu_pe/rho = 0.001735
# At 0.1 MeV: 0.0164
# At 0.02 MeV: 0.742
# Power law: ~ E^{-3.5} approximately
def mu_pe(E):
    """Photoelectric linear attenuation coefficient in cm^-1"""
    if E <= 0:
        return 0.0
    # Fit to data
    # Energy (MeV): mu_pe/rho (cm^2/g)
    # 0.01: 5.33
    # 0.02: 0.742
    # 0.05: 0.0321
    # 0.1: 0.0164
    # 0.2: 0.0097
    # 0.3: 0.0057
    # 0.5: 0.0026
    # 1.0: 0.0011
    # 1.1732: 0.001735
    # 1.3325: 0.001524
    # Let's interpolate in log-log
    e_vals = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 1.1732, 1.3325, 2.0])
    mu_vals = np.array([5.33, 0.742, 0.0321, 0.0164, 0.0097,0.0057, 0.0026, 0.0011, 0.001735, 0.001524, 0.00078])
    if E <= 0.01:
        # Extrapolate: ~ E^{-3.5}
        return mu_vals[0] * (E/0.01)**(-3.5) * 1.0
    elif E >= 2.0:
        return 0.00078 * (E/2.0)**(-3.0) * 1.0
    else:
        logE = math.log(E)
        logE_vals = np.log(e_vals)
        logmu_vals = np.log(mu_vals)
        return math.exp(np.interp(logE, logE_vals, logmu_vals)) * 1.0

# Rayleigh (coherent)
# Mass attenuation coefficient for Rayleigh in water:
# At 1.1732: 0.00253
# At 0.1: 0.0016
# At 1.0: 0.0025
# At 2.0: 0.0012
# Very roughly constant at low E then decreases at high E
def mu_rayleigh(E):
    """Rayleigh linear attenuation coefficient in cm^-1"""
    if E <= 0:
        return 0.
    e_vals = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 1.1732, 1.3325, 2.0, 5.0])
    mu_vals = np.array([0.0008, 0.0010, 0.0012, 0.0016, 0.0021, 0.0024, 0.0025, 0.00253, 0.00213, 0.0012, 0.0003])
    if E <= 0.01:
        return 0.0008 * (E/0.01)**0.3 * 1.0
    elif E >= 5.0:
        return 0.0003 * (E/5.0)**(-1.5) * 1.0
    else:
        logE = math.log(E)
        logE_vals = np.log(e_vals)
        logmu_vals = np.log(mu_vals)
        return math.exp(np.interp(logE, logE_vals, logmu_vals)) * 1.0

# Test
for E in [1.1732, 1.3325, 0.5, 0.2, 0.1]:
    mu_c = mu_compton(E)
    mu_p = mu_pair(E)
    mu_pe_val = mu_pe(E)
    mu_r = mu_rayleigh(E)
    mu_t = mu_c + mu_p + mu_pe_val + mu_r
    print(f"E={E:.4f} MeV: total={mu_t:.6f}, compton={mu_c:.6f}, pair={mu_p:.6f}, pe={mu_pe_val:.6f}, rayleigh={mu_r:.6f}")