import numpy as np
import scipy.integrate as integrate
import math

# ===== CONSTANTS =====
mec2 = 0.511
r0 = 2.8179403262e-13
r0_sq = r0**2
NA = 6.02214076e23
M_H2O = 18.01528
rho = 1.0
electrons_per_gram_H2O = 10 * NA / M_H2O
electrons_per_cm3 = electrons_per_gram_H2O * rho
F_COMP = 0.95

# ===== CROSS-SECTION MODEL =====
def sigma_kn(E):
    if E <= 0: return 0
    alpha = E / mec2
    return (2*np.pi*r0_sq * ((1+alpha)/alpha**2 * (2*(1+alpha)/(1+2*alpha) - np.log(1+2*alpha)/alpha) 
                            + np.log(1+2*alpha)/(2*alpha) - (1+3*alpha)/(1+2*alpha)**2))

def mu_comp_rho(E):
    return F_COMP * sigma_kn(E) * electrons_per_gram_H2O

def mu_pe_rho(E):
    E = max(float(E), 0.01)
    E = min(E, 5.0)
    table_E = np.array([0.01,0.015,0.02,0.03,0.04,0.05,0.06,0.08,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.8,1.0,1.25,1.5,2.0,3.0,5.0])
    table_M = np.array([4.79,1.41,0.519,0.169,0.0770,0.0261,0.0137,0.00480,0.00213,0.000388,0.000111,0.0000235,0.00000881,0.00000412,0.00000212,0.000000838,0.000000443,0.000000220,0.000000123,0.0000000581,0.0000000191,0.00000000392])
    return float(np.exp(np.interp(np.log(E), np.log(table_E), np.log(table_M))))

def mu_pair_rho(E):
    if E <= 1.022: return 0.0
    E = min(E, 10.0)
    table_E = np.array([1.022,1.25,1.5,2.0,2.5,3.0,4.0,5.0,6.0,8.0,10.0])
    table_M = np.array([0.0,0.000639,0.00151,0.00318,0.00496,0.00720,0.0124,0.0169,0.0211,0.0286,0.0340])
    logE = np.log(E)
    logE_table = np.log(table_E)
    logM_table = np.log(np.maximum(table_M, 1e-12))
    return float(np.exp(np.interp(logE, logE_table, logM_table)))

def mu_ray_rho(E):
    E = max(float(E), 0.01)
    E = min(E, 5.0)
    table_E = np.array([0.01,0.015,0.02,0.03,0.05,0.06,0.08,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.8,1.0,1.25,1.5,2.0,3.0,5.0])
    table_M = np.array([0.000404,0.000882,0.00154,0.00349,0.00616,0.00716,0.00722,0.00682,0.00574,0.00467,0.00331,0.00250,0.00199,0.00162,0.00116,0.000873,0.000651,0.000502,0.000326,0.000151,0.0000650])
    return float(np.exp(np.interp(np.log(E), np.log(table_E), np.log(table_M))))

def mu_total_rho(E):
    return mu_comp_rho(E) + mu_pe_rho(E) + mu_pair_rho(E) + mu_ray_rho(E)

def mu_linear(E):
    return mu_total_rho(E) * rho

# ===== COMPTON DIFFERENTIAL CROSS SECTION =====
def compton_dsigma_dOmega(E, costheta):
    """dσ/dΩ per electron (cm^2/sr), returns (dsigma, E_scattered)"""
    alpha = E / mec2
    k = 1.0 / (1.0 + alpha * (1.0 - costheta))
    Ep = E * k
    return F_COMP * (r0_sq/2.0) * k**2 * (k + 1.0/k - (1.0 - costheta**2)), Ep

def compton_energy_avg(E):
    """(sigma_single_e, avg_E_scattered, avg_E_transfer)"""
    costheta = np.linspace(-1, 1, 50000)
    ds, Ep = compton_dsigma_dOmega(E, costheta)
    dfdx = 2*np.pi * ds
    sigma = integrate.simpson(dfdx, costheta)
    if sigma <= 0:
        return 0, 0, 0
    avg_Ep = integrate.simpson(dfdx * Ep, costheta) / sigma
    return sigma, avg_Ep, E - avg_Ep

# ===== ENERGY ABSORPTION COEFFICIENTS (use XCOM-like values) =====
# Mass energy absorption for water (cm^2/g), from XCOM
xcom_en_E = np.array([0.01,0.02,0.03,0.05,0.08,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.8,1.0,1.25,1.5,2.0,3.0])
xcom_en_mu = np.array([4.78,0.503,0.1536,0.0403,0.0241,0.0255,0.0497,0.0303,0.0318,0.0325,0.0326,0.0328,0.0326,0.0310,0.0292,0.0280,0.0233,0.0200])
# Let me fix some values - 0.15 looks wrong (0.0497), probably should be ~0.0315

def mu_en_rho_xcom(E):
    """Mass energy absorption (cm^2/g) for water, interpolated from XCOM"""
    table_E = np.array([0.01,0.03,0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.8,1.0,1.25,1.5,2.0,3.0])
    table_M = np.array([4.78,0.160,0.0405,0.0255,0.0314,0.0303,0.0318,0.0325,0.0326,0.0327,0.0326,0.0310,0.0292,0.0280,0.0233,0.0200])
    return float(np.exp(np.interp(np.log(E), np.log(table_E), np.log(table_M))))

def mu_tr_rho_xcom(E):
    """Mass energy transfer (cm^2/g) for water"""
    # Approximate mu_tr = mu_en/(1-g), where g is the radiative fraction
    g_rad = radiative_fraction_water(E)
    mu_en = mu_en_rho_xcom(E)
    return mu_en / (1 - g_rad)

def radiative_fraction_water(E):
    """Radiative fraction g for photons in water (approximate)"""
    # g is small at low energies and increases slowly
    # Approximate values: 2 MeV: 0.005, 1 MeV: 0.005, 0.5 MeV: 0.005
    return 0.005

# Check
print("XCOM-like mu_en/rho values:")
for E in [0.1, 0.5, 1.0, 2.0]:
    print(f"  E={E}: {mu_en_rho_xcom(E):.5f}")

# ===== GEOMETRY =====
SSD = 100.0
depth_poi = 10.0
field_half = 5.0
phantom_half = 15.0
phantom_depth = 30.0

# Source at (0,0,-SSD), phantom surface at z=0
# Field half-angles
theta_x_half = math.atan(field_half / SSD)  # radians
print(f"\nField half-angle: {theta_x_half*180/math.pi:.4f} deg")

# ===== PRIMARY PHOTON FLUENCE =====
# Phi0 = primary fluence at POI (110 cm) without phantom
# With phantom, primary fluence at POI:
# Phi_prim(10cm) = Phi0 * exp(-mu_total(2MeV) * 10)
mu_2MeV = mu_total_rho(2.0) * rho  # per cm
Phi_prim_ratio = np.exp(-mu_2MeV * depth_poi)
print(f"\nmu_total(2MeV) = {mu_2MeV:.5f} cm^-1")
print(f"Primary fluence ratio at 10 cm: Phi_prim/Phi0 = {Phi_prim_ratio:.4f}")

# Also check what fraction enters the beam at the surface
# The beam cross-section at the POI is 11x11 cm (assuming divergence)
# But Phi0 is defined at the POI itself (110 cm from source), so this is already accounted for.

print("\nChecking geometry at POI:")
print(f"  Field at POI (110 cm): {2*field_half*110/SSD:.2f} x {2*field_half*110/SSD:.2f} cm")