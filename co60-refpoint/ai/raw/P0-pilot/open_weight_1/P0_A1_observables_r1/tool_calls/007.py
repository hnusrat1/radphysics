import numpy as np
import scipy.integrate as integrate
import math

# ===== CONSTANTS =====
mec2 = 0.511  # MeV
r0 = 2.8179403262e-13  # cm
r0_sq = r0**2
NA = 6.02214076e23
M_H2O = 18.01528
rho = 1.0  # g/cm^3

# ===== GEOMETRY =====
SSD = 100.0
depth_poi = 10.0
field_half = 5.0  # half-width of field at surface (cm)
# Phantom: x,y in [-15,15], z in [0,30]
phantom_half = 15.0
phantom_depth_max = 30.0

# Point of interest: (0,0,z=10)
r_poi = SSD + depth_poi  # 110 cm from source

# ===== CROSS SECTIONS FOR WATER (cm^2/g) =====
# Approximate NIST/XCOM values for 2 MeV
# These are the mass attenuation coefficients (total interaction, per unit mass)
# μ_Compton/ρ ≈ 0.04450
# μ_photo/ρ ≈ 0.00059
# μ_pair/ρ ≈ 0.00316 (nuclear + electron field)
# μ_Rayleigh/ρ ≈ 0.00103
# Total ≈ 0.04928

# I'll compute Compton exactly from Klein-Nishina, and use approximate values for others
def mu_total_water(E_MeV):
    """Total linear attenuation coefficient (cm^-1) in water"""
    # Piecewise approximation based on XCOM data (need to build model)
    # For now, use data points I recall
    pass

# Let me define the cross section model
# For Compton: Klein-Nishina per electron × electrons/g
def sigma_compton_cm2(E):
    """Total Compton cross section per electron in cm^2"""
    alpha = E / mec2
    return 2*np.pi*r0_sq * ((1+alpha)/alpha**2 * (2*(1+alpha)/(1+2*alpha) - np.log(1+2*alpha)/alpha) 
                            + np.log(1+2*alpha)/(2*alpha) - (1+3*alpha)/(1+2*alpha)**2)

def mu_compton_rho(E):
    """Compton mass attenuation for water (cm^2/g)"""
    return sigma_compton_cm2(E) * electrons_per_gram_H2O

electrons_per_gram_H2O = 10 * NA / M_H2O

# Photoelectric mass attenuation for water (cm^2/g) - approximate
# Data points (E_MeV, mu_photo/rho):
# Known XCOM data for water:
# 0.030: 3.62, 0.050: 0.677, 0.100: 0.1250, 0.200: 0.0239, 0.300: 0.00859
# 0.400: 0.00404, 0.500: 0.00219, 0.600: 0.00128, 0.800: 0.000548
# 1.000: 0.000271, 1.500: 0.0000706, 2.000: 0.0000261
# Let me use interpolation

pe_data_x = np.array([0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0])
pe_data_y = np.array([3.62, 0.677, 0.125, 0.0239, 0.00859, 0.00404, 0.00219, 0.00128, 0.000548, 0.000271, 0.0000706, 0.0000261])

def mu_pe_rho(E):
    """Photoelectric mass attenuation for water (cm^2/g)"""
    if E <= 0.03:
        return pe_data_y[0] * (E/0.03)**(-3.5)
    logx = np.log(pe_data_x)
    logy = np.log(pe_data_y)
    logE = np.log(np.clip(E, pe_data_x[0], pe_data_x[-1]))
    return np.exp(np.interp(logE, logx, logy))

# Pair production mass attenuation for water (cm^2/g)
# Above threshold 1.022 MeV
# Data (E_MeV, mu_pair/rho): 
# 1.25: 0.00060, 1.5: 0.00132, 2.0: 0.00316, 3.0: 0.00849, 4.0: 0.0143, 5.0: 0.0196
pp_data_x = np.array([1.02, 1.25, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0])
pp_data_y = np.array([0.0, 0.00060, 0.00132, 0.00316, 0.00849, 0.0143, 0.0196, 0.0245, 0.0323, 0.0387])

def mu_pair_rho(E):
    """Pair production mass attenuation for water (cm^2/g)"""
    if E < 1.022:
        return 0.0
    logx = np.log(pp_data_x)
    logy = np.log(pp_data_y)
    logE = np.log(np.clip(E, pp_data_x[0], pp_data_x[-1]))
    return np.exp(np.interp(logE, logx, logy))

# Rayleigh mass attenuation for water (cm^2/g)
# Data: 0.03: 0.00267, 0.05: 0.00268, 0.1: 0.00256, 0.2: 0.00242, 0.3: 0.00227
# 0.4: 0.00214, 0.5: 0.00200, 0.6: 0.00187, 0.8: 0.00162, 1.0: 0.00140
# 1.5: 0.00100, 2.0: 0.00072
ray_data_x = np.array([0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0])
ray_data_y = np.array([0.00267, 0.00268, 0.00256, 0.00242, 0.00227, 0.00214, 0.00200, 0.00187, 0.00162, 0.00140, 0.00100, 0.00072])

def mu_rayleigh_rho(E):
    """Rayleigh mass attenuation for water (cm^2/g)"""
    logx = np.log(ray_data_x)
    logy = np.log(ray_data_y)
    logE = np.log(np.clip(E, ray_data_x[0], ray_data_x[-1]))
    return np.exp(np.interp(logE, logx, logy))

# Total mass attenuation
def mu_total_rho(E):
    return mu_compton_rho(E) + mu_pe_rho(E) + mu_pair_rho(E) + mu_rayleigh_rho(E)

E0 = 2.0
print(f"Mass attenuation components for water at 2 MeV:")
print(f"  Compton: {mu_compton_rho(E0):.6f} cm^2/g")
print(f"  Photo:   {mu_pe_rho(E0):.6f} cm^2/g")
print(f"  Pair:    {mu_pair_rho(E0):.6f} cm^2/g")
print(f"  Rayleigh:{mu_rayleigh_rho(E0):.6f} cm^2/g")
print(f"  Total:   {mu_total_rho(E0):.6f} cm^2/g")

# Test at other energies
for E_test in [0.1, 0.3, 0.5, 1.0, 1.5, 2.0]:
    print(f"  E={E_test} MeV: total mu/rho = {mu_total_rho(E_test):.6f}")

# Linear attenuation coefficient in water
def mu_linear(E):
    return mu_total_rho(E) * rho

print(f"\nLinear attenuation at 2 MeV: {mu_linear(E0):.5f} cm^-1")
print(f"Primary transmission at 10 cm: {np.exp(-mu_linear(E0)*10):.4f}")