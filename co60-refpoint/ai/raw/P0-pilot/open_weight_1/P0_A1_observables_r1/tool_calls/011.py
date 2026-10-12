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

# ===== CROSS-SECTION MODEL (calibrated to XCOM) =====
F_COMP = 0.95  # incoherent scattering function correction (approximate average)

# Klein-Nishina total cross section per electron (cm^2)
def sigma_kn(E):
    if E <= 0: return 0
    alpha = E / mec2
    return (2*np.pi*r0_sq * ((1+alpha)/alpha**2 * (2*(1+alpha)/(1+2*alpha) - np.log(1+2*alpha)/alpha) 
                            + np.log(1+2*alpha)/(2*alpha) - (1+3*alpha)/(1+2*alpha)**2))

def mu_comp_rho(E):
    """Compton (incoherent) mass attenuation for water, cm^2/g"""
    return F_COMP * sigma_kn(E) * electrons_per_gram_H2O

def mu_pe_rho(E):
    """Photoelectric mass attenuation for water, cm^2/g"""
    E = np.clip(E, 0.01, 5.0)
    table_E = np.array([0.01,0.015,0.02,0.03,0.04,0.05,0.06,0.08,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.8,1.0,1.25,1.5,2.0,3.0,5.0])
    table_M = np.array([4.79,1.41,0.519,0.169,0.0770,0.0261,0.0137,0.00480,0.00213,0.000388,0.000111,0.0000235,0.00000881,0.00000412,0.00000212,0.000000838,0.000000443,0.000000220,0.000000123,0.0000000581,0.0000000191,0.00000000392])
    return np.exp(np.interp(np.log(E), np.log(table_E), np.log(table_M)))

def mu_pair_rho(E):
    """Pair production mass attenuation for water, cm^2/g"""
    if E <= 1.022: return 0.0
    E = np.clip(E, 1.022, 10.0)
    table_E = np.array([1.022,1.25,1.5,2.0,2.5,3.0,4.0,5.0,6.0,8.0,10.0])
    table_M = np.array([0.0,0.000639,0.00151,0.00318,0.00496,0.00720,0.0124,0.0169,0.0211,0.0286,0.0340])
    # interpolate in log
    mask = E > 1.022
    result = np.zeros_like(E)
    if np.isscalar(E):
        if E <= 1.022: return 0.0
        idx = np.searchsorted(table_E, E)
        if idx == 0: return 0.0
        if idx >= len(table_E): idx = len(table_E)-1
        E1, E2 = table_E[idx-1], table_E[idx]
        M1, M2 = table_M[idx-1], table_M[idx]
        if M1 == 0: M1 = 1e-10
        return np.exp(np.interp(np.log(E), np.log([E1,E2]), np.log([max(M1,1e-10),M2])))
    else:
        result = np.zeros_like(E)
        mask = E > 1.022
        E_masked = E[mask]
        logE = np.log(E_masked)
        logE_table = np.log(table_E)
        logM_table = np.log(np.maximum(table_M, 1e-12))
        result[mask] = np.exp(np.interp(logE, logE_table, logM_table))
        return result

def mu_ray_rho(E):
    """Rayleigh mass attenuation for water, cm^2/g"""
    E = np.clip(E, 0.01, 5.0)
    table_E = np.array([0.01,0.015,0.02,0.03,0.05,0.06,0.08,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.8,1.0,1.25,1.5,2.0,3.0,5.0])
    table_M = np.array([0.000404,0.000882,0.00154,0.00349,0.00616,0.00716,0.00722,0.00682,0.00574,0.00467,0.00331,0.00250,0.00199,0.00162,0.00116,0.000873,0.000651,0.000502,0.000326,0.000151,0.0000650])
    return np.exp(np.interp(np.log(E), np.log(table_E), np.log(table_M)))

def mu_total_rho(E):
    """Total mass attenuation coefficient for water (cm^2/g)"""
    if np.isscalar(E):
        return mu_comp_rho(E) + mu_pe_rho(E) + mu_pair_rho(E) + mu_ray_rho(E)
    E = np.array(E)
    return mu_comp_rho(E) + mu_pe_rho(E) + mu_pair_rho(E) + mu_ray_rho(E)

def mu_linear(E):
    """Linear attenuation coefficient (cm^-1)"""
    return mu_total_rho(E) * rho

# Check values
print("Model cross sections for water (cm^2/g):")
print(f"  E=2.000: comp={mu_comp_rho(2.0):.5f}, pe={mu_pe_rho(2.0):.2e}, pair={mu_pair_rho(2.0):.5f}, ray={mu_ray_rho(2.0):.5f}, tot={mu_total_rho(2.0):.5f}")
print(f"  E=1.000: comp={mu_comp_rho(1.0):.5f}, pe={mu_pe_rho(1.0):.2e}, pair={mu_pair_rho(1.0):.5f}, ray={mu_ray_rho(1.0):.5f}, tot={mu_total_rho(1.0):.5f}")
print(f"  E=0.500: comp={mu_comp_rho(0.5):.5f}, pe={mu_pe_rho(0.5):.2e}, pair={mu_pair_rho(0.5):.5f}, ray={mu_ray_rho(0.5):.5f}, tot={mu_total_rho(0.5):.5f}")
print(f"  E=0.100: comp={mu_comp_rho(0.1):.5f}, pe={mu_pe_rho(0.1):.2e}, pair={mu_pair_rho(0.1):.5f}, ray={mu_ray_rho(0.1):.5f}, tot={mu_total_rho(0.1):.5f}")
print(f"  E=0.200: comp={mu_comp_rho(0.2):.5f}, pe={mu_pe_rho(0.2):.2e}, pair={mu_pair_rho(0.2):.5f}, ray={mu_ray_rho(0.2):.5f}, tot={mu_total_rho(0.2):.5f}")

# Mass energy absorption/transfer coefficients
# For computing kerma
def mu_en_rho(E):
    """Mass energy absorption coefficient for water (cm^2/g)"""
    E = np.array(E)
    # Distributed: Compton energy transfer + photo + pair (minus electron energy lost to bremsstrahlung)
    # For the collision kerma, we need mu_en (energy absorption), 
    # for total kerma we need mu_tr (energy transfer)
    
    # Approximate energy transfer fractions per process:
    # Compton: 1 - <E_sc>/E
    # Photo: (E - E_bind)/E ≈ 1
    # Pair: 1 - 2*mec2/E
    # Rayleigh: 0
    
    result = np.zeros_like(E)
    for i, e in enumerate(np.atleast_1d(E)):
        if e <= 0:
            continue
        # Compton average energy transfer (compute exactly)
        _, _, avg_ET = compton_energy_quantities_cached(e)
        comp_tr = mu_comp_rho(e) * (avg_ET / e)
        pe_tr = mu_pe_rho(e) * 1.0
        pair_tr = mu_pair_rho(e) * max(0, 1 - 2*mec2/e) if e > 2*mec2 else 0
        # Rayleigh: 0
        mu_tr = comp_tr + pe_tr + pair_tr
        # energy absorption = energy transfer × (1 - g)
        # g is the radiative fraction, which for water at 2 MeV is small (~0.5%)
        g_frac = 0.005  # approximate
        mu_en = mu_tr * (1 - g_frac)
        result[i] = mu_en
    return result

# Cache for compton energy transfer
_compton_cache = {}
def compton_energy_quantities_cached(E):
    if E in _compton_cache:
        return _compton_cache[E]
    costheta = np.linspace(-1, 1, 50000)
    alpha = E / mec2
    k = 1.0 / (1.0 + alpha * (1.0 - costheta))
    Ep = E * k
    ds = (r0_sq/2.0) * k**2 * (k + 1/k - (1 - costheta**2))
    dfdx = 2*np.pi * ds
    sigma = integrate.simpson(dfdx, costheta)
    avg_Ep = integrate.simpson(dfdx * Ep, costheta) / sigma
    avg_ET = E - avg_Ep
    _compton_cache[E] = (sigma, avg_Ep, avg_ET)
    return _compton_cache[E]

# Test
sigma, avgEp, avgET = compton_energy_quantities_cached(2.0)
print(f"\nCompton at 2 MeV: sigma={sigma:.4e}, <E_sc>={avgEp:.4f}, <E_tr>={avgET:.4f}")
print(f"mu_en_rho(2 MeV) = {mu_en_rho(2.0):.5f}")
print(f"mu_en_rho(1 MeV) = {mu_en_rho(1.0):.5f}")
print(f"mu_en_rho(0.5 MeV) = {mu_en_rho(0.5):.5f}")
print(f"mu_en_rho(0.1 MeV) = {mu_en_rho(0.1):.5f}")

# Known XCOM values for comparison:
# mu_en_rho(2 MeV) ≈ 0.0233, mu_en_rho(1 MeV) ≈ 0.0310, mu_en_rho(0.5) ≈ 0.0326, mu_en_rho(0.1) ≈ 0.0256
print("\n(For reference, XCOM mu_en/rho: 2 MeV=0.0233, 1 MeV=0.0310, 0.5=0.0326, 0.1=0.0256)")