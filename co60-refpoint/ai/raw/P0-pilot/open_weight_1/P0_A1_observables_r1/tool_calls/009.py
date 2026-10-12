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
electrons_per_gram_H2O = 10 * NA / M_H2O
electrons_per_cm3_H2O = electrons_per_gram_H2O * rho

# ===== CROSS SECTIONS =====

# Klein-Nishina total cross section per electron (cm^2)
def sigma_kn(E):
    alpha = E / mec2
    return 2*np.pi*r0_sq * ((1+alpha)/alpha**2 * (2*(1+alpha)/(1+2*alpha) - np.log(1+2*alpha)/alpha) 
                            + np.log(1+2*alpha)/(2*alpha) - (1+3*alpha)/(1+2*alpha)**2)

# Klein-Nishina differential cross section dσ/dΩ per electron (cm^2/sr)
# Returns (dsigma, E_scattered_MeV)
def dsigma_kn_dOmega(E, costheta):
    alpha = E / mec2
    k = 1.0 / (1.0 + alpha * (1.0 - costheta))
    Ep = E * k
    return (r0_sq/2.0) * k**2 * (k + 1.0/k - (1.0 - costheta**2)), Ep

# Incoherent scattering function S(q,Z) for water (approximate)
# At 2 MeV, for most angles, S ≈ Z (free electron limit)
# At small angles (long wavelengths), S is reduced
# For simplicity in this scope, we'll use a rough model or just S ≈ Z for E >= 0.3 MeV
# at high energies.

# Compton mass attenuation coefficient for water (cm^2/g)
# Calibrated to match NIST XCOM values
# XCOM values: E(MeV): mu_incoh/rho
xcom_comp_E = np.array([0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0])
xcom_comp_mu = np.array([0.0225, 0.0410, 0.0585, 0.0900, 0.129, 0.149, 0.198, 0.234, 0.271, 0.278, 0.274, 0.264, 0.240, 0.218, 0.176, 0.04451, 0.0979, 0.0616, 0.0430])

# Hmm, I clearly have wrong data in the table. Let me think.

# The Compton( incoherent) mass attenuation coefficient in XCOM for water:
# E=0.1 MeV: 0.149 cm^2/g
# E=0.2 MeV: 0.219 cm^2/g? No, that can't be right.

# Let me compute from KN:
# sigma_kn per electron: at low energy approaches 2*pi*r0^2 = 6.65e-25 cm^2
# electrons/g water = 3.343e23
# At 0.01 MeV (alpha = 0.0196), KN sigma ~ 6.65e-25 * (1 - ...) ~ 6.7e-25
# mu = 6.7e-25 * 3.343e23 = 0.224 cm^2/g

print("Let me recompute the proper cross sections.")

# Compute KN mu/rho over a grid
E_grid2 = np.array([0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0])
mu_kn_grid = np.array([sigma_kn(E) * electrons_per_gram_H2O for E in E_grid2])
for E, m in zip(E_grid2, mu_kn_grid):
    print(f"  E={E:.3f} MeV: KN mu_comp/rho = {m:.6f} cm^2/g")