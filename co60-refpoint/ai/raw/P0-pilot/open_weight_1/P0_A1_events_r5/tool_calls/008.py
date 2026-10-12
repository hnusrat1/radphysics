import numpy as np
import math

m_e_c2 = 0.511  # MeV

# Cross section data (cm^-1 for density 1.0)
E_data = np.array([0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0])
mu_R_data = np.array([1.114, 0.0709, 0.0195, 0.00512, 0.00091, 0.000247, 0.000119, 0.000032, 0.000013])
mu_C_data = np.array([0.2941, 0.1379, 0.1470, 0.1277, 0.09477, 0.06958, 0.04877, 0.02377, 0.01301])
mu_PE_data = np.array([1.274, 0.00574, 0.00422, 0.00397, 0.00127, 0.000798, 0.000383, 0.000053, 0.000008])
mu_PP_data = np.array([0.0, 0.0, 0.0, 0.0, 0.000013, 0.000049, 0.000089, 0.000524, 0.001764])

# Check
print("Data check:")
print(f"E_data shape: {E_data.shape}")
print(f"mu_R_data: {mu_R_data}")
print(f"mu_C_data: {mu_C_data}")
print(f"mu_PE_data: {mu_PE_data}")
print(f"mu_PP_data: {mu_PP_data}")

# Interpolation function
def interp_mu(E, mu_data):
    """Interpolate mu in log-log space"""
    # Clip to data range, extrapolate if needed
    log_E = np.log(E_data)
    log_mu = np.log(mu_data)
    log_E_new = np.log(np.clip(E, E_data[0], E_data[-1]))
    result = np.exp(np.interp(log_E_new, log_E, log_mu))
    # Handle extrapolation below minimum
    if np.isscalar(E):
        if E < E_data[0]:
            # Extrapolate using first two points
            slope = (log_mu[1] - log_mu[0]) / (log_E[1] - log_E[0])
            result = np.exp(slope * (np.log(E) - log_E[0]) + log_mu[0])
        elif E > E_data[-1]:
            slope = (log_mu[-1] - log_mu[-2]) / (log_E[-1] - log_E[-2])
            result = np.exp(slope * (np.log(E) - log_E[-1]) + log_mu[-1])
    else:
        result = np.array(result)
        for i, e in enumerate(np.atleast_1d(E)):
            if e < E_data[0]:
                slope = (log_mu[1] - log_mu[0]) / (log_E[1] - log_E[0])
                result[i] = np.exp(slope * (np.log(e) - log_E[0]) + log_mu[0])
            elif e > E_data[-1]:
                slope = (log_mu[-1] - log_mu[-2]) / (log_E[-1] - log_E[-2])
                result[i] = np.exp(slope * (np.log(e) - log_E[-1]) + log_mu[-1])
    return result

def mu_R(E): return float(interp_mu(E, mu_R_data))
def mu_C(E): return float(interp_mu(E, mu_C_data))
def mu_PE(E): return float(interp_mu(E, mu_PE_data))
def mu_PP(E): return float(interp_mu(E, mu_PP_data))
def mu_total(E): return mu_R(E) + mu_C(E) + mu_PE(E) + mu_PP(E)

# Test
print("\nCross sections at 2.0 MeV:")
t = mu_total(2.0)
print(f"Total: {t:.6f}")
print(f"R: {mu_R(2.0):.6f}, C: {mu_C(2.0):.6f}, PE: {mu_PE(2.0):.6f}, PP: {mu_PP(2.0):.6f}")
print(f"R%: {mu_R(2.0)/t*100:.3f}, C%: {mu_C(2.0)/t*100:.3f}, PE%: {mu_PE(2.0)/t*100:.3f}, PP%: {mu_PP(2.0)/t*100:.3f}")

print("\nCross sections at 0.5 MeV:")
t = mu_total(0.5)
print(f"Total: {t:.6f}")
print(f"R%: {mu_R(0.5)/t*100:.3f}, C%: {mu_C(0.5)/t*100:.3f}, PE%: {mu_PE(0.5)/t*100:.3f}, PP%: {mu_PP(0.5)/t*100:.3f}")

print("\nCross sections at 0.1 MeV:")
t = mu_total(0.1)
print(f"Total: {t:.6f}")
print(f"R%: {mu_R(0.1)/t*100:.3f}, C%: {mu_C(0.1)/t*100:.3f}, PE%: {mu_PE(0.1)/t*100:.3f}, PP%: {mu_PP(0.1)/t*100:.3f}")