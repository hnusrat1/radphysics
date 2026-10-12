import numpy as np

# Cross section data
E_data = np.array([0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0])
mu_R_data = np.array([1.114, 0.0709, 0.0195, 0.00512, 0.00091, 0.000247, 0.000119, 0.000032, 0.000013])
mu_C_data = np.array([0.2941, 0.1379, 0.1470, 0.1277, 0.09477, 0.06958, 0.04877, 0.02377, 0.01301])
mu_PE_data = np.array([1.274, 0.00574, 0.00422, 0.00397, 0.00127, 0.000798, 0.000383, 0.000053, 0.000008])
mu_PP_data = np.array([0.0, 0.0, 0.0, 0.0, 0.000013, 0.000049, 0.000089, 0.000524, 0.001764])

def interp_mu_safe(E, mu_data):
    """Interpolate in log-log space"""
    mask = mu_data > 0
    E_pos = E_data[mask]
    mu_pos = mu_data[mask]
    
    if len(E_pos) < 2:
        return 0.0 if np.isscalar(E) else np.zeros_like(np.atleast_1d(E), dtype=float)
    
    log_E = np.log(E_pos)
    log_mu = np.log(mu_pos)
    
    is_scalar = np.isscalar(E)
    E_arr = np.atleast_1d(np.asarray(E, dtype=float))
    result = np.zeros_like(E_arr)
    
    for i, e in enumerate(E_arr):
        if e < E_pos[0]:
            slope = (log_mu[1] - log_mu[0]) / (log_E[1] - log_E[0])
            result[i] = np.exp(slope * (np.log(e) - log_E[0]) + log_mu[0])
        elif e > E_pos[-1]:
            slope = (log_mu[-1] - log_mu[-2]) / (log_E[-1] - log_E[-2])
            result[i] = np.exp(slope * (np.log(e) - log_E[-1]) + log_mu[-1])
        else:
            result[i] = np.exp(np.interp(np.log(e), log_E, log_mu))
    
    return float(result[0]) if is_scalar else result

def mu_R(E): return interp_mu_safe(E, mu_R_data)
def mu_C(E): return interp_mu_safe(E, mu_C_data)
def mu_PE(E): return interp_mu_safe(E, mu_PE_data)
def mu_PP(E): return interp_mu_safe(E, mu_PP_data)
def mu_total(E): return mu_R(E) + mu_C(E) + mu_PE(E) + mu_PP(E)

# Check
for E in [0.3, 0.4, 0.6, 0.8, 1.5]:
    t = mu_total(E)
    print(f"E={E:.1f}: total={t:.5f}, R%={mu_R(E)/t*100:.2f}, C%={mu_C(E)/t*100:.2f}, PE%={mu_PE(E)/t*100:.2f}")

print(f"\nE=0.6: total={mu_total(0.6):.5f}")
print(f"At 0.6 MeV: R%={mu_R(0.6)/mu_total(0.6)*100:.2f}%, C%={mu_C(0.6)/mu_total(0.6)*100:.2f}%, PE%={mu_PE(0.6)/mu_total(0.6)*100:.2f}%")

# mu_en/rho data
E_en = np.array([0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0])
mu_en_data = np.array([4.94, 0.0417, 0.0261, 0.0298, 0.0327, 0.0309, 0.0263, 0.0160, 0.0132])
print(f"\nmu_en/rho at 2.0 MeV: {interp_mu_safe(2.0, mu_en_data):.5f} cm^2/g")
print(f"mu_en/rho at 0.6 MeV: {interp_mu_safe(0.6, mu_en_data):.5f} cm^2/g")