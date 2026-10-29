import numpy as np
import math
import json

np.random.seed(42)

# Constants
m_e_c2 = 0.511  # MeV
E_source = 2.0  # MeV

# Cross section data (linear attenuation, cm^-1 for water at density 1.0)
E_xs = np.array([0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0])
mu_R = np.array([1.114, 0.0709, 0.0195, 0.00512, 0.00091, 0.000247, 0.000119, 0.000032, 0.000013])
mu_C = np.array([0.2941, 0.1379, 0.1470, 0.1277, 0.09477, 0.06958, 0.04877, 0.02377, 0.01301])
mu_PE = np.array([1.274, 0.00574, 0.00422, 0.00397, 0.00127, 0.000798, 0.000383, 0.000053, 0.000008])
mu_PP = np.array([0.0, 0.0, 0.0, 0.0, 0.000013, 0.000049, 0.000089, 0.000524, 0.001764])

def get_mu(E, mu_data):
    """Log-log interpolation of mu at energy E"""
    mask = mu_data > 0
    E_pos = E_xs[mask]
    mu_pos = mu_data[mask]
    if len(E_pos) < 2:
        return 0.0
    log_E = np.log(E_pos)
    log_mu = np.log(mu_pos)
    e = np.clip(E, E_pos[0], E_pos[-1])
    return float(np.exp(np.interp(np.log(e), log_E, log_mu)))

def mu_total(E):
    return get_mu(E, mu_R) + get_mu(E, mu_C) + get_mu(E, mu_PE) + get_mu(E, mu_PP)

def sample_interaction_process(E):
    """Sample which interaction occurs, return process string"""
    r = get_mu(E, mu_R)
    c = get_mu(E, mu_C)
    p = get_mu(E, mu_PE)
    pp = get_mu(E, mu_PP)
    total = r + c + p + pp
    if total <= 0:
        return None
    xi = np.random.random() * total
    if xi < r:
        return "rayleigh"
    xi -= r
    if xi < c:
        return "compton"
    xi -= c
    if xi < p:
        return "photoelectric"
    return "pair"

# Test
print("Interaction probabilities at 2.0 MeV:")
r = get_mu(2.0, mu_R)
c = get_mu(2.0, mu_C)
p = get_mu(2.0, mu_PE)
pp = get_mu(2.0, mu_PP)
t = r + c + p + pp
print(f"  Rayleigh: {r/t*100:.3f}%")
print(f"  Compton: {c/t*100:.3f}%")
print(f"  Photoelectric: {p/t*100:.3f}%")
print(f"  Pair production: {pp/t*100:.3f}%")
print(f"  Total mu: {t:.5f} cm^-1")
print(f"  Mean free path: {1/t:.3f} cm")

print("\nInteraction probabilities at 0.6 MeV:")
r = get_mu(0.6, mu_R)
c = get_mu(0.6, mu_C)
p = get_mu(0.6, mu_PE)
pp = get_mu(0.6, mu_PP)
t = r + c + p + pp
print(f"  Rayleigh: {r/t*100:.3f}%")
print(f"  Compton: {c/t*100:.3f}%")
print(f"  Photoelectric: {p/t*100:.3f}%")
print(f"  Pair production: {pp/t*100:.3f}%")
print(f"  Total mu: {t:.5f} cm^-1")