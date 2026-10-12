import numpy as np
from scipy.interpolate import interp1d
import math

np.random.seed(12345)

# Cross section data for water (mass attenuation * density = linear, density=1.0)
data_E = np.array([0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0])
data_coh = np.array([1.114, 0.0709, 0.0195, 0.00512, 0.00091, 0.000247, 0.000119, 0.000032, 0.000013])
data_incoh = np.array([0.2941, 0.1379, 0.1470, 0.1277, 0.09477, 0.06958, 0.04877, 0.02377, 0.01301])
data_photo = np.array([1.274, 0.00574, 0.00422, 0.00397, 0.00127, 0.000798, 0.000383, 0.000053, 0.000008])
data_pair = np.array([0.0, 0.0, 0.0, 0.0, 0.000013, 0.000049, 0.000089, 0.000524, 0.001764])

data_total = data_coh + data_incoh + data_photo + data_pair

def interp_log_log(x_vals, y_vals, x_new):
    """Interpolate in log-log space"""
    # Filter to positive values
    mask = y_vals > 0
    x_pos = x_vals[mask]
    y_pos = y_vals[mask]
    
    x_new = np.atleast_1d(np.asarray(x_new, dtype=float))
    result = np.empty_like(x_new)
    
    for i, x in enumerate(x_new):
        if x < x_pos.min():
            # Extrapolate using first two points
            logx = np.log(x_pos[:2])
            logy = np.log(y_pos[:2])
            slope = (logy[1] - logy[0]) / (logx[1] - logx[0])
            result[i] = np.exp(slope * (np.log(x) - logx[0]) + logy[0])
        elif x > x_pos.max():
            # Extrapolate using last two points
            logx = np.log(x_pos[-2:])
            logy = np.log(y_pos[-2:])
            slope = (logy[1] - logy[0]) / (logx[1] - logx[0])
            result[i] = np.exp(slope * (np.log(x) - logx[0]) + logy[0])
        else:
            # Interpolate
            logx = np.log(x_pos)
            logy = np.log(y_pos)
            result[i] = np.exp(np.interp(np.log(x), logx, logy))
    
    return result.item() if result.size == 1 else result

# Cross section functions (linear attenuation, cm^-1)
def mu_R(E):
    return float(interp_log_log(data_E, data_coh, E))

def mu_C(E):
    return float(interp_log_log(data_E, data_incoh, E))

def mu_PE(E):
    return float(interp_log_log(data_E, data_photo, E))

def mu_PP(E):
    return float(interp_log_log(data_E, data_pair, E))

def mu_total(E):
    return mu_R(E) + mu_C(E) + mu_PE(E) + mu_PP(E)

# Test
print("Cross sections at key energies:")
for E in [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]:
    t = mu_total(E)
    r = mu_R(E) / t * 100
    c = mu_C(E) / t * 100
    p = mu_PE(E) / t * 100
    pp = mu_PP(E) / t * 100
    print(f"E={E:.3f} MeV: mu={t:.5f}, R={r:.2f}%, C={c:.2f}%, PE={p:.2f}%, PP={pp:.2f}%")

print(f"Mean free path at 2.0 MeV: {1/mu_total(2.0):.3f} cm")