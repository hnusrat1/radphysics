import numpy as np

me = 0.5109989461
r_e = 2.8179403227e-13

# Compton sampling
def sample_compton_cos(E):
    alpha = E / me
    while True:
        mu = np.random.uniform(-1, 1)
        f = 1.0 / (1.0 + alpha * (1.0 - mu))
        dcs = f**2 * (f + 1.0/f - (1.0 - mu**2))
        if np.random.random() < dcs / 2.0:
            return mu

def compton_electron_cos(E, cos_theta):
    sin_theta = np.sqrt(1 - cos_theta**2)
    tan_half = sin_theta / (1 + cos_theta)
    alpha = E / me
    cot_phi = (1 + alpha) * tan_half
    if cot_phi == 0:
        return 0.0
    tan_phi = 1.0 / cot_phi
    cos_phi = 1.0 / np.sqrt(1.0 + tan_phi**2)
    return cos_phi

# Test
cos_t = sample_compton_cos(1.1732)
print("cos_theta:", cos_t)
E_out = 1.1732 / (1 + (1.1732/me)*(1-cos_t))
print("E_out:", E_out)
cos_e = compton_electron_cos(1.1732, cos_t)
print("cos_electron:", cos_e)