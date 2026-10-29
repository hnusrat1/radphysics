import numpy as np

me = 0.5109989461

# Rayleigh - very forward
def sample_rayleigh_cos():
    # Use cos(theta) close to 1
    return np.random.uniform(0.999, 1.0)

# Photoelectron angle (Sauter distribution simplified)
def sample_pe_electron_cos(E_kin):
    E_tot = E_kin + me
    beta = np.sqrt(1 - (me/E_tot)**2)
    # Use rejection sampling with max at mu=1
    max_val = (1 - 0) / (1 - beta)**4  # at mu=1, sin^2=0, but limit is 0? Actually at mu=1, (1-mu^2)=0, val=0. Max is elsewhere.
    # Find approximate maximum by scanning
    mus = np.linspace(-1, 1, 1001)
    vals = (1 - mus**2) / (1 - beta*mus)**4
    max_val = np.max(vals)
    while True:
        mu = np.random.uniform(-1, 1)
        val = (1 - mu**2) / (1 - beta*mu)**4
        if np.random.random() < val / max_val:
            return mu

# Pair production
def sample_pair(K_avail):
    t_e = np.random.uniform(0, K_avail)
    t_p = K_avail - t_e
    # Both particles emitted forward
    cos_e = np.random.uniform(0.98, 1.0)
    cos_p = np.random.uniform(0.98, 1.0)
    return t_e, t_p, cos_e, cos_p

# Test
print("Rayleigh cos:", sample_rayleigh_cos())
print("PE electron cos:", sample_pe_electron_cos(1.0))
print("Pair:", sample_pair(0.15))