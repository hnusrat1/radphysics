import numpy as np

# ============================================================
# Constants and cross sections for water
# ============================================================
m_e = 0.510998950  # MeV
r_e = 2.8179403262e-13  # cm
NA = 6.02214076e23
e_per_cm3 = 10 * NA / 18.01528  # electrons per cm^3

def mu_compton(E):
    """Klein-Nishina Compton coefficient (cm^-1)"""
    a = E / m_e
    t1 = (1+a)/a**2 * (2*(1+a)/(1+2*a) - np.log(1+2*a)/a)
    t2 = np.log(1+2*a)/(2*a)
    t3 = (1+3*a)/(1+2*a)**2
    return 2*np.pi*r_e**2 * (t1 + t2 - t3) * e_per_cm3

def mu_photo(E):
    """Photoelectric coefficient (cm^-1) - NIST data for water"""
    xs = np.array([0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.1732, 1.25, 1.3325, 1.5, 2.0])
    ys = np.array([4.91, 1.67, 0.755, 0.216, 0.0881, 0.0442, 0.0255, 0.0113, 0.00642, 0.00254, 0.00152, 0.00154, 0.00228, 0.00406, 0.00555, 0.00450, 0.00377, 0.00300, 0.00258, 0.00230, 0.00179, 0.00100])
    return np.interp(E, xs, ys) * 1.0

def mu_pair(E):
    """Pair production coefficient (cm^-1)"""
    if E < 2*m_e:
        return 0.0
    xs = np.array([1.022, 1.1, 1.1732, 1.25, 1.3325, 1.5, 2.0, 3.0, 5.0])
    ys = np.array([0.0, 0.0002, 0.00035, 0.00071, 0.00100, 0.00146, 0.00383, 0.0085, 0.0175])
    return np.interp(E, xs, ys) * 1.0

def mu_rayleigh(E):
    """Rayleigh coefficient (cm^-1)"""
    xs = np.array([0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.1732, 1.25, 1.3325, 1.5, 2.0])
    ys = np.array([0.037, 0.034, 0.032, 0.028, 0.025, 0.022, 0.019, 0.015, 0.012, 0.0085, 0.0065, 0.0050, 0.0040, 0.0055, 0.0045, 0.0035, 0.00303, 0.00250, 0.00242, 0.00210, 0.00186, 0.00114])
    return np.interp(E, xs, ys) * 1.0

def mu_total(E):
    return mu_compton(E) + mu_photo(E) + mu_pair(E) + mu_rayleigh(E)

# Print cross sections
print("Cross sections at 1.1732 MeV:")
mu_c = mu_compton(1.1732)
mu_p = mu_photo(1.1732)
mu_pp = mu_pair(1.1732)
mu_r = mu_rayleigh(1.1732)
mu_t = mu_total(1.1732)
print(f"  C={mu_c:.6f} P={mu_p:.6f} PP={mu_pp:.6f} R={mu_r:.6f} T={mu_t:.6f}")

print("Cross sections at 1.3325 MeV:")
mu_c = mu_compton(1.3325)
mu_p = mu_photo(1.3325)
mu_pp = mu_pair(1.3325)
mu_r = mu_rayleigh(1.3325)
mu_t = mu_total(1.3325)
print(f"  C={mu_c:.6f} P={mu_p:.6f} PP={mu_pp:.6f} R={mu_r:.6f} T={mu_t:.6f}")

# ============================================================
# Sampling functions
# ============================================================
def sample_interaction_type(E):
    mc = mu_compton(E)
    mp = mu_photo(E)
    mpp = mu_pair(E)
    mr = mu_rayleigh(E)
    total = mc + mp + mpp + mr
    r = np.random.random() * total
    if r < mc:
        return 'compton'
    r -= mc
    if r < mp:
        return 'photoelectric'
    r -= mp
    if r < mpp:
        return 'pair'
    return 'rayleigh'

def sample_compton(E_in):
    a = E_in / m_e
    while True:
        cos_theta = 2*np.random.random() - 1
        E_out = E_in / (1 + a * (1 - cos_theta))
        kn = (E_out/E_in)**2 * (E_out/E_in + E_in/E_out - (1 - cos_theta**2))
        if np.random.random() <= kn:
            return cos_theta, E_out, E_in - E_out

def sample_rayleigh(E_in):
    """Rayleigh scattering: elastic, very forward-peaked."""
    # For 1.17 MeV photons on water, typical scattering angle ~ 0.001 rad
    # Use form-factor-based approximate sampling
    while True:
        theta = np.random.exponential(0.002)
        if theta > 0.1:
            continue
        cos_theta = np.cos(theta)
        # Form factor squared ~ exp(-(theta/k)^2) with k ~ 0.001
        ff2 = np.exp(-(theta/0.001)**2)
        w = (1 + cos_theta**2) / 2 * ff2
        if np.random.random() <= w:
            return cos_theta, E_in

# Test sampling
print("\nTesting Compton sampling:")
np.random.seed(123)
for _ in range(3):
    cos, eout, te = sample_compton(1.1732)
    print(f"  cos={cos:.4f}, E_out={eout:.4f}, t_e={te:.4f}")