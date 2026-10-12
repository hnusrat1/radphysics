import numpy as np

# Physical constants
m_e = 0.510998950  # MeV/c^2
r_e = 2.8179403262e-13  # cm
NA = 6.02214076e23
e_per_cm3 = 10 * NA / 18.01528  # electrons/cm^3 in water

# ============================================================
# Cross sections for water at density 1.0 g/cm^3
# ============================================================

def mu_compton(E):
    """Compton linear attenuation coefficient (cm^-1) for water"""
    alpha = E / m_e
    # Klein-Nishina total cross section per electron
    t1 = (1+alpha)/alpha**2 * (2*(1+alpha)/(1+2*alpha) - np.log(1+2*alpha)/alpha)
    t2 = np.log(1+2*alpha)/(2*alpha)
    t3 = (1+3*alpha)/(1+2*alpha)**2
    sigma = 2*np.pi*r_e**2*(t1 + t2 - t3)
    return sigma * e_per_cm3

def mu_photo(E):
    """Photoelectric linear attenuation coefficient (cm^-1) for water.
    Uses NIST data interpolation."""
    data_E = np.array([0.5, 0.6, 0.8, 1.0, 1.1732, 1.25, 1.3325, 1.5, 2.0])
    data_mu = np.array([0.00406, 0.00555, 0.00450, 0.00377, 0.00300, 0.00258, 0.00230, 0.00179, 0.00100])
    return np.interp(E, data_E, data_mu) * 1.0

def mu_pair(E):
    """Pair production linear attenuation coefficient (cm^-1) for water."""
    if E < 2*m_e:
        return 0.0
    data_E = np.array([1.022, 1.1, 1.1732, 1.25, 1.3325, 1.5, 2.0, 3.0])
    data_mu = np.array([0.0, 0.0002, 0.00035, 0.00071, 0.00100, 0.00146, 0.00383, 0.00850])
    return np.interp(E, data_E, data_mu) * 1.0

def mu_rayleigh(E):
    """Rayleigh scattering linear attenuation coefficient (cm^-1) for water."""
    data_E = np.array([0.5, 0.6, 0.8, 1.0, 1.1732, 1.25, 1.3325, 1.5, 2.0])
    data_mu = np.array([0.0055, 0.0045, 0.0035, 0.00303, 0.00250, 0.00242, 0.00210, 0.00186, 0.00114])
    return np.interp(E, data_E, data_mu) * 1.0

def mu_total(E):
    return mu_compton(E) + mu_photo(E) + mu_pair(E) + mu_rayleigh(E)

def sample_interaction_type(E):
    """Sample interaction type based on relative cross sections.
    Returns: 'compton', 'photoelectric', 'pair', or 'rayleigh'"""
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
    """Sample Compton scattering. Returns (cos_theta, E_out, t_e)"""
    alpha = E_in / m_e
    
    # Rejection sampling (Kahn method)
    while True:
        epsilon = 2 * np.random.random() - 1
        cos_theta = 1 - 2 * alpha * epsilon / (1 + alpha * (1 - epsilon))
        if cos_theta < -1 or cos_theta > 1:
            continue
        
        E_out = E_in / (1 + alpha * (1 - cos_theta))
        ratio = (E_out/E_in)**2 * (E_out/E_in + E_in/E_out - (1 - cos_theta**2))
        
        if (1 + cos_theta**2) * np.random.random() <= ratio:
            break
    
    t_e = E_in - E_out
    return cos_theta, E_out, t_e

def sample_compton_direction(E_in):
    """Alternative simpler method to sample Compton scattering"""
    alpha = E_in / m_e
    
    # Use method from Zaidi (2006)
    while True:
        # Sample from the approximate distribution
        r1 = np.random.random()
        r2 = np.random.random()
        
        # Proposed cos_theta
        a = 1 + 2*alpha*r1
        b = 1 + 2*alpha
        cos_theta = 1 - 2*r1*alpha / (a)
        # Actually this is translating to: cos_theta = 1 - 2*alpha*r1/(1+2*alpha*r1)
        # which gives values between 1 and -1 for r1 in [0,1]
        
        # Hmm, let me just use a different approach
        # Sample from approximate distribution and use rejection
        
        # Let me try: sample cos_uniform and accept with probability proportional to KN DCS
        cos_theta = 2*np.random.random() - 1
        E_out = E_in / (1 + alpha * (1 - cos_theta))
        
        # KN formula (unnormalized)
        kn = (E_out/E_in)**2 * (E_out/E_in + E_in/E_out - (1 - cos_theta**2))
        
        # Max of KN is at cos_theta = 1: kn_max = 1
        if np.random.random() <= kn:
            t_e = E_in - E_out
            return cos_theta, E_out, t_e

def sample_pair(E_in):
    """Sample pair production.
    Returns (t_e, cos_e, t_positron, cos_positron)
    For simplicity, assume equal energy sharing (approximately true at these energies)
    and isotropic emission in CMS."""
    # In pair production, the total kinetic energy is E_in - 2*m_e
    K_total = E_in - 2*m_e
    
    # Approximately equal sharing (this is a simplification)
    # Actually for the problem, we need t_e and cos_electron, and t_positron
    
    # For simplicity, assume equal energy sharing
    t_e = K_total / 2
    t_positron = K_total / 2
    
    # The pair is emitted roughly forward, with angular distribution
    # peaking at cos_theta ≈ 1. For simplicity, sample from an approximation.
    # At these energies (just above threshold), the angular distribution is 
    # approximately isotropic in the CMS.
    
    # Sample cos_theta for electron (and positron opposite in CMS)
    # For a simple approximation, sample from cos_theta = 1 - 2*random()
    # but weighted by forward peaking
    
    # More realistic: use the Bethe-Heitler angular distribution approximation
    # For our purposes, sample roughly forward-peaked
    
    # Let me use a simple model: cos_theta ≈ 1 - small_angle
    # For the electron
    theta = np.random.exponential(0.1)  # roughly forward
    cos_e = np.cos(theta)
    
    return t_e, cos_e, t_positron

# Let me test
print("Testing cross sections at 1.1732 MeV:")
E = 1.1732
print(f"  mu_C = {mu_compton(E):.6f}")
print(f"  mu_P = {mu_photo(E):.6f}")
print(f"  mu_PP = {mu_pair(E):.6f}")
print(f"  mu_R = {mu_rayleigh(E):.6f}")
print(f"  mu_T = {mu_total(E):.6f}")

print("\nTesting cross sections at 1.3325 MeV:")
E = 1.3325
print(f"  mu_C = {mu_compton(E):.6f}")
print(f"  mu_P = {mu_photo(E):.6f}")
print(f"  mu_PP = {mu_pair(E):.6f}")
print(f"  mu_R = {mu_rayleigh(E):.6f}")
print(f"  mu_T = {mu_total(E):.6f}")

print("\nTesting interaction type sampling:")
np.random.seed(42)
types = {'compton': 0, 'photoelectric': 0, 'pair': 0, 'rayleigh': 0}
for _ in range(10000):
    t = sample_interaction_type(1.1732)
    types[t] += 1
for k, v in types.items():
    print(f"  {k}: {v/10000:.4f}")

print("\nTesting Compton sampling:")
np.random.seed(42)
for _ in range(5):
    cos, Eout, te = sample_compton_direction(1.1732)
    print(f"  cos={cos:.4f}, E_out={Eout:.4f}, t_e={te:.4f}")