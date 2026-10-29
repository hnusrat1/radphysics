import numpy as np
from scipy import interpolate

# ---- Compton scattering sampling ----
def sample_compton(energy, rng=np.random):
    """
    Sample Compton scattering.
    Returns: (e_out, cos_scatter, e_electron, cos_electron)
    e_out: outgoing photon energy (MeV)
    cos_scatter: cosine of photon scattering angle
    e_electron: kinetic energy of electron (MeV)
    cos_electron: cosine of electron angle relative to incident photon direction
    """
    E = energy
    alpha = E / m_e
    # Use Kahn's method (rejection sampling)
    # Based on algorithm from EGS4 / PENELOPE
    
    # Constants
    a1 = 1.0 / (1.0 + 2.0 * alpha)
    a2 = 1.0 / (1.0 + 0.0001)  # not used?
    
    # Sample from the distribution using the "mixed method"
    while True:
        # Generate two random numbers
        xi1 = rng.uniform()
        xi2 = rng.uniform()
        
        # First branch (p1 = probability of using first distribution)
        # The distribution is split into two components:
        # f1(cosθ) = 1/(1+α(1-cosθ))^2 (which can be sampled analytically)
        # f2(cosθ) = 1/(1+α(1-cosθ))^2 * (1-cos^2θ) (rejection)
        
        # Actually, let me use a simpler algorithm:
        # Sample from p(μ) ∝ (1/μ0^2 + (1-μ^2)/2) where μ = cosθ, μ0 = 1+α(1-μ)
        # using direct rejection.
        
        # Sample cosθ uniformly in [-1,1]
        mu = 2.0 * xi1 - 1.0
        # Compute E'/E
        ratio = 1.0 / (1.0 + alpha * (1.0 - mu))
        # Klein-Nishina rejection function
        f_kn = ratio**2 * (ratio + 1.0/ratio - (1.0 - mu**2))
        # Maximum value of f_kn occurs at mu=1 (forward)
        # At mu=1: ratio=1, f=1*(1+1-0)=2
        # Actually the maximum is at mu=1, f=2
        # At mu=-1: ratio=1/(1+2α), f = ratio^2 * (ratio+1/ratio-0) = ratio^2*(ratio+1/ratio) = ratio + 1/ratio? Wait:
        # At mu=-1: ratio = 1/(1+2α)
        # f = ratio^2 * (ratio + 1/ratio - 0) = ratio^3 + ratio = ratio*(ratio^2+1)
        # For alpha=3.9 (2 MeV), ratio~0.1136, f~0.1136*(0.0129+1)=0.115
        # So max is 2 at mu=1.
        if xi2 <= f_kn / 2.0:
            # Accept
            e_out = E * ratio
            cos_scatter = mu
            break
    
    # Compute electron kinetic energy
    e_electron = E - e_out
    
    # Compute electron angle
    # cosθ_e = (E - E' cosθ) / sqrt(E^2 + E'^2 - 2 E E' cosθ)
    denom = np.sqrt(E**2 + e_out**2 - 2.0 * E * e_out * cos_scatter)
    if denom > 0:
        cos_electron = (E - e_out * cos_scatter) / denom
    else:
        cos_electron = 1.0
    
    return e_out, cos_scatter, e_electron, cos_electron

# Test sampling
print("Testing Compton sampling for 2 MeV:")
for i in range(5):
    e_out, mu, e_el, mu_el = sample_compton(2.0)
    print(f"  E_out={e_out:.4f} MeV, cosθ={mu:.4f}, T_e={e_el:.4f}, cosθ_e={mu_el:.4f}")