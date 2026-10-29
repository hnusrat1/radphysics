import numpy as np
from scipy import interpolate

# ---- Physical constants ----
m_e = 0.510998950  # MeV
r_e = 2.8179403262e-13  # cm
N_A = 6.02214076e23

# Water composition
N_e_per_g = N_A * 10 / 18.01528

# ---- Cross sections ----
def klein_nishina_sigma(E_MeV):
    alpha = E_MeV / m_e
    if alpha <= 0:
        return 0.0
    term1 = (1+alpha)/alpha**2 * (2*(1+alpha)/(1+2*alpha) - np.log(1+2*alpha)/alpha)
    term2 = np.log(1+2*alpha) / (2*alpha)
    term3 = (1+3*alpha) / (1+2*alpha)**2
    sigma = 2*np.pi * r_e**2 * (term1 + term2 - term3)
    return sigma

def mu_compton(E):
    return klein_nishina_sigma(E) * N_e_per_g

# Photoelectric interpolation
pe_energies = np.array([0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0])
pe_murho = np.array([4.89, 1.23, 0.359, 0.0559, 0.0150, 0.0050, 0.0019, 0.00038, 0.000163, 0.000044, 0.000020, 0.000008, 0.000004, 0.0000025, 0.0000018, 0.0000010, 0.0000007, 0.0000003, 0.0000002, 0.0000001, 0.00000003, 0.00000001])
f_pe_log = interpolate.interp1d(np.log(pe_energies[pe_energies>0]), np.log(np.maximum(pe_murho[pe_energies>0], 1e-20)), kind='linear', bounds_error=False, fill_value='extrapolate')
def mu_pe(E):
    if E <= 0:
        return 0.0
    if E < 0.01:
        return pe_murho[0] * (E/0.01)**(-3.5)
    if E > 10.0:
        return 1e-10
    return max(0.0, float(np.exp(f_pe_log(np.log(E)))))

# Pair production
pair_energies = np.array([1.022, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0])
pair_murho = np.array([0.0, 0.00004, 0.00013, 0.00046, 0.0015, 0.0048, 0.0077])
f_pair = interpolate.interp1d(pair_energies, pair_murho, kind='linear', bounds_error=False, fill_value=0.0)
def mu_pair(E):
    if E <= 2*m_e:
        return 0.0
    if E > 20.0:
        return 0.0077 + 0.00002 * (E-20)
    return float(f_pair(E))

def mu_rayleigh(E):
    if E < 0.01:
        return 0.16
    elif E < 0.1:
        return 0.16 * (E/0.01)**(-3.5)
    elif E < 0.5:
        return 0.0002 * (E/0.1)**(-4)
    else:
        return 0.0

def mu_total(E):
    return mu_compton(E) + mu_pe(E) + mu_pair(E) + mu_rayleigh(E)

# ---- Compton scattering ----
def sample_compton(energy, rng=np.random):
    E = energy
    alpha = E / m_e
    
    while True:
        xi1 = rng.uniform()
        xi2 = rng.uniform()
        mu = 2.0 * xi1 - 1.0  # cosθ in [-1,1]
        ratio = 1.0 / (1.0 + alpha * (1.0 - mu))
        f_kn = ratio**2 * (ratio + 1.0/ratio - (1.0 - mu**2))
        # Maximum f_kn is 2 at mu=1 (forward scattering)
        if xi2 <= f_kn / 2.0:
            e_out = E * ratio
            cos_scatter = mu
            break
    
    e_electron = E - e_out
    
    # Electron direction
    denom = np.sqrt(E**2 + e_out**2 - 2.0 * E * e_out * cos_scatter)
    if denom > 0:
        cos_electron = (E - e_out * cos_scatter) / denom
    else:
        cos_electron = 1.0
    
    return e_out, cos_scatter, e_electron, cos_electron

# ---- Rayleigh scattering ----
def sample_rayleigh(energy, rng=np.random):
    """
    Rayleigh scattering: no energy loss, just direction change.
    Differential cross section dσ/dΩ ∝ (1+cos^2θ) * F(q,Z)^2
    For water at low energies, we approximate with a simple form factor.
    Here we just sample cosθ from a distribution that approximates Rayleigh.
    At low energies, Rayleigh is strongly forward-peaked.
    We'll use a simple model: sample cosθ from distribution ∝ (1+cos^2θ) * exp(-a*(1-cosθ))
    where a depends on energy.
    """
    # For simplicity, use a forward-peaked distribution
    # Sample cosθ close to 1
    # Use sigma ~ 0.02 / (E * Z^(1/3)) in radians, roughly
    # For water, Z=10, at 0.1 MeV, sigma_θ ~ 0.02/(0.1*10^(1/3)) ~ 0.02/0.215 = 0.093 rad ~ 5 deg
    # At lower energies, wider.
    E = energy
    # Simple approximation: theta in Gaussian with sigma ~ 0.02 / (E * Z_eff^(1/3))
    Z_eff = 10.0  # sum Z for H2O
    theta_sigma = 0.02 / (max(E, 0.01) * Z_eff**(1/3))
    # Sample theta from Rayleigh distribution? Use simple rejection
    while True:
        mu = 1.0 - 0.001 * rng.exponential()  # not great
        # Actually let's use a simple model: cosθ = 1 - 2*u^2 where u is uniform in [0, a]
        # Better: use the form factor approximation
        # For simplicity, just sample mu from uniform in [cos(θ_max), 1] where θ_max is small
        # But this is approximate. Let's use a more physical approach.
        
        # Physical Rayleigh cross section (Thomson) times form factor squared
        # Form factor F(q,Z) ~ Z / (1 + (q*r_a)^2) where r_a ~ a0/Z^(1/3)
        # For water, the molecular form factor is complex.
        # Let's just use a simple rejection sampling with (1+cos^2θ) and a cutoff.
        
        # Sample cosθ uniformly in [0,1] (forward hemisphere only)
        mu = rng.uniform(0.0, 1.0)
        # The differential cross section is roughly (1+mu^2) times form factor squared
        # Form factor squared is ~ exp(-b*(1-mu)) approximately
        # For simplicity, accept with probability proportional to (1+mu^2)
        if rng.uniform() <= (1.0 + mu**2) / 2.0:
            return mu, E  # cos_scatter, e_out
    
    # If we get here, just return forward
    return 1.0, E

# ---- Photoelectric ----
# Photoelectric: photon absorbed, electron emitted
# Angular distribution: approx dσ/dΩ ∝ sin^2θ / (1-βcosθ)^4 (non-relativistic)
# but we can approximate as isotropic or forward peaked
def sample_photoelectric(energy, rng=np.random):
    """Return (cos_electron) where electron gets kinetic energy = energy - binding energy"""
    E = energy
    # Binding energy for water ~ 0.5 keV for K-shell (oxygen), negligible
    t_e = E  # approximately, ignoring binding energy
    # Angular distribution: at 2 MeV, forward peaked but we approximate as isotropic
    cos_electron = rng.uniform(-1.0, 1.0)
    return t_e, cos_electron

# ---- Pair production ----
# Photon -> e- + e+
# Energy sharing: the differential cross section for pair production is approximately
# dσ/dT_e ≈ constant (if we ignore screening and other effects)
# So sample T_e (kinetic energy of electron) uniformly in [0, E - 2m_e]
# Then T_pos = (E - 2m_e) - T_e
# Angular distribution: both particles are forward peaked with angle ~ m_e/E
def sample_pair(energy, rng=np.random):
    E = energy
    if E <= 2*m_e:
        return 0.0, 1.0, 0.0  # shouldn't happen
    T_total = E - 2*m_e
    # Sample electron kinetic energy uniformly
    t_e = rng.uniform(0.0, T_total)
    t_pos = T_total - t_e
    # Angular distribution: approximate as forward peaked with cosθ ~ 1 - (m_e/E)^2
    # Actually the average angle is ~ m_e/E
    # For simplicity, sample cosθ close to 1
    # Use a small deviation
    theta_avg = m_e / E  # radians
    # Sample from distribution: p(θ) ∝ θ / (θ^2 + (m_e/E)^2)^2
    # For simplicity, use Gaussian with sigma = m_e/E
    theta = rng.rayleigh(scale=theta_avg)
    cos_electron = np.cos(theta)
    # Positron angle is approximately opposite? Actually both go forward.
    # For pair, we only need t_pos (kinetic energy of positron), not its angle.
    return t_e, cos_electron, t_pos

# Test sampling
print("Testing samplers:")
e_out, mu_s, t_e, mu_e = sample_compton(2.0)
print(f"Compton: E_out={e_out:.4f}, cosθ={mu_s:.4f}, T_e={t_e:.4f}, cosθ_e={mu_e:.4f}")

t_e, mu_e = sample_photoelectric(2.0)
print(f"PE: T_e={t_e:.4f}, cosθ_e={mu_e:.4f}")

t_e, mu_e, t_pos = sample_pair(5.0)
print(f"Pair: T_e={t_e:.4f}, cosθ_e={mu_e:.4f}, T_pos={t_pos:.4f}")

mu_s, e_out = sample_rayleigh(0.1)
print(f"Rayleigh: cosθ={mu_s:.4f}, E_out={e_out}")