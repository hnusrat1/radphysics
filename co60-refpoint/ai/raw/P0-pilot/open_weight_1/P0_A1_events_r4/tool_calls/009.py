import numpy as np

# Let's trace a few photons manually
from scipy import interpolate

# Replicate necessary functions
m_e = 0.510998950
r_e = 2.8179403262e-13
N_A = 6.02214076e23
N_e_per_g = N_A * 10 / 18.01528

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

pe_energies = np.array([0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0])
pe_murho = np.array([4.89, 1.23, 0.359, 0.0559, 0.0150, 0.0050, 0.0019, 0.00038, 0.000163, 0.000044, 0.000020, 0.000008, 0.000004, 0.0000025, 0.0000018, 0.0000010, 0.0000007, 0.0000003, 0.0000002, 0.0000001, 0.00000003, 0.00000001])
f_pe_log = interpolate.interp1d(np.log(pe_energies[1:]), np.log(np.maximum(pe_murho[1:], 1e-20)), kind='linear', bounds_error=False, fill_value='extrapolate')
def mu_pe(E):
    if E <= 0:
        return 0.0
    if E < 0.01:
        return 4.89 * (E/0.01)**(-3.5)
    if E > 10.0:
        return 1e-10
    return max(0.0, float(np.exp(f_pe_log(np.log(E)))))

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

# Precompute
E_grid = np.logspace(-2, 1, 500)
mu_C_grid = np.array([mu_compton(E) for E in E_grid])
mu_PE_grid = np.array([mu_pe(E) for E in E_grid])
mu_PP_grid = np.array([mu_pair(E) for E in E_grid])
mu_R_grid = np.array([mu_rayleigh(E) for E in E_grid])
mu_T_grid = mu_C_grid + mu_PE_grid + mu_PP_grid + mu_R_grid

f_muT = interpolate.interp1d(E_grid, mu_T_grid, kind='linear', bounds_error=False, fill_value=0.0)
f_muC = interpolate.interp1d(E_grid, mu_C_grid, kind='linear', bounds_error=False, fill_value=0.0)

def get_mu(E):
    mu_t = float(f_muT(E))
    mu_c = float(f_muC(E))
    return mu_t, mu_c

# Test one photon
rng = np.random.RandomState(42)

# Sample direction
u = rng.uniform(-5, 5)
v = rng.uniform(-5, 5)
dir_vec = np.array([u, v, 100.0])
dir_vec = dir_vec / np.linalg.norm(dir_vec)

pos = np.array([0.0, 0.0, 0.0])
energy = 2.0
dir = dir_vec

# Move to surface
t_surf = (100.0 - pos[2]) / dir[2]
pos = pos + t_surf * dir
print(f"At surface: pos={pos}, dir={dir}, energy={energy}")

# Now track
for step in range(10):
    mu_t, mu_c = get_mu(energy)
    print(f"  Step {step}: pos={pos}, energy={energy:.4f}, mu_t={mu_t:.6f}")
    if mu_t <= 0:
        break
    mfp = 1.0 / mu_t
    print(f"    mfp={mfp:.2f} cm")
    s = -mfp * np.log(1.0 - rng.uniform())
    print(f"    sampled s={s:.2f} cm")
    new_pos = pos + s * dir
    print(f"    new_pos={new_pos}")
    if new_pos[2] >= 130.0:
        print("    beyond phantom exit, break")
        break
    if new_pos[2] < 100.0:
        print("    before surface, break")
        break
    
    # Check scoring volume
    z = new_pos[2]
    r = np.sqrt(new_pos[0]**2 + new_pos[1]**2)
    in_scoring = (z >= 109.9 and z <= 110.1 and r <= 1.0)
    print(f"    in_scoring={in_scoring}, z={z:.2f}, r={r:.4f}")
    
    # Just sample a dummy interaction
    xi = rng.uniform()
    if xi < 0.99:  # mostly Compton for simplicity
        # Simplistic Compton
        alpha = energy / m_e
        mu_sc = 2.0 * rng.uniform() - 1.0
        ratio = 1.0 / (1.0 + alpha * (1.0 - mu_sc))
        e_out = energy * ratio
        energy = e_out
        # New direction (simplified)
        azimuth = rng.uniform(0, 2*np.pi)
        # Rotate dir
        z_axis = dir
        if abs(z_axis[2]) < 0.9:
            x_axis = np.cross(np.array([0,0,1]), z_axis)
            x_axis = x_axis / np.linalg.norm(x_axis)
        else:
            x_axis = np.cross(np.array([1,0,0]), z_axis)
            x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        sin_theta = np.sqrt(max(0, 1-mu_sc**2))
        dir = (sin_theta * np.cos(azimuth) * x_axis + 
               sin_theta * np.sin(azimuth) * y_axis +
               mu_sc * z_axis)
        print(f"    Compton: e_out={e_out:.4f}, new_dir={dir}")
    else:
        print("    Absorption")
        break
    
    pos = new_pos
    if step > 5:
        break