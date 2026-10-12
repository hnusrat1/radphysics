import numpy as np
import json

# Constants
m_ec2 = 0.510998950 # MeV

# Cross section ratios for water at ~1.25 MeV (NIST XCOM data approximations)
# Compton: 0.0607, Rayleigh: 0.0037, PE: 0.0007
sigma_c = 0.0607
sigma_r = 0.0037
sigma_pe = 0.0007
sigma_total = sigma_c + sigma_r + sigma_pe

p_c = sigma_c / sigma_total
p_r = sigma_r / sigma_total
p_pe = sigma_pe / sigma_total

def sample_compton_kinematics(e_inc):
    alpha = e_inc / m_ec2
    while True:
        cos_t = 2 * np.random.rand() - 1
        ep = e_inc / (1 + alpha * (1 - cos_t))
        kn = (ep/e_inc)**2 * (ep/e_inc + e_inc/ep - (1 - cos_t)**2)
        ep_min = e_inc / (1 + 2*alpha)
        kn_max = (ep_min/e_inc)**2 * (ep_min/e_inc + e_inc/ep_min) 
        if np.random.rand() <= kn / kn_max:
            cot_te = (1 + alpha/2) * np.sqrt((1-cos_t)/(1+cos_t))
            cos_te = 1.0 / np.sqrt(1.0 + cot_te**2)
            return ep, e_inc - ep, float(cos_t), float(cos_te)

events = []
target = 200
energies = [1.1732, 1.3325]

D_source = 0.0
D_surf = 100.0
D_phantom_end = 130.0
D_score_center = 105.0
D_score_half = 0.1
half_field = 5.0
voxel_r = 1.0

# Pre-calculate inverse cross section for sampling mean free path
inv_sigma = 1.0 / sigma_total 

while len(events) < target:
    x = np.random.uniform(-half_field, half_field)
    y = np.random.uniform(-half_field, half_field)
    z = D_source
    
    tx = np.random.uniform(-half_field, half_field)
    ty = np.random.uniform(-half_field, half_field)
    td = D_surf
    
    dx = tx - x
    dy = ty - y
    dd = td - z
    L = np.hypot(np.hypot(dx, dy), dd)
    
    ux, uy, ud = dx/L, dy/L, dd/L
    
    e = np.random.choice(energies)
    current_depth = z
    
    for _ in range(20): # Max scatters
        s = np.random.exponential(inv_sigma)
        delta_D = s * ud
        
        new_D = current_depth + delta_D
        if new_D > D_phantom_end: break
            
        delta_x = delta_D * (ux/ud)
        delta_y = delta_D * (uy/ud)
        x_new = x + delta_x
        y_new = y + delta_y
        
        in_vox = (new_D >= D_score_center - D_score_half) and \
                 (new_D <= D_score_center + D_score_half) and \
                 (x_new**2 + y_new**2 < voxel_r**2)
                 
        if in_vox:
            proc_rand = np.random.rand()
            if proc_rand < p_c:
                ep, ek, cos_s, cos_e = sample_compton_kinematics(e)
                events.append({
                    "process": "compton",
                    "e_in_MeV": round(float(e), 6),
                    "e_out_MeV": round(float(ep), 6),
                    "cos_scatter": round(cos_s, 6),
                    "t_e_MeV": round(float(ek), 6),
                    "cos_electron": round(float(cos_e), 6),
                    "t_positron_MeV": None
                })
                e = ep
                phi_s = np.random.rand() * 2 * np.pi
                ud_new = ud * cos_s + np.sqrt(1-ud**2) * np.sqrt(1-cos_s**2) * np.cos(phi_s)
                sin_ud_new = np.sqrt(max(0, 1-ud_new**2))
                if sin_ud_new > 1e-6:
                    ux = sin_ud_new * np.cos(phi_s)
                    uy = sin_ud_new * np.sin(phi_s)
                    ud = ud_new
                else:
                    ud = 1.0; ux=0; uy=0
                
            elif proc_rand < p_c + p_r:
                cos_s = 2*np.random.rand() - 0.5 
                if cos_s > 1.0: cos_s = 1.0
                events.append({
                    "process": "rayleigh",
                    "e_in_MeV": round(float(e), 6),
                    "e_out_MeV": round(float(e), 6),
                    "cos_scatter": round(float(cos_s), 6),
                    "t_e_MeV": 0.0,
                    "cos_electron": None,
                    "t_positron_MeV": None
                })
                phi_s = np.random.rand() * 2 * np.pi
                ud_new = ud * cos_s + np.sqrt(1-ud**2) * np.sqrt(1-cos_s**2) * np.cos(phi_s)
                sin_ud_new = np.sqrt(max(0, 1-ud_new**2))
                if sin_ud_new > 1e-6:
                    ux = sin_ud_new * np.cos(phi_s)
                    uy = sin_ud_new * np.sin(phi_s)
                    ud = ud_new
                else:
                    ud = 1.0; ux=0; uy=0
                
            else:
                events.append({
                    "process": "photoelectric",
                    "e_in_MeV": round(float(e), 6),
                    "e_out_MeV": None,
                    "cos_scatter": None,
                    "t_e_MeV": round(float(e), 6),
                    "cos_electron": round(float(np.random.uniform(0.85, 1.0)), 6),
                    "t_positron_MeV": None
                })
                break
            current_depth = new_D
            x, y = x_new, y_new
        else:
            current_depth = new_D
            x, y = x_new, y_new

output = {
    "method": "Analytical Monte Carlo photon transport simulation. Mass attenuation coefficients for liquid water at 1.17/1.33 MeV used to determine interaction probabilities (Compton dominant). Klein-Nishina distribution applied for Compton scattering kinematics. Primary and scattered photon contributions sampled based on exponential path lengths. Events recorded when interaction points fell within the specified scoring cylinder (r=1 cm, depth 5.0 +/- 0.1 cm).",
    "events": events
}

print(json.dumps(output, indent=2))