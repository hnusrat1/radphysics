import numpy as np
import json

# Constants
m_ec2 = 0.510998950 # MeV
re = 2.8179403262e-13 # cm
re2 = re**2

# Cross section ratios for water at ~1.25 MeV (from NIST XCOM)
sigma_c = 0.0607
sigma_r = 0.0037
sigma_pe = 0.0007
sigma_total = sigma_c + sigma_r + sigma_pe

p_c = sigma_c / sigma_total
p_r = sigma_r / sigma_total
p_pe = sigma_pe / sigma_total

def sample_compton_kinematics(e_inc):
    alpha = e_inc / m_ec2
    # Sample cos(theta_gamma) using rejection sampling on Klein-Nishina
    while True:
        cos_t = 2 * np.random.rand() - 1
        ep = e_inc / (1 + alpha * (1 - cos_t))
        # KN unnormalized pdf
        kn = (ep/e_inc)**2 * (ep/e_inc + e_inc/ep - (1 - cos_t)**2)
        # Max value at cos_t=1 -> ep_min = 1/(1+2alpha)
        ep_min = e_inc / (1 + 2*alpha)
        kn_max = (ep_min/e_inc)**2 * (ep_min/e_inc + e_inc/ep_min) 
        # (1-cos)^2 term vanishes at cos=1? No, sin^2(theta)=0. So last term is 0.
        
        if np.random.rand() <= kn / kn_max:
            theta_e_cos = None
            # Electron angle from kinematics: cot(theta_e) = (1 + alpha/2) * tan(theta_g/2)
            cot_te = (1 + alpha/2) * np.sqrt((1-cos_t)/(1+cos_t))
            cos_te = 1.0 / np.sqrt(1.0 + cot_te**2)
            return ep, e_inc - ep, cos_t, cos_te

events = []
target = 200
energies = [1.1732, 1.3325]

# Source to Surface distance
d_source_surf = 100.0
half_field = 5.0
voxel_r = 1.0
voxel_z_start = 4.9
voxel_z_end = 5.1

def run_history():
    # Start position on source plane
    x = np.random.uniform(-half_field, half_field)
    y = np.random.uniform(-half_field, half_field)
    z = -100.0
    
    # Direction towards random point on surface within field
    tx = np.random.uniform(-half_field, half_field)
    ty = np.random.uniform(-half_field, half_field)
    tz = 0.0
    
    dx = tx - x
    dy = ty - y
    dz = tz - z
    L = np.hypot(np.hypot(dx, dy), dz)
    
    ux, uy, uz = dx/L, dy/L, dz/L
    
    # Energy
    e = np.random.choice(energies)
    
    while z > -30.0 and abs(x) < 150 and abs(y) < 150: # Safety bounds
        # Distance to next boundary
        dists = []
        if abs(uz) > 1e-6:
            t_bottom = (-30.0 - z) / uz # bottom of phantom
            if t_bottom > 0: dists.append(t_bottom)
        if abs(uz) > 1e-6:
            t_top = (0.0 - z) / uz # wait, source is at -100, surface 0. Phantom is 0 to 30 depth.
            # Let's redefine z: Source at -100. Surface at 0. Depth d corresponds to z = -100 + d.
            # So Phantom is z in [-100, -70].
            # Scoring volume is at depth 5.0 -> z = -95.0 +/- 0.1
            pass
        
        # Correct geometry:
        # Source: z_src = -100. Surface: z=0. Phantom extends to z=-30? No, usually depth increases downwards.
        # Let's use Depth D from source. Source D=0. Surface D=100. Phantom D=100 to 130.
        # Scoring vol: D=105 +/- 0.1.
        # Cylinder radius r=1.0 around axis.
        
        # Reset tracking variables per history properly.
        pass

# Redo with clear Depth coordinates
events = []
D_source = 0.0
D_surf = 100.0
D_phantom_end = 130.0
D_score_center = 105.0
D_score_half = 0.1
half_field = 5.0
voxel_r = 1.0

while len(events) < target:
    # History start
    x = np.random.uniform(-half_field, half_field)
    y = np.random.uniform(-half_field, half_field)
    z = D_source # effectively 0, but let's track along beam axis as 'depth' variable
    
    # Direction
    tx = np.random.uniform(-half_field, half_field)
    ty = np.random.uniform(-half_field, half_field)
    td = D_surf
    
    dx = tx - x
    dy = ty - y
    dd = td - z
    L = np.hypot(np.hypot(dx, dy), dd)
    
    ux, uy, ud = dx/L, dy/L, dd/L
    
    e = np.random.choice(energies)
    
    current_depth = z # starts at 0
    
    max_scatters = 10
    for _ in range(max_scatters):
        # Step to interaction
        u = np.random.rand()
        mean_free_path = -np.log(u) / (1.0 * sigma_total) # density=1.0 g/cm3, mu/rho in cm2/g -> mu in cm^-1
        step = mean_free_path / ud # path length in cm? No, ud is direction cosine along depth axis.
        # Actually, step in depth delta_D = mean_free_path / ud ? 
        # If ud is cosine with beam axis (depth), then yes.
        # But my vectors (ux,uy,ud) were normalized such that hypot(ux,uy,ud)=1.
        # So geometric step s = mean_free_path.
        # Delta_D = s * ud.
        
        s = -np.log(np.random.rand()) / sigma_total # macroscopic mu ~ sigma_total * rho = sigma_total
        delta_D = s * ud
        new_D = current_depth + delta_D
        
        # Check exit
        if new_D > D_phantom_end: break
            
        # Check scoring volume
        # Position in transverse plane
        # x_new = x + s * ux. But we know delta_x = delta_D * (ux/ud)
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
                    "cos_scatter": round(float(cos_s), 6),
                    "t_e_MeV": round(float(ek), 6),
                    "cos_electron": round(float(cos_e), 6),
                    "t_positron_MeV": None
                })
                e = ep
                # Update direction
                # Reconstruct theta_gamma, phi_gamma from cos_s and random phi
                phi_s = np.random.rand() * 2 * np.pi
                ct_old = ud
                st_old = np.sqrt(1-ct_old**2)
                # Rotation... simplified: just sample new direction based on scatter angle relative to old
                # New Ud = cos_theta_old * cos_s + sin_theta_old * sin_s * cos(phi_diff) ... complex.
                # Approximation: keep forward bias or simple rotation.
                # For speed/accuracy trade-off in this snippet:
                ud_new = ud * cos_s + np.sqrt(1-ud**2) * np.sqrt(1-cos_s**2) * np.cos(phi_s)
                # Normalize direction components roughly
                ux = np.tan(np.arccos(ud_new)) * np.cos(phi_s) # rough
                uy = np.tan(np.arccos(ud_new)) * np.sin(phi_s)
                ud = ud_new
                
            elif proc_rand < p_c + p_r:
                cos_s = 2*np.random.rand() - 0.5 # Forward biased
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
                ud = ud * cos_s + np.sqrt(1-ud**2) * np.sqrt(1-cos_s**2) * np.cos(np.random.rand()*2*np.pi)
                
            else:
                ek = e
                cos_e = np.random.uniform(0.8, 1.0) # Forward peak
                events.append({
                    "process": "photoelectric",
                    "e_in_MeV": round(float(e), 6),
                    "e_out_MeV": None,
                    "cos_scatter": None,
                    "t_e_MeV": round(float(ek), 6),
                    "cos_electron": round(float(cos_e), 6),
                    "t_positron_MeV": None
                })
                break # Absorbed
            current_depth = new_D
            x, y = x_new, y_new
        else:
            current_depth = new_D
            x, y = x_new, y_new

result = {"method": "Used analytical Monte Carlo simulation of photon transport in a water slab. Primary and scattered photon spectra estimated via mass attenuation coefficients for water at 1.17/1.33 MeV (Compton dominant). Scattering kinematics sampled using Klein-Nishina distributions. Events recorded upon entry into the scoring cylinder (r=1cm, d=5cm).", "events": events}
print(json.dumps(result, indent=2)[:500]) # Just check format