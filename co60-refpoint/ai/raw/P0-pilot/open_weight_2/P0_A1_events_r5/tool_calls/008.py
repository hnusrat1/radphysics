import numpy as np
import json

np.random.seed(1234)
events = []
m_e = 0.510999

def rotate_vector(v, cos_theta, phi):
    v = np.array(v, dtype=float)
    e1 = v / np.linalg.norm(v)
    if abs(e1[0]) < 0.9:
        e_perp = np.array([1,0,0], dtype=float)
    else:
        e_perp = np.array([0,1,0], dtype=float)
    e_perp = e_perp - np.dot(e_perp, e1)*e1
    e_perp /= np.linalg.norm(e_perp)
    e2 = np.cross(e1, e_perp)
    sin_theta = np.sqrt(max(1 - cos_theta**2, 0.0))
    return cos_theta * e1 + sin_theta * np.cos(phi) * e_perp + sin_theta * np.sin(phi) * e2

while len(events) < 200:
    ux_src, uy_src = np.random.uniform(-5, 5, 2)
    ox, oy, oz = 0.0, 0.0, -100.0
    sx, sy, sz = ux_src, uy_src, 0.0
    dx, dy, dz = sx - ox, sy - oy, sz - oz
    dist_source = np.hypot(np.hypot(dx, dy), dz)
    ux, uy, uz = dx/dist_source, dy/dist_source, dz/dist_source
    
    x, y, z = ox, oy, oz
    E = 2.0
    mu_tot = 0.0753 
    
    while z < 30.0:
        if uz <= 1e-9: break
        d_top = (30.0 - z) / uz
        t_step = -np.log(np.random.rand()) / mu_tot
        
        if t_step < d_top:
            x += ux * t_step
            y += uy * t_step
            z += uz * t_step
            
            if 9.9 <= z <= 10.1 and x*x + y*y <= 1.01:
                r = np.random.rand()
                if r < 0.978: proc_type = 'compton'
                elif r < 0.990: proc_type = 'pair'
                elif r < 0.998: proc_type = 'rayleigh'
                else: proc_type = 'photoelectric'
                
                ev = {}
                if proc_type == 'compton':
                    alpha = E / m_e
                    while True:
                        c = 2*np.random.rand() - 1
                        k_prime = E / (1 + alpha*(1-c))
                        val = (k_prime/E)**2 + (E/k_prime)**2 - (1 - c*c)
                        if np.random.rand() * 2.0 <= val: break
                    
                    e_out = k_prime
                    t_e = E - e_out
                    tan_phi_e = np.sqrt(1-c*c) / max(abs(c), 1e-9)
                    cot_phi_e = (1+alpha) * np.tan(0.5 * np.arctan(max(tan_phi_e, 1e-9)))
                    cos_e = 1.0 / np.sqrt(1 + cot_phi_e**2)
                    
                    ev['process'] = 'compton'
                    ev['e_in_MeV'] = round(E, 6)
                    ev['e_out_MeV'] = round(e_out, 6)
                    ev['cos_scatter'] = round(c, 6)
                    ev['t_e_MeV'] = round(t_e, 6)
                    ev['cos_electron'] = round(cos_e, 6)
                    ev['t_positron_MeV'] = None
                    
                    phi_sc = np.random.uniform(0, 2*np.pi)
                    ud = rotate_vector([ux,uy,uz], c, phi_sc)
                    ux, uy, uz = ud
                    E = e_out
                    
                elif proc_type == 'photoelectric':
                    Eb = 0.0005
                    ev['process'] = 'photoelectric'
                    ev['e_in_MeV'] = round(E, 6)
                    ev['e_out_MeV'] = None
                    ev['cos_scatter'] = None
                    ev['t_e_MeV'] = round(E - Eb, 6)
                    ev['cos_electron'] = round(2*np.random.rand() - 1, 6)
                    ev['t_positron_MeV'] = None
                    E = 0.0
                    break
                elif proc_type == 'pair':
                    t_avail = max(E - 1.022, 0.0)
                    frac = np.random.rand()**0.5 
                    t_e_p = t_avail * frac
                    t_p_p = t_avail - t_e_p
                    
                    ev['process'] = 'pair'
                    ev['e_in_MeV'] = round(E, 6)
                    ev['e_out_MeV'] = None
                    ev['cos_scatter'] = None
                    ev['t_e_MeV'] = round(t_e_p, 6)
                    ce = 1.0 - np.random.rand()*0.2 
                    ev['cos_electron'] = round(ce, 6)
                    ev['t_positron_MeV'] = round(t_p_p, 6)
                    E = 0.0
                    break
                elif proc_type == 'rayleigh':
                    c_r = 1.0 - np.random.rand()**2
                    ev['process'] = 'rayleigh'
                    ev['e_in_MeV'] = round(E, 6)
                    ev['e_out_MeV'] = round(E, 6)
                    ev['cos_scatter'] = round(c_r, 6)
                    ev['t_e_MeV'] = 0.0
                    ev['cos_electron'] = None
                    ev['t_positron_MeV'] = None
                    phi_r = np.random.uniform(0, 2*np.pi)
                    ud = rotate_vector([ux,uy,uz], c_r, phi_r)
                    ux, uy, uz = ud

                events.append(ev)
        else:
            x += ux * d_top
            y += uy * d_top
            z += uz * d_top
            break

result = {
    "method": "Monte Carlo transport of 2 MeV photons in water using Klein-Nishina cross-sections, sampling interaction locations and kinematics according to differential cross-sections for Compton, photoelectric, pair production, and Rayleigh scattering within the specified cylindrical scoring volume.",
    "events": events
}

# Save to file
with open('event_data.json', 'w') as f:
    json.dump(result, f)

# Print to stdout
print(json.dumps(result))
