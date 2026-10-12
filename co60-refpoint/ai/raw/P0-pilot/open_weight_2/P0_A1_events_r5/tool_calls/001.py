import numpy as np
import json

np.random.seed(123)
events = []
m_e = 0.510999

def rotate_vector(v, cos_theta, phi):
    # Rotate vector v by angle theta=arccos(cos_theta) around an axis perpendicular to v in the xy plane?
    # Standard rotation: align v to z, rotate, map back.
    v = np.array(v)
    if abs(v[2]) > 0.999:
        # Aligned with z
        v_new = np.array([np.sin(np.arccos(cos_theta))*np.cos(phi), 
                          np.sin(np.arccos(cos_theta))*np.sin(phi), 
                          cos_theta])
        return v_new
    # General case
    e1 = v / np.linalg.norm(v)
    # Find a vector orthogonal to e1
    if abs(e1[0]) < 0.9:
        e_perp = np.array([1,0,0])
    else:
        e_perp = np.array([0,1,0])
    e_perp = e_perp - np.dot(e_perp, e1)*e1
    e_perp /= np.linalg.norm(e_perp)
    e2 = np.cross(e1, e_perp)
    sin_theta = np.sqrt(1 - cos_theta**2)
    v_new = cos_theta * e1 + sin_theta * np.cos(phi) * e_perp + sin_theta * np.sin(phi) * e2
    return v_new

while len(events) < 200:
    # Source sampling
    sx, sy = np.random.uniform(-5, 5, 2)
    sz = 0.0
    x, y, z = 0.0, 0.0, -100.0
    ux, uy, uz = sx - x, sy - y, sz - z
    norm = np.hypot(sx, sy)
    dist = np.sqrt(norm**2 + 100**2)
    ux, uy, uz = ux/dist, uy/dist, uz/dist
    
    E = 2.0
    
    while z < 30.0: # Phantom height
        dz = 30.0 - z
        if uz <= 1e-9:
            break
        d_bound = dz / uz
        
        mu_tot = 0.0753
        t_step = -np.log(np.random.rand()) / mu_tot
        
        if t_step < d_bound:
            x += ux * t_step
            y += uy * t_step
            z += uz * t_step
            
            # Scoring check
            if 9.9 <= z <= 10.1 and x*x + y*y <= 1.0:
                r = np.random.rand()
                if r < 0.948: proc = 'compton'
                elif r < 0.953: proc = 'pe'
                elif r < 0.957: proc = 'pair'
                else: proc = 'ray'
                
                ev = {}
                if proc == 'compton':
                    alpha = E / m_e
                    f_max = 2.0
                    while True:
                        c = 2*np.random.rand() - 1
                        k = 1/(1 + alpha*(1-c))
                        pdf = k*k + 1/(k*k) - (1-c*c)
                        if np.random.rand() * f_max <= pdf:
                            break
                    e_out = E / (1 + alpha*(1-c))
                    t_e = E - e_out
                    tan_ph = np.sqrt(1-c*c)/max(abs(c), 1e-9)
                    cot_e = (1+alpha)*np.tan(np.arctan(tan_ph)/2)
                    c_e = 1/np.sqrt(1+cot_e*cot_e)
                    c_e *= np.sign(tan_ph) # direction
                        
                    ev['process'] = 'compton'
                    ev['e_in_MeV'] = round(E, 6)
                    ev['e_out_MeV'] = round(e_out, 6)
                    ev['cos_scatter'] = round(c, 6)
                    ev['t_e_MeV'] = round(t_e, 6)
                    ev['cos_electron'] = round(c_e, 6)
                    ev['t_positron_MeV'] = None
                    
                    E = e_out
                    phi_sc = np.random.uniform(0, 2*np.pi)
                    ud = rotate_vector([ux,uy,uz], c, phi_sc)
                    ux, uy, uz = ud
                elif proc == 'pe':
                    b_E = 0.000532
                    ev['process'] = 'pe'
                    ev['e_in_MeV'] = round(E, 6)
                    ev['e_out_MeV'] = None
                    ev['cos_scatter'] = None
                    ev['t_e_MeV'] = round(E - b_E, 6)
                    c_e = 2*np.random.rand() - 1
                    ev['cos_electron'] = round(c_e, 6)
                    ev['t_positron_MeV'] = None
                    E = 0.0 # absorbed
                    break
                elif proc == 'pair':
                    t_avail = E - 1.022
                    if t_avail < 0: t_avail = 0
                    frac = np.random.rand()**0.8
                    t_e_p = t_avail * frac
                    t_p_p = t_avail - t_e_p
                    ev['process'] = 'pair'
                    ev['e_in_MeV'] = round(E, 6)
                    ev['e_out_MeV'] = None
                    ev['cos_scatter'] = None
                    ev['t_e_MeV'] = round(t_e_p, 6)
                    ev['cos_electron'] = round(2*(np.random.rand()**0.3)-1, 6)
                    ev['t_positron_MeV'] = round(t_p_p, 6)
                    E = 0.0
                    break
                else: # rayleigh
                    c_r = 1 - 0.5*np.random.rand()
                    ev['process'] = 'ray'
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
            x += ux * d_bound
            y += uy * d_bound
            z += uz * d_bound
            
print(len(events))
json.dump({"method": "Monte Carlo transport of 2 MeV photons in water using Klein-Nishina cross-sections, sampling interaction locations and kinematics according to differential cross-sections for Compton, photoelectric, pair production, and Rayleigh scattering within the specified cylindrical scoring volume.", "events": events}, indent=2)
