import sys, os
sys.path.insert(0, os.getcwd())
import numpy as np
import math
import time
import pickle
from mc_physics import *

elec_collect_radius = 2.5

def make_scoring_full():
    return {
        'ph_prim_fluence': 0.0, 'ph_scat_fluence': 0.0,
        'ph_prim_spec': np.zeros(28), 'ph_scat_spec': np.zeros(28),
        'ph_prim_ang': np.zeros(10), 'ph_scat_ang': np.zeros(10),
        'cyl_int': np.zeros(4), 'cyl_int_prim': np.zeros(4), 'cyl_int_scat': np.zeros(4),
        'cyl_int_energy_transfer': np.zeros(4),
        'electrons_cylinder': [],
        'electron_sources': [],
        'n_annihilation': 0,
    }

def transport_photon_full(p_start, u_start, E_start, primary_start, scoring):
    electrons_in_cyl = []
    p = p_start.copy(); u = u_start.copy(); E = E_start; primary = primary_start
    for _ in range(300):
        if E <= 1e-4: break
        mu = float(np.interp(E, E_lut, mu_lut))
        if mu <= 0: break
        s_interact = -math.log(np.random.rand())/mu
        s_boundary = 1e30
        if u[2] > 0: s_boundary = min(s_boundary, (phantom_depth-p[2])/u[2])
        elif u[2] < 0: s_boundary = min(s_boundary, (-p[2])/u[2])
        if u[0] > 0: s_boundary = min(s_boundary, (phantom_half-p[0])/u[0])
        elif u[0] < 0: s_boundary = min(s_boundary, (-phantom_half-p[0])/u[0])
        if u[1] > 0: s_boundary = min(s_boundary, (phantom_half-p[1])/u[1])
        elif u[1] < 0: s_boundary = min(s_boundary, (-phantom_half-p[1])/u[1])
        s_step = min(s_interact, s_boundary)
        if s_step < 0: s_step = 0
        L = segment_cylinder_track(p, u, s_step)
        if L > 1e-12:
            if primary:
                scoring['ph_prim_fluence'] += L
                idx = np.searchsorted(photon_energy_edges, E, side='right')-1
                if 0 <= idx < 28: scoring['ph_prim_spec'][idx] += L
                idx_a = int((u[2]+1)/0.2); idx_a = min(9,max(0,idx_a))
                scoring['ph_prim_ang'][idx_a] += L
            else:
                scoring['ph_scat_fluence'] += L
                idx = np.searchsorted(photon_energy_edges, E, side='right')-1
                if 0 <= idx < 28: scoring['ph_scat_spec'][idx] += L
                idx_a = int((u[2]+1)/0.2); idx_a = min(9,max(0,idx_a))
                scoring['ph_scat_ang'][idx_a] += L
        if s_step >= s_boundary - 1e-12: break
        p = p + s_step*u
        in_cyl = is_in_cylinder(p[0],p[1],p[2])
        dist_to_cyl = math.sqrt(p[0]**2+p[1]**2+(p[2]-depth_poi)**2)
        near_cyl = dist_to_cyl < elec_collect_radius
        idx_mu = min(max(np.searchsorted(E_lut, E),1), len(E_lut)-1)
        mc = mu_comp_lut[idx_mu]; mpe = mu_pe_lut[idx_mu]
        mpair = mu_pair_lut[idx_mu]; mray = mu_ray_lut[idx_mu]
        mtot = mu_lut[idx_mu]
        r = np.random.rand()*mtot
        if r < mc:
            ct_sc, E_sc = sample_compton_fast(E)
            T_e = E - E_sc
            phi = np.random.rand()*2*np.pi
            u_e, u_out = compton_electron_dir(u, ct_sc, E, E_sc, phi)
            u = u_out
            if in_cyl:
                scoring['cyl_int'][0]+=1
                if primary: scoring['cyl_int_prim'][0]+=1
                else: scoring['cyl_int_scat'][0]+=1
                scoring['cyl_int_energy_transfer'][0]+=T_e
                electrons_in_cyl.append((T_e, u_e[2], primary))
            if near_cyl and T_e > 0.010:
                scoring['electron_sources'].append((p.copy(), u_e.copy(), T_e, primary))
            E = E_sc
            primary = False
        elif r < mc+mpe:
            T_e = E - 0.0005
            if in_cyl:
                scoring['cyl_int'][1]+=1
                if primary: scoring['cyl_int_prim'][1]+=1
                else: scoring['cyl_int_scat'][1]+=1
                scoring['cyl_int_energy_transfer'][1]+=T_e
                cos_e = 2*np.random.rand()-1
                electrons_in_cyl.append((T_e, cos_e, primary))
            if near_cyl and T_e > 0.010:
                cos_e = 2*np.random.rand()-1
                u_e = np.array([math.sqrt(1-cos_e**2)*math.cos(np.random.rand()*2*math.pi),
                                math.sqrt(1-cos_e**2)*math.sin(np.random.rand()*2*math.pi), cos_e])
                scoring['electron_sources'].append((p.copy(), u_e.copy(), T_e, primary))
            break
        elif r < mc+mpe+mpair:
            T_total = E - 2*mec2
            if in_cyl:
                scoring['cyl_int'][2]+=1
                if primary: scoring['cyl_int_prim'][2]+=1
                else: scoring['cyl_int_scat'][2]+=1
                scoring['cyl_int_energy_transfer'][2]+=T_total
                if T_total > 0:
                    f = np.random.rand()
                    electrons_in_cyl.append((f*T_total, u[2]*0.98, primary))
                    electrons_in_cyl.append(((1-f)*T_total, u[2]*0.98, primary))
            if near_cyl and T_total > 0.010:
                f = np.random.rand()
                u_e1 = rotate_dir(u, 0.98, np.random.rand()*2*np.pi)
                u_e2 = rotate_dir(u, 0.98, np.random.rand()*2*np.pi)
                scoring['electron_sources'].append((p.copy(), u_e1.copy(), f*T_total, primary))
                scoring['electron_sources'].append((p.copy(), u_e2.copy(), (1-f)*T_total, primary))
            theta1 = np.arccos(2*np.random.rand()-1)
            phi1 = np.random.rand()*2*np.pi
            u_ann1 = np.array([math.sin(theta1)*math.cos(phi1), math.sin(theta1)*math.sin(phi1), math.cos(theta1)])
            scoring['n_annihilation'] += 2
            transport_annihilation(p, u_ann1, scoring)
            transport_annihilation(p, -u_ann1, scoring)
            break
        else:
            if in_cyl:
                scoring['cyl_int'][3]+=1
                if primary: scoring['cyl_int_prim'][3]+=1
                else: scoring['cyl_int_scat'][3]+=1
            ct_sc = math.cos(np.random.normal(0, 0.002))
            phi = np.random.rand()*2*np.pi
            u = rotate_dir(u, ct_sc, phi)
            primary = False
    return electrons_in_cyl

def transport_annihilation(p_start, u_start, scoring):
    p = p_start.copy(); u = u_start.copy(); E = 0.511
    for _ in range(100):
        if E <= 1e-4: break
        mu = float(np.interp(E, E_lut, mu_lut))
        if mu <= 0: break
        s_interact = -math.log(np.random.rand())/mu
        s_boundary = 1e30
        if u[2] > 0: s_boundary = min(s_boundary, (phantom_depth-p[2])/u[2])
        elif u[2] < 0: s_boundary = min(s_boundary, (-p[2])/u[2])
        if u[0] > 0: s_boundary = min(s_boundary, (phantom_half-p[0])/u[0])
        elif u[0] < 0: s_boundary = min(s_boundary, (-phantom_half-p[0])/u[0])
        if u[1] > 0: s_boundary = min(s_boundary, (phantom_half-p[1])/u[1])
        elif u[1] < 0: s_boundary = min(s_boundary, (-phantom_half-p[1])/u[1])
        s_step = min(s_interact, s_boundary)
        L = segment_cylinder_track(p, u, s_step)
        if L > 1e-12:
            scoring['ph_scat_fluence'] += L
            idx = np.searchsorted(photon_energy_edges, E, side='right')-1
            if 0 <= idx < 28: scoring['ph_scat_spec'][idx] += L
            idx_a = int((u[2]+1)/0.2); idx_a = min(9,max(0,idx_a))
            scoring['ph_scat_ang'][idx_a] += L
        if s_step >= s_boundary - 1e-12: break
        p = p + s_step*u
        in_cyl = is_in_cylinder(p[0],p[1],p[2])
        dist_to_cyl = math.sqrt(p[0]**2+p[1]**2+(p[2]-depth_poi)**2)
        near_cyl = dist_to_cyl < elec_collect_radius
        idx_mu = min(max(np.searchsorted(E_lut,E),1),len(E_lut)-1)
        mc=mu_comp_lut[idx_mu]; mpe=mu_pe_lut[idx_mu]; mpair=mu_pair_lut[idx_mu]; mray=mu_ray_lut[idx_mu]; mtot=mu_lut[idx_mu]
        r = np.random.rand()*mtot
        if r < mc:
            ct_sc, E_sc = sample_compton_fast(E)
            T_e = E - E_sc
            phi = np.random.rand()*2*np.pi
            u_e, u_out = compton_electron_dir(u, ct_sc, E, E_sc, phi)
            u = u_out; E = E_sc
            if in_cyl:
                scoring['cyl_int'][0]+=1; scoring['cyl_int_scat'][0]+=1
                scoring['cyl_int_energy_transfer'][0]+=T_e
            if near_cyl and T_e > 0.010:
                scoring['electron_sources'].append((p.copy(), u_e.copy(), T_e, False))
        elif r < mc+mpe:
            if in_cyl:
                scoring['cyl_int'][1]+=1; scoring['cyl_int_scat'][1]+=1
                T_e = E - 0.0005
                scoring['cyl_int_energy_transfer'][1]+=T_e
            if near_cyl:
                T_e = E - 0.0005
                cos_e = 2*np.random.rand()-1
                u_e = np.array([math.sqrt(1-cos_e**2)*math.cos(np.random.rand()*2*math.pi),
                                math.sqrt(1-cos_e**2)*math.sin(np.random.rand()*2*math.pi), cos_e])
                if T_e > 0.010:
                    scoring['electron_sources'].append((p.copy(), u_e.copy(), T_e, False))
            break
        elif r < mc+mpe+mpair:
            if in_cyl:
                scoring['cyl_int'][2]+=1; scoring['cyl_int_scat'][2]+=1
                T_total = E - 2*mec2
                scoring['cyl_int_energy_transfer'][2]+=T_total
            if near_cyl:
                T_total = E - 2*mec2
                if T_total > 0.010:
                    f = np.random.rand()
                    u_e1 = rotate_dir(u, 0.98, np.random.rand()*2*np.pi)
                    u_e2 = rotate_dir(u, 0.98, np.random.rand()*2*np.pi)
                    scoring['electron_sources'].append((p.copy(), u_e1.copy(), f*T_total, False))
                    scoring['electron_sources'].append((p.copy(), u_e2.copy(), (1-f)*T_total, False))
            break
        else:
            if in_cyl:
                scoring['cyl_int'][3]+=1; scoring['cyl_int_scat'][3]+=1
            ct_sc = math.cos(np.random.normal(0, 0.002))
            phi = np.random.rand()*2*np.pi
            u = rotate_dir(u, ct_sc, phi)
    return

def make_elec_scoring():
    return {'e_fluence': 0.0, 'e_spec': np.zeros(25), 'e_ang': np.zeros(10), 'e_dose': 0.0,
            'delta_fluence': 0.0}

def transport_electron(p0, u0, T0, elec_scoring, is_delta=False, ds=0.025):
    p = p0.copy(); u = u0.copy(); T = T0
    if T < 0.010: return
    range_max = R_csda_w(T)
    max_steps = int(1.5*range_max/ds) + 50
    if max_steps > 8000: max_steps = 8000
    for _ in range(max_steps):
        if T < 0.010: break
        s = min(ds, range_max)
        L = segment_cylinder_track(p, u, s)
        if L > 1e-12:
            elec_scoring['e_fluence'] += L
            idx = np.searchsorted(electron_energy_edges, T, side='right')-1
            if 0 <= idx < 25: elec_scoring['e_spec'][idx] += L
            idx_a = int((u[2]+1)/0.2); idx_a = min(9,max(0,idx_a))
            elec_scoring['e_ang'][idx_a] += L
            elec_scoring['e_dose'] += L*S_col_w(T)
            if is_delta:
                elec_scoring['delta_fluence'] += L
        new_p = p + s*u
        if new_p[2] < 0 or new_p[2] > phantom_depth: break
        if abs(new_p[0]) > phantom_half or abs(new_p[1]) > phantom_half: break
        p = new_p
        dE = (S_col_w(T)+S_rad_w(T))*s
        T_new = T - dE
        if T_new < 0.010: break
        T = T_new
        ct_scat = multiple_scatter_angle(T, s)
        phi = np.random.rand()*2*np.pi
        u = rotate_dir(u, ct_scat, phi)
    return

# ====== RUN THE FULL MC ======
print("Starting 500k photon MC...")
t0 = time.time()
np.random.seed(123)
scoring = make_scoring_full()
elec_scoring = make_elec_scoring()
N = 500000
for i in range(N):
    x_t = (np.random.rand()-0.5)*10
    y_t = (np.random.rand()-0.5)*10
    norm = math.sqrt(x_t*x_t+y_t*y_t+SSD*SSD)
    u = np.array([x_t,y_t,SSD])/norm
    p_surf = np.array([x_t,y_t,0.0])
    els = transport_photon_full(p_surf, u, 2.0, True, scoring)
    scoring['electrons_cylinder'].extend(els)
    if (i+1) % 100000 == 0:
        print(f"  {i+1}/{N}, {time.time()-t0:.1f}s")
print(f"Photon MC done in {time.time()-t0:.1f}s")
print(f"Primary fluence track: {scoring['ph_prim_fluence']:.2f}")
print(f"Scattered fluence track: {scoring['ph_scat_fluence']:.2f}")
print(f"SPR fluence: {scoring['ph_scat_fluence']/scoring['ph_prim_fluence']:.4f}")

# Save intermediate
with open('mc_results.pkl', 'wb') as f:
    pickle.dump({'scoring': scoring, 'N': N}, f)

# Transport electrons
print(f"\nTransporting {len(scoring['electron_sources'])} electron sources...")
t1 = time.time()
for (p, u, T, parent_prim) in scoring['electron_sources']:
    transport_electron(p, u, T, elec_scoring, is_delta=False)
print(f"Electron transport done in {time.time()-t1:.1f}s")
print(f"Electron fluence track: {elec_scoring['e_fluence']:.3f} cm")
print(f"Electron dose: {elec_scoring['e_dose']:.3f} MeV")
print(f"Delta fluence: {elec_scoring['delta_fluence']:.3f} cm")

with open('mc_results.pkl', 'wb') as f:
    pickle.dump({'scoring': scoring, 'elec_scoring': elec_scoring, 'N': N}, f)
print("All results saved.")