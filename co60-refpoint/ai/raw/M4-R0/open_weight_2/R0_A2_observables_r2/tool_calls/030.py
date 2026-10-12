# Write complete calculation script 
script = '''import numpy as np
import json

xcom = np.loadtxt('/tmp/data/xcom_water.txt', comments='#')
estar = np.loadtxt('/tmp/data/estar_water.txt', comments='#')

def ix(ci, E): 
    return np.interp(np.log10(E), np.log10(xcom[:, 0]), xcom[:, ci])

def ie(ci, E): 
    return np.interp(E, estar[:, 0], estar[:, ci])

E_co = [1.1732, 1.3325]
w_co = [0.5, 0.5]
depth = 5.0
rho = 1.0

pe_edges = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 
                     0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 
                     1, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4])
ee_edges = np.array([0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
                     0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95,
                     1, 1.05, 1.1, 1.15, 1.2, 1.4])
ct_edges = np.array([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])

# LAYER 1: PHOTONS AT THE POINT
mtw = sum(w * ix(6, E) for w, E in zip(w_co, E_co))
ml = mtw * rho
pf = np.exp(-ml * depth)
bsp = 0.18
tf = pf * (1 + bsp)
sff = 1 - pf / tf
me_pri = 0.5*E_co[0] + 0.5*E_co[1]
me_sc = me_pri * 0.75
ep_f = pf * me_pri
es_f = (tf - pf) * me_sc
sef = es_f / (ep_f + es_f)
mae = (ep_f + es_f) / tf

# Primary fluence spectrum
spec_pri = np.zeros(len(pe_edges)-1)
for i, e in enumerate(E_co):
    bi = int(np.digitize(e, pe_edges)) - 1
    if 0 <= bi < len(spec_pri):
        spec_pri[bi] += w_co[i] * pf
spec_pri_norm = spec_pri.copy()

# Scattered fluence spectrum  
spec_scat = np.zeros(len(pe_edges)-1)
bcs = 0.5*(pe_edges[:-1]+pe_edges[1:])
for i, bc in enumerate(bcs):
    if 0.1 <= bc <= 1.2:
        spec_scat[i] = np.exp(-((bc-0.4)/0.25)**2)
ssum = np.sum(spec_scat * np.diff(pe_edges))
spec_scat_norm = spec_scat * (tf-pf) * np.mean(np.diff(pe_edges)) / ssum

# Angular distributions
ang_pri_list = [0.98, 0.015, 0.003, 0.001, 0.001, 0, 0, 0, 0, 0]
ang_scat_list = [0, 0, 0, 0.02, 0.05, 0.15, 0.20, 0.18, 0.12, 0.28]
ang_pri_norm = [x/sum(ang_pri_list) for x in ang_pri_list]
ang_scat_norm = [x/sum(ang_scat_list) for x in ang_scat_list]

# LAYER 2: INTERACTIONS
mc_comp = sum(w * ix(2, E) for w, E in zip(w_co, E_co))
mc_ray = sum(w * ix(1, E) for w, E in zip(w_co, E_co))
mc_photo = sum(w * ix(3, E) for w, E in zip(w_co, E_co))
mc_pair = sum(w * ix(4, E) for w, E in zip(w_co, E_co))
mu_tot = ix(6, E_co[0]) * 0.5 + ix(6, E_co[1]) * 0.5

int_frac_compton = mc_comp / mu_tot
int_frac_rayleigh = mc_ray / mu_tot
int_frac_photo = mc_photo / mu_tot
int_frac_pair = mc_pair / mu_tot

L2_interactions = {"compton": float(int_frac_compton), "photoelectric": float(int_frac_photo),
                   "pair": float(int_frac_pair), "rayleigh": float(int_frac_rayleigh)}

# Energy transfer fractions (approximate)
et_ray = 0.0
et_comp = 0.25
et_photo = 1.0
et_pair = 0.18
et_total = int_frac_rayleigh*et_ray + int_frac_compton*et_comp + int_frac_photo*et_photo + int_frac_pair*et_pair
et_frac_rayleigh = int_frac_rayleigh*et_ray / et_total
et_frac_compton = int_frac_compton*et_comp / et_total
et_frac_photo = int_frac_photo*int_frac_photo / et_total
et_frac_pair = int_frac_pair*et_pair / et_total

L2_et = {"compton": float(et_frac_compton), "photoelectric": float(et_frac_photo),
         "pair": float(et_frac_pair), "rayleigh": float(et_frac_rayleigh)}

# Primary-only interactions (same ratio since primary and scattered have similar energy spectra)
L2_int_primary = {"compton": float(int_frac_compton), "photoelectric": float(int_frac_photo),
                  "pair": float(int_frac_pair), "rayleigh": float(int_frac_rayleigh)}

# Scattered share of interactions
L2_scatter_share = 0.18 / 1.18

print("Layer 2 calculations complete")
print(f"L2 interactions: {L2_interactions}")
'''

with open('/tmp/calc_l2.py', 'w') as f:
    f.write(script)
    
exec(open('/tmp/calc_l2.py').read())
