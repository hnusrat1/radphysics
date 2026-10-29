# Write full calculation script to file
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

# Layer 1 calculations
mtw = sum(w * ix(6, E) for w, E in zip(w_co, E_co))
ml = mtw * rho
pf = np.exp(-ml * depth)
bsp = 0.18
tf = pf * (1 + bsp)
sff = 1 - pf / tf
me_pri = 0.5*E_co[0] + 0.5*E_co[1]
me_sc = me_pri * 0.75
ep = pf * me_pri
es = (tf - pf) * me_sc
sef = es / (ep + es)
mae = (ep + es) / tf

# Photon fluence spectra (primary and scattered)
spec_pri = np.zeros(len(pe_edges)-1)
for i, e in enumerate(E_co):
    bi = int(np.digitize(e, pe_edges)) - 1
    if 0 <= bi < len(spec_pri):
        spec_pri[bi] += w_co[i] * pf
spec_pri_sum = np.sum(spec_pri * np.diff(pe_edges))
spec_pri_norm = spec_pri * np.mean(np.diff(pe_edges)) / spec_pri_sum

# Scattered spectrum estimate
spec_scat = np.zeros(len(pe_edges)-1)
bcs = 0.5*(pe_edges[:-1]+pe_edges[1:])
for i, bc in enumerate(bcs):
    if 0.1 <= bc <= 1.2:
        spec_scat[i] = np.exp(-((bc-0.4)/0.25)**2)
spec_scat_sum = np.sum(spec_scat * np.diff(pe_edges))
spec_scat_norm = spec_scat * (tf-pf) * np.mean(np.diff(pe_edges)) / spec_scat_sum

# Angular distributions
ang_pri = [0.98, 0.015, 0.003, 0.001, 0.001, 0, 0, 0, 0, 0]  # Highly forward peaked
ang_scat = [0, 0, 0, 0.02, 0.05, 0.15, 0.20, 0.18, 0.12, 0.28]  # More isotropic after scattering

# Normalize angular distributions
ang_pri = [x/sum(ang_pri) for x in ang_pri]
ang_scat = [x/sum(ang_scat) for x in ang_scat]

print("Layer 1 complete")
print(f"L1_pf={pf:.6f}, L1_tf={tf:.6f}")
'''

with open('/tmp/calc_layer1.py', 'w') as f:
    f.write(script)

exec(open('/tmp/calc_layer1.py').read())
