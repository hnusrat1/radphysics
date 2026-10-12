script = r'''
import numpy as np
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

# Layer 1
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

print(f"L1: pf={pf:.6f}, tf={tf:.6f}, sff={sff:.6f}")
print(f"L1 sef={sef:.6f}, mae={mae:.4f}")
'''

with open('/tmp/calc_full.py', 'w') as f:
    f.write(script)
    
exec(open('/tmp/calc_full.py').read())
