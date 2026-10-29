import numpy as np
xcom = np.loadtxt('/tmp/data/xcom_water.txt', comments='#')
estar = np.loadtxt('/tmp/data/estar_water.txt', comments='#')
def ix(ci, E): return np.interp(np.log10(E), np.log10(xcom[:, 0]), xcom[:, ci])
E_co = [1.1732, 1.3325]
w_co = [0.5, 0.5]
depth = 5.0
rho = 1.0
# Layer 1 calculation
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
print(f"pf={pf:.6f}, tf={tf:.6f}, sff={sff:.6f}")
print(f"sef={sef:.6f}, mae={mae:.4f}")
print(f"me_pri={me_pri:.4f}, me_sc={me_sc:.4f}")
