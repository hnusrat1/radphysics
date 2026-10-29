
import numpy as np

me = 0.51099895
re = 2.8179403262e-13
NA = 6.02214076e23
n_e_g = NA*10/18.01528

def kn_total(E):
    eta = np.asarray(E,dtype=float)/me
    a = 1.0+2.0*eta
    return 2*np.pi*re**2*((1+eta)/eta**2*(2*(1+eta)/a - np.log(a)/eta) + np.log(a)/(2*eta) - (1+3*eta)/a**2)

Egrid = np.array([0.01,0.02,0.03,0.05,0.08,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.8,1.0,1.1732,1.25,1.3325,1.5,2.0,3.0])
muR  = np.array([0.028,0.012,0.0085,0.0060,0.0052,0.0045,0.0033,0.0028,0.0022,0.0019,0.0017,0.0015,0.0012,0.0010,0.0009,0.00085,0.0008,0.00075,0.00055,0.00040])
tau  = np.array([0.60,0.15,0.045,0.0093,0.0026,0.0019,0.00070,0.00042,0.00016,0.000085,0.000052,0.000036,0.000021,0.000013,0.000010,0.000009,0.000008,0.000007,0.000004,0.0000015])
kap  = np.zeros_like(Egrid)
sel = Egrid>1.022
kap[sel]=np.interp(np.clip(Egrid[sel],1.1732,3.0),[1.1732,1.3325,1.5,2.0,3.0],[0.00012,0.00075,0.0011,0.0026,0.0055])

def interp1(x, xp, fp):
    return np.interp(np.clip(x, xp[0], xp[-1]), xp, fp)
def mu_comp(E): return kn_total(np.clip(E,0.01,3.0))*n_e_g
def mu_ray(E): return interp1(E, Egrid, muR)
def mu_photo(E): return interp1(E, Egrid, tau)
def mu_pair(E): 
    E=np.asarray(E,float)
    out = np.where(E>2*me, interp1(E, Egrid, kap), 0.0)
    return out
def mu_totX(E):
    return mu_comp(E)+mu_ray(E)+mu_photo(E)+mu_pair(E)

for E in [0.1,0.5,1.1732,1.3325]:
    print('%6.4f MeV: mu_c=%.5f mu_R=%.5f mu_ph=%.6f mu_pair=%.6f tot=%.5f'%(E,mu_comp(E),mu_ray(E),mu_photo(E),mu_pair(E),mu_totX(E)))

# check pair cap continuity
print('kap at grid',kap)
