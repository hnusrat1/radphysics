
import numpy as np, time

me = 0.51099895
re = 2.8179403262e-13
NA = 6.02214076e23
n_e_g = NA*10/18.01528

def kn_dsigma_dmu(E, mu):
    eta = E/me
    r = 1.0/(1.0+eta*(1.0-mu))
    sint2 = 1.0-mu*mu
    return np.pi*re**2 * r**2 * (r + 1.0/r - sint2)
def kn_total(E):
    eta = E/me
    a = 1.0+2.0*eta
    return 2*np.pi*re**2*((1+eta)/eta**2*(2*(1+eta)/a - np.log(a)/eta) + np.log(a)/(2*eta) - (1+3*eta)/a**2)

Eanch = np.array([0.05,0.1,0.2,0.3,0.5,1.0,1.25,1.5,2.0,3.0])
uanch = np.array([0.2269,0.1707,0.1370,0.1187,0.0969,0.0707,0.0632,0.0577,0.0494,0.0397])
def mu_tot(E):
    return np.exp(np.interp(np.log(np.clip(E,0.02,3.0)), np.log(Eanch), np.log(uanch)))

# coherent, photo, pair mass attenuation coefficients (interpolation tables)
Egrid = np.array([0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.8,1.0,1.1732,1.25,1.3325,1.5,2.0])
muR  = np.array([0.0063,0.0050,0.0045,0.0042,0.0039,0.0038,0.0037,0.0036,0.0035,0.0035,0.0035,0.0035,0.0034,0.0033,0.0030])
tau  = np.array([0.0093,0.0019,0.00070,0.00042,0.00016,0.000085,0.000052,0.000036,0.000021,0.000013,0.000010,0.000009,0.000008,0.000007,0.000004])
kap  = np.array([0,0,0,0,0,0,0,0,0,0,0.00012,0.00042,0.00075,0.0011,0.0026])
def interp1(x, xp, fp):
    return np.interp(np.clip(x, xp[0], xp[-1]), xp, fp)
def mu_comp(E): return kn_total(E)*n_e_g
def mu_ray(E): return interp1(E, Egrid, muR)
def mu_photo(E): return interp1(E, Egrid, tau)
def mu_pair(E): return np.where(E>2*me, interp1(E, Egrid, kap), 0.0)
def mu_tot2(E):
    return mu_comp(E)+mu_ray(E)+mu_photo(E)+mu_pair(E)

# check mu_tot2 vs mu_tot at line energies
for E in [1.1732,1.3325]:
    print(E, "mu_tot(NIST fit)", mu_tot(E), "mu_tot2(components)", mu_tot2(E))
