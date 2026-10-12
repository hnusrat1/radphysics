
import numpy as np

# Constants
me = 0.51099895  # MeV
re = 2.8179403262e-13  # cm
NA = 6.02214076e23
M_H2O = 18.01528  # g/mol
n_e_g = NA*10/M_H2O   # electrons per gram
rho = 1.0
n_e = n_e_g*rho

def kn_dsigma_dmu(E, mu):
    """Klein-Nishina dsigma/dmu per electron, mu=cos of photon scatter angle"""
    eta = E/me
    r = 1.0/(1.0+eta*(1.0-mu))
    sint2 = 1.0-mu*mu
    return np.pi*re**2 * r**2 * (r + 1.0/r - sint2)

def kn_total(E):
    eta = E/me
    a = 1.0+2.0*eta
    return 2*np.pi*re**2*((1+eta)/eta**2*(2*(1+eta)/a - np.log(a)/eta) + np.log(a)/(2*eta) - (1+3*eta)/a**2)

# check KN values
for E in [0.1,1.0,1.1732,1.25,1.3325]:
    print(E, kn_total(E)*1e24, "barn/electron", kn_total(E)*n_e_g, "cm2/g")

# Approximate mu_R (coherent) and tau (photo) and kappa (pair) per g water
Egrid = np.array([0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.8,1.0,1.1732,1.25,1.3325,1.5,2.0])
muR = np.array([0.0063,0.0050,0.0045,0.0042,0.0039,0.0038,0.0037,0.0036,0.0035,0.0035,0.0035,0.0035,0.0034,0.0033,0.0030])
tau = np.array([0.0093,0.0019,0.00070,0.00042,0.00016,0.000085,0.000052,0.000036,0.000021,0.000013,0.000010,0.000009,0.000008,0.000007,0.000004])
kappa = np.array([0,0,0,0,0,0,0,0,0,0,0.00012,0.00042,0.00075,0.0011,0.0026])
from scipy.interpolate import interp1d
f_muR = interp1d(Egrid, muR, kind='linear', bounds_error=False, fill_value=(muR[0],muR[-1]))
f_tau = interp1d(Egrid, tau, kind='linear', bounds_error=False, fill_value=(tau[0],tau[-1]))
f_kap = interp1d(Egrid, kappa, kind='linear', bounds_error=False, fill_value=(kappa[0],kappa[-1]))

def mu_comp(E): return kn_total(E)*n_e_g
def mu_ray(E): return f_muR(E)
def mu_photo(E): return f_tau(E)
def mu_pair(E): return f_kap(E)
def mu_tot(E): return mu_comp(E)+mu_ray(E)+mu_photo(E)+mu_pair(E)

for E in [0.1,0.5,1.1732,1.3325]:
    print(E, "mu_comp",round(mu_comp(E),5),"mu_R",round(mu_ray(E),5),"mu_ph",round(mu_photo(E),6),"mu_pair",round(mu_pair(E),5),"mu_tot",round(mu_tot(E),5))
