
import numpy as np

# ---------- Constants ----------
mec2 = 0.5110  # MeV
re2 = 2.8179403e-13**2  # cm^2, classical electron radius squared
twopi_re2 = 2*np.pi*re2  # cm^2
N_e_per_g = 6.02214076e23 * 10/18.01528  # electrons per gram for water ~3.343e23
rho_w = 1.0  # g/cm3
N_e_per_cm3 = N_e_per_g * rho_w

# ---------- Klein-Nishina ----------
def sig_KN(E):  # barns per electron
    al = E/mec2
    a2 = al*al
    # f(al)
    t1 = (1+al)/a2 * (2*(1+al)/(1+2*al) - np.log(1+2*al)/al)
    t2 = np.log(1+2*al)/(2*al)
    t3 = (1+3*al)/(1+2*al)**2
    f = t1 + t2 - t3
    return 0.4987 * f  # 2*pi*r_e^2 = 0.4987 barns

def kn_diff(cosT, E):  # dsigma/dOmega (per electron, relative) for photon scatter angle cosT
    al = E/mec2
    x = 1 + al*(1-cosT)
    f = (1.0/x)**2 * (0.5*(1+cosT*cosT) + (al*al*(1-cosT)**2)/(2*x))
    return f

def mu_compton_cm2g(E):
    return sig_KN(E)*1e-24*N_e_per_g  # cm2/g

# ---------- NIST-like total attenuation for water ----------
E_tab = np.array([0.01,0.02,0.03,0.04,0.05,0.06,0.08,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.8,1.0,1.25,1.5,2.0])
mu_tab = np.array([4.951, 1.374?] if False else [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
# fill with NIST-like values (mass attenuation, cm2/g, water)
mu_tab = np.array([4.951,1.374,0.7031,0.4470,0.2269,0.2059,0.1837,0.1706,0.1505,0.1370,0.1186,0.1061,0.0969,0.0896,0.0786,0.0707,0.0636,0.0576,0.0494])
# Coherent (Rayleigh) approximate for water
coh_tab = np.array([0.0, 0.0, 0.0, 0.032, 0.0205,0.0148,0.0102,0.0084,0.0047,0.0030,0.0014,0.0009,0.0006,0.00045,0.00030,0.00024,0.00019,0.00016,0.0001])
# pair production approximate for water
pair_tab = np.zeros_like(E_tab)
for i,E in enumerate(E_tab):
    if E>1.022:
        pair_tab[i] = 0.00153*((E-1.022)/0.228)**0.618

def interp(x, xe, ye):
    return np.interp(x, xe, ye)

def mu_tot_cm2g(E):
    return interp(E, E_tab, mu_tab)

def mu_coh_cm2g(E):
    return interp(E, E_tab, coh_tab)

def mu_pair_cm2g(E):
    return np.where(E>1.022, interp(E, E_tab, pair_tab), 0.0)

def mu_pe_cm2g(E):
    tot = mu_tot_cm2g(E)
    c = mu_compton_cm2g(E)
    return np.maximum(0, tot - c - mu_coh_cm2g(E) - mu_pair_cm2g(E))

# check at 1.1732 and 1.3325
for E in [0.05,0.1,0.2,0.5,1.0,1.1732,1.25,1.3325,1.4]:
    print(f"E={E:.4f}: mu_tot={mu_tot_cm2g(E):.4f} comp={mu_compton_cm2g(E):.4f} pe={mu_pe_cm2g(E):.5f} "
          f"pair={mu_pair_cm2g(E):.5f} coh={mu_coh_cm2g(E):.5f} sum={mu_compton_cm2g(E)+mu_pe_cm2g(E)+mu_pair_cm2g(E)+mu_coh_cm2g(E):.4f}")

# primary fluence per phi0
mu1 = mu_tot_cm2g(1.1732); mu2 = mu_tot_cm2g(1.3325)
print("mu1,mu2:",mu1,mu2)
print("primary/Phi0 =", 0.5*(np.exp(-mu1*5)+np.exp(-mu2*5)))

# check phi0 per history
Omega = 100.0/100.0**2  # 10x10 at 100 cm -> 100 cm^2/10000 cm^2 = 0.01 sr
phi0_per_hist = 1.0/(Omega*105.0**2)
print("phi0 per history:", phi0_per_hist)
