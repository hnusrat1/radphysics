
import numpy as np, math, time

# ---------------- photon cross-section tables (water, rho=1 g/cm3) ----------------
Etab = np.array([0.05,0.06,0.08,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.8,1.0,1.25,1.5,2.0],float)
mu_tab   = np.array([0.2269,0.2059,0.1838,0.1707,0.1505,0.1373,0.1186,0.1061,0.0969,0.0896,0.0787,0.0707,0.0632,0.0576,0.0494],float)
mu_coh   = np.array([0.0222,0.0193,0.0161,0.0154,0.0118,0.0097,0.0073,0.0061,0.0053,0.0047,0.0039,0.0033,0.0028,0.0024,0.0019],float)
mu_en_tab= np.array([0.0406,0.0323,0.0260,0.0256,0.0277,0.0297,0.0319,0.0328,0.0330,0.0329,0.0320,0.0309,0.0297,0.0285,0.0262],float)

def interp1(x, xe, ye):
    return np.exp(np.interp(np.log(x), np.log(xe), np.log(ye)))

def mu_total(E): return interp1(E,Etab,mu_tab)
def mu_re(E): return interp1(E,Etab,mu_coh)
def mu_en(E): return interp1(E,Etab,mu_en_tab)

# photoelectric approx calibrated
def mu_pe(E):
    # power law ~ E^-3.2, value ~0.0064 at 0.1 MeV
    return 0.0064*(E/0.1)**(-3.2)

# pair approx: nonzero above 1.022
def mu_pair(E):
    if E<=1.022: return 0.0
    # approx 0.00025*(E-1.022)/0.5 at 1.5, scaled
    return 0.00025 + 0.00015*(E-1.25)  # rough around 1.25-1.5

def comp_frac(E):
    m = mu_total(E)
    c = mu_re(E); p = mu_pe(E); q = mu_pair(E)
    comp = m - c - p - q
    if comp<0: comp = m*0.85; c*0.1
    return comp,m,c,p,q

# sanity
for E in [0.1,0.5,1.0,1.1732,1.25,1.3325,1.5]:
    comp,m,c,p,q = comp_frac(E)
    print(E, "mu",m,"comp",comp,"coh",c,"pe",p,"pair",q, "fracs", comp/m, c/m, p/m, q/m)
