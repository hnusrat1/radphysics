import numpy as np

# ------------------------------------------------ data
def load_xcom():
    with open('/tmp/data/xcom_water.txt') as f:
        lines = [l for l in f if not l.startswith('#') and l.strip()]
    x = np.loadtxt(lines)
    return x[:,0], x[:,1], x[:,2], x[:,3], x[:,4], x[:,5], x[:,6]
E_x, mu_coh, mu_incoh, mu_pe, mu_pairn, mu_paire, mu_tot = load_xcom()

def mu_interp(E, col):
    # log-log interpolation
    return np.exp(np.interp(np.log(E), np.log(E_x), np.log(col)))

def mu_total(E): return mu_interp(E, mu_tot)
def mu_incoh_E(E): return mu_interp(E, mu_incoh)
def mu_coh_E(E): return mu_interp(E, mu_coh)
def mu_pe_E(E): return mu_interp(E, mu_pe)
def mu_pair_E(E): return mu_interp(E, mu_pairn+mu_paire)

def load_estar():
    with open('/tmp/data/estar_water.txt') as f:
        lines = [l for l in f if not l.startswith('#') and l.strip()]
    est = np.loadtxt(lines)
    return est[:,0], est[:,1], est[:,2], est[:,3], est[:,4], est[:,5], est[:,6]
Ee, Scol, Srad, Stot, csda, yld, den = load_estar()
def interp_estar(E, col):
    return np.exp(np.interp(np.log(E), np.log(Ee), np.log(col)))
def scol_w(E): return interp_estar(E, Scol)
def srad_w(E): return interp_estar(E, Srad)

rho = 1.0  # g/cm3
mc2 = 0.51099895  # MeV
re2 = 7.9408e-26  # cm^2

# ------------------------------------------------ KN
def kn_dsig(E0, cosT):
    # dsigma/dOmega (per electron) for photon scattering angle theta, cosT = cos theta
    a = E0/mc2
    Ep = E0/(1.0 + a*(1.0 - cosT))
    r = Ep/E0
    return 0.5*re2 * r**2 * (r + 1.0/r - (1.0-cosT**2))

def kn_total(E0):
    # numeric integrate over theta
    th = np.linspace(1e-6, np.pi, 20000)
    c = np.cos(th)
    return np.trapz(kn_dsig(E0, c)*2*np.pi*np.sin(th), th)

def compton_Eprime(E0, cosT):
    a = E0/mc2
    return E0/(1.0 + a*(1.0 - cosT))

def compton_T_e(E0, cosT):
    return E0 - compton_Eprime(E0, cosT)

def compton_phi_e(E0, cosT):
    # recoil electron angle from incoming photon direction
    a = E0/mc2
    th = np.arccos(np.clip(cosT,-1,1))
    # cot(phi) = (1+a) tan(theta/2)
    cotp = (1.0+a)*np.tan(th/2.0)
    return np.arctan2(1.0, cotp)  # phi in [0, pi/2]

# check KN total for E0=1.1732: compute vs XCOM total incoherent per electron
for E0 in [1.1732, 1.3325]:
    print(E0, "KN total per electron cm2:", kn_total(E0))
    # xcom incoherent total per molecule /10 electrons:
    print("  xcom incoh per electron:", mu_incoh_E(E0)/ (3.343e23*1e-24*1))  # mu cm2/g / (electrons/g) -> cm2/e ... rough check