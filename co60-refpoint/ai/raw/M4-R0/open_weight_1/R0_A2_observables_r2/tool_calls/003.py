import numpy as np

def load_xcom():
    with open('/tmp/data/xcom_water.txt') as f:
        lines = [l for l in f if not l.startswith('#') and l.strip()]
    x = np.loadtxt(lines)
    return x[:,0], x[:,1], x[:,2], x[:,3], x[:,4], x[:,5], x[:,6]
E_x, mu_coh, mu_incoh, mu_pe, mu_pairn, mu_paire, mu_tot = load_xcom()
def mu_interp(E, col):
    return np.exp(np.interp(np.log(E), np.log(E_x), np.log(col)))
def mu_total(E): return mu_interp(E, mu_tot)
def mu_incoh_E(E): return mu_interp(E, mu_incoh)
def mu_coh_E(E): return mu_interp(E, mu_coh)
def mu_pe_E(E): return mu_interp(E, mu_pe)
def mu_pair_E(E): return mu_interp(E, mu_pairn+mu_paire)

rho=1.0; mc2=0.51099895; re2=7.9408e-26

def kn_dsig(E0, cosT):
    a = E0/mc2
    Ep = E0/(1.0 + a*(1.0 - cosT))
    r = Ep/E0
    return 0.5*re2 * r**2 * (r + 1.0/r - (1.0-cosT**2))
def kn_total(E0):
    th = np.linspace(1e-6, np.pi, 20000)
    c = np.cos(th)
    return np.trapezoid(kn_dsig(E0, c)*2*np.pi*np.sin(th), th)
def compton_Eprime(E0, cosT):
    a = E0/mc2
    return E0/(1.0 + a*(1.0 - cosT))
def compton_phi_e(E0, cosT):
    a = E0/mc2
    th = np.arccos(np.clip(cosT,-1,1))
    cotp = (1.0+a)*np.tan(th/2.0)
    return np.arctan2(1.0, cotp)

for E0 in [1.1732, 1.3325]:
    print(E0, "KN total per electron cm2:", kn_total(E0))
    # per electron from XCOM: mu_incoh [cm2/g] * rho / (electrons per g)
    Ne_g = 3.343e23
    print("  xcom incoh per electron:", mu_incoh_E(E0)/Ne_g)

# mean Compton scattered energy and energy transfer at E0
for E0 in [1.1732,1.3325]:
    th=np.linspace(1e-6,np.pi,20000); c=np.cos(th)
    w = kn_dsig(E0,c)*2*np.pi*np.sin(th)
    Ep = compton_Eprime(E0,c)
    meanEp = np.trapezoid(w*Ep,th)/np.trapezoid(w,th)
    meanTr = E0-meanEp
    print(f"E0={E0}: <Ep>={meanEp:.4f}, <T>={meanTr:.4f}, Tmax={compton_T_e(E0,-1):.4f}")