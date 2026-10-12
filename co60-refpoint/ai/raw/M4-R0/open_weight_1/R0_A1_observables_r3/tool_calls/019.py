
import numpy as np

mec2=0.5110
N_e_per_g = 6.02214076e23*10/18.01528
n_e = N_e_per_g

E_tab = np.array([0.01,0.02,0.03,0.04,0.05,0.06,0.08,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.8,1.0,1.25,1.5,2.0])
mu_tab = np.array([4.951,1.374,0.7031,0.4470,0.2269,0.2059,0.1837,0.1706,0.1505,0.1370,0.1186,0.1061,0.0969,0.0896,0.0786,0.0707,0.0636,0.0576,0.0494])
def interp(x,xe,ye): return np.interp(x,xe,ye)
def mu_tot(E): return interp(E,E_tab,mu_tab)
def sig_KN(E):
    al=E/mec2; a2=al*al
    f=(1+al)/a2*(2*(1+al)/(1+2*al)-np.log(1+2*al)/al)+np.log(1+2*al)/(2*al)-(1+3*al)/(1+2*al)**2
    return 0.4987*f
def mu_C(E): return sig_KN(E)*1e-24*N_e_per_g
coh_tab = np.array([0.0,0.0,0.0,0.03,0.0205,0.0148,0.0102,0.0084,0.0047,0.0030,0.0014,0.0009,0.0006,0.00045,0.00030,0.00024,0.00019,0.00016,0.00010])
def mu_coh(E): return interp(E,E_tab,coh_tab)
def mu_pair(E): return np.where(E>1.022, 0.00052*((E-1.022)/0.978)**2.0, 0.0)
def mu_pe(E): return np.maximum(0, mu_tot(E)-mu_C(E)-mu_coh(E)-mu_pair(E))

Scol_water = lambda E: np.interp(E, [0.01,0.02,0.03,0.05,0.07,0.1,0.2,0.3,0.5,0.7,1.0,1.25,1.4],
                                  [22.7,11.3,7.97,6.36,5.14,4.098,2.63,2.16,1.94,1.89,1.847,1.847,1.85])
Srad_water = lambda E: 0.0155*(E/1.0)**1.15

C = 0.4987e-24 * 0.511

def moler_dsde(T, E0):
    if T <= 0.011 or E0<=0.010 or E0>=T/2: return 0.0
    beta2 = 1.0 - 1.0/(1+T/mec2)**2
    if beta2 <= 0: return 0.0
    factor = C * n_e / beta2
    term = 1/E0**2 - 1/(E0*(T-E0)) + 1/(T-E0)**2
    return factor*term

def S_rest_water(E):
    if E <= 0.011: return Scol_water(E)
    dEint = 0.00025
    E0vals = np.arange(0.0105, E/2 - 1e-9, dEint)
    integ = np.sum(E0vals * np.array([moler_dsde(E,x) for x in E0vals]) * dEint)
    return Scol_water(E) - integ

Eg = np.arange(0.010, 1.401, 0.002)
dE = 0.002
nE = len(Eg)
S_rest_arr = np.array([S_rest_water(e) for e in Eg])
Scol_arr = np.array([Scol_water(e) for e in Eg])
Srad_arr = np.array([Srad_water(e) for e in Eg])
print("S_rest at 0.1,0.5,1.0,1.25:", [S_rest_water(e) for e in [0.1,0.5,1.0,1.25]])

ph_bins = np.array([0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4])
ph_c = (ph_bins[:-1]+ph_bins[1:])/2
prim_spec = np.load('prim_spec.npy')
scat_spec = np.load('scat_spec.npy')

rng = np.random.default_rng(0)
def compton_samples(E, n, rng):
    x = rng.uniform(-1,1,size=n)
    def kndiff(cosT):
        al=E/mec2; x0=1+al*(1-cosT)
        return (1.0/x0)**2*(0.5*(1+cosT*cosT)+(al*al*(1-cosT)**2)/(2*x0))
    acc = rng.uniform(0,1,size=n) < kndiff(x)
    while not acc.all():
        rem=~acc
        nx=rng.uniform(-1,1,size=rem.sum())
        nacc = rng.uniform(0,1,size=rem.sum()) < kndiff(nx)
        x[rem]=nx; acc[rem]=nacc
    al=E/mec2
    Ep = E/(1+al*(1-x))
    return E-Ep

def build_q(spec):
    qq = np.zeros(nE)
    for i, Ec in enumerate(ph_c):
        wf = spec[i]
        if wf <= 0: continue
        nCt = wf * mu_C(Ec)
        if nCt > 0:
            ns = 40000
            Ee = compton_samples(Ec, ns, rng)
            h, _ = np.histogram(Ee, bins=Eg, weights=np.full(ns, nCt/ns))
            qq += h/dE
        nPE = wf * mu_pe(Ec)
        if nPE > 0:
            T = Ec - 0.0005
            if T >= 0.01:
                bi = np.searchsorted(Eg, T, side='right')-1
                if 0 <= bi < nE:
                    qq[bi] += nPE/dE
        nPA = wf * mu_pair(Ec)
        if nPA > 0:
            T = Ec - 1.022
            if T > 0:
                for _ in range(2000):
                    f = rng.uniform(0,1)
                    Te = f*T
                    if Te >= 0.01:
                        bi = np.searchsorted(Eg, Te, side='right')-1
                        if 0 <= bi < nE:
                            qq[bi] += nPA/2000/dE
    return qq

q = build_q(prim_spec+scat_spec)
print("q integrated (interactions per cm3 per Phi0, electrons only):", q.sum()*dE)

# solve
Phi = np.zeros(nE)
for ite in range(12):
    qsec = np.zeros(nE)
    imax = np.searchsorted(Eg, 1.401, side='right')
    start = np.searchsorted(Eg, 0.010, side='right')
    hi = np.searchsorted(Eg, 1.401, side='left')
    for ite_idx in range(start, hi):
        T = Eg[ite_idx]
        if T < 0.012: continue
        max_idx = np.searchsorted(Eg, T/2, side='right')-1
        for ie in range(start, max_idx+1):
            E0 = Eg[ie]
            if E0 >= T/2 - 1e-12: continue
            dsd = moler_dsde(T, E0)
            if dsd > 0:
                qsec[ie] += Phi[ite_idx]*dsd*dE
    cum = np.cumsum((q+qsec)[::-1])[::-1]*dE
    new = cum/S_rest_arr
    new = np.where(S_rest_arr>0, new, 0.0)
    diff = np.max(np.abs(new-Phi))
    Phi = new
    print(f"iter {ite+1}: diff {diff:.3e}")
    if diff < 1e-7: break

cum_p = np.cumsum(q[::-1])[::-1]*dE
Phi_p = cum_p/S_rest_arr
Phi_p = np.where(S_rest_arr>0, Phi_p, 0.0)

print("Phi_total:", Phi.sum()*dE, " Phi_primary:", Phi_p.sum()*dE)
print("delta fraction:", (Phi.sum()-Phi_p.sum())/Phi.sum())
print("mean electron energy:", np.sum(Eg*Phi*dE)/np.sum(Phi*dE))
print("rad/coll:", np.sum(Phi*Srad_arr*dE)/np.sum(Phi*Scol_arr*dE))

el_bins = np.array([0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.4])
Phi_spec = np.zeros(25)
for i in range(25):
    Phi_spec[i] = np.sum(Phi[(Eg>=el_bins[i])&(Eg<el_bins[i+1])]*dE)
Phi_prim_spec = np.zeros(25)
for i in range(25):
    Phi_prim_spec[i] = np.sum(Phi_p[(Eg>=el_bins[i])&(Eg<el_bins[i+1])]*dE)
print("Phi_spec sum:", Phi_spec.sum())
print("Phi_spec:", np.round(Phi_spec,6))
print("Phi_prim_spec:", np.round(Phi_prim_spec,6))

np.savez('L4_data.npz', Eg=Eg, Phi=Phi, Phi_p=Phi_p, q=q, S_rest=S_rest_arr, Scol=Scol_arr, Srad=Srad_arr,
         Phi_spec=Phi_spec, Phi_prim_spec=Phi_prim_spec)
print("saved")
