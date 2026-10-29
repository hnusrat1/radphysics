import numpy as np

mec2=0.5110
N_e_per_g = 6.02214076e23*10/18.01528
n_e = N_e_per_g  # per cm3 since density 1

E_tab = np.array([0.01,0.02,0.03,0.04,0.05,0.06,0.08,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.8,1.0,1.25,1.5,2.0])
mu_tab = np.array([4.951,1.374,0.7031,0.4470,0.2269,0.2059,0.1837,0.1706,0.1505,0.1370,0.1186,0.1061,0.0969,0.0896,0.0786,0.0707,0.0636,0.0576,0.0494])
def interp(x,xe,ye): return np.interp(x,xe,ye)
def mu_tot(E): return interp(E,E_tab,mu_tab)
def sig_KN(E):
    al=E/mec2; a2=al*al
    f=(1+al)/a2*(2*(1+al)/(1+2*al)-np.log(1+2*al)/al)+np.log(1+2*al)/(2*al)-(1+3*al)/(1+2*al)**2
    return 0.4987*f
def mu_C(E): return sig_KN(E)*1e-24*N_e_per_g
def mu_pe(E): return np.maximum(0, mu_tot(E)-mu_C(E)-mu_coh(E)-mu_pair(E))
def mu_coh(E): return interp(E, E_tab, np.array([0.0,0.0,0.0,0.03,0.0205,0.0148,0.0102,0.0084,0.0047,0.0030,0.0014,0.0009,0.0006,0.00045,0.00030,0.00024,0.00019,0.00016,0.00010]))
def mu_pair(E): return np.where(E>1.022, 0.00052*((E-1.022)/0.978)**2.0, 0.0)

# Stopping powers (approximate ESTAR water)
Scol_water = lambda E: np.interp(E, [0.01,0.02,0.03,0.05,0.07,0.1,0.2,0.3,0.5,0.7,1.0,1.25,1.4], 
                                  [22.7,11.3,7.97,6.36,5.14,4.098,2.63,2.16,1.94,1.89,1.847,1.847,1.85])
Srad_water = lambda E: 0.0155*(E/1.0)**1.15  # approx
Scol_air = lambda E: np.interp(E, [0.01,0.02,0.03,0.05,0.07,0.1,0.2,0.3,0.5,0.7,1.0,1.25,1.4],
                                   [19.6,9.65,6.87,5.30,4.30,3.66,2.30,1.90,1.84,1.79,1.765,1.75,1.75])
Srad_air = lambda E: 0.01*(E/1.0)**1.05  # approximate

# Grid
Eg = np.arange(0.010, 1.401, 0.002)  # 696 points
dE = 0.002
nE = len(Eg)

# Precompute Moller
# dSigma/dm = n_e * dsigma/dE0
# dsigma/dE0 = (2π r_e^2 m c^2 / β^2) * [1/E0^2 - 1/(E0(T-E0)) + 1/(T-E0)^2] for E0<=T/2
# Prefactor: C = 2πr_e2 * m c^2 = 0.4987e-24 * 0.511 = 2.547e-25 cm^2 MeV
C = 0.4987e-24 * 0.511  # cm^2 MeV

def moler_dsde(T, E0):
    """ dSigma/dE0 per cm for incident T in MeV, secondary E0 (E0<=T/2), returns cm^-1 MeV^-1 """
    if T <= 0.011: return 0.0
    beta2 = 1.0 - 1.0/(1+T/mec2)**2
    if beta2 <= 0: return 0.0
    factor = C * n_e / beta2  # cm^-1 MeV?  C*ne = 2.547e-25*3.343e23 = 8.518e-2 cm^-1 MeV? Wait: C*ne: 2.547e-25 * 3.343e23 = 8.517e-2. units: cm^2 MeV * cm^-3 = cm^-1 MeV. / beta^2. OK.
    # factor = n_e * C / beta^2
    # The three-term formula
    term = 1/E0**2 - 1/(E0*(T-E0)) + 1/(T-E0)**2
    return factor * term  # cm^-1 MeV^-1

# Precompute S_rest on grid
def S_rest_water(E):
    if E <= 0.011: return Scol_water(E)
    # integral ∫_Δ^{E/2} E0 dσ/dE0 dE0
    dEint = 0.00025  # fine subgrid for integration
    E0vals = np.arange(0.010+1e-8, E/2, dEint)
    integ = np.sum(E0vals * moler_dsde(E, E0vals) * dEint)
    return Scol_water(E) - integ

# Precompute S_rest for all grid points
S_rest_arr = np.array([S_rest_water(e) for e in Eg])
Scol_arr = np.array([Scol_water(e) for e in Eg])
Srad_arr = np.array([Srad_water(e) for e in Eg])

# ---- q(E0) from photon spectra ----
ph_bins = np.array([0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4])
ph_c = (ph_bins[:-1]+ph_bins[1:])/2
prim_spec = np.load('prim_spec.npy')
scat_spec = np.load('scat_spec.npy')
tot_spec = prim_spec+scat_spec

q = np.zeros(nE)
# For each photon bin
rng = np.random.default_rng(0)
def build_q(spec):
    qq = np.zeros(nE)
    for i, Ec in enumerate(ph_c):
        wf = spec[i]
        if wf <= 0: continue
        # Compton
        nC = wf * mu_C(Ec) * (0.05)  # fluence in bin = spec[i] (dimensionless/Phi0). Multiply by width for per MeV? 
        # Wait: spec[i] is fluence in the bin divided by Phi0: dimensionless. fluence per MeV = spec[i]/0.05. Then interactions per cm^3 per phi0 = (spec[i]/0.05)*mu_C(Ec). So nC per cm^3 per phi0 in this bin (integrated) = spec[i]*mu_C(Ec). That's the total interactions in the bin per cm^3 per Phi0. 
        # For q(E0) per MeV per cm^3 per Phi0, we need d(N_interactions)/dE0: N_tot_C × f(E0). 
        nCt = spec[i] * mu_C(Ec)  # per cm^3 per Phi0
        if nCt > 0:
            # sample distribution of T_e
            ns = 40000
            Ee, _ = compton_samples(Ec, ns, rng)
            h, _ = np.histogram(Ee, bins=Eg, weights=np.full(ns, nCt/ns))
            qq += h / dE  # histogram gives sum per bin (integral over energy bin); divide by dE to get per MeV
        # PE
        nPE = spec[i] * mu_pe(Ec)
        if nPE > 0:
            T = Ec - 0.0005
            if T >= 0.01:
                bi = np.searchsorted(Eg, T, side='right')-1
                if bi >= 0 and bi < nE:
                    qq[bi] += nPE / dE
        # Pair: only the electron (positron excluded for L4)
        nPA = spec[i] * mu_pair(Ec)
        if nPA > 0:
            T = Ec - 1.022
            if T > 0:
                for _ in range(2000):  # sample splitting
                    f = rng.uniform(0,1)
                    Te = f*T
                    if Te >= 0.01:
                        bi = np.searchsorted(Eg, Te, side='right')-1
                        if bi >= 0 and bi < nE:
                            qq[bi] += nPA / 2000 / dE
    return qq

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

q = build_q(tot_spec)
print("q integrated:", q.sum()*dE, " (should be ~total interactions per cm^3 per Phi0)")
print("q shape:", q.shape, "Eg range:", Eg[0], Eg[-1])

# ---- Solve cascade ----
iter=0
Phi = np.zeros(nE)
while True:
    # compute q_sec
    qsec = np.zeros(nE)
    for ite, T in enumerate(Eg):
        if T < 0.012: continue
        # secondary E0 in [Δ, T/2]
        max_idx = min(np.searchsorted(Eg, T/2, side='right')-1, nE-1)
        for ie0 in range(np.searchsorted(Eg,0.010,side='right'), max_idx+1):
            E0 = Eg[ie0]
            if E0 <= 0.010 or E0 >= T/2-1e-12: continue
            dsdE = moler_dsde(T, E0)
            if dsdE > 0:
                qsec[ie0] += Phi[ite] * dsdE * dE
    # integral: ∫_E∞ (q+qsec) dE' = reversed cumsum * dE
    cum = np.cumsum((q+qsec)[::-1])[::-1] * dE  # integral from E to ∞
    newPhi = cum / S_rest_arr
    newPhi = np.where(S_rest_arr>0, newPhi, 0.0)
    diff = np.max(np.abs(newPhi-Phi))
    Phi = newPhi
    iter += 1
    print(f"  iter {iter}: max diff {diff:.6e}")
    if diff < 1e-6 and iter>2: break

# Primary-only (no deltas)
cum_p = np.cumsum(q[::-1])[::-1]*dE
Phi_p = cum_p / S_rest_arr
Phi_p = np.where(S_rest_arr>0, Phi_p, 0.0)

# Total electron fluence
Phi_total = Phi.copy()
print("Phi_total integrated:", Phi_total.sum()*dE)
print("Phi_primary integrated:", Phi_p.sum()*dE)

# Delta fraction
delta_f = (Phi_total.sum() - Phi_p.sum())/Phi_total.sum()
print("delta fraction:", delta_f)

# mean energy:
mean_e = np.sum(Eg * Phi_total * dE) / np.sum(Phi_total*dE)
print("mean electron energy:", mean_e)

# radiative/collision:
rad_col = np.sum(Phi_total * Srad_arr * dE) / np.sum(Phi_total * Scol_arr * dE)
print("radiative/collision:", rad_col)

# Spectrum per 25 bins
el_bins = np.array([0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.4])
Phi_spectrum = np.zeros(25)
for i in range(len(el_bins)-1):
    lo, hi = el_bins[i], el_bins[i+1]
    mask = (Eg>=lo)&(Eg<hi)
    Phi_spectrum[i] = np.sum(Phi_total[mask]*dE)
print("Phi_spectrum sum:", Phi_spectrum.sum())
print("Phi_spectrum:", np.round(Phi_spectrum,6))

# Also for primary:
Phi_prim_spectrum = np.zeros(25)
for i in range(len(el_bins)-1):
    lo, hi = el_bins[i], el_bins[i+1]
    mask = (Eg>=lo)&(Eg<hi)
    Phi_prim_spectrum[i] = np.sum(Phi_p[mask]*dE)

print("Phi_prim_spectrum sum:", Phi_prim_spectrum.sum())

np.savez('L4_data.npz', Eg=Eg, Phi=Phi_total, Phi_p=Phi_p, q=q,
         S_rest=S_rest_arr, Scol=Scol_arr, Srad=Srad_arr,
         Phi_spec=Phi_spectrum, Phi_prim_spec=Phi_prim_spectrum,
         delta_f=delta_f, mean_e=mean_e, rad_col=rad_col)
print("saved L4")