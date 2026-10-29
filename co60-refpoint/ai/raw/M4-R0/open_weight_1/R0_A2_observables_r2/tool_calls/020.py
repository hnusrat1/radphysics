import numpy as np, pickle
with open('mc_results.pkl','rb') as f:
    d = pickle.load(f)
depth28=d['depth28']; fine5=d['fine5']; score28_at5=d['score28_at5']; nh=d['nh']; depths=d['depths']
print("loaded. nh=",nh)
Om=110.25

# ---------- cross sections
def load_xcom():
    with open('/tmp/data/xcom_water.txt') as f:
        lines = [l for l in f if not l.startswith('#') and l.strip()]
    x = np.loadtxt(lines)
    return x[:,0], x[:,1], x[:,2], x[:,3], x[:,4], x[:,5], x[:,6]
E_x, mu_coh, mu_incoh, mu_pe, mu_pairn, mu_paire, mu_tot = load_xcom()
def mu_interp(E, col):
    return np.exp(np.interp(np.log(E), np.log(E_x), np.log(col)))
def mup(E):  # dictionary of mass attenuation coefficients cm2/g
    return dict(compton=mu_interp(E,mu_incoh), pe=mu_interp(E,mu_pe), pair=mu_interp(E,mu_pairn+mu_paire),
                rayleigh=mu_interp(E,mu_coh), total=mu_interp(E,mu_tot))
def load_estar():
    with open('/tmp/data/estar_water.txt') as f:
        lines = [l for l in f if not l.startswith('#') and l.strip()]
    est = np.loadtxt(lines)
    return est[:,0], est[:,1], est[:,2], est[:,3], est[:,4], est[:,5], est[:,6]
Ee, Scol, Srad, Stot, csda, yld, den = load_estar()
def interp_e(E, col): return np.exp(np.interp(np.log(E), np.log(Ee), np.log(col)))
def scol_w(E): return interp_e(E, Scol)
def srad_w(E): return interp_e(E, Srad)
def yield_w(E): return interp_e(E, yld)
def csda_w(E): return interp_e(E, csda)

mc2=0.51099895; re2=7.9408e-26
def kn_dsig(E0, cosT):
    a=E0/mc2; Ep=E0/(1.0+a*(1.0-cosT)); r=Ep/E0
    return 0.5*re2*r**2*(r+1.0/r-(1.0-cosT**2))
def kn_total(E0):
    th=np.linspace(1e-6,np.pi,20000)
    return np.trapezoid(kn_dsig(E0,np.cos(th))*2*np.pi*np.sin(th),th)
def compton_Ep(E0,cosT):
    a=E0/mc2; return E0/(1.0+a*(1.0-cosT))
def compton_mean_transfer(E0):
    th=np.linspace(1e-6,np.pi,4000); c=np.cos(th)
    w=kn_dsig(E0,c)*2*np.pi*np.sin(th)
    Tr = E0-compton_Ep(E0,c)
    return np.trapezoid(w*Tr,th)/np.trapezoid(w,th)

# recoil electron energy distribution p(T) (per MeV, normalized to 1) for photon energy E0
def compton_recoil_pdf(E0, Tgrid):
    th=np.linspace(1e-6,np.pi,20000)
    c=np.cos(th); st=np.sin(th)
    T = E0-compton_Ep(E0,c)
    dsdo = kn_dsig(E0,c)*2*np.pi*st        # dsigma/dtheta
    dTdth = np.gradient(T, th)
    # p(T) dT = (1/sig) dsdo/|dT/dth| dT
    pT = dsdo/np.abs(dTdth)
    sig = np.trapezoid(dsdo,th)
    pT = pT/sig
    # map onto Tgrid via histogram-like accumulation (since T monotonic decreasing)
    # T goes from ~Tmax (theta~0) down to 0 (theta~pi); reverse
    # interpolate pT at Tgrid
    # sort by T
    order=np.argsort(T)
    Ts=T[order]; Ps=pT[order]
    # p(T) dT: use interpolation, careful with normalization
    from numpy import interp
    # use histogram accumulation
    vals=np.interp(Tgrid, Ts, Ps, left=0, right=0)
    return vals  # per MeV

# test compton_recoil_pdf normalization and mean
E0=1.1732
Tg=np.linspace(0,1.1,1000)
p=compton_recoil_pdf(E0,Tg)
print("Compton pdf integral:", np.trapezoid(p,Tg))
print("mean T from pdf:", np.trapezoid(p*Tg,Tg), " vs direct:", compton_mean_transfer(E0))

# photon energy representation for scattered: fine5 edges
fine_edges=np.arange(0.01,1.405,0.005); nf=len(fine_edges)-1
fine_mid=(fine_edges[:-1]+fine_edges[1:])/2
df=fine_edges[1]-fine_edges[0]
# scattered fine fluence spectrum at z=5 (sum over cos): per Phi0 per MeV
sc_fine = fine5[:,:,1].sum(axis=1)  # per fine bin (fluence in bin)
sc_fine_perMeV = sc_fine/df

# primary line weights from MC:
prim28 = score28_at5[:,0,:].sum(axis=1)  # 28 bins primary
w1 = prim28[23]; w2 = prim28[26]
print("primary line weights:", w1, w2, "sum", w1+w2)

# ---------------- L1
prim_total = prim28.sum()
scat_total = sc_fine.sum()
total_ph = prim_total + scat_total
print("primary/Phi0", prim_total, "scattered/Phi0", scat_total, "total", total_ph)

# energy fluence
E_prim_lines = np.array([1.1732,1.3325]); w_lines=np.array([w1,w2])
Eprim_energy = (w_lines*E_prim_lines).sum()
Esc_energy = (sc_fine*fine_mid).sum()
print("primary energy fluence/Phi0:", Eprim_energy, "scattered energy fluence/Phi0:", Esc_energy)
print("scattered energy fraction:", Esc_energy/(Eprim_energy+Esc_energy))
print("scattered fluence fraction:", scat_total/total_ph)
mean_E_ph = (Eprim_energy+Esc_energy)/total_ph
print("mean photon energy (fluence-weighted):", mean_E_ph)

# L1 spectra 28 bins
pe28 = np.array([0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4])
sc28 = np.histogram(fine_mid, bins=pe28, weights=sc_fine)[0]
print("scattered 28-bin spectrum sum:", sc28.sum())
print("primary 28-bin nonzero:", prim28[prim28>0])

# angular
sc_ang = fine5[:,:,1].sum(axis=0)/scat_total
print("scattered angular frac:", sc_ang)
print("sum:", sc_ang.sum())