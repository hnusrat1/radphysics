import numpy as np, pickle

with open('mc_results.pkl','rb') as f:
    d=pickle.load(f)
fine5=d['fine5']; score28_at5=d['score28_at5']

def load_xcom():
    with open('/tmp/data/xcom_water.txt') as f:
        lines=[l for l in f if not l.startswith('#') and l.strip()]
    x=np.loadtxt(lines); return x[:,0],x[:,1],x[:,2],x[:,3],x[:,4],x[:,5],x[:,6]
E_x,mu_coh,mu_incoh,mu_pe,mu_pairn,mu_paire,mu_tot=load_xcom()
def mu_interp(E,col): return np.exp(np.interp(np.log(E),np.log(E_x),np.log(col)))
def mup(E): return dict(compton=mu_interp(E,mu_incoh), pe=mu_interp(E,mu_pe), pair=mu_interp(E,mu_pairn+mu_paire), rayleigh=mu_interp(E,mu_coh), total=mu_interp(E,mu_tot))
def load_estar():
    with open('/tmp/data/estar_water.txt') as f:
        lines=[l for l in f if not l.startswith('#') and l.strip()]
    est=np.loadtxt(lines); return est[:,0],est[:,1],est[:,2],est[:,3],est[:,4],est[:,5],est[:,6]
Ee,Scol,Srad,Stot,csda,yld,den=load_estar()
def interp_e(E,col): return np.exp(np.interp(np.log(E),np.log(Ee),np.log(col)))
def scol_w(E): return interp_e(E,Scol)
def srad_w(E): return interp_e(E,Srad)

mc2=0.51099895; re2=7.9408e-26
def kn_dsig(E0,cosT):
    a=E0/mc2; Ep=E0/(1.0+a*(1.0-cosT)); r=Ep/E0
    return 0.5*re2*r**2*(r+1.0/r-(1.0-cosT**2))
def kn_total(E0):
    th=np.linspace(1e-6,np.pi,20000); return np.trapezoid(kn_dsig(E0,np.cos(th))*2*np.pi*np.sin(th),th)
def compton_Ep(E0,cosT):
    a=E0/mc2; return E0/(1.0+a*(1.0-cosT))

fine_edges=np.arange(0.01,1.405,0.005); nf=len(fine_edges)-1
fine_mid=(fine_edges[:-1]+fine_edges[1:])/2; df=0.005
sc_fine=fine5[:,:,1].sum(axis=1)
w1=score28_at5[23,0,:].sum(); w2=score28_at5[26,0,:].sum(); wlines=np.array([w1,w2]); Elines=np.array([1.1732,1.3325])

# ---------- fine electron energy grid
Tgrid = np.concatenate([np.linspace(0.01,0.1,100,endpoint=False), np.linspace(0.1,1.4,600)])
# make it strictly increasing and cover bins; we'll use edges for binning from bin edges
Tg=Tgrid
nT=Tg.size

# production q_v on T grid (per volume per Phi0 per MeV)
q = np.zeros(nT)

def add_photon_fine(E, w, qarr):
    d=mup(E)
    # Compton recoil pdf on Tgrid
    th=np.linspace(1e-5,np.pi,20000); c=np.cos(th); st=np.sin(th)
    T=E-compton_Ep(E,c)
    dsdo=kn_dsig(E,c)*2*np.pi*st
    dTdth=np.gradient(T,th)
    pT=dsdo/np.abs(dTdth); sig=np.trapezoid(dsdo,th); pT=pT/sig
    order=np.argsort(T); Ts=T[order]; Ps=pT[order]
    pvals=np.interp(Tg,Ts,Ps,left=0,right=0)
    qarr += w*d['compton']*pvals
    # PE at T=E
    idx=np.argmin(np.abs(Tg-E))
    qarr[idx]+=w*d['pe']/ (Tg[1]-Tg[0]) if idx>0 else 0
    # pair: two at Tp
    Tp=(E-1.022)/2
    if Tp>0.01:
        idx=np.argmin(np.abs(Tg-Tp))
        qarr[idx]+=2*w*d['pair']/(Tg[1]-Tg[0])
    # note: delta representation - will integrate over bins

for E,w in zip(Elines,wlines):
    add_photon_fine(E,w,q)
for E,w in zip(fine_mid,sc_fine):
    if w>0: add_photon_fine(E,w,q)

print("q total integral:", np.trapezoid(q,Tg), "(expect ~0.085 per cm3)")

# units: q [1/(cm3 MeV)] per Phi0; per mass: /rho (=1) -> [1/(g MeV)]
q_m = q  # per g

# Solve CSDA equilibrium equation including Moller secondaries
S = scol_w(Tg)  # MeV cm2/g
# electron density per g for water
Ne_g = 3.343e23  # electrons/g

def moller_sigma(Tp_, T):
    # per electron dsigma/dT (cm2/MeV) at secondary energy T from primary Tp_
    gamma=1+Tp_/mc2
    beta2 = 1.0-1.0/gamma**2
    A=(2*np.pi*re2*mc2)/(beta2*Tp_)
    term=A*(1.0/T**2 + 1.0/(Tp_-T)**2 - (2*gamma-1)/gamma**2 * 1.0/(T*(Tp_-T)))
    return np.maximum(term,0.0)

# Q(T) = integral of q_m from T to Tmax
Q = np.array([np.trapezoid(q_m[Tg>=t], Tg[Tg>=t]) if (Tg>=t).any() else 0.0 for t in Tg])
# solve with delta: Phi[i] depends on Phi[j] for Tj>=2Ti
Phi=np.zeros(nT)
for i in range(nT-1,-1,-1):
    Ti=Tg[i]
    # secondary source: sum over j with Tj >= 2 Ti
    jmax = np.searchsorted(Tg, 2*Ti, side='left')
    if jmax < nT:
        Tj=Tg[jmax:]
        # dSigma/dT(Tj -> Ti) per g: (Ne_g)*moller_sigma
        sec = Phi[jmax:]*Ne_g*moller_sigma(Tj, Ti)
        src = np.trapezoid(sec, Tj)
    else:
        src=0.0
    Phi[i]=(Q[i]+src)/S[i]

# no-delta solution
Phi0=np.zeros(nT)
for i in range(nT-1,-1,-1):
    Phi0[i]=Q[i]/S[i]

# integrate into 25 electron bins
el_edges=np.array([0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.4])
def bin_fluence(Phi_):
    out=np.zeros(len(el_edges)-1)
    for i in range(len(el_edges)-1):
        lo,hi=el_edges[i],el_edges[i+1]
        sel=(Tg>=lo)&(Tg<hi)
        out[i]=np.trapezoid(Phi_[sel],Tg[sel]) if sel.sum()>1 else 0
    return out
spec_tot=bin_fluence(Phi)
spec_nod=bin_fluence(Phi0)
Phi_tot_int=spec_tot.sum(); Phi_nod_int=spec_nod.sum()
print("electron fluence per Phi0 (with delta):", Phi_tot_int)
print("electron fluence per Phi0 (no delta):", Phi_nod_int)
print("delta fraction of fluence:", (Phi_tot_int-Phi_nod_int)/Phi_tot_int)
meanE=np.sum(spec_tot*(el_edges[:-1]+el_edges[1:])/2)/Phi_tot_int
print("mean electron energy (fluence-weighted):", meanE)
# radiative/collision
Srad=srad_w(Tg)
num=np.trapezoid(Phi*Srad,Tg); den=np.trapezoid(Phi*Scol if False else Phi*scol_w(Tg),Tg)
# careful: Scol already = scol_w; use directly
den2=np.trapezoid(Phi*Scol,Tg)
print("radiative/collision:", num/den2)
print("spec_tot nonzero in bins:", np.round(spec_tot,6))

np.save('L4_spec.npy', spec_tot)
np.save('L4_Phi.npy', Phi)
np.save('L4_Tg.npy', Tg)