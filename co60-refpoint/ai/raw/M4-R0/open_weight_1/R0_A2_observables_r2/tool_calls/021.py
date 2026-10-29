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
def yield_w(E): return interp_e(E,yld)

mc2=0.51099895; re2=7.9408e-26
def kn_dsig(E0,cosT):
    a=E0/mc2; Ep=E0/(1.0+a*(1.0-cosT)); r=Ep/E0
    return 0.5*re2*r**2*(r+1.0/r-(1.0-cosT**2))
def kn_total(E0):
    th=np.linspace(1e-6,np.pi,20000); return np.trapezoid(kn_dsig(E0,np.cos(th))*2*np.pi*np.sin(th),th)
def compton_Ep(E0,cosT):
    a=E0/mc2; return E0/(1.0+a*(1.0-cosT))
def compton_mean_transfer(E0):
    th=np.linspace(1e-6,np.pi,4000); c=np.cos(th)
    w=kn_dsig(E0,c)*2*np.pi*np.sin(th); Tr=E0-compton_Ep(E0,c)
    return np.trapezoid(w*Tr,th)/np.trapezoid(w,th)

# Compton recoil tables for primary lines and scattered fine energies, plus electron angle
def compton_tables(E0, Tg):
    th=np.linspace(1e-5,np.pi,20000); c=np.cos(th); st=np.sin(th)
    T=E0-compton_Ep(E0,c)
    dsdo=kn_dsig(E0,c)*2*np.pi*st
    dTdth=np.gradient(T,th)
    pT=dsdo/np.abs(dTdth); sig=np.trapezoid(dsdo,th); pT=pT/sig
    order=np.argsort(T); Ts=T[order]; Ps=pT[order]
    # electron angle phi_e vs photon theta
    a=E0/mc2
    cotp=(1.0+a)*np.tan(th/2.0)
    phi=np.arctan2(1.0,cotp)
    cosphi=np.cos(phi)
    # p(T) on grid
    pvals=np.interp(Tg,Ts,Ps,left=0,right=0)
    return pvals, Ts, Ps, th, cosphi, T

# electron bin edges
el_edges=np.array([0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.4])
nel=len(el_edges)-1

# fine photon energy grid for scattered
fine_edges=np.arange(0.01,1.405,0.005); nf=len(fine_edges)-1
fine_mid=(fine_edges[:-1]+fine_edges[1:])/2; df=fine_edges[1]-fine_edges[0]
sc_fine=fine5[:,:,1].sum(axis=1)          # fluence per fine bin per Phi0
sc_fine_perMeV=sc_fine/df

w1=score28_at5[23,0,:].sum(); w2=score28_at5[26,0,:].sum()
wlines=np.array([w1,w2]); Elines=np.array([1.1732,1.3325])

# ---------------- L2 interaction fractions
# total interaction integrals
def inter_rate(Earr, warr):
    # returns dict rates
    tot=0; comp=0; pe=0; pair=0; ray=0
    for E,w in zip(Earr,warr):
        d=mup(E); tot+=w*d['total']; comp+=w*d['compton']; pe+=w*d['pe']; pair+=w*d['pair']; ray+=w*d['rayleigh']
    return dict(total=tot,compton=comp,pe=pe,pair=pair,rayleigh=ray)

prim_rate = inter_rate(Elines,wlines)
scat_rate = inter_rate(fine_mid, sc_fine)
tot_rate = {k: prim_rate[k]+scat_rate[k] for k in prim_rate}

L2_int = {k: tot_rate[k]/tot_rate['total'] for k in ['compton','pe','pair','rayleigh']}
L2_int_prim = {k: prim_rate[k]/prim_rate['total'] for k in ['compton','pe','pair','rayleigh']}
L2_int_scat = {k: scat_rate[k]/scat_rate['total'] for k in ['compton','pe','pair','rayleigh']}
L2_scat_share = scat_rate['total']/tot_rate['total']
print("L2 interaction fractions:", L2_int)
print("  prim:", L2_int_prim, "scat:", L2_int_scat)
print("  scattered share of interactions:", L2_scat_share)

# energy transfer weighted
def transfer(E, proc):
    if proc=='compton': return compton_mean_transfer(E)
    if proc=='pe': return E-0.0005
    if proc=='pair': return max(E-1.022,0)
    if proc=='rayleigh': return 0.0
def energy_transfer_rates(Earr,warr):
    res={'compton':0.,'pe':0.,'pair':0.,'rayleigh':0.}
    for E,w in zip(Earr,warr):
        d=mup(E)
        res['compton']+=w*d['compton']*transfer(E,'compton')
        res['pe']+=w*d['pe']*transfer(E,'pe')
        res['pair']+=w*d['pair']*transfer(E,'pair')
        res['rayleigh']+=0
    return res
et_tot = {k: energy_transfer_rates(Elines,wlines)[k]+energy_transfer_rates(fine_mid,sc_fine)[k] for k in ['compton','pe','pair','rayleigh']}
s=sum(et_tot.values())
L2_et = {k: et_tot[k]/s for k in et_tot}
print("L2 energy transfer fractions:", L2_et)

# ---------------- L3 initial electron production
# Accumulate q_v per electron bin (per unit volume per Phi0)
q_all=np.zeros(nel); q_prim=np.zeros(nel)
# helper: add a photon energy E with weight w (fluence per Phi0) and flag include
def add_photon(E, w, q):
    if w<=0: return
    d=mup(E)
    # Compton
    Tg=np.linspace(1e-4, E*2*(E/mc2)/(1+2*(E/mc2)), 4000)  # up to Tmax
    pT,Ts,Ps,th,cosphi,T = compton_tables(E, Tg)
    # accumulate into electron bins
    for i in range(nel):
        lo,hi=el_edges[i],el_edges[i+1]
        # integral of pT over [lo,hi] within [0,Tmax]
        sel=(Tg>=lo)&(Tg<hi)
        val=np.trapezoid(pT[sel],Tg[sel]) if sel.sum()>1 else 0
        q[i]+=w*d['compton']*val
    # photoelectric (T=E)
    idx=np.clip(np.searchsorted(el_edges,E,side='right')-1,0,nel-1)
    q[idx]+=w*d['pe']
    # pair: two particles at Tp
    Tp=(E-1.022)/2
    if Tp>0:
        idx2=np.clip(np.searchsorted(el_edges,Tp,side='right')-1,0,nel-1)
        q[idx2]+=2*w*d['pair']

for E,w in zip(Elines,wlines):
    add_photon(E,w,q_all); add_photon(E,w,q_prim)
for E,w in zip(fine_mid,sc_fine):
    add_photon(E,w,q_all)

print("total particles per volume per Phi0:", q_all.sum())
print("primary particles:", q_prim.sum())
L3_init = q_all/q_all.sum()
L3_init_prim = q_prim/q_prim.sum()
mean_T_all = np.sum(q_all*(el_edges[:-1]+el_edges[1:])/2)/q_all.sum()
mean_T_prim = np.sum(q_prim*(el_edges[:-1]+el_edges[1:])/2)/q_prim.sum()
print("L3 mean initial energy all:", mean_T_all, "prim:", mean_T_prim)
print("L3 init spectrum sum:", L3_init.sum(), L3_init_prim.sum())
print("L3 bins nonzero counts:", np.sum(L3_init>0))