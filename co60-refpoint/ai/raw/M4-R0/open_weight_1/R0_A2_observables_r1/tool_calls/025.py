import numpy as np, math
A = np.load('/tmp/mc_tallies_5e6.npy', allow_pickle=True)
flu_p,flu_s,inter_a,inter_p,inter_s,trans,e_all,e_prim,e_cos,e_sum,e_sum_prim,slab_src,Ntot = A
Ntot=int(Ntot); V_cyl=math.pi*0.2; PHI0=1/10.5**2
ph_edges=np.array([0,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,1,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4])
el_edges=np.array([0.01,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,1,1.05,1.1,1.15,1.2,1.4])
cos_edges=np.linspace(-1,1,11)
Emid=(ph_edges[:-1]+ph_edges[1:])/2
fp = flu_p.sum(1)/Ntot/V_cyl/PHI0
fs = flu_s.sum(1)/Ntot/V_cyl/PHI0
mc2=0.5109989461
d = np.loadtxt('/tmp/data/xcom_water.txt', comments='#')
E_x=d[:,0]; mu_co=d[:,1]; mu_in=d[:,2]; mu_pe=d[:,3]; mu_pn=d[:,4]; mu_pef=d[:,5]; mu_tot=d[:,6]
def interp(x,xp,yp): return np.interp(x,xp,yp)
def mu_inc(E): return interp(E,E_x,mu_in)
def mu_pe_(E): return interp(E,E_x,mu_pe)
def KNf(alpha,cos):
    x=1+alpha*(1-cos); r=1/x; return r*r*(r+x-(1-cos*cos))
def sample_kn(Earr, n, rng):
    cos_s=np.zeros(n); Te=np.zeros(n); cose=np.zeros(n); done=np.zeros(n,bool); idx=np.arange(n)
    while True:
        sub=idx[~done]
        if sub.size==0: break
        E=Earr[0]
        alpha=E/mc2; cs=rng.uniform(-1,1,sub.size)
        f=KNf(alpha,cs); acc=rng.random(sub.size)<(f/2.0)
        if acc.any():
            ai=sub[acc]; cos_s[ai]=cs[acc]
            Ep=E/(1+alpha*(1-cs[acc])); Te[ai]=E-Ep
            cose[ai]=(E-Ep*cs[acc])/np.sqrt(E**2+Ep**2-2*E*Ep*cs[acc]); done[ai]=True
    return Te,cose
rng=np.random.default_rng(42)
e_hist_all=np.zeros(25); e_hist_prim=np.zeros(25); cos_hist_all=np.zeros(10); cos_hist_prim=np.zeros(10)
e_sum_an=0.0; e_cnt_an=0; e_sum_an_p=0.0; e_cnt_an_p=0
# primary lines
for Eline,bi in ((1.1732,23),(1.3325,26)):
    w=fp[bi]
    mu=mu_inc(Eline)
    n=int(w*mu*2.0e5)
    if n>0:
        Te,cose=sample_kn(np.array([Eline]),n,rng)
        wt=w*mu/n
        eb=np.clip(np.searchsorted(el_edges,Te,side='right')-1,0,24); cb=np.clip(np.searchsorted(cos_edges,cose,side='right')-1,0,9)
        np.add.at(e_hist_all, eb, wt); np.add.at(e_hist_prim, eb, wt)
        np.add.at(cos_hist_all, cb, wt); np.add.at(cos_hist_prim, cb, wt)
        e_sum_an += (Te*wt).sum(); e_cnt_an += wt.sum(); e_sum_an_p += (Te*wt).sum(); e_cnt_an_p += wt.sum()
    wt_pe=w*mu_pe_(Eline)
    eb=el_bin(Eline-0.0005)
    np.add.at(e_hist_all,eb,wt_pe); np.add.at(e_hist_prim,eb,wt_pe); np.add.at(cos_hist_all,9,wt_pe); np.add.at(cos_hist_prim,9,wt_pe)
    e_sum_an += (Eline-0.0005)*wt_pe; e_cnt_an += wt_pe; e_sum_an_p += (Eline-0.0005)*wt_pe; e_cnt_an_p += wt_pe
# scattered photons
for bi,E in enumerate(Emid):
    w=fs[bi]
    if w<=0 or E<=0.011: continue
    mu=mu_inc(E); n=int(w*mu*2.0e5)
    if n>0:
        Te,cose=sample_kn(np.array([E]),n,rng)
        wt=w*mu/n
        eb=np.clip(np.searchsorted(el_edges,Te,side='right')-1,0,24); cb=np.clip(np.searchsorted(cos_edges,cose,side='right')-1,0,9)
        np.add.at(e_hist_all, eb, wt); np.add.at(cos_hist_all, cb, wt)
        e_sum_an += (Te*wt).sum(); e_cnt_an += wt.sum()
    wt_pe=w*mu_pe_(E)
    if wt_pe>0 and E>0.0105:
        eb=el_bin(E-0.0005); np.add.at(e_hist_all,eb,wt_pe); np.add.at(cos_hist_all,9,wt_pe)
        e_sum_an += (E-0.0005)*wt_pe; e_cnt_an += wt_pe
print('L3_initial_spectrum:', e_hist_all/e_hist_all.sum())
print('L3_initial_spectrum_parent_primary:', e_hist_prim/e_hist_prim.sum())
print('L3_mean_initial_energy_MeV:', e_sum_an/e_cnt_an)
print('L3_mean_initial_energy_parent_primary_MeV:', e_sum_an_p/e_cnt_an_p)
print('L3_initial_direction:', cos_hist_all/cos_hist_all.sum())
print('MC check means:', e_sum/e_all.sum(), e_sum_prim/e_prim.sum())

# ---------------- L4 angular model ----------------
fine=np.arange(0.01,1.401,0.005)
# recompute Phi (all generations) quickly from analytic q̃ (from earlier cell logic, compact)
re2=2.8179403262e-13**2; Ne=3.343e23
qr=np.zeros_like(fine)
for i,E in enumerate(Emid):
    if E<=0.011: continue
    w=fp[i]+fs[i]
    if w<=0: continue
    alpha=E/mc2; Epmax=E/(1+2*alpha); E0s=E-fine
    valid=(E0s>=Epmax)&(E0s<=E)&(fine>0)&(fine<E)
    if not valid.any(): continue
    xg=E/E0s[valid]; cosg=1-(xg-1)/alpha; sin2=1-cosg**2
    dsdc=math.pi*re2*(E0s[valid]/E)**2*((E0s[valid]/E)+(E/E0s[valid])-sin2)
    dsdE0=dsdc*(E/(alpha*E0s[valid]**2)); qr[valid]+=w*Ne*dsdE0
    if mu_pe_(E)>0 and E>0.0005:
        e0=E-0.0005
        if e0>=0.01: qr[np.argmin(np.abs(fine-e0))]+=w*mu_pe_(E)/0.005
s2=np.loadtxt('/tmp/data/estar_water.txt', comments='#'); Ee=s2[:,0]; Scol=s2[:,1]
Phi=np.zeros_like(fine)
def moller_n(T,W):
    tau=T/mc2; beta2=tau*(tau+2)/(tau+1)**2; Wmax=T/2.0
    out=np.zeros_like(W); v=(W>0.009)&(W<Wmax-1e-9)
    if v.any():
        Wv=W[v]; out[v]=(2.5504e-25*Ne/beta2)*(1/Wv**2)*(1-beta2*Wv/Wmax+(Wv/Wmax)**2/2)
    return out
Qn_=0.0
for i in range(len(fine)-1,-1,-1):
    E0=fine[i]; s=0.0
    for j in range(len(fine)-1,i,-1):
        if E0<fine[j]/2 and Phi[j]>0: s+=Phi[j]*moller_n(fine[j],E0)*0.005
    Q=(Qn_ if i+1<len(fine) else 0)+(qr[i]+s)*0.005; Qn_=Q
    Phi[i]=Q/interp(fine[i],Ee,Scol)
p0=cos_hist_all/cos_hist_all.sum()
ang=np.zeros(10)
for i,E in enumerate(fine):
    w_iso=max(0,min(1,1-E/1.3))
    ang += Phi[i]*( (1-w_iso)*p0 + w_iso*0.1 )*0.005
ang/=ang.sum()
print('L4_angular:', ang)
print('L4 fluence integral:', np.trapezoid(Phi,fine))