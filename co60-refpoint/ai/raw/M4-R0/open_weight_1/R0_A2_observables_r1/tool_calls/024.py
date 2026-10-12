import numpy as np, math
A = np.load('/tmp/mc_tallies_5e6.npy', allow_pickle=True)
flu_p,flu_s,inter_a,inter_p,inter_s,trans,e_all,e_prim,e_cos,e_sum,e_sum_prim,slab_src,Ntot = A
Ntot=int(Ntot)
V_cyl=math.pi*0.2; PHI0=1/10.5**2
ph_edges=np.array([0,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,1,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4])
el_edges=np.array([0.01,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,1,1.05,1.1,1.15,1.2,1.4])
cos_edges=np.linspace(-1,1,11)
Emid=(ph_edges[:-1]+ph_edges[1:])/2
fp = flu_p.sum(1)/Ntot/V_cyl/PHI0
fs = flu_s.sum(1)/Ntot/V_cyl/PHI0
ftot=fp+fs
mc2=0.5109989461
def KNf(alpha,cos):
    x=1+alpha*(1-cos); r=1/x; return r*r*(r+x-(1-cos*cos))
def sample_kn_electrons(E, n, rng):
    # sample Compton electron energies and cos(e) for photon energy E (array)
    # returns Te array, cose array
    cos_s=np.zeros(n); Te=np.zeros(n); cose=np.zeros(n); done=np.zeros(n,bool); idx=np.arange(n)
    alpha=E/mc2
    while True:
        sub=idx[~done]
        if sub.size==0: break
        cs=rng.uniform(-1,1,sub.size)
        f=KNf(alpha,cs); acc=rng.random(sub.size)<(f/2.0)
        if acc.any():
            ai=sub[acc]; cos_s[ai]=cs[acc]
            Ep=E/(1+alpha*(1-cs[acc])); Te[ai]=E-Ep
            cose[ai]=(E-Ep*cs[acc])/np.sqrt(E**2+Ep**2-2*E*Ep*cs[acc]); done[ai]=True
    return Te,cose
rng=np.random.default_rng(42)
# L3 analytic from photon spectra
nper_bin=20000
e_hist_all=np.zeros(25); e_hist_prim=np.zeros(25); cos_hist_all=np.zeros(10); cos_hist_prim=np.zeros(10)
e_sum_an=0.0; e_cnt_an=0; e_sum_an_p=0.0; e_cnt_an_p=0
# primary lines
for bi in (23,26):
    E=Emid[bi]; w=fp[bi]*mui_(E:=E) if False else fp[bi]
    # actual primary photon energy: line energies
    Eline = 1.1732 if bi==23 else 1.3325
    w = fp[bi]  # fluence per Phi0 in bin
    n=int(w*mu_inc(Eline)*nper_bin*1000)  # scale to get counts ~ proportional to mu*fluence
    if n<=0: continue
    Te,cose=sample_kn_electrons(np.array([Eline])*np.ones(n), n, rng)
    # weight per electron: w*mu_inc(Eline)/n
    wt = w*mu_inc(Eline)/n
    eb=np.clip(np.searchsorted(el_edges,Te,side='right')-1,0,24); cb=np.clip(np.searchsorted(cos_edges,cose,side='right')-1,0,9)
    np.add.at(e_hist_all, eb, wt); np.add.at(e_hist_prim, eb, wt)
    np.add.at(cos_hist_all, cb, wt); np.add.at(cos_hist_prim, cb, wt)
    e_sum_an += (Te*wt).sum(); e_cnt_an += wt.sum(); e_sum_an_p += (Te*wt).sum(); e_cnt_an_p += wt.sum()
    # photoelectric primary (tiny)
    wt_pe = w*mu_pe_(Eline)
    if wt_pe>0:
        eb=el_bin(Eline-0.0005); np.add.at(e_hist_all, eb, wt_pe); np.add.at(e_hist_prim, eb, wt_pe)
        cb=9; np.add.at(cos_hist_all, cb, wt_pe); np.add.at(cos_hist_prim, cb, wt_pe)
        e_sum_an += (Eline-0.0005)*wt_pe; e_cnt_an += wt_pe; e_sum_an_p += (Eline-0.0005)*wt_pe; e_cnt_an_p += wt_pe
# scattered photons
for bi,E in enumerate(Emid):
    w=fs[bi]
    if w<=0 or E<=0.011: continue
    n=int(w*mu_inc(E)*nper_bin)
    if n<=0: continue
    Te,cose=sample_kn_electrons(np.full(n,E), n, rng)
    wt=w*mu_inc(E)/n
    eb=np.clip(np.searchsorted(el_edges,Te,side='right')-1,0,24); cb=np.clip(np.searchsorted(cos_edges,cose,side='right')-1,0,9)
    np.add.at(e_hist_all, eb, wt); np.add.at(cos_hist_all, cb, wt)
    e_sum_an += (Te*wt).sum(); e_cnt_an += wt.sum()
    wt_pe=w*mu_pe_(E)
    if wt_pe>0 and E>0.0105:
        eb=el_bin(E-0.0005); np.add.at(e_hist_all,eb,wt_pe); cb=9; np.add.at(cos_hist_all,cb,wt_pe)
        e_sum_an += (E-0.0005)*wt_pe; e_cnt_an += wt_pe
print('L3 mean initial E all (analytic):', e_sum_an/e_cnt_an)
print('L3 mean initial E prim (analytic):', e_sum_an_p/e_cnt_an_p)
print('L3 spectrum all (normalized):', e_hist_all/e_hist_all.sum())
print('L3 spectrum prim (normalized):', e_hist_prim/e_hist_prim.sum())
print('L3 direction all:', cos_hist_all/cos_hist_all.sum())
print('MC direct: mean all', e_sum/e_all.sum(), 'prim', e_sum_prim/e_prim.sum())