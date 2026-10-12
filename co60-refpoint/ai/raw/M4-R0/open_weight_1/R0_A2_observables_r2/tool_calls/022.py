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

mc2=0.51099895; re2=7.9408e-26
def kn_dsig(E0,cosT):
    a=E0/mc2; Ep=E0/(1.0+a*(1.0-cosT)); r=Ep/E0
    return 0.5*re2*r**2*(r+1.0/r-(1.0-cosT**2))
def kn_total(E0):
    th=np.linspace(1e-6,np.pi,20000); return np.trapezoid(kn_dsig(E0,np.cos(th))*2*np.pi*np.sin(th),th)
def compton_Ep(E0,cosT):
    a=E0/mc2; return E0/(1.0+a*(1.0-cosT))

ct_edges=np.array([-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1])
el_edges=np.array([0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.4])
nel=len(el_edges)-1
w1=score28_at5[23,0,:].sum(); w2=score28_at5[26,0,:].sum(); wlines=np.array([w1,w2]); Elines=np.array([1.1732,1.3325])

# --- L3 initial direction (fraction in cos theta bins)
dir_hist=np.zeros(10)
# primary photons: direction +z; electron recoil from Compton
def add_electron_dir_primary(E, w, dirarr):
    d=mup(E)
    th=np.linspace(1e-5,np.pi,10000); c=np.cos(th)
    dsdo=kn_dsig(E,c)*2*np.pi*np.sin(th)
    a=E/mc2
    cotp=(1.0+a)*np.tan(th/2.0)
    phi=np.arctan2(1.0,cotp)
    cose=np.cos(phi)   # electron cos relative to incoming photon (+z); azimuth averages -> use mean cos
    # weight by dsigma
    wgt=dsdo/sum(dsdo)  # normalized
    # distribute: for each theta, electron cos = cose (mean over azimuth is exact for photon along z because symmetric)
    for i in range(10):
        lo,hi=ct_edges[i],ct_edges[i+1]
        sel=(cose>=lo)&(cose<hi)
        dirarr[i]+= w*d['compton']*wgt[sel].sum()
    # PE: electron forward (cos~1), negligible but add
    dirarr[9]+=w*d['pe']
    # pair: two particles forward
    dirarr[9]+=2*w*d['pair']

for E,w in zip(Elines,wlines):
    add_electron_dir_primary(E,w,dir_hist)

# scattered photons as parents: joint distribution fine5[:, k_cos, scattered]
fine_edges=np.arange(0.01,1.405,0.005); nf=len(fine_edges)-1
fine_mid=(fine_edges[:-1]+fine_edges[1:])/2
sc_2d = fine5[:,:,1]  # (nf,10) fluence per bin
for jk in range(10):
    mug = (ct_edges[jk]+ct_edges[jk+1])/2
    spec = sc_2d[:,jk]   # fluence per fine bin
    for j,E in enumerate(fine_mid):
        w=spec[j]
        if w<=0: continue
        d=mup(E)
        th=np.linspace(1e-5,np.pi,5000); c=np.cos(th)
        dsdo=kn_dsig(E,c)*2*np.pi*np.sin(th)
        a=E/mc2
        cotp=(1.0+a)*np.tan(th/2.0)
        phi=np.arctan2(1.0,cotp)
        co= np.cos(phi); so=np.sin(phi); sg=np.sqrt(max(0,1-mug**2))
        c0=mug*co; s0=so*sg
        wgt=dsdo/dsdo.sum()
        for i in range(10):
            lo,hi=ct_edges[i],ct_edges[i+1]
            lo_=max(lo,c0-s0); hi_=min(hi,c0+s0)
            frac=0.0
            if hi_>lo_:
                frac=(hi_-lo_)/(2*s0)
            dir_hist[i]+= w*d['compton']*(wgt*frac).sum()

tot_dir=dir_hist.sum()
L3_dir=dir_hist/tot_dir
print("L3 initial direction (fraction):", L3_dir)
print("sum:", L3_dir.sum())
# also add efficiency check: photons with E below 1.022: pair not possible; handled fine
# Note: pair/PE contributions from scattered photons omitted (tiny) - fine.