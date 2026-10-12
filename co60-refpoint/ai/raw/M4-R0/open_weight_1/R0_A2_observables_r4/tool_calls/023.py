import numpy as np, time
from scipy.interpolate import interp1d

# ---------- electron data & physics ----------
el=np.loadtxt('/tmp/data/estar_water.txt',comments='#')
Ee=el[:,0]; Sc_=el[:,1]; Sr_=el[:,2]; St_=el[:,3]; R_=el[:,4]; Y_=el[:,5]
def Scol(E): return np.interp(E,Ee,Sc_)
def Srad(E): return np.interp(E,Ee,Sr_)
def Yrad(E): return np.interp(E,Ee,Y_)
mec2=0.5109989461
r0=2.8179403262e-13
Ne_per_g=3.34283e23   # electrons/g for water
Delta=0.010  # 10 keV cutoff

# precompute restricted stopping and delta production tables on energy grid
Egrid=np.logspace(np.log10(0.0105),np.log10(1.5),600)
from scipy.integrate import quad
def moler_pref(E):
    gam=1+E/mec2; b2=1-1/(gam*gam); return 2*np.pi*r0*r0*mec2/b2
def dsigma(E,T):
    gam=1+E/mec2; A=moler_pref(E)
    return A*(1/T**2+1/(E-T)**2-(2*gam-1)/(gam*gam)/(T*(E-T)))
S_rest_tab=np.zeros_like(Egrid); Ndel_tab=np.zeros_like(Egrid); Sdel_tab=np.zeros_like(Egrid)
for j,E in enumerate(Egrid):
    if E<=2*Delta:
        S_rest_tab[j]=Scol(E); Ndel_tab[j]=0; Sdel_tab[j]=0
    else:
        sd=quad(lambda T: T*dsigma(E,T),Delta,E/2)[0]
        nd=quad(lambda T: dsigma(E,T),Delta,E/2)[0]
        Sdel_tab[j]=Ne_per_g*sd; Ndel_tab[j]=Ne_per_g*nd
        S_rest_tab[j]=max(Scol(E)-Ne_per_g*sd,0.05)
def Srest(E): return np.interp(E,Egrid,S_rest_tab)
def Scol_f(E): return np.interp(E,Ee,Sc_)
def Srad_f(E): return np.interp(E,Ee,Sr_)
def Sdel(E): return np.interp(E,Egrid,Sdel_tab)
def Ndel(E): return np.interp(E,Egrid,Ndel_tab)

def sample_delta_energy(E,n,rng):
    # sample T in [Delta, E/2] from Møller dσ via rejection on 1/T^2 proposal
    Tmax=E/2; out=np.empty(n); got=np.zeros(n,bool)
    invD=1/Delta
    A=moler_pref(E); gam=1+E/mec2
    def f(T): return A*(1/T**2+1/(E-T)**2-(2*gam-1)/(gam*gam)/(T*(E-T)))
    # max of f*(T^2)/A... proposal g(T)=1/(T^2 * ln(Tmax/Delta))? use T distribution 1/T^2 on [Delta,Tmax]
    # normalized proposal p(T)= (1/T^2)/(1/Delta-1/Tmax)
    norm=1/Delta-1/Tmax
    while not got.all():
        u=rng.uniform(0,1,n)
        T=1.0/(1/Delta - u*norm)   # inverse CDF of p ~ 1/T^2
        # acceptance: f(T)/(A*1/T^2) = [1 + T^2/(E-T)^2 - (2gam-1)/(gam^2) * T/(E-T)]
        w=1+(T*T)/(E-T)**2 - (2*gam-1)/(gam*gam)*T/(E-T)
        # w max at T=Delta: w(T=Delta) ~ 1+ (D/E)^2 ~1 ; at T→E/2, (T/(E-T))^2 ~1 and last term -(2gam-1)/gam^2*1 ~ -0.47 → 0.53. Actually max might be 1.
        acc=rng.uniform(0,1,n)*1.02 < w
        new=(~got)&acc
        out[new]=T[new]; got[new]=True
    return out

def highland_sigma(E, ds):
    # theta rms for step ds (cm); X0 water 36.08 cm
    X0=36.08
    pc=np.sqrt(E*(E+2*mec2)); Etot=E+mec2; beta=pc/Etot
    betacp=pc*beta
    if betacp<1e-6: return 0.0
    x=ds/X0
    t0=(13.6/betacp)*np.sqrt(x)*(1+0.038*np.log(max(x,1e-6)))
    return min(t0,1.5)

# e- bin edges and cos bins
eb_edges=np.array([0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.4])
ct_edges=np.array([-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1])
nEB=len(eb_edges)-1; nCT=len(ct_edges)-1

def cyl_chord(p0,p1,R,zc,hh):
    d=p1-p0; L=np.sqrt(np.sum(d*d,axis=1))
    dx=d[:,0]; dy=d[:,1]; dz=d[:,2]; x0=p0[:,0]; y0=p0[:,1]; z0=p0[:,2]
    a=dx*dx+dy*dy; b=x0*dx+y0*dy; c=x0*x0+y0*y0-R*R
    big=a>1e-16
    sq=np.sqrt(np.maximum(b*b-a*c,0))
    t1=(-b-sq)/np.maximum(a,1e-30); t2=(-b+sq)/np.maximum(a,1e-30)
    lo=np.minimum(t1,t2); hi=np.maximum(t1,t2)
    lo[~big]=-1e9; hi[~big]=1e9
    outside=(~big)&(x0*x0+y0*y0>=R*R)
    invdz=np.where(np.abs(dz)>1e-12,1.0/dz,0.0)
    z1=(zc-hh-z0)*invdz; z2=(zc+hh-z0)*invdz
    zlo=np.minimum(z1,z2); zhi=np.maximum(z1,z2)
    dz0=np.abs(dz)<1e-12
    zlo[dz0]=0.; zhi[dz0]=1.
    bad=dz0&(np.abs(z0-zc)>hh); zlo[bad]=1.; zhi[bad]=0.
    lo=np.maximum(np.maximum(lo,zlo),0); hi=np.minimum(np.minimum(hi,zhi),1)
    chord=L*np.maximum(0,hi-lo); chord[outside]=0
    return chord

# ---------- electron source sampling from photon MC data ----------
def build_source_fns(photon_res, Omega105=109.97517633116931):
    N=photon_res['N']; slab_prim=photon_res['slab_prim']; slab_sec=photon_res['slab_sec']
    zc=photon_res['zc_slabs']; hh=photon_res['hh_slab']; Vslab=np.pi*1.0*2*hh
    # fluence per Phi0 per line at slab centers
    f1=slab_prim[0]/N/Vslab*Omega105; f2=slab_prim[1]/N/Vslab*Omega105
    fs=slab_sec/N/Vslab*Omega105   # (nz,28)
    z_fn1=interp1d(zc,f1,kind='linear',bounds_error=False,fill_value=(f1[0],f1[-1]))
    z_fn2=interp1d(zc,f2,kind='linear',bounds_error=False,fill_value=(f2[0],f2[-1]))
    def fs_fn(z):
        # interpolate each bin
        zz=np.atleast_1d(np.asarray(z,float))
        out=np.empty((len(zz),28))
        for k in range(28):
            out[:,k]=np.interp(zz,zc,fs[:,k])
        return out if np.ndim(z)>0 else out[0]
    return dict(zc=zc,f1=f1,f2=f2,fs=fs,z_fn1=z_fn1,z_fn2=z_fn2,fs_fn=fs_fn)

# photon cross-section lookups (need for source creation)
ph=np.loadtxt('/tmp/data/xcom_water.txt',comments='#')
Eph=ph[:,0]; coh=ph[:,1]; inc=ph[:,2]; pe=ph[:,3]; pn=ph[:,4]; pel=ph[:,5]; tot=ph[:,6]
Elg=np.logspace(np.log10(0.01),np.log10(2.0),2500)
def interp_log(x,y,xq):
    ok=(y>0)&(x>0); fi=interp1d(np.log(x[ok]),np.log(y[ok]),kind='linear',bounds_error=False,fill_value='extrapolate')
    return np.exp(fi(np.log(xq)))
COH=interp_log(Eph,coh,Elg); INC_i=interp_log(Eph,inc,Elg); PE=interp_log(Eph,pe,Elg); PN0=interp_log(Eph,pn,Elg); PEL0=interp_log(Eph,pel,Elg); TOT=interp_log(Eph,tot,Elg)
logElg=np.log(Elg)
def xs_lookup(arr,E): return np.exp(np.interp(np.log(np.maximum(E,2e-3)),logElg,np.log(arr)))
def pair_xs(E):
    E=np.maximum(E,0); return np.where(E>1.022,xs_lookup(PN0,E)+xs_lookup(PEL0,E),0.0)
E1=1.1732; E2=1.3325

def KN_weight(c,E):
    a=E/mec2; Ep=E/(1+a*(1-c)); rr=Ep/E
    return rr*rr*(rr+1/rr-(1-c*c))
def sample_KN_single(E,rng):
    got=False
    fmax=KN_weight(-1.0,E)
    while not got:
        c=rng.uniform(-1,1); a_=E/mec2
        if rng.uniform(0,1)*fmax < KN_weight(c,E):
            got=True
    Ep=E/(1+(E/mec2)*(1-c)); T=E-Ep
    # electron direction
    pe= np.sqrt(T*(T+2*mec2))
    cose=(E-Ep*c)/pe if pe>0 else 1.0
    cose=np.clip(cose,-1,1)
    return T,cose,Ep,c

def sample_electron_state(Egamma, rng):
    """Return list of (T, cos_theta_parent_electron, process) for one photon interaction (excluding coherent).
    T in MeV, direction relative to photon travel direction."""
    winc=xs_lookup(INC_i,np.array([Egamma]))[0]; wpe=xs_lookup(PE,np.array([Egamma]))[0]; wpair=pair_xs(np.array([Egamma]))[0]
    W=winc+wpe+wpair
    r=rng.uniform(0,1)
    if r < winc/W:  # Compton
        T,cose,Ep,c=sample_KN_single(Egamma,rng)
        return [('inc',T,cose)]
    elif r < (winc+wpe)/W:  # photoelectric
        T=Egamma-0.0005
        # photoelectron direction: forward-peaked; sample cos from sin^2 weighted forward? approximate:
        cose=rng.uniform(0.7,1.0)
        return [('pe',T,cose)]
    else:  # pair
        Te=(Egamma-1.022)/2.0
        # electron forward, positron backward approx
        return [('pair',Te,0.6),('pair',Te,-0.6)]

# source sampling
def sample_sources(z0, src, N_el, rng, R_lat=2.5, zr=1.2):
    """src: dict with z_fn1,z_fn2,fs_fn. Sample electron births in region z in [z0-zr,z0+zr] (clip to 0..8), radius R_lat.
    Returns arrays pos(N,3), dir(N,3), E0(N), parent(N), ispos(N,bool), plus w_el."""
    pe_edges=np.array([0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4])
    zlo=max(z0-zr,0.01); zhi=min(z0+zr,7.9)
    # interaction-rate density S(z) on fine grid
    zg=np.linspace(zlo,zhi,400)
    mu1=xs_lookup(TOT,np.array([E1]))[0]; mu2=xs_lookup(TOT,np.array([E2]))[0]
    f1=src['z_fn1'](zg); f2=src['z_fn2'](zg); fs=src['fs_fn'](zg)
    Ec=0.5*(pe_edges[:-1]+pe_edges[1:])
    mut_sec=xs_lookup(TOT,Ec)
    S=(f1*mu1+f2*mu2)+fs@mut_sec   # per cm (per unit Phi0)
    # integrate
    dzg=zg[1]-zg[0]
    norm=np.trapz(S,zg)
    N_birth=np.pi*R_lat**2*norm
    w_el=N_birth/N_el
    # sample z from S via inverse CDF
    cdf=np.concatenate([[0],np.cumsum(S*dzg)]); cdf/=cdf[-1]
    u=rng.uniform(0,1,N_el)
    zsel=np.interp(u,cdf,zg)
    # lateral: uniform disk
    r_l=rng.uniform(0,R_lat,N_el); phi_l=rng.uniform(0,2*np.pi,N_el)
    xl=r_l*np.cos(phi_l); yl=r_l*np.sin(phi_l)
    # sample energy/process for each electron
    E0=np.empty(N_el); parent=np.empty(N_el); ispos=np.zeros(N_el,bool)
    cos0=np.empty(N_el); # direction relative to +z
    # sample source type: mixture
    m1=f1[np.interp(zsel,zg,np.arange(len(zg))).astype(int)]*mu1 if False else None
    # simpler: per electron, compute weights at its z
    zidx=np.interp(zsel,zg,np.arange(len(zg)))
    zi=zidx.astype(int); frac=zidx-zi; zi=np.clip(zi,0,len(zg)-2)
    f1s=src['z_fn1'](zsel); f2s=src['z_fn2'](zsel)
    fs_sel=src['fs_fn'](zsel)  # (N,28)
    w1=f1s*mu1; w2=f2s*mu2; ws=fs_sel*mut_sec[None,:]
    Wtot=w1+w2+ws.sum(axis=1)
    uu=rng.uniform(0,1,N_el)
    # assign source
    cw1=w1/Wtot; cw2=(w1+w2)/Wtot
    is_src=np.where(uu<cw1,0,np.where(uu<cw2,1,2))
    Egamma=np.empty(N_el)
    for mode in (0,1):
        m=is_src==mode
        Egamma[m]= E1 if mode==0 else E2
        parent[m]=0
    ms=is_src==2
    # scattered: choose bin
    cdfb=np.cumsum(ws,axis=1)
    cdfb/=cdfb[:,-1][:,None]
    us=rng.uniform(0,1,ms.sum())
    bidx=np.sum(us[:,None]>cdfb[ms],axis=1)
    bidx=np.clip(bidx,0,27)
    Egamma[ms]=Ec[bidx]
    parent[ms]=1
    # process: sample electron state
    for i in range(N_el):
        st=sample_electron_state(Egamma[i],rng)
        # take first particle (for most, one particle; pair gives two - store second by pushing extra)
        E0[i]=st[0][1]; cos0[i]=st[0][2]; ispos[i]=(st[0][0]=='pair' and False)  # only mark positron for second particle
    # build directions: photon was traveling along +z (nearly); electron direction relative to +z with az uniform
    az=rng.uniform(0,2*np.pi,N_el)
    sint=np.sqrt(np.maximum(0,1-cos0*cos0))
    dx=sint*np.cos(az); dy=sint*np.sin(az); dz=cos0
    pos=np.stack([xl,yl,100.0+zsel],axis=1)
    d=np.stack([dx,dy,dz],axis=1)
    d/=np.linalg.norm(d,axis=1,keepdims=True)
    # handle pair second particles: sample extra few (rare, ignore for main run; they're positrons which we exclude from fluence but should be in dose)
    # For simplicity: if a sampled 'pair' had a positron, skip (tiny) - but include electron as primary particle above.
    return pos,d,E0,parent,Egamma,ispos,w_el,N_birth