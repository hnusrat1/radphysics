import numpy as np
from scipy.interpolate import interp1d
import time, os

# ---------- data loading ----------
def load_data():
    d = np.loadtxt('/tmp/data/xcom_water.txt', comments='#')
    E_ph = d[:,0]; mu_coh=d[:,1]; mu_incoh=d[:,2]; mu_pe=d[:,3]; mu_pairN=d[:,4]; mu_pairE=d[:,5]; mu_tot=d[:,6]
    f_coh = interp1d(E_ph, mu_coh, kind='linear', bounds_error=False, fill_value=(mu_coh[0], mu_coh[-1]))
    f_incoh = interp1d(E_ph, mu_incoh, kind='linear', bounds_error=False, fill_value=(mu_incoh[0], mu_incoh[-1]))
    f_pe = interp1d(E_ph, mu_pe, kind='linear', bounds_error=False, fill_value=(mu_pe[0], mu_pe[-1]))
    f_pair = interp1d(E_ph, mu_pairN+mu_pairE, kind='linear', bounds_error=False, fill_value=(mu_pairN[0], mu_pairN[-1]))
    f_tot = interp1d(E_ph, mu_tot, kind='linear', bounds_error=False, fill_value=(mu_tot[0], mu_tot[-1]))
    d2 = np.loadtxt('/tmp/data/estar_water.txt', comments='#')
    Ee=d2[:,0]; Sc=d2[:,1]; Sr=d2[:,2]; St=d2[:,3]; R=d2[:,4]; Y=d2[:,5]; de=d2[:,6]
    f_Sc = interp1d(Ee, Sc, kind='linear', bounds_error=False, fill_value=(Sc[0],Sc[-1]))
    f_Sr = interp1d(Ee, Sr, kind='linear', bounds_error=False, fill_value=(Sr[0],Sr[-1]))
    f_St = interp1d(Ee, St, kind='linear', bounds_error=False, fill_value=(St[0],St[-1]))
    f_R  = interp1d(Ee, R, kind='linear', bounds_error=False, fill_value=(R[0],R[-1]))
    f_Y  = interp1d(Ee, Y, kind='linear', bounds_error=False, fill_value=(Y[0],Y[-1]))
    return dict(E_ph=E_ph, mu_coh=mu_coh, mu_incoh=mu_incoh, mu_pe=mu_pe, mu_pairN=mu_pairN,
                mu_pairE=mu_pairE, mu_tot=mu_tot, f_coh=f_coh, f_incoh=f_incoh, f_pe=f_pe,
                f_pair=f_pair, f_tot=f_tot, Ee=Ee, Sc=Sc, Sr=Sr, St=St, R=R, Y=Y, de=de,
                f_Sc=f_Sc, f_Sr=f_Sr, f_St=f_St, f_R=f_R, f_Y=f_Y)

DATA = load_data()

PH_EDGES = np.array([0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,
                     0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4])
CT_EDGES = np.array([-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1])
EL_EDGES = np.array([0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,
                     0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.4])

# ---------- geometry ----------
def seg_cyl_len(p0, p1, z0=5.0, R=1.0, hz=0.2):
    x0=p0[:,0]; y0=p0[:,1]; z0p=p0[:,2]
    sx=p1[:,0]-x0; sy=p1[:,1]-y0; sz=p1[:,2]-z0p
    L=np.sqrt(sx*sx+sy*sy+sz*sz)
    N=len(L)
    dx=np.zeros(N); dy=np.zeros(N); dz=np.zeros(N)
    nz=L>1e-15
    dx[nz]=sx[nz]/L[nz]; dy[nz]=sy[nz]/L[nz]; dz[nz]=sz[nz]/L[nz]
    a = (dx**2+dy**2)*L**2
    b = 2*(x0*dx+y0*dy)*L
    c = x0**2+y0**2-R**2
    t1=np.zeros(N); t2=np.ones(N)
    par = a<1e-15
    npar=~par
    disc = b[npar]**2-4*a[npar]*c[npar]
    sq=np.sqrt(np.maximum(disc,0))
    t1[npar] = (-b[npar]-sq)/(2*a[npar])
    t2[npar] = (-b[npar]+sq)/(2*a[npar])
    t1[par]=0.0; t2[par]=1.0
    q_ok = disc>0
    bad = np.zeros(N,bool)
    bad[par] = c[par]>0
    bad[npar] = ~q_ok
    tLo_r = np.clip(t1,0,1); tHi_r = np.clip(t2,0,1)
    zlo=z0-hz/2; zhi=z0+hz/2
    tz_lo = np.where(np.abs(dz)>1e-12, (zlo-z0p)/(dz*L), -1e9)
    tz_hi = np.where(np.abs(dz)>1e-12, (zhi-z0p)/(dz*L),  1e9)
    tLo_z = np.minimum(tz_lo,tz_hi); tHi_z=np.maximum(tz_lo,tz_hi)
    tLo = np.maximum(tLo_r, np.clip(tLo_z,0,1))
    tHi = np.minimum(tHi_r, np.clip(tHi_z,0,1))
    good = (tHi>tLo)&~bad
    length=np.zeros(N); length[good]=L[good]*(tHi[good]-tLo[good])
    return length, good

def rotate_dir(d, cosT, sinT, phi):
    N=d.shape[0]
    ref=np.tile(np.array([0.,0.,1.]),(N,1))
    par = np.abs(d[:,2])>0.9
    if par.any():
        ref[par]=np.array([1.,0.,0.])
    e1 = np.cross(ref,d)
    nrm=np.linalg.norm(e1,axis=1,keepdims=True)
    e1 = e1/(nrm+1e-30)
    e2 = np.cross(d,e1)
    cT=cosT[:,None]; sT=sinT[:,None]; cp=np.cos(phi)[:,None]; sp=np.sin(phi)[:,None]
    return cT*d + sT*(cp*e1+sp*e2)

def sample_iso_dir(N):
    z=np.random.uniform(-1,1,N); phi=np.random.uniform(0,2*np.pi,N)
    r=np.sqrt(1-z*z)
    return np.stack([r*np.cos(phi), r*np.sin(phi), z],axis=1)

def sample_compton_angle(E_arr, seed=None):
    # E_arr array of photon energies; returns mu (cos of photon scattering angle)
    E_arr=np.asarray(E_arr,float); N=len(E_arr)
    out=np.empty(N); got=np.zeros(N,bool)
    tries=0
    while not got.all():
        mu=np.random.uniform(-1,1,N)
        alpha=E_arr/0.51099895
        r=1.0/(1.0+alpha*(1.0-mu))
        f=r*r*(r+1.0/r-(1.0-mu**2))
        acc=np.random.uniform(0,3.0,N)<f
        sel=(~got)&acc
        out[sel]=mu[sel]; got[sel]=True
        tries+=1
        if tries>200000:
            out[~got]=1.0; got[~got]=True
    return out

def compton_after(E_arr, mu_arr):
    # returns scattered photon energy E', electron energy Te, electron cos(phi) w.r.t. incident dir
    alpha=E_arr/0.51099895
    Ep=E_arr/(1.0+alpha*(1.0-mu_arr))
    Te=E_arr-Ep
    # electron angle: cos(phi) = (E - E' cos mu)/sqrt((E-E'cos mu)^2 + (E' sin mu)^2)
    nu=np.cos(np.arccos(mu_arr)) # mu
    num = E_arr - Ep*mu_arr
    den = np.sqrt((E_arr-Ep*mu_arr)**2 + (Ep*np.sin(np.arccos(mu_arr)))**2)
    cos_phi = num/den
    return Ep, Te, cos_phi

def boundary_dist(p, d):
    # p (N,3), d (N,3) unit; distance to leaving the 30x30x30 cube (x,y in [-15,15], z in [0,30])
    N=p.shape[0]
    t=np.full(N,1e9)
    # z planes
    z=p[:,2]; dz=d[:,2]
    m=dz>1e-12; t[m]=np.minimum(t[m],(30.0-z[m])/dz[m])
    m=dz<-1e-12; t[m]=np.minimum(t[m],(0.0-z[m])/dz[m])
    for axis in (0,1):
        pos=p[:,axis]; dd=d[:,axis]
        m=dd>1e-12; t[m]=np.minimum(t[m],(15.0-pos[m])/dd[m])
        m=dd<-1e-12; t[m]=np.minimum(t[m],(-15.0-pos[m])/dd[m])
    t[t<0]=0
    return t

def run_photon_mc(N=200000, depths=(5.0,), seed=12345):
    rng=np.random.default_rng(seed)
    np.random.seed(seed)
    E1=1.1732; E2=1.3325
    # sample field photons
    u=rng.uniform(-5,5,N); v=rng.uniform(-5,5,N)
    # direction from source (0,0,-100) to (u,v,0)
    dx0=u; dy0=v; dz0=100.0
    norm=np.sqrt(dx0**2+dy0**2+dz0**2)
    dx=dx0/norm; dy=dy0/norm; dz=dz0/norm
    x=u.copy(); y=v.copy(); z=np.zeros(N)
    E=np.where(rng.random(N)<0.5, E1, E2)
    primary=np.ones(N,bool)
    alive=np.ones(N,bool)
    # tallies
    nz=len(depths)
    prim_tl=np.zeros(nz); scat_tl=np.zeros(nz)
    # main cylinder at index of depth 5.0 (if present)
    imain = np.argmin(np.abs(np.array(depths)-5.0)) if np.any(np.abs(np.array(depths)-5.0)<1e-9) else None
    spec_prim=np.zeros(28); spec_scat=np.zeros(28)
    ang_prim=np.zeros(10); ang_scat=np.zeros(10)
    eflu_prim=0.0; eflu_scat=0.0
    int_main={'comp':0,'pe':0,'pair':0,'ray':0,'comp_p':0,'pe_p':0,'pair_p':0,'ray_p':0}
    spawn=[]
    step_count=0
    t0=time.time()
    maxiter=60
    for it in range(maxiter):
        n_alive=alive.sum()
        if n_alive==0: break
        mu_tot = DATA['f_tot'](E)
        s = -np.log(np.maximum(rng.random(N),1e-30))/np.maximum(mu_tot,1e-10)
        # clamp to boundary
        tb = boundary_dist(np.stack([x,y,z],axis=1), np.stack([dx,dy,dz],axis=1))
        s = np.minimum(s, tb)
        s = np.maximum(s, 1e-12)
        # positions before
        p0=np.stack([x,y,z],axis=1)
        x2=x+dx*s; y2=y+dy*s; z2=z+dz*s
        p1=np.stack([x2,y2,z2],axis=1)
        # score main cylinder
        if imain is not None:
            length,good = seg_cyl_len(p0,p1, z0=depths[imain])
            # energy bins, cos bins
            ebin=np.clip(np.searchsorted(PH_EDGES, E, side='right')-1,0,27)
            cbin=np.clip(np.searchsorted(CT_EDGES, dz, side='right')-1,0,9)
            m=good&alive
            # separate primary/scattered
            mp=m&primary; ms=m&(~primary)
            spec_prim += np.bincount(ebin[mp], weights=length[mp], minlength=28)
            spec_scat += np.bincount(ebin[ms], weights=length[ms], minlength=28)
            ang_prim += np.bincount(cbin[mp], weights=length[mp], minlength=10)
            ang_scat += np.bincount(cbin[ms], weights=length[ms], minlength=10)
            eflu_prim += np.sum(length[mp]*E[mp]); eflu_scat += np.sum(length[ms]*E[ms])
        # score all depths (track length total)
        for j,z0j in enumerate(depths):
            length,_=seg_cyl_len(p0,p1,z0=z0j)
            mp=(length>0)&alive&primary; ms=(length>0)&alive&(~primary)
            prim_tl[j]+=length[mp].sum(); scat_tl[j]+=length[ms].sum()
        # move
        x=x2; y=y2; z=z2
        # check exit
        exited = (x<-15)|(x>15)|(y<-15)|(y>15)|(z<0)|(z>30)
        alive &= ~exited
        # interactions for still-alive photons
        ia=alive.copy()
        if ia.sum()==0: continue
        Ea=E[ia]; 
        mu_tot_a=DATA['f_tot'](Ea)
        pcoh=DATA['f_coh'](Ea)/mu_tot_a
        pincoh=DATA['f_incoh'](Ea)/mu_tot_a
        ppe=DATA['f_pe'](Ea)/mu_tot_a
        ppair=DATA['f_pair'](Ea)/mu_tot_a
        r=rng.random(ia.sum())
        # locate interaction point for main-cylinder interaction tallies
        incyl_main=False
        if imain is not None:
            z0j=depths[imain]
            incyl_main = (np.abs(z[ia]-z0j)<=0.1)&(x[ia]**2+y[ia]**2<=1.0)
        # --- incoherent ---
        mcomp = (r<pincoh)&ia
        if mcomp.sum()>0:
            idx=np.where(mcomp)[0]
            E_now=E[idx]; is_prim=primary[idx]
            mu_arr=sample_compton_angle(E_now)
            Ep,Te,cos_phi = compton_after(E_now, mu_arr)
            # photon direction update
            cosT=mu_arr; sinT=np.sqrt(np.maximum(1-mu_arr**2,0))
            newd=rotate_dir(np.stack([dx[idx],dy[idx],dz[idx]],axis=1), cosT, sinT, rng.uniform(0,2*np.pi,len(idx)))
            dx[idx]=newd[:,0]; dy[idx]=newd[:,1]; dz[idx]=newd[:,2]
            E[idx]=Ep
            primary[idx]=False
            # interaction tallies in main cylinder
            if imain is not None:
                ici = incyl_main[idx]
                int_main['comp']+=ici.sum()
                int_main['comp_p']+= (ici&is_prim).sum()
        # --- coherent ---
        mcoh = (r>=pincoh)&(r<pincoh+pcoh)&ia
        if mcoh.sum()>0:
            idx=np.where(mcoh)[0]
            # Thomson-like sampling of cos from (1+cos^2)
            mu_c=np.empty(len(idx)); got=np.zeros(len(idx),bool)
            while not got.all():
                cand=np.random.uniform(-1,1,len(idx))
                f=1+cand**2
                acc=np.random.uniform(0,2,len(idx))<f
                sel=(~got)&acc
                mu_c[sel]=cand[sel]; got[sel]=True
            cosT=mu_c; sinT=np.sqrt(np.maximum(1-mu_c**2,0))
            newd=rotate_dir(np.stack([dx[idx],dy[idx],dz[idx]],axis=1), cosT, sinT, rng.uniform(0,2*np.pi,len(idx)))
            dx[idx]=newd[:,0]; dy[idx]=newd[:,1]; dz[idx]=newd[:,2]
            primary[idx]=False
            if imain is not None:
                ici = incyl_main[idx]
                int_main['ray']+=ici.sum()
                int_main['ray_p']+=(ici&primary[idx]).sum()  # primary stays? after scatter it's not primary; but interaction of primary photon: parent was primary. classify by pre-state
        # --- photoelectric ---
        mpe = (r>=pincoh+pcoh)&(r<pincoh+pcoh+ppe)&ia
        if mpe.sum()>0:
            idx=np.where(mpe)[0]
            if imain is not None:
                ici = incyl_main[idx]
                int_main['pe']+=ici.sum()
                int_main['pe_p']+=(ici&primary[idx]).sum()
            alive[idx]=False
        # --- pair ---
        mpair = (r>=pincoh+pcoh+ppe)&ia
        if mpair.sum()>0:
            idx=np.where(mpair)[0]
            if imain is not None:
                ici = incyl_main[idx]
                int_main['pair']+=ici.sum()
                int_main['pair_p']+=(ici&primary[idx]).sum()
            # create two annihilation photons 0.511, replace one, spawn other
            nd=idx.size
            for j in range(nd):
                nvec=rng.random(3); nrm=np.linalg.norm(nvec); nvec/=nrm
                # keep current slot as first annihilation photon
                dx[idx[j]]=nvec[0]; dy[idx[j]]=nvec[1]; dz[idx[j]]=nvec[2]
                E[idx[j]]=0.511; primary[idx[j]]=False
                is_prim_pre = None
                spawn.append((x[idx[j]],y[idx[j]],z[idx[j]],-nvec[0],-nvec[1],-nvec[2],0.511,False))
        step_count+=1
        if it%10==0 and it>0:
            pass
    # append spawn photons and continue? For simplicity, we already absorbed them via spawn list; process them now in same loop? 
    # Actually we allowed spawning only at the end; simpler: process spawn by extending arrays and continuing loop.
    # We'll restructure: process spawn inside loop by extending arrays.
    return dict(prim_tl=prim_tl, scat_tl=scat_tl, spec_prim=spec_prim, spec_scat=spec_scat,
                ang_prim=ang_prim, ang_scat=ang_scat, eflu_prim=eflu_prim, eflu_scat=eflu_scat,
                int_main=int_main, N=N, steps=step_count, iters=it)

if __name__=='__main__':
    # quick test 20k photons
    res=run_photon_mc(20000, depths=(5.0,), seed=1)
    print(res['prim_tl'], res['scat_tl'])
    V=np.pi*0.2
    print("prim fluence/hist per depth:", res['prim_tl']/V/res['N'])
    print("analytic prim/Phi0:", 0.5*(np.exp(-DATA['f_tot'](1.1732)*5)+np.exp(-DATA['f_tot'](1.3325)*5)))
    print("scat/prim track ratio at 5:", res['scat_tl'][0]/res['prim_tl'][0])
    print("int_main:", res['int_main'])
    print("steps:", res['steps'], "iters:", res['iters'])