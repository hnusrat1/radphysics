import numpy as np, time, json
from scipy.interpolate import interp1d

rng = np.random.default_rng(12345)

# ---------------- data ----------------
ph = np.loadtxt('/tmp/data/xcom_water.txt', comments='#')
Eph = ph[:,0]; coh=ph[:,1]; inc=ph[:,2]; pe=ph[:,3]; pn=ph[:,4]; pel=ph[:,5]; tot=ph[:,6]
def mklogf(x,y):
    ok=(y>0)&(x>0)
    return interp1d(np.log(x[ok]),np.log(y[ok]),kind='linear',bounds_error=False,fill_value='extrapolate')
fcoh=mklogf(Eph,coh); finc=mklogf(Eph,inc); fpe=mklogf(Eph,pe); fpn=mklogf(Eph,pn); fpel=mklogf(Eph,pel); ftot=mklogf(Eph,tot)
def mu_coh(E): return float(np.exp(fcoh(np.log(float(E)))))
def mu_inc(E): return float(np.exp(finc(np.log(float(E)))))
def mu_pe(E):  return float(np.exp(fpe(np.log(float(E)))))
def mu_pn(E):  return float(np.exp(fpn(np.log(float(E)))))
def mu_pel(E): return float(np.exp(fpel(np.log(float(E)))))
def mu_tot(E): return float(np.exp(ftot(np.log(float(E)))))

el = np.loadtxt('/tmp/data/estar_water.txt', comments='#')
Ee=el[:,0]; Sc=el[:,1]; Sr=el[:,2]; St=el[:,3]; R=el[:,4]; Y=el[:,5]
def Scol(E): return float(np.interp(E,Ee,Sc))
def Yrad(E): return float(np.interp(E,Ee,Y))

mec2=0.5109989461
E1,E2 = 1.1732, 1.3325

# bin edges
pe_edges = np.array([0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4])
ct_edges = np.array([-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1])
ZE = np.array([0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4])

def ebin(E):
    return np.searchsorted(pe_edges, E, side='right')-1

# scoring cylinder constants
Rcyl=1.0; zc_cyl=105.0; hh=0.1
Vcyl = np.pi*Rcyl**2*2*hh

# ----- geometry helpers (vectorized) -----
def cyl_chord(p0, p1, R, zc, hh):
    # p0,p1: (N,3); segment
    d = p1-p0
    L = np.sqrt(np.sum(d*d,axis=1))
    dx=d[:,0]; dy=d[:,1]; dz=d[:,2]
    x0=p0[:,0]; y0=p0[:,1]; z0=p0[:,2]
    a = dx*dx+dy*dy
    b = x0*dx+y0*dy
    c = x0*x0+y0*y0-R*R
    # t in [0,1]
    tlo = np.zeros_like(L); thi = np.ones_like(L)
    # xy condition
    s = b*b-a*c
    big = a>1e-16
    # for a>0
    sq = np.sqrt(np.maximum(s,0))
    t1 = (-b-sq)/np.maximum(a,1e-30)
    t2 = (-b+sq)/np.maximum(a,1e-30)
    lo = np.minimum(t1,t2); hi=np.maximum(t1,t2)
    # for a~0 lines (dz along axis): need x0²+y0²<R²
    axzero = ~big
    lo[axzero] = -1e9; hi[axzero] = 1e9
    outside = axzero & (x0*x0+y0*y0>=R*R)
    # z condition
    z1=(zc-hh-z0)/np.where(np.abs(dz)>1e-12,dz,1.0); z2=(zc+hh-z0)/np.where(np.abs(dz)>1e-12,dz,1.0)
    zlo=np.minimum(z1,z2); zhi=np.maximum(z1,z2)
    dz0 = np.abs(dz)<1e-12
    zlo[dz0]=0; zhi[dz0]=1
    zlo[dz0 & (np.abs(z0-zc)>hh)] = 1.0; zhi[dz0 & (np.abs(z0-zc)>hh)] = 0.0
    lo=np.maximum(lo,zlo); hi=np.minimum(hi,zhi)
    lo=np.maximum(lo,tlo); hi=np.minimum(hi,thi)
    chord = L*np.maximum(0,hi-lo)
    chord[outside]=0
    return chord

def sample_dir_uniform_pyramid(n, rng):
    # uniform in solid angle within square pyramid half-slope a=0.05
    a=0.05
    u=np.empty(n); v=np.empty(n)
    got=np.zeros(n,bool)
    maxw=(1+2*a*a)**1.5
    while not got.all():
        uu=rng.uniform(-a,a,n); vv=rng.uniform(-a,a,n)
        acc = rng.uniform(0,1,n)*maxw < (1+uu*uu+vv*vv)**1.5
        new=(~got)&acc
        u[new]=uu[new]; v[new]=vv[new]; got[new]=True
    d=np.stack([u,v,np.ones_like(u)],axis=1)
    d/=np.sqrt(np.sum(d*d,axis=1,keepdims=True))
    return d, u, v

def rotate_vec(d, cost, sint, cosphi, sinphi):
    # Rodrigues-style rotate d by polar angle theta (cos cost), azimuth phi in plane perpendicular to d
    # build basis: n = d; pick arbitrary axis not parallel, e1 = normalize(cross(n, aux))
    aux = np.tile(np.array([0.0,0.0,1.0]),(len(d),1))
    aux[np.abs(d[:,2])>0.9]=np.array([1.0,0.0,0.0])
    e1 = np.cross(aux,d); e1/=np.sqrt(np.sum(e1*e1,axis=1,keepdims=True))
    e2 = np.cross(d,e1)
    return cost[:,None]*d + sint[:,None]*(cosphi[:,None]*e1 + sinphi[:,None]*e2)

def KN_weight(cosT, E):
    a = E/mec2
    Ep = E/(1+a*(1-cosT))
    r = Ep/E
    return r*r*(r + 1/r - (1-cosT*cosT))

def sample_KN_batch(Earr, rng):
    n=len(Earr)
    cosT=np.empty(n); Ep=np.empty(n)
    fmax=np.array([KN_weight(-1.0,E) for E in Earr])
    got=np.zeros(n,bool)
    while not got.all():
        c=rng.uniform(-1,1,n)
        acc = (rng.uniform(0,1,n)*fmax) < KN_weight(c,Earr)
        new=(~got)&acc
        cosT[new]=c[new]; got[new]=True
    a=Earr/mec2
    Ep = Earr/(1+a*(1-cosT))
    return cosT, Ep

# ---------- photon MC ----------
def run_photon_mc(N, seed, nmax_extra=200000):
    rng = np.random.default_rng(seed)
    # initial photons
    d,u,v = sample_dir_uniform_pyramid(N, rng)
    E = np.where(rng.uniform(0,1,N)<0.5, E1, E2)
    pos = np.stack([u*100.0, v*100.0, np.full(N,100.0)],axis=1)
    prim = np.ones(N,bool)
    alive = np.ones(N,bool)
    # tallies
    flu_prim=np.zeros(len(pe_edges)-1); flu_sec=np.zeros(len(pe_edges)-1)
    ang_prim=np.zeros(len(ct_edges)-1); ang_sec=np.zeros(len(ct_edges)-1)
    Eflu_prim=0.0; Eflu_sec=0.0
    nz=32; z_edges=np.linspace(0,8,nz+1); zc_slabs=0.5*(z_edges[:-1]+z_edges[1:]); hh_slab=0.5*(z_edges[1]-z_edges[0])
    slab_prim=np.zeros((2,nz)); slab_sec=np.zeros((nz,len(pe_edges)-1))
    # interaction counting in scoring volume
    int_count = {'inc':0,'pe':0,'pair':0,'ray':0,'prim':0,'sec':0}  # total, primary/scattered separate below
    int_prim = {'inc':0,'pe':0,'pair':0,'ray':0}
    int_sec  = {'inc':0,'pe':0,'pair':0,'ray':0}
    # depth Kcol (analytic muenE) per slab
    slab_Kcol=np.zeros(nz)
    from scipy.integrate import quad
    def KN_av(E):
        fmax=KN_weight(-1.0,E)
        Z=quad(lambda ct: KN_weight(ct,E),-1,1)[0]
        mT=quad(lambda ct: KN_weight(ct,E)*(E-E/(1+(E/mec2)*(1-ct))),-1,1)[0]/Z
        mTY=quad(lambda ct: KN_weight(ct,E)*(E-E/(1+(E/mec2)*(1-ct)))*(1-Yrad(E-E/(1+(E/mec2)*(1-ct)))),-1,1)[0]/Z
        return mT,mTY
    def mutrE(E):
        mT,_=KN_av(E)
        val=mu_pe(E)*(E-0.0005)+mu_inc(E)*mT
        if E>1.022: val+=(mu_pn(E)+mu_pel(E))*(E-1.022)
        return val
    def muenE(E):
        mT,mTY=KN_av(E)
        val=mu_pe(E)*(E-0.0005)*(1-Yrad(E-0.0005))+mu_inc(E)*mTY
        if E>1.022:
            Te=(E-1.022)/2; val+=(mu_pn(E)+mu_pel(E))*2*Te*(1-Yrad(Te))
        return val
    extra_new=[]
    t0=time.time()
    it=0
    while np.any(alive):
        idx=np.nonzero(alive)[0]
        Ei=E[idx]
        mut=np.array([mu_tot(e) for e in Ei])
        s_int=-np.log(rng.uniform(1e-300,1,size=len(idx)))/np.maximum(mut,1e-10)
        # boundary distance to cube [-15,15]x[-15,15]x[100,130]
        p0=pos[idx]; dd=dir_all[idx] if False else None
        d0=d[idx]
        # slab method for box: tmin for each axis
        inv=np.abs(d0)>1e-12
        t1=(np.array([-15.0,100.0, -15.0])-p0)/d0
        t2=(np.array([15.0,130.0,15.0])-p0)/d0
        tlo=np.maximum(np.minimum(t1,t2),0); thi=np.maximum(t1,t2)
        tb=np.min(thi,axis=1)
        s=np.minimum(s_int,tb)
        s=s.reshape(-1,1)
        p1=p0+d0*s
        # score primary/scattered in scoring volume
        ch=cyl_chord(p0,p1,Rcyl,zc_cyl,hh)
        eb=ebin(Ei)
        cb=np.digitize(d0[:,2],ct_edges)-1
        pe_prim=prim[idx]
        for i in range(len(pe_edges)-1):
            m=(eb==i)&(ch>0)
            if m.any():
                chm=ch[m]
                p_=pe_prim[m]
                flu_prim[i]+=np.sum(chm[p_])
                flu_sec[i]+=np.sum(chm[~p_])
        for j in range(len(ct_edges)-1):
            m=(cb==j)&(ch>0)
            if m.any():
                chm=ch[m]; p_=pe_prim[m]
                ang_prim[j]+=np.sum(chm[p_]); ang_sec[j]+=np.sum(chm[~p_])
        Eflu_prim+=np.sum(ch[pe_prim]*Ei[pe_prim])
        Eflu_sec +=np.sum(ch[~pe_prim]*Ei[~pe_prim])
        # slab scoring
        for k in range(nz):
            chs=cyl_chord(p0,p1,1.0,100.0+zc_slabs[k],hh_slab)
            if chs.sum()>0:
                for i in range(len(pe_edges)-1):
                    m=(eb==i)&(chs>0)
                    if m.any():
                        slab_sec[k,i]+=np.sum(chs[m][~pe_prim[m]])
                slab_prim[0,k]+=np.sum(chs[pe_prim & (Ei==E1)])
                slab_prim[1,k]+=np.sum(chs[pe_prim & (Ei==E2)])
                # kcol contribution (energy transferred, minus radiative) - using muenE
                slab_Kcol[k]+=np.sum(chs*muenE(Ei)) # MeV per g? chs*muenE: cm * MeV cm^2/g / cm? wait muenE is mu_en*E in MeV cm^2/g -> chs[cm]*muenE[MeV cm^2/g]... need /cm? mu_en [cm^2/g]; chs [cm]; per volume? This gives track*mu_en*E in units cm * MeV cm^2/g = MeV cm^3/g... divide by V_slab later for per volume.
        # interaction handling for interacting photons
        inter = (s_int < tb)[:,0]
        np_inter=np.sum(inter)
        if np_inter>0:
            iidx=idx[inter]
            Ei_i=Ei[inter]
            pr_i=pe_prim[inter]
            pi=p1[inter]; di=d0[inter]
            mut_i=mut[inter]
            rp=rng.uniform(0,1,np_inter)
            # process thresholds
            pc=mu_coh(Ei_i)/mut_i; pi_=mu_inc(Ei_i)/mut_i; pp=mu_pe(Ei_i)/mut_i; ppair=(mu_pn(Ei_i)+mu_pel(Ei_i))/mut_i
            isc = rp < pc; iin = (rp>=pc)&(rp<pc+pi_); ipe=(rp>=pc+pi_)&(rp<pc+pi_+pp); ipair=rp>=pc+pi_+pp
            # scattered photons (compton or rayleigh) keep alive with new state
            csc = (isc|iin)&(~alive[iidx])  # will set via alive update below
            # count interactions inside scoring cylinder
            in_cyl = np.abs(pi[:,2]-zc_cyl)<hh
            radius=np.sqrt(pi[:,0]**2+pi[:,1]**2)
            in_cyl &= radius<Rcyl
            for proc, mask in [('ray',isc),('inc',iin),('pe',ipe),('pair',ipair)]:
                if mask.any():
                    mm=mask&in_cyl
                    if mm.any():
                        prm=pr_i[mm]
                        pidx=np.where(mm)[0]
                        for q in pidx:
                            key='prim' if pr_i[q] else 'sec'
                            int_count[proc]+=1  # total by process
                            if pr_i[q]: int_prim[proc]+=1
                            else: int_sec[proc]+=1
            # narrow to those continuing (compton or rayleigh)
            cont = isc|iin
            if cont.any():
                # Rayleigh: new direction
                # Compton: new energy
                cidx=np.nonzero(cont)[0]
                Econt=Ei[cont]; dcont=di[cont]
                is_inc = iin[cont]
                nE=Econt.copy()
                # sample KN for incoherent ones
                cosT,Ep = sample_KN_batch(Econt[is_inc], rng)
                nE[is_inc]=Ep
                # angles: polar theta
                th=np.arccos(cosT) if False else None
                cost = cosT if True else None
                # We'll do: sample scattering angle for all continuing; rayleigh: sample forward thomson
                # Build polar cos for rayleigh (use (1+cos^2), sample below)
                raymask=~is_inc
                nraym=np.sum(raymask)
                cR=np.empty(nraym)
                got=np.zeros(nraym,bool)
                while not got.all():
                    cc=rng.uniform(-1,1,nraym)
                    # use form-factor-ish forward bias: accept with (1+cc^2)*(1-cc*0.98)? keep simple Thomson (1+cc^2)
                    accept = rng.uniform(0,1,nraym)*2.0 < (1+cc*cc)
                    new=(~got)&accept
                    cR[new]=cc[new]; got[new]=True
                cc=np.concatenate([cosT,cR]) if False else None
                # assemble polar cos for all continuing
                c_all=np.empty(len(cidx))
                c_all[is_inc]=cosT
                c_all[raymask]=cR
                sint=np.sqrt(np.maximum(0,1-c_all*c_all))
                phi=rng.uniform(0,2*np.pi,len(cidx))
                dnew=rotate_vec(dcont, c_all, sint, np.cos(phi), np.sin(phi))
                d[iidx[cont]]=dnew
                E[iidx[cont]]=nE
                prim[iidx[cont]]=False
                # keep alive
                alive[iidx[cont]]=True
            # photoelectric / pair: kill and spawn
            kill = ipe|ipair
            if kill.any():
                kk=np.nonzero(kill)[0]
                alive[iidx[kk]]=False
            if ipair.any():
                # spawn two annihilation photons at 0.511
                pidx=np.nonzero(ipair)[0]
                pid=iidx[pidx]
                for q in range(len(pidx)):
                    # random perpendicular direction
                    u2=rng.normal(size=3); u2/=np.linalg.norm(u2)
                    u2-=di[pidx[q]]*np.dot(u2,di[pidx[q]]); u2/=np.linalg.norm(u2)
                    n1=u2.copy(); n2=-u2.copy()
                    # place at interaction point
                    extra_new.append((pi[pidx[q]].copy(),n1,0.511,False))
                    extra_new.append((pi[pidx[q]].copy(),n2,0.511,False))
        # photons that left the phantom (not interacting): kill
        alive[idx[~inter]]=False
        alive[idx[inter]]=False  # will re-set continuing ones above
        # (we already set continuing ones alive; kill flag for interacting all then re-alive)
        # re-enable continuing
        if False: pass
        # Actually careful: we set alive[iidx[cont]]=True inside, then here override:
        # fix: kill all interacting, then re-enable continuing again
        # The lines above set alive[idx[inter]]=False after having set some true; so redo:
        # (We'll just re-enable here)
        for ci in np.nonzero(cont)[0]:
            alive[iidx[ci]]=True
        # add extra photons when slots become free (replace dead)
        # For simplicity: append extras to global stacks handled at end of batch
        it+=1
        if it%50==0:
            pass
        if len(extra_new)>nmax_extra:
            break
    # return (scaled per photon later)
    res=dict(N=N, flu_prim=flu_prim, flu_sec=flu_sec, ang_prim=ang_prim, ang_sec=ang_sec,
             Eflu_prim=Eflu_prim, Eflu_sec=Eflu_sec, slab_prim=slab_prim, slab_sec=slab_sec,
             slab_Kcol=slab_Kcol, zc_slabs=zc_slabs, hh_slab=hh_slab, int_count=int_count,
             int_prim=int_prim, int_sec=int_sec, nnew=len(extra_new), time=time.time()-t0)
    return res