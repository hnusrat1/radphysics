# Save the working MC module to disk for reuse
module_code = r'''
import numpy as np, time, os

PH_EDGES = np.array([0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,
                     0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4])
CT_EDGES = np.array([-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1])

def seg_cyl_len(p0, p1, z0=5.0, R=1.0, hz=0.2):
    x0=p0[:,0]; y0=p0[:,1]; z0p=p0[:,2]
    sx=p1[:,0]-x0; sy=p1[:,1]-y0; sz=p1[:,2]-z0p
    L=np.sqrt(sx*sx+sy*sy+sz*sz); N=len(L)
    dx=np.zeros(N); dy=np.zeros(N); dz=np.zeros(N)
    nz=L>1e-15
    dx[nz]=sx[nz]/L[nz]; dy[nz]=sy[nz]/L[nz]; dz[nz]=sz[nz]/L[nz]
    a=(dx**2+dy**2)*L**2; b=2*(x0*dx+y0*dy)*L; c=x0**2+y0**2-R**2
    t1=np.zeros(N); t2=np.ones(N)
    par=a<1e-15; npar=~par
    disc=b[npar]**2-4*a[npar]*c[npar]; sq=np.sqrt(np.maximum(disc,0))
    t1[npar]=(-b[npar]-sq)/(2*a[npar]); t2[npar]=(-b[npar]+sq)/(2*a[npar])
    t1[par]=0.0; t2[par]=1.0
    q_ok=disc>0
    bad=np.zeros(N,bool); bad[par]=c[par]>0; bad[npar]=~q_ok
    tLo_r=np.clip(t1,0,1); tHi_r=np.clip(t2,0,1)
    zlo=z0-hz/2; zhi=z0+hz/2
    with np.errstate(divide='ignore',invalid='ignore'):
        tz_lo=np.where((np.abs(dz)*L)>1e-15,(zlo-z0p)/(dz*L),-1e9)
        tz_hi=np.where((np.abs(dz)*L)>1e-15,(zhi-z0p)/(dz*L),1e9)
    tLo_z=np.minimum(tz_lo,tz_hi); tHi_z=np.maximum(tz_lo,tz_hi)
    tLo=np.maximum(tLo_r,np.clip(tLo_z,0,1)); tHi=np.minimum(tHi_r,np.clip(tHi_z,0,1))
    good=(tHi>tLo)&~bad
    length=np.zeros(N); length[good]=L[good]*(tHi[good]-tLo[good])
    return length,good

def rotate_dir(d,cosT,sinT,phi):
    N=d.shape[0]
    ref=np.tile(np.array([0.,0.,1.]),(N,1))
    par=np.abs(d[:,2])>0.9
    if par.any(): ref[par]=np.array([1.,0.,0.])
    e1=np.cross(ref,d); e1/=np.linalg.norm(e1,axis=1,keepdims=True)+1e-30
    e2=np.cross(d,e1)
    return cosT[:,None]*d+sinT[:,None]*(np.cos(phi)[:,None]*e1+np.sin(phi)[:,None]*e2)

def sample_compton_angle(E_arr):
    E_arr=np.asarray(E_arr,float); N=len(E_arr)
    out=np.empty(N); got=np.zeros(N,bool); tries=0
    while not got.all():
        mu=np.random.uniform(-1,1,N)
        alpha=E_arr/0.51099895
        r=1.0/(1.0+alpha*(1.0-mu))
        f=r*r*(r+1.0/r-(1.0-mu**2))
        acc=np.random.uniform(0,3.0,N)<f
        sel=(~got)&acc; out[sel]=mu[sel]; got[sel]=True; tries+=1
        if tries>200000: out[~got]=1.0; got[~got]=True
    return out

def compton_after(E_arr,mu_arr):
    alpha=E_arr/0.51099895
    Ep=E_arr/(1.0+alpha*(1.0-mu_arr))
    Te=E_arr-Ep
    sinmu=np.sqrt(np.maximum(1-mu_arr**2,0))
    num=E_arr-Ep*mu_arr
    den=np.sqrt((E_arr-Ep*mu_arr)**2+(Ep*sinmu)**2)
    return Ep,Te,num/den

def boundary_dist(p,d,HW=15.0,ZD=30.0):
    N=p.shape[0]; t=np.full(N,1e9)
    z=p[:,2]; dz=d[:,2]
    m=dz>1e-12; t[m]=np.minimum(t[m],(ZD-z[m])/dz[m])
    m=dz<-1e-12; t[m]=np.minimum(t[m],(0.0-z[m])/dz[m])
    for axis in (0,1):
        pos=p[:,axis]; dd=d[:,axis]
        m=dd>1e-12; t[m]=np.minimum(t[m],(HW-pos[m])/dd[m])
        m=dd<-1e-12; t[m]=np.minimum(t[m],(-HW-pos[m])/dd[m])
    t[t<0]=0; return t

def load_xcom():
    from scipy.interpolate import interp1d
    d=np.loadtxt('/tmp/data/xcom_water.txt',comments='#')
    _E=d[:,0]; _coh=d[:,1]; _incoh=d[:,2]; _pe=d[:,3]; _pn=d[:,4]; _pef=d[:,5]; _tot=d[:,6]
    return dict(f_coh=interp1d(_E,_coh,bounds_error=False,fill_value=(_coh[0],_coh[-1])),
                f_incoh=interp1d(_E,_incoh,bounds_error=False,fill_value=(_incoh[0],_incoh[-1])),
                f_pe=interp1d(_E,_pe,bounds_error=False,fill_value=(_pe[0],_pe[-1])),
                f_pair=interp1d(_E,_pn+_pef,bounds_error=False,fill_value=(_pn[0]+_pef[0],_pn[-1]+_pef[-1])),
                f_tot=interp1d(_E,_tot,bounds_error=False,fill_value=(_tot[0],_tot[-1])))

def run_photon_mc(N=200000, depths=(5.0,), seed=12345, maxiter=80, HW=15.0, ZD=30.0, field_half=5.0):
    D=load_xcom(); rng=np.random.default_rng(seed); np.random.seed(seed)
    E1=1.1732; E2=1.3325
    u=rng.uniform(-field_half,field_half,N); v=rng.uniform(-field_half,field_half,N)
    norm=np.sqrt(u**2+v**2+100**2)
    dx=u/norm; dy=v/norm; dz=100.0/norm
    x=u.copy(); y=v.copy(); z=np.zeros(N)
    E=np.where(rng.random(N)<0.5,E1,E2)
    primary=np.ones(N,bool); alive=np.ones(N,bool)
    nz=len(depths)
    prim_tl=np.zeros(nz); scat_tl=np.zeros(nz)
    spec_prim=np.zeros(28); spec_scat=np.zeros(28)
    ang_prim=np.zeros(10); ang_scat=np.zeros(10)
    eflu_prim=0.0; eflu_scat=0.0
    imain=None
    for j,z0j in enumerate(depths):
        if abs(z0j-5.0)<1e-9: imain=j
    int_main={'comp':0,'pe':0,'pair':0,'ray':0,'comp_prim':0,'pe_prim':0,'pair_prim':0,'ray_prim':0}
    t0=time.time()
    for it in range(maxiter):
        ncur=len(x)
        if int(alive.sum())==0: break
        mu_tot=D['f_tot'](E)
        s=-np.log(np.maximum(rng.random(ncur),1e-30))/np.maximum(mu_tot,1e-10)
        tb=boundary_dist(np.stack([x,y,z],axis=1),np.stack([dx,dy,dz],axis=1),HW,ZD)
        s=np.minimum(s,tb); s=np.maximum(s,1e-12)
        p0=np.stack([x,y,z],axis=1)
        x2=x+dx*s; y2=y+dy*s; z2=z+dz*s
        p1=np.stack([x2,y2,z2],axis=1)
        if imain is not None:
            length,good=seg_cyl_len(p0,p1,z0=depths[imain])
            ebin=np.clip(np.searchsorted(PH_EDGES,E,side='right')-1,0,27)
            cbin=np.clip(np.searchsorted(CT_EDGES,dz,side='right')-1,0,9)
            m=good&alive
            mp=m&primary; ms=m&(~primary)
            spec_prim+=np.bincount(ebin[mp],weights=length[mp],minlength=28)
            spec_scat+=np.bincount(ebin[ms],weights=length[ms],minlength=28)
            ang_prim+=np.bincount(cbin[mp],weights=length[mp],minlength=10)
            ang_scat+=np.bincount(cbin[ms],weights=length[ms],minlength=10)
            eflu_prim+=float(np.sum(length[mp]*E[mp])); eflu_scat+=float(np.sum(length[ms]*E[ms]))
        for j,z0j in enumerate(depths):
            length,_=seg_cyl_len(p0,p1,z0=z0j)
            mp=(length>0)&alive&primary; ms=(length>0)&alive&(~primary)
            prim_tl[j]+=float(length[mp].sum()); scat_tl[j]+=float(length[ms].sum())
        x=x2; y=y2; z=z2
        exited=(x<-HW)|(x>HW)|(y<-HW)|(y>HW)|(z<0)|(z>ZD)
        alive &= ~exited
        ia=np.where(alive)[0]; nia=len(ia)
        if nia==0: continue
        Ea=E[ia]; mu_a=D['f_tot'](Ea)
        pcoh=D['f_coh'](Ea)/mu_a; pincoh=D['f_incoh'](Ea)/mu_a
        ppe=D['f_pe'](Ea)/mu_a; ppair=D['f_pair'](Ea)/mu_a
        r=rng.random(nia)
        if imain is not None:
            z0j=depths[imain]
            incyl=(np.abs(z[ia]-z0j)<=0.1)&(x[ia]**2+y[ia]**2<=1.0)
        else:
            incyl=np.zeros(nia,bool)
        prim_a=primary[ia]
        mcomp=(r<pincoh)
        if mcomp.sum()>0:
            idx=ia[mcomp]; is_prim=prim_a[mcomp]
            mu_arr=sample_compton_angle(E[idx])
            Ep,Te,cf=compton_after(E[idx],mu_arr)
            newd=rotate_dir(np.stack([dx[idx],dy[idx],dz[idx]],axis=1),mu_arr,np.sqrt(np.maximum(1-mu_arr**2,0)),rng.uniform(0,2*np.pi,len(idx)))
            dx[idx]=newd[:,0]; dy[idx]=newd[:,1]; dz[idx]=newd[:,2]
            E[idx]=Ep; primary[idx]=False
            ici=incyl[mcomp]; int_main['comp']+=int(ici.sum()); int_main['comp_prim']+=int((ici&is_prim).sum())
        mcoh=(r>=pincoh)&(r<pincoh+pcoh)
        if mcoh.sum()>0:
            idx=ia[mcoh]; n=len(idx)
            mu_c=np.empty(n); got=np.zeros(n,bool); tr=0
            while not got.all():
                cand=np.random.uniform(-1,1,n)
                acc=np.random.uniform(0,2,n)<(1+cand**2)
                sel=(~got)&acc; mu_c[sel]=cand[sel]; got[sel]=True; tr+=1
                if tr>200000: mu_c[~got]=1; got[~got]=True
            newd=rotate_dir(np.stack([dx[idx],dy[idx],dz[idx]],axis=1),mu_c,np.sqrt(np.maximum(1-mu_c**2,0)),rng.uniform(0,2*np.pi,n))
            dx[idx]=newd[:,0]; dy[idx]=newd[:,1]; dz[idx]=newd[:,2]
            is_prim=prim_a[mcoh]; primary[idx]=False
            ici=incyl[mcoh]; int_main['ray']+=int(ici.sum()); int_main['ray_prim']+=int((ici&is_prim).sum())
        mpe=(r>=pincoh+pcoh)&(r<pincoh+pcoh+ppe)
        if mpe.sum()>0:
            idx=ia[mpe]; is_prim=prim_a[mpe]
            ici=incyl[mpe]; int_main['pe']+=int(ici.sum()); int_main['pe_prim']+=int((ici&is_prim).sum())
            alive[idx]=False
        mpair=(r>=pincoh+pcoh+ppe)
        if mpair.sum()>0:
            idx=ia[mpair]; is_prim=prim_a[mpair]
            ici=incyl[mpair]; int_main['pair']+=int(ici.sum()); int_main['pair_prim']+=int((ici&is_prim).sum())
            alive[idx]=False
            for j in range(len(idx)):
                nvec=rng.standard_normal(3); nvec/=np.linalg.norm(nvec)+1e-30
                for sgn in (1.0,-1.0):
                    x=np.append(x,x[idx[j]]); y=np.append(y,y[idx[j]]); z=np.append(z,z[idx[j]])
                    dx=np.append(dx,sgn*nvec[0]); dy=np.append(dy,sgn*nvec[1]); dz=np.append(dz,sgn*nvec[2])
                    E=np.append(E,0.511); primary=np.append(primary,False); alive=np.append(alive,True)
    alive[:]=False
    return dict(prim_tl=prim_tl,scat_tl=scat_tl,spec_prim=spec_prim,spec_scat=spec_scat,
                ang_prim=ang_prim,ang_scat=ang_scat,eflu_prim=eflu_prim,eflu_scat=eflu_scat,
                int_main=int_main,N=N,time=time.time()-t0,iters=it)
'''
open('mc_module.py','w').write(module_code)
print("saved mc_module.py")