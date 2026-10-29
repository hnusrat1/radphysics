
mc2 = r'''
import numpy as np, time
from physics import *
V=np.pi*1.0**2*0.2
def seg_in_cyl(A,d,L,zc,half=0.1,R=1.0):
    dx,dy,dz=d[:,0],d[:,1],d[:,2]
    Ax,Ay,Az=A[:,0],A[:,1],A[:,2]
    a=dx*dx+dy*dy; b=2*(Ax*dx+Ay*dy); c=Ax*Ax+Ay*Ay-R*R
    lo=np.zeros(len(L)); hi=L.copy()
    # radial
    with np.errstate(divide='ignore',invalid='ignore'):
        disc=b*b-4*a*c
        sq=np.sqrt(np.maximum(disc,0))
        t1=(-b-sq)/(2*a); t2=(-b+sq)/(2*a)
    rad_ok=np.where(a>1e-14, disc>=0, c<=0)
    t1=np.where(a>1e-14,t1,-np.inf); t2=np.where(a>1e-14,t2,np.inf)
    lo=np.maximum(lo,t1); hi=np.minimum(hi,t2)
    # slab
    with np.errstate(divide='ignore',invalid='ignore'):
        tz1=(zc-half-Az)/dz; tz2=(zc+half-Az)/dz
    zlo=np.minimum(tz1,tz2); zhi=np.maximum(tz1,tz2)
    inslab=np.abs(Az-zc)<=half
    zlo=np.where(np.abs(dz)>1e-14,zlo,np.where(inslab,-np.inf,np.inf))
    zhi=np.where(np.abs(dz)>1e-14,zhi,np.where(inslab,np.inf,-np.inf))
    lo=np.maximum(lo,zlo); hi=np.minimum(hi,zhi)
    out=np.where(rad_ok, np.maximum(hi-lo,0), 0.0)
    return out
def run(N, zpts, seed=7, Emin=0.005, store_main=None, maxgen=300, sampling_w=True):
    rng=np.random.default_rng(seed)
    npt=len(zpts)
    A_E=np.zeros((npt,len(Eb)-1)); A_c=np.zeros((npt,len(cb)-1))
    A_tot=np.zeros(npt); A_Ef=np.zeros(npt)
    A_K=np.zeros(npt); A_Kc=np.zeros(npt)
    A_int=np.zeros((npt,4)); A_tr=np.zeros((npt,4))
    A_Kc_scat=np.zeros(npt)
    store=[]
    u=rng.uniform(-0.05,0.05,N); v=rng.uniform(-0.05,0.05,N)
    w=(1+u*u+v*v)**(-1.5)
    d=np.stack([u,v,np.ones(N)],axis=1); d/=np.linalg.norm(d,axis=1)[:,None]
    pos=np.stack([100*u,100*v,np.zeros(N)],axis=1)
    E=lines[(rng.random(N)<0.5).astype(int)]
    isprim=np.ones(N,bool)
    box_lo=np.array([-15,-15,0.]); box_hi=np.array([15,15,30.])
    gen=0; nColl=0
    zc=np.array(zpts)
    while E.size>0:
        gen+=1
        mu=mutot(E)
        s=-np.log(rng.random(E.size))/mu
        with np.errstate(divide='ignore',invalid='ignore'):
            tx=np.where(d>0,(box_hi-pos)/d,np.where(d<0,(box_lo-pos)/d,np.inf))
        texit=np.min(tx,axis=1)
        inside=s<texit
        L=np.minimum(s,texit)
        # segment scoring for scattered photons
        sc_mask=~isprim
        if sc_mask.any():
            As=pos[sc_mask]; ds=d[sc_mask]; Ls=L[sc_mask]; Es=E[sc_mask]; ws=w[sc_mask]
            for j,z in enumerate(zc):
                Lin=seg_in_cyl(As,ds,Ls,z)
                m=Lin>0
                if not m.any(): continue
                Wj=ws[m]*Lin[m]/V
                Em=Es[m]; dm=ds[m]
                hb=np.clip(np.digitize(Em,Eb)-1,0,len(Eb)-2)
                np.add.at(A_E[j],hb,Wj)
                cbin=np.clip(np.digitize(dm[:,2],cb)-1,0,len(cb)-2)
                np.add.at(A_c[j],cbin,Wj)
                A_tot[j]+=Wj.sum(); A_Ef[j]+=np.sum(Wj*Em)
                A_K[j]+=np.sum(Wj*ET(Em)); A_Kc[j]+=np.sum(Wj*KC(Em)); A_Kc_scat[j]+=np.sum(Wj*KC(Em))
                A_int[j,0]+=np.sum(Wj*muC(Em)); A_tr[j,0]+=np.sum(Wj*muC(Em)*Tmean(Em))
                A_int[j,1]+=np.sum(Wj*mupe(Em)); A_tr[j,1]+=np.sum(Wj*mupe(Em)*Em)
                A_int[j,2]+=np.sum(Wj*mupair(Em)); A_tr[j,2]+=np.sum(Wj*mupair(Em)*(Em-2*me))
                A_int[j,3]+=np.sum(Wj*muR(Em))
                if j==store_main:
                    store.append((Em.copy(),dm.copy(),Wj.copy()))
        # advance: keep only photons that interact inside
        pos=pos+L[:,None]*d
        keep=inside
        pos=pos[keep]; d=d[keep]; E=E[keep]; w=w[keep]; isprim=isprim[keep]; mu=mu[keep]
        if E.size==0: break
        nColl+=E.size
        X=pos
        xi=rng.random(E.size)*mu
        pC=muC(E); ppe=mupe(E); ppair=mupair(E)
        isC=xi<pC
        isPE=(xi>=pC)&(xi<pC+ppe)
        isPair=(xi>=pC+ppe)&(xi<pC+ppe+ppair)
        isR=xi>=pC+ppe+ppair
        newE=E.copy(); newd=d.copy(); alive=np.ones(E.size,bool)
        ci=np.where(isC)[0]
        if ci.size>0:
            eps,cc=sample_KN(E[ci],rng)
            phi=2*np.pi*rng.random(ci.size)
            newd[ci]=rotate(d[ci],cc,phi); newE[ci]=E[ci]*eps
        ri=np.where(isR)[0]
        if ri.size>0:
            c=np.empty(ri.size); idx=np.arange(ri.size)
            while idx.size>0:
                cc=2*rng.random(idx.size)-1
                acc=rng.random(idx.size)*2<(1+cc**2)
                c[idx[acc]]=cc[acc]; idx=idx[~acc]
            phi=2*np.pi*rng.random(ri.size)
            newd[ri]=rotate(d[ri],c,phi)
        alive[isPE]=False; alive[isPair]=False
        pi_=np.where(isPair)[0]
        if pi_.size>0:
            nd=isotropic(pi_.size,rng)
            hpos=np.concatenate([X[pi_],X[pi_]]); hdir=np.concatenate([nd,-nd])
            hE=np.full(2*pi_.size,0.511); hw=np.concatenate([w[pi_],w[pi_]])
        else:
            hpos=np.zeros((0,3)); hdir=np.zeros((0,3)); hE=np.zeros(0); hw=np.zeros(0)
        alive &= newE>Emin
        pos=np.concatenate([X[alive],hpos]); d=np.concatenate([newd[alive],hdir])
        E=np.concatenate([newE[alive],hE]); w=np.concatenate([w[alive],hw])
        isprim=np.concatenate([np.zeros(alive.sum(),bool),np.zeros(hE.size,bool)])
        if gen>=maxgen: break
    return dict(A_E=A_E,A_c=A_c,A_tot=A_tot,A_Ef=A_Ef,A_K=A_K,A_Kc=A_Kc,A_int=A_int,A_tr=A_tr,A_Kc_scat=A_Kc_scat,store=store,nColl=nColl,N=N)
'''
open('/tmp/work/mc2.py','w').write(mc2)
import sys, time; sys.path.insert(0,'/tmp/work')
import mc2 as M2, physics as P, numpy as np
zpts=[0.25,0.5,0.75,1.0,1.5,2.0,3.0,4.0,5.0,6.0,8.0,10.0]
t=time.time()
R2=M2.run(300000, zpts, seed=5, store_main=zpts.index(5.0))
print("time",time.time()-t)
N=R2['N']
print([round(P.SF/N*x,4) for x in R2['A_tot']])
print(R2['nColl'], len(R2['store']))
