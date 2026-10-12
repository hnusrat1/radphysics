
code2 = r'''
import numpy as np
from phys import *
ZMIN,ZMAX=0.0,30.0
HALF=15.0
def run_photons(N, seed=0, Emin=0.01, maxcoll=200):
    rng=np.random.default_rng(seed)
    # generate primaries: uniform in solid angle in pyramid: sample tan-plane uniform, accept cos^3
    n_gen=0; tx=[];ty=[]
    need=N
    tx_all=[];ty_all=[]
    while sum(len(a) for a in tx_all)<N:
        m=int(N*1.2)+100
        a=rng.uniform(-0.05,0.05,m); b=rng.uniform(-0.05,0.05,m)
        cos3=(1+a*a+b*b)**-1.5
        acc=rng.random(m)<cos3
        tx_all.append(a[acc]); ty_all.append(b[acc])
    tx=np.concatenate(tx_all)[:N]; ty=np.concatenate(ty_all)[:N]
    d=np.stack([tx,ty,np.ones(N)],axis=1); d/=np.linalg.norm(d,axis=1,keepdims=True)
    # entry at z=0: position = 100*tan
    r=np.stack([100*tx,100*ty,np.zeros(N)],axis=1)
    # primaries start at the surface: note start at z=0 (entry); ray from source
    E=np.full(N,2.0)
    flag=np.zeros(N,bool)  # scattered flag
    # active photon arrays
    seg_r=[];seg_d=[];seg_E=[];seg_f=[];seg_l=[]
    # annihilation photons appended
    act = dict(r=r,d=d,E=E,f=flag)
    alive=np.ones(len(E),bool)
    nproc=dict(coh=0,comp=0,pe=0,pair=0)
    it=0
    while act['E'].size>0 and it<maxcoll:
        it+=1
        r=act['r']; d=act['d']; En=act['E']; f=act['f']
        coh,inc,pe,pr,mt = mu_all(En)
        # distance to exit box
        with np.errstate(divide='ignore', invalid='ignore'):
            tx_=np.where(d[:,0]>0,(HALF-r[:,0])/d[:,0],np.where(d[:,0]<0,(-HALF-r[:,0])/d[:,0],np.inf))
            ty_=np.where(d[:,1]>0,(HALF-r[:,1])/d[:,1],np.where(d[:,1]<0,(-HALF-r[:,1])/d[:,1],np.inf))
            tz_=np.where(d[:,2]>0,(ZMAX-r[:,2])/d[:,2],np.where(d[:,2]<0,(ZMIN-r[:,2])/d[:,2],np.inf))
        texit=np.minimum(np.minimum(tx_,ty_),tz_)
        s=-np.log(rng.random(En.size))/mt
        leave = s>=texit
        L = np.where(leave,texit,s)
        seg_r.append(r); seg_d.append(d); seg_E.append(En); seg_f.append(f); seg_l.append(L)
        # interacting ones
        keep=~leave
        idx=np.where(keep)[0]
        if idx.size==0:
            act={'r':np.zeros((0,3)),'d':np.zeros((0,3)),'E':np.zeros(0),'f':np.zeros(0,bool)}
            break
        rn=r[idx]+s[idx,None]*d[idx]
        dn=d[idx]; En_=En[idx]; fn=f[idx]
        u=rng.random(idx.size)*mt[idx]
        c1=coh[idx]; c2=c1+inc[idx]; c3=c2+pe[idx]
        isC = (u>=c1)&(u<c2)
        isR = u<c1
        isP = (u>=c2)&(u<c3)
        isPair = u>=c3
        # Compton
        newr=[];newd=[];newE=[];newf=[]
        if isC.any():
            m=isC
            ep,ct = sample_KN(En_[m],rng)
            phi=rng.uniform(0,2*np.pi,m.sum())
            nd=rotate(dn[m],ct,phi)
            newr.append(rn[m]); newd.append(nd); newE.append(En_[m]*ep); newf.append(np.ones(m.sum(),bool))
            nproc['comp']+=m.sum()
        if isR.any():
            m=isR
            ct=sample_Rayleigh_cos(m.sum(),rng)
            phi=rng.uniform(0,2*np.pi,m.sum())
            nd=rotate(dn[m],ct,phi)
            newr.append(rn[m]); newd.append(nd); newE.append(En_[m]); newf.append(np.ones(m.sum(),bool))
            nproc['coh']+=m.sum()
        if isP.any():
            nproc['pe']+=isP.sum()
        if isPair.any():
            m=isPair
            nproc['pair']+=m.sum()
            # two annihilation photons back to back, isotropic
            cz=rng.uniform(-1,1,m.sum()); ph=rng.uniform(0,2*np.pi,m.sum())
            sz=np.sqrt(1-cz*cz)
            v=np.stack([sz*np.cos(ph),sz*np.sin(ph),cz],axis=1)
            newr += [rn[m],rn[m]]; newd += [v,-v]; newE += [np.full(m.sum(),0.511),np.full(m.sum(),0.511)]
            newf += [np.ones(m.sum(),bool),np.ones(m.sum(),bool)]
        # Low energy absorb: photons with E< Emin are absorbed at the collision (handled: they are not continued unless Compton/Rayleigh)
        if newr:
            nr=np.concatenate(newr); nd_=np.concatenate(newd); ne=np.concatenate(newE); nf=np.concatenate(newf)
            good = ne>=Emin
            act={'r':nr[good],'d':nd_[good],'E':ne[good],'f':nf[good]}
        else:
            act={'r':np.zeros((0,3)),'d':np.zeros((0,3)),'E':np.zeros(0),'f':np.zeros(0,bool)}
    SR=np.concatenate(seg_r); SD=np.concatenate(seg_d); SE=np.concatenate(seg_E); SF=np.concatenate(seg_f); SL=np.concatenate(seg_l)
    return dict(r=SR,d=SD,E=SE,f=SF,l=SL,N=N,nproc=nproc,iters=it)
'''
open('/tmp/work/photon.py','w').write(code2)
import importlib, photon
importlib.reload(phys); importlib.reload(photon)
import time
t=time.time()
res=photon.run_photons(100000, seed=1)
print(time.time()-t, len(res['E']), res['nproc'], res['iters'], res['f'].mean())
