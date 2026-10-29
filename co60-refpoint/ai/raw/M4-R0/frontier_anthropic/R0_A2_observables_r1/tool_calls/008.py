
sim = r'''
import numpy as np, lib, sys, time
from lib import *
def vogel(M,z0):
    k=np.arange(M); r=np.sqrt((k+0.5)/M); th=k*np.pi*(3-np.sqrt(5))
    return np.stack([r*np.cos(th), r*np.sin(th), np.full(M,z0)],1)
def run(N, seed, depths, M=7, Ebin=0.005, nE=280, nC=10):
    rng=np.random.default_rng(seed)
    Pts=[vogel(M,z0) for z0 in depths]
    H=np.zeros((len(depths),nE*nC))
    # emission
    uv=[]
    while len(uv)<N:
        u=rng.uniform(-0.05,0.05,2*N); v=rng.uniform(-0.05,0.05,2*N)
        acc=rng.uniform(0,1,2*N)< (1+u*u+v*v)**-1.5
        uv.append(np.stack([u[acc],v[acc]],1))
    uv=np.concatenate(uv)[:N]
    Eline=np.where(rng.uniform(size=N)<0.5, 1.1732, 1.3325)
    d=np.stack([uv[:,0],uv[:,1],np.ones(N)],1); d/=np.linalg.norm(d,axis=1)[:,None]
    pos=np.stack([100*uv[:,0],100*uv[:,1],np.zeros(N)],1)
    E=Eline.copy()
    nprim_events=0; nevents=0
    while len(E)>0:
        mt=mutot(E)
        s=-np.log(rng.uniform(size=len(E)))/mt
        tex=ray_exit(pos,d)
        alive=s<tex
        pos=pos[alive]; d=d[alive]; E=E[alive]; s=s[alive]; mt=mt[alive]
        if len(E)==0: break
        x=pos+s[:,None]*d
        nevents+=len(E)
        # contributions to each depth set
        pC=mu_inc(E)/mt; pR=mu_coh(E)/mt; pP=mu_pair(E)/mt
        kk=E/mec2
        for j,P in enumerate(Pts):
            for p in P:
                D=p[None,:]-x
                r=np.linalg.norm(D,axis=1)
                n=D/r[:,None]
                mus=np.sum(d*n,axis=1)
                Es=E/(1+kk*(1-mus))
                dsig=re**2/2*(Es/E)**2*(Es/E+E/Es-(1-mus*mus))  # per sr
                wC=ne*dsig/mt*np.exp(-mutot(Es)*r)/r**2
                wR=pR*3/(16*np.pi)*(1+mus*mus)*np.exp(-mt*r)/r**2
                wP=pP*2/(4*np.pi)*np.exp(-mutot(np.full(len(E),0.511))*r)/r**2
                cz=n[:,2]
                ib=np.clip(np.floor((cz+1)/0.2).astype(int),0,nC-1)
                ieC=np.clip(np.floor(Es/Ebin).astype(int),0,nE-1)
                ieR=np.clip(np.floor(E/Ebin).astype(int),0,nE-1)
                ieP=np.full(len(E),int(0.511/Ebin))
                wt=(wC+wR+wP)/M
                # accumulate
                H[j]+=np.bincount(ieC*nC+ib,weights=wC/M,minlength=nE*nC)
                H[j]+=np.bincount(ieR*nC+ib,weights=wR/M,minlength=nE*nC)
                H[j]+=np.bincount(ieP*nC+ib,weights=wP/M,minlength=nE*nC)
        # transport sampling
        u=rng.uniform(size=len(E))
        newpos=[];newd=[];newE=[]
        isC=u<pC
        isR=(u>=pC)&(u<pC+pR)
        isP=(u>=pC+pR)&(u<pC+pR+pP)
        # Compton
        if isC.any():
            idx=np.where(isC)[0]
            mu=kn_sample_mu(E[idx],rng)
            Es=E[idx]/(1+E[idx]/mec2*(1-mu))
            dn=rot(d[idx],mu,rng.uniform(0,2*np.pi,len(idx)))
            newpos.append(x[idx]); newd.append(dn); newE.append(Es)
        if isR.any():
            idx=np.where(isR)[0]
            mu=np.empty(len(idx)); todo=np.arange(len(idx))
            while len(todo):
                m_=rng.uniform(-1,1,len(todo)); a=rng.uniform(size=len(todo))<(1+m_*m_)/2
                mu[todo[a]]=m_[a]; todo=todo[~a]
            dn=rot(d[idx],mu,rng.uniform(0,2*np.pi,len(idx)))
            newpos.append(x[idx]); newd.append(dn); newE.append(E[idx])
        if isP.any():
            idx=np.where(isP)[0]
            for _ in range(2):
                newpos.append(x[idx]); newd.append(iso(len(idx),rng)); newE.append(np.full(len(idx),0.511))
        if newpos:
            pos=np.concatenate(newpos); d=np.concatenate(newd); E=np.concatenate(newE)
            keep=E>0.01
            pos=pos[keep]; d=d[keep]; E=E[keep]
        else:
            E=np.array([]); pos=np.zeros((0,3)); d=np.zeros((0,3))
    return H, nevents
if __name__=="__main__":
    N=int(sys.argv[1]); seed=int(sys.argv[2]); tag=sys.argv[3]
    depths=[0.25,0.5,0.75,1.0,2.0,5.0]
    t=time.time()
    H,ne_=run(N,seed,depths)
    np.savez(f'/tmp/work/H_{tag}.npz',H=H,N=N,depths=np.array(depths),nev=ne_)
    print("done",N,ne_,time.time()-t)
'''
open('/tmp/work/sim.py','w').write(sim)
import subprocess
r=subprocess.run(['bash','-lc','cd /tmp/work && python sim.py 20000 1 test'],capture_output=True,text=True)
print(r.stdout, r.stderr[-3000:])
