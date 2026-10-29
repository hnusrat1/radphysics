
mc_code = r'''
import numpy as np, sys
sys.path.insert(0,'/tmp/work')
import phys
from phys import *
def kn_sample_cos(E, rng):
    # rejection sampling of cos for KN; f/(re^2/2) <= 2
    n=E.size; out=np.empty(n); todo=np.arange(n)
    while todo.size:
        c=rng.uniform(-1,1,todo.size)
        f=fKN(E[todo],c)/(re**2/2)
        acc=rng.uniform(0,2,todo.size)<f
        out[todo[acc]]=c[acc]; todo=todo[~acc]
    return out
def rayleigh_cos(n,rng):
    out=np.empty(n); todo=np.arange(n)
    while todo.size:
        c=rng.uniform(-1,1,todo.size)
        acc=rng.uniform(0,1,todo.size)<(1+c*c)/2
        out[todo[acc]]=c[acc]; todo=todo[~acc]
    return out
def rotate(d, cth, phi):
    sth=np.sqrt(np.clip(1-cth**2,0,None)); cph=np.cos(phi); sph=np.sin(phi)
    dx,dy,dz=d[:,0],d[:,1],d[:,2]
    tmp=np.sqrt(np.clip(1-dz**2,1e-30,None))
    nx=sth*(dx*dz*cph-dy*sph)/tmp+dx*cth
    ny=sth*(dy*dz*cph+dx*sph)/tmp+dy*cth
    nz=-sth*cph*tmp+dz*cth
    # near-axis fallback
    ax=np.abs(dz)>0.99999
    nx=np.where(ax, sth*cph, nx); ny=np.where(ax, sth*sph, ny); nz=np.where(ax, cth*np.sign(dz), nz)
    n=np.stack([nx,ny,nz],1)
    return n/np.linalg.norm(n,axis=1)[:,None]
def run(N, depths, rng, store_for=5.0):
    # photon starts
    line=rng.integers(0,2,N); E=Lines[line].copy()
    # uniform in solid angle within pyramid: (u,v) in [-0.05,0.05]^2 weight (1+u^2+v^2)^-1.5
    acc_all=[]
    u=np.empty(0); v=np.empty(0)
    while u.size<N:
        uu=rng.uniform(-0.05,0.05,2*N); vv=rng.uniform(-0.05,0.05,2*N)
        ok=rng.uniform(0,1,2*N)<(1+uu*uu+vv*vv)**-1.5
        u=np.concatenate([u,uu[ok]]); v=np.concatenate([v,vv[ok]])
    u=u[:N]; v=v[:N]
    d=np.stack([u,v,np.ones(N)],1); d/=np.linalg.norm(d,axis=1)[:,None]
    pos=np.stack([u*100.0, v*100.0, np.zeros(N)],1)
    order=np.zeros(N,int)
    P=np.array([[0,0,z] for z in depths],float)  # points on axis (x=y=0)
    nP=len(depths)
    dose=np.zeros(nP)     # sum W*E*mu_tr (scattered arrivals) per point, per N
    dose_scat_by_pt=np.zeros(nP)
    kerma_scat_by_pt=np.zeros(nP)
    jstore=int(np.argmin(np.abs(np.array(depths)-store_for)))
    arr=dict(E=[],u=[],W=[],typ=[])
    nev=0
    # pair annihilation spawns
    alive=np.ones(N,bool)
    while True:
        idx=np.nonzero(alive)[0]
        if idx.size==0: break
        Ec=E[idx]; mt=mu_tot(Ec)
        sint=-np.log(rng.uniform(size=idx.size))/mt
        # exit distance inside box [-15,15]^2 x [0,30]
        p=pos[idx]; dd=d[idx]
        with np.errstate(divide='ignore',invalid='ignore'):
            tx=np.where(dd[:,0]>0,(15-p[:,0])/dd[:,0],np.where(dd[:,0]<0,(-15-p[:,0])/dd[:,0],np.inf))
            ty=np.where(dd[:,1]>0,(15-p[:,1])/dd[:,1],np.where(dd[:,1]<0,(-15-p[:,1])/dd[:,1],np.inf))
            tz=np.where(dd[:,2]>0,(30-p[:,2])/dd[:,2],np.where(dd[:,2]<0,(0-p[:,2])/dd[:,2],np.inf))
        sex=np.minimum(np.minimum(tx,ty),tz)
        sex=np.maximum(sex,0)
        L=np.minimum(sint,sex)
        # NEE for each point
        s_r=rng.uniform(size=idx.size)*L
        x=p+s_r[:,None]*dd
        for j in range(nP):
            r_vec=P[j][None,:]-x
            r=np.linalg.norm(r_vec,axis=1)
            uu_=r_vec/r[:,None]
            cth=np.sum(dd*uu_,axis=1)
            # Compton
            Ep=Ec/(1+Ec/mc2*(1-cth))
            fC=fKN(Ec,cth)  # per sr per electron
            WC=L*ne*fC*np.exp(-phys.mu_tot(Ep)*r)/r**2
            WR=L*mu_coh(Ec)*3*(1+cth**2)/(16*np.pi)*np.exp(-mu_tot(Ec)*r)/r**2
            if j==jstore:
                # store arrivals (Compton and Rayleigh)
                arr['E'].append(np.concatenate([Ep,Ec])); arr['u'].append(np.concatenate([uu_,uu_])); 
                arr['W'].append(np.concatenate([WC,WR])); arr['typ'].append(np.concatenate([np.zeros(idx.size,int),np.ones(idx.size,int)]))
            # dose from scattered arrivals (energy transfer approx mu_tr)
            dose[j]+=np.sum(WC*Ep*0)  # placeholder
            kerma_scat_by_pt[j]+=np.sum(WC*(Ec-Ep)*0)  # unused
            dose_scat_by_pt[j]+=np.sum(WC*(Ec-Ep))
        # collisions
        hit=sint<sex
        # move
        pos[idx]=p+L[:,None]*dd
        alive[idx[~hit]]=False
        hid=idx[hit]
        if hid.size==0: continue
        Eh=E[hid]; mt=mu_tot(Eh); pc=mu_c(Eh)/mt; pr=mu_coh(Eh)/mt; pp=mu_pair(Eh)/mt
        r=rng.uniform(size=hid.size)
        isC=r<pc; isR=(r>=pc)&(r<pc+pr); isP=(r>=pc+pr)&(r<pc+pr+pp); isE=~(isC|isR|isP)
        alive[hid[isE]]=False
        # Compton
        ic=hid[isC]
        if ic.size:
            Ei=E[ic]; c=kn_sample_cos(Ei,rng); phi=rng.uniform(0,2*np.pi,ic.size)
            Eout=Ei/(1+Ei/mc2*(1-c))
            d[ic]=rotate(d[ic],c,phi); E[ic]=Eout; order[ic]+=1
        ir=hid[isR]
        if ir.size:
            c=rayleigh_cos(ir.size,rng); phi=rng.uniform(0,2*np.pi,ir.size)
            d[ir]=rotate(d[ir],c,phi); order[ir]+=1
        ip=hid[isP]
        alive[ip]=False
        if ip.size:
            # annihilation photons: spawn two 0.511 MeV photons
            n2=ip.size
            cz=rng.uniform(-1,1,n2); ph=rng.uniform(0,2*np.pi,n2)
            s=np.sqrt(1-cz**2)
            dirA=np.stack([s*np.cos(ph),s*np.sin(ph),cz],1)
            newpos=np.concatenate([pos[ip],pos[ip]]); newd=np.concatenate([dirA,-dirA])
            newE=np.full(2*n2,0.511); neworder=np.ones(2*n2,int)*2
            pos=np.concatenate([pos,newpos]); d=np.concatenate([d,newd]); E=np.concatenate([E,newE])
            order=np.concatenate([order,neworder]); alive=np.concatenate([alive,np.ones(2*n2,bool)])
            # the spawned photons now; keep going
    return dict(arr={k:np.concatenate(v) for k,v in arr.items()}, dose=dose, dscat=dose_scat_by_pt, N=N, depths=depths)
'''
open('/tmp/work/mc.py','w').write(mc_code)
import importlib, mc; importlib.reload(mc)
import time
t=time.time()
res=mc.run(20000, [0.5,5.0], np.random.default_rng(2), store_for=5.0)
print(time.time()-t, res['arr']['E'].size, res['dscat'])
