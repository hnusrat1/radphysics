
mc_code = r'''
import numpy as np
from phys import *
EDGES_PH=np.array([0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4])
EDGES_CT=np.array([-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1.0])
LINES=np.array([1.1732,1.3325])
DEPTHS=[0.25,0.5,0.75,1.0,5.0]
def perp(u,rng):
    a=np.where(np.abs(u[:,0:1])<0.9, np.array([[1.0,0,0]]), np.array([[0,1.0,0]]))
    e1=np.cross(u,a); e1/=np.linalg.norm(e1,axis=1)[:,None]
    e2=np.cross(u,e1)
    return e1,e2
def rotate(u,c,phi,rng):
    e1,e2=perp(u,rng)
    s=np.sqrt(np.maximum(0,1-c*c))
    n=c[:,None]*u+s[:,None]*(np.cos(phi)[:,None]*e1+np.sin(phi)[:,None]*e2)
    return n/np.linalg.norm(n,axis=1)[:,None]
def sample_KN(E,rng):
    k=E/mc2; L=np.log(1+2*k)
    n=E.size; eps=np.empty(n); cs=np.empty(n)
    todo=np.arange(n)
    while todo.size:
        kk=k[todo]; LL=L[todo]
        e=np.exp(-rng.random(todo.size)*LL)
        c=1-(1/e-1)/kk
        s2=1-c*c
        acc=2*rng.random(todo.size) < (e*e+1-e*s2)
        idx=todo[acc]
        eps[idx]=e[acc]; cs[idx]=c[acc]
        todo=todo[~acc]
    return eps,cs
def kn_dsig_dOmega(E,c):
    k=E/mc2; eps=1/(1+k*(1-c)); s2=1-c*c
    return 0.5*re**2*eps**2*(eps+1/eps-s2)
def isotropic(n,rng):
    v=rng.normal(size=(n,3)); return v/np.linalg.norm(v,axis=1)[:,None]

def run(N,seed,depths=DEPTHS,keep_detail_depth=5.0):
    rng=np.random.default_rng(seed)
    # source sampling: uniform in solid angle within pyramid |tx|,|ty|<=0.05
    tx=rng.uniform(-0.05,0.05,N); ty=rng.uniform(-0.05,0.05,N)
    acc=rng.random(N) < (1+tx*tx+ty*ty)**-1.5
    tx,ty=tx[acc],ty[acc]; N=tx.size
    E0=LINES[rng.integers(0,2,N)]
    d=np.stack([tx,ty,np.ones(N)],1); d/=np.linalg.norm(d,axis=1)[:,None]
    pos=np.stack([100*tx,100*ty,np.zeros(N)],1)
    pos=pos+1e-9*d
    dirs=d; E=E0.copy()
    out={k:[] for k in ['det','g','Ea','nz','typ','nx','ny','nzv']}
    # store arrivals: for each det
    detpos=[np.array([0,0,z]) for z in depths]
    nlive=N; 
    # initial "incoming" photons have no prior collisions
    act_pos=pos; act_dir=dirs; act_E=E
    ncoll=0; nmax=60
    it=0
    while act_E.size>0 and it<nmax:
        it+=1
        n=act_E.size
        # distance to exit box
        with np.errstate(divide='ignore',invalid='ignore'):
            tx_=np.where(act_dir[:,0]>0,(15-act_pos[:,0])/act_dir[:,0],np.where(act_dir[:,0]<0,(-15-act_pos[:,0])/act_dir[:,0],np.inf))
            ty_=np.where(act_dir[:,1]>0,(15-act_pos[:,1])/act_dir[:,1],np.where(act_dir[:,1]<0,(-15-act_pos[:,1])/act_dir[:,1],np.inf))
            tz_=np.where(act_dir[:,2]>0,(30-act_pos[:,2])/act_dir[:,2],np.where(act_dir[:,2]<0,(0-act_pos[:,2])/act_dir[:,2],np.inf))
        texit=np.minimum(np.minimum(tx_,ty_),tz_)
        texit=np.maximum(texit,0)
        mu=mu_tot(act_E)
        s=-np.log(rng.random(n))/mu
        inside=s<texit
        if not inside.any(): break
        idx=np.nonzero(inside)[0]
        E_=act_E[idx]; u=act_dir[idx]; x=act_pos[idx]+s[idx,None]*u; mu_=mu[idx]
        ncoll+=idx.size
        # NEE
        pc=mu_inc(E_)/mu_; pr=mu_coh(E_)/mu_; pe=mu_pe(E_)/mu_; pp=mu_pair(E_)/mu_
        for j,dz in enumerate(depths):
            dvec=np.array([0,0,dz])[None,:]-x
            R=np.linalg.norm(dvec,axis=1); nh=dvec/R[:,None]
            c=np.sum(u*nh,axis=1)
            # Compton
            k=E_/mc2; ep=E_/(1+k*(1-c))
            dsig=kn_dsig_dOmega(E_,c)
            p=dsig/kn_total(E_)
            gC=pc*p*np.exp(-mu_tot(ep)*R)/R**2
            # Rayleigh (Thomson-shape approx)
            pR=3*(1+c*c)/(16*np.pi)
            gR=pr*pR*np.exp(-mu_*R)/R**2
            # annihilation photons (2 per pair event), isotropic
            gA=pp*2/(4*np.pi)*np.exp(-mu_tot(0.511)*R)/R**2
            # tally
            for typ,g,Ea in [(0,gC,ep),(1,gR,E_),(2,gA,np.full(E_.size,0.511))]:
                sel=g>0
                out['det'].append(np.full(sel.sum(),j,dtype=np.int8))
                out['g'].append(g[sel]); out['Ea'].append(Ea[sel]); out['typ'].append(np.full(sel.sum(),typ,dtype=np.int8))
                out['nz'].append(nh[sel,2]); out['nx'].append(nh[sel,0]); out['ny'].append(nh[sel,1])
                out['nzv'].append(np.zeros(0))
        # sample interaction for each collision
        u2=rng.random(idx.size)
        isC=u2<pc
        isR=(~isC)&(u2<pc+pr)
        isPE=(~isC)&(~isR)&(u2<pc+pr+pe)
        isPP=(~isC)&(~isR)&(~isPE)
        newpos=[];newdir=[];newE=[]
        # Compton
        ci=np.nonzero(isC)[0]
        if ci.size:
            eps,cs=sample_KN(E_[ci],rng)
            phi=rng.uniform(0,2*np.pi,ci.size)
            nd=rotate(u[ci],cs,phi,rng)
            newpos.append(x[ci]); newdir.append(nd); newE.append(E_[ci]*eps)
        ri=np.nonzero(isR)[0]
        if ri.size:
            cc=np.empty(ri.size); todo=np.arange(ri.size)
            while todo.size:
                cr=rng.uniform(-1,1,todo.size); a=rng.random(todo.size)<(1+cr*cr)/2
                cc[todo[a]]=cr[a]; todo=todo[~a]
            phi=rng.uniform(0,2*np.pi,ri.size)
            nd=rotate(u[ri],cc,phi,rng)
            newpos.append(x[ri]); newdir.append(nd); newE.append(E_[ri])
        pi_=np.nonzero(isPP)[0]
        if pi_.size:
            for _ in range(2):
                newpos.append(x[pi_]); newdir.append(isotropic(pi_.size,rng)); newE.append(np.full(pi_.size,0.511))
        if newE:
            ne_=np.concatenate(newE); nd_=np.concatenate(newdir); npos=np.concatenate(newpos)
            keep=ne_>0.005
            act_E=ne_[keep]; act_dir=nd_[keep]; act_pos=npos[keep]
        else:
            act_E=np.zeros(0); act_dir=np.zeros((0,3)); act_pos=np.zeros((0,3))
    res={}
    for k in ['det','g','Ea','typ','nx','ny','nz']:
        res[k]=np.concatenate(out[k]) if out[k] else np.zeros(0)
    res['N']=N; res['Ntotal']=N
    return res
'''
open('/tmp/work/mc.py','w').write(mc_code)
import importlib, sys, time
sys.path.insert(0,'/tmp/work')
import mc
importlib.reload(mc)
t=time.time(); r=mc.run(200000,1); print(time.time()-t, r['g'].size, r['N'])
