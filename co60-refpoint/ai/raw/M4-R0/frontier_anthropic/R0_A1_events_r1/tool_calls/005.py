
sim = r'''
import numpy as np, json, sys, time
sys.path.insert(0,'.')
from xs import mu_parts, me
seed=int(sys.argv[1]); N=int(sys.argv[2])
rng=np.random.default_rng(seed)
def sample_dirs(n):
    got=[]; tot=0
    cmin=1/np.sqrt(1+2*0.05**2)
    while tot<n:
        m=2*(n-tot)+100
        c=rng.uniform(cmin,1,m); ph=rng.uniform(0,2*np.pi,m)
        s=np.sqrt(1-c*c); tn=s/c
        ok=(np.abs(tn*np.cos(ph))<=0.05)&(np.abs(tn*np.sin(ph))<=0.05)
        d=np.stack([s*np.cos(ph),s*np.sin(ph),c],1)[ok]
        got.append(d); tot+=len(d)
    return np.concatenate(got)[:n]
def rotate(d,ct,ph):
    st=np.sqrt(np.clip(1-ct*ct,0,None))
    a=np.where(np.abs(d[:,0:1])<0.9, np.array([[1.0,0,0]]), np.array([[0,1.0,0]]))
    u=np.cross(d,a); u/=np.linalg.norm(u,axis=1,keepdims=True)
    v=np.cross(d,u)
    nd=ct[:,None]*d+(st*np.cos(ph))[:,None]*u+(st*np.sin(ph))[:,None]*v
    return nd/np.linalg.norm(nd,axis=1,keepdims=True)
def in_cyl(p):
    return (p[:,0]**2+p[:,1]**2<=1.0)&(p[:,2]>=4.9)&(p[:,2]<=5.1)
N_init=N
lines=np.array([1.1732,1.3325])
E=lines[rng.integers(0,2,N)]
D=sample_dirs(N)
P=np.zeros((N,3)); P[:,2]=-100.0
P=P+D*(100.0/D[:,2])[:,None]     # point on surface z=0
P[:,2]=0.0
ev=[]   # list of dicts
t1=time.time(); gen=0
while len(E)>0 and gen<500:
    gen+=1
    inc,pe,pair,coh=mu_parts(E)
    mu=inc+pe+pair+coh
    with np.errstate(divide='ignore',invalid='ignore'):
        tx=np.where(D[:,0]>0,(15-P[:,0])/D[:,0],np.where(D[:,0]<0,(-15-P[:,0])/D[:,0],np.inf))
        ty=np.where(D[:,1]>0,(15-P[:,1])/D[:,1],np.where(D[:,1]<0,(-15-P[:,1])/D[:,1],np.inf))
        tz=np.where(D[:,2]>0,(30-P[:,2])/D[:,2],np.where(D[:,2]<0,(0-P[:,2])/D[:,2],np.inf))
    tb=np.minimum(np.minimum(tx,ty),tz)
    s=-np.log(rng.uniform(size=len(E)))/mu
    idx=np.nonzero(s<tb)[0]
    n=len(idx)
    if n==0: break
    Ei=E[idx]; di=D[idx]
    Pi=P[idx]+s[idx,None]*di
    cyl=in_cyl(Pi)
    r=rng.uniform(size=n)*mu[idx]
    pc=inc[idx]; pp=pe[idx]; pa=pair[idx]
    isC=r<pc
    isP=(~isC)&(r<pc+pp)
    isA=(~isC)&(~isP)&(r<pc+pp+pa)
    isR=(~isC)&(~isP)&(~isA)
    newE=np.zeros(n); newD=np.zeros((n,3))
    # Compton
    ci=np.nonzero(isC)[0]
    if len(ci):
        Ec=Ei[ci]; a=Ec/me; c=np.empty(len(ci)); todo=np.arange(len(ci))
        while len(todo):
            cc=rng.uniform(-1,1,len(todo))
            Ep_=Ec[todo]/(1+a[todo]*(1-cc)); eps=Ep_/Ec[todo]
            f=eps**2*(eps+1/eps-(1-cc*cc))
            ok=rng.uniform(size=len(todo))*2<f
            c[todo[ok]]=cc[ok]; todo=todo[~ok]
        Ep_=Ec/(1+a*(1-c)); ph=rng.uniform(0,2*np.pi,len(ci))
        newE[ci]=Ep_; newD[ci]=rotate(di[ci],c,ph)
        pex=(Ec-Ep_*c)/np.sqrt(Ec**2+Ep_**2-2*Ec*Ep_*c)
        sel=cyl[ci]
        k=ci[sel]
        for j in np.nonzero(sel)[0]:
            ev.append(('compton',float(Ec[j]),float(Ep_[j]),float(c[j]),float(Ec[j]-Ep_[j]),float(pex[j]),None))
    # Rayleigh
    ri=np.nonzero(isR)[0]
    if len(ri):
        c=np.empty(len(ri)); todo=np.arange(len(ri))
        while len(todo):
            cc=rng.uniform(-1,1,len(todo))
            ok=rng.uniform(size=len(todo))*2<(1+cc*cc)
            c[todo[ok]]=cc[ok]; todo=todo[~ok]
        ph=rng.uniform(0,2*np.pi,len(ri))
        newE[ri]=Ei[ri]; newD[ri]=rotate(di[ri],c,ph)
        sel=cyl[ri]
        for j in np.nonzero(sel)[0]:
            ev.append(('rayleigh',float(Ei[ri][j]),float(Ei[ri][j]),float(c[j]),0.0,None,None))
    # photoelectric
    pi_=np.nonzero(isP)[0]
    if len(pi_):
        sel=cyl[pi_]
        cz=np.empty(len(pi_)); todo=np.arange(len(pi_))
        while len(todo):
            cc=rng.uniform(-1,1,len(todo)); ok=rng.uniform(size=len(todo))<(1-cc*cc)
            cz[todo[ok]]=cc[ok]; todo=todo[~ok]
        for j in np.nonzero(sel)[0]:
            ev.append(('photoelectric',float(Ei[pi_][j]),None,None,float(Ei[pi_][j]),float(cz[j]),None))
    # pair
    ai=np.nonzero(isA)[0]
    if len(ai):
        Tt=Ei[ai]-2*me
        U=rng.uniform(size=len(ai))
        Tp_all=Tt*U; Te_all=Tt*(1-U)
        th=np.abs(rng.normal(size=len(ai)))*me/(Te_all+me)
        ce_all=np.cos(np.minimum(th,np.pi))
        sel=cyl[ai]
        for j in np.nonzero(sel)[0]:
            ev.append(('pair',float(Ei[ai][j]),None,None,float(Te_all[j]),float(ce_all[j]),float(Tp_all[j])))
        dn=rng.normal(size=(len(ai),3)); dn/=np.linalg.norm(dn,axis=1,keepdims=True)
        Pa=Pi[ai]
    else:
        dn=np.zeros((0,3)); Pa=np.zeros((0,3))
    # next generation: scattered (Compton/Rayleigh) photons above cutoff, plus annihilation photons
    surv=(isC|isR)&(newE>0.01)
    sidx=np.nonzero(surv)[0]
    na=len(ai)
    P=np.concatenate([Pi[sidx],Pa,Pa]); D=np.concatenate([newD[sidx],dn,-dn])
    E=np.concatenate([newE[sidx],np.full(2*na,me)])
print('gen',gen,'time',round(time.time()-t1,1),'events',len(ev),'N',N_init)
json.dump(ev,open(f'ev_{seed}.json','w'))
'''
open('sim.py','w').write(sim)
import subprocess
r=subprocess.run(['python','sim.py','1','200000'],capture_output=True,text=True,cwd='.')
print(r.stdout, r.stderr[-2000:])
