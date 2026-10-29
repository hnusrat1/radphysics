
sim = r'''
import numpy as np, json, sys, time
sys.path.insert(0,'.')
from xs import mu_parts, me
seed=int(sys.argv[1]); N=int(sys.argv[2])
rng=np.random.default_rng(seed)
H=30.0
def sample_dirs(n):
    # uniform in solid angle within pyramid of half-angle 5 cm /100 cm (square)
    out=np.zeros((0,3)); 
    got=[]
    tot=0
    cmin=1/np.sqrt(1+2*0.05**2)  # cos of corner direction
    while tot<n:
        m=2*(n-tot)+100
        c=rng.uniform(cmin,1,m); ph=rng.uniform(0,2*np.pi,m)
        s=np.sqrt(1-c*c); tn=s/c
        ok=(np.abs(tn*np.cos(ph))<=0.05)&(np.abs(tn*np.sin(ph))<=0.05)
        d=np.stack([s*np.cos(ph),s*np.sin(ph),c],1)[ok]
        got.append(d); tot+=len(d)
    return np.concatenate(got)[:n]
def perp_basis(d):
    a=np.where(np.abs(d[:,0:1])<0.9, np.array([[1.0,0,0]]), np.array([[0,1.0,0]]))
    u=np.cross(d,a); u/=np.linalg.norm(u,axis=1,keepdims=True)
    v=np.cross(d,u)
    return u,v
def rotate(d,ct,ph):
    st=np.sqrt(np.clip(1-ct*ct,0,None))
    u,v=perp_basis(d)
    nd=ct[:,None]*d+(st*np.cos(ph))[:,None]*u+(st*np.sin(ph))[:,None]*v
    return nd/np.linalg.norm(nd,axis=1,keepdims=True)
def in_cyl(p):
    return (p[:,0]**2+p[:,1]**2<=1.0)&(p[:,2]>=4.9)&(p[:,2]<=5.1)
# initial photons
lines=np.array([1.1732,1.3325])
E0=lines[rng.integers(0,2,N)]
D0=sample_dirs(N)
t0=-(-100.0)/D0[:,2]  # distance from source (z=-100) to surface z=0
P0=np.stack([0*D0[:,0],0*D0[:,1],-100.0+0*D0[:,2]],1)+D0*(100.0/D0[:,2])[:,None]
P=P0; Dd=D0; E=E0.copy()
prim=np.ones(N,bool)
ev={k:[] for k in ['process','e_in','e_out','cos_s','t_e','cos_e','t_p']}
def record(proc,ein,eout,cs,te,ce,tp):
    n=len(ein)
    ev['process']+= [proc]*n
    ev['e_in']+=list(ein); ev['e_out']+=list(eout); ev['cos_s']+=list(cs)
    ev['t_e']+=list(te); ev['cos_e']+=list(ce); ev['t_p']+=list(tp)
nAnn=0
t1=time.time()
gen=0
while len(E)>0 and gen<200:
    gen+=1
    inc,pe,pair,coh=mu_parts(E)
    mu=inc+pe+pair+coh
    # distance to boundary of cube
    with np.errstate(divide='ignore',invalid='ignore'):
        tx=np.where(Dd[:,0]>0,(15-P[:,0])/Dd[:,0],np.where(Dd[:,0]<0,(-15-P[:,0])/Dd[:,0],np.inf))
        ty=np.where(Dd[:,1]>0,(15-P[:,1])/Dd[:,1],np.where(Dd[:,1]<0,(-15-P[:,1])/Dd[:,1],np.inf))
        tz=np.where(Dd[:,2]>0,(30-P[:,2])/Dd[:,2],np.where(Dd[:,2]<0,(0-P[:,2])/Dd[:,2],np.inf))
    tb=np.minimum(np.minimum(tx,ty),tz)
    s=-np.log(rng.uniform(size=len(E)))/mu
    inside=s<tb
    idx=np.nonzero(inside)[0]
    if len(idx)==0: break
    Pi=P[idx]+s[idx,None]*Dd[idx]
    Ei=E[idx]; di=Dd[idx]; mui=mu[idx]
    cyl=in_cyl(Pi)
    r=rng.uniform(size=len(idx))*mui
    pc=inc[idx]; pp=pe[idx]; pa=pair[idx]; pco=coh[idx]
    isC=r<pc
    isP=(~isC)&(r<pc+pp)
    isA=(~isC)&(~isP)&(r<pc+pp+pa)
    isR=(~isC)&(~isP)&(~isA)
    # Compton sampling
    newE=Ei.copy(); newD=di.copy(); alive=np.ones(len(idx),bool)
    # Compton
    ci=np.nonzero(isC)[0]
    if len(ci):
        Ec=Ei[ci]; a=Ec/me
        c=np.empty(len(ci)); acc=np.zeros(len(ci),bool)
        todo=np.arange(len(ci))
        while len(todo):
            cc=rng.uniform(-1,1,len(todo))
            Ep=Ec[todo]/(1+a[todo]*(1-cc))
            eps=Ep/Ec[todo]
            f=eps**2*(eps+1/eps-(1-cc*cc))
            ok=rng.uniform(size=len(todo))*2<f
            c[todo[ok]]=cc[ok]; acc[todo[ok]]=True
            todo=todo[~ok]
        Ep=Ec/(1+a*(1-c))
        ph=rng.uniform(0,2*np.pi,len(ci))
        newE[ci]=Ep
        newD[ci]=rotate(di[ci],c,ph)
        pex=(Ec-Ep*c)/np.sqrt(Ec**2+Ep**2-2*Ec*Ep*c)
        Te=Ec-Ep
        rec_c=ci[cyl[ci]]
    # Rayleigh
    ri=np.nonzero(isR)[0]
    if len(ri):
        c=np.empty(len(ri)); todo=np.arange(len(ri))
        while len(todo):
            cc=rng.uniform(-1,1,len(todo))
            ok=rng.uniform(size=len(todo))*2<(1+cc*cc)
            c[todo[ok]]=cc[ok]; todo=todo[~ok]
        ph=rng.uniform(0,2*np.pi,len(ri))
        newD[ri]=rotate(di[ri],c,ph)
    # photo
    pi_=np.nonzero(isP)[0]
    # pair
    ai=np.nonzero(isA)[0]
    # record events in cylinder
    m=cyl[ci] if len(ci) else np.zeros(0,bool)
    rec_idx=[]
    if len(ci):
        sel=cyl[ci]
        q=ci[sel]
        record(['compton']*len(q), Ei[q], Ep[sel], c[sel], Te[sel], pex[sel], [None]*len(q))
    if len(ri):
        sel=cyl[ri]; q=ri[sel]
        record(['rayleigh']*len(q), Ei[q], Ei[q], c[sel], [0.0]*len(q), [None]*len(q), [None]*len(q))
    if len(pi_):
        sel=cyl[pi_]; q=pi_[sel]
        # photoelectron cos from sin^2 distribution (rejection)
        cz=np.empty(len(q)); todo=np.arange(len(q))
        while len(todo):
            cc=rng.uniform(-1,1,len(todo)); ok=rng.uniform(size=len(todo))<(1-cc*cc)
            cz[todo[ok]]=cc[ok]; todo=todo[~ok]
        record(['photoelectric']*len(q), Ei[q], [None]*len(q), [None]*len(q), Ei[q], cz, [None]*len(q))
    if len(ai):
        sel=cyl[ai]; q=ai[sel]
        Tt=Ei[ai]-2*me
        U=rng.uniform(size=len(ai))
        Tp_all=Tt*U; Te_all=Tt*(1-U)
        th=np.abs(rng.normal(size=len(ai)))*me/(Te_all+me)
        ce_all=np.cos(np.minimum(th,np.pi))
        record(['pair']*len(q), Ei[q], [None]*len(q), [None]*len(q), Te_all[sel], ce_all[sel], Tp_all[sel])
    # apply outcomes: Compton continues, Rayleigh continues, photo kills, pair kills and makes 2 photons
    keep=np.ones(len(idx),bool)
    keep[isP]=False; keep[isA]=False
    # low energy cutoff: kill photons below 0.01 MeV
    newE_keep=newE
    # secondary annihilation photons
    if len(ai):
        n=len(ai)
        dn=rng.normal(size=(n,3)); dn/=np.linalg.norm(dn,axis=1,keepdims=True)
        Pa=Pi[ai]
        # new photons
        newP=np.concatenate([Pa,Pa]); newD2=np.concatenate([dn,-dn]); newE2=np.full(2*n,me)
    else:
        newP=np.zeros((0,3)); newD2=np.zeros((0,3)); newE2=np.zeros(0)
    # build next generation: scattered photons continuing
    cont=idx[keep & (newE>0.01)]
    P_next=Pi[keep & (newE>0.01)]
    D_next=newD[keep & (newE>0.01)]
    E_next=newE[keep & (newE>0.01)]
    # photons whose interaction location is at Pi; in next gen they start at Pi
    # Non-interacting photons (escaping) are dropped; photons not interacting leave the pool
    # Build full pool: continuing scattered photons + annihilation photons
    P=np.concatenate([P_next,newP]); Dd=np.concatenate([D_next,newD2]); E=np.concatenate([E_next,newE2])
    # clear and fine
print('gen',gen,'time',time.time()-t1, 'events',len(ev['process']))
json.dump({k:[ (float(x) if isinstance(x,(float,np.floating)) else x) for x in v] for k,v in ev.items()}, open(f'ev_{seed}.json','w'))
'''
open('sim.py','w').write(sim)
import subprocess
print(subprocess.run(['python','sim.py','1','200000'],capture_output=True,text=True,cwd='.'))
