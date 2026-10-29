
mc = r'''
import numpy as np, sys, time, pickle
from phys import *
trap=np.trapezoid
seed=int(sys.argv[1]); N=int(sys.argv[2])
rng=np.random.default_rng(seed)
ug=np.linspace(-0.05,0.05,801)
UU,VV=np.meshgrid(ug,ug)
Omega=trap(trap((1+UU**2+VV**2)**-1.5,ug,axis=1),ug)
def sample_uv(n):
    out=[];m=0
    while m<n:
        k=2*n
        u=rng.uniform(-0.05,0.05,k); v=rng.uniform(-0.05,0.05,k)
        acc=rng.uniform(0,1,k)<(1+u*u+v*v)**-1.5
        out.append(np.stack([u[acc],v[acc]],1)); m+=acc.sum()
    return np.concatenate(out)[:n]
uv=sample_uv(N)
d=np.column_stack([uv[:,0],uv[:,1],np.ones(N)]); d/=np.linalg.norm(d,axis=1)[:,None]
pos=np.column_stack([100*uv[:,0],100*uv[:,1],np.zeros(N)])
E=np.full(N,2.0); prim=np.ones(N,bool); alive=np.ones(N,bool)
Vc=np.pi*1.0**2*0.2
NORM=Omega*110.0**2/(N*Vc)
Eb=np.array([0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4])
cb=np.array([-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1.0])
pk=['c','pe','pair','coh']
inter={(p,t):0.0 for p in 'PS' for t in pk}
transf={(p,t):0.0 for p in 'PS' for t in pk}
Lsum={'P':0.0,'S':0.0}; ESum={'P':0.0,'S':0.0}
hist_prim=np.zeros(28); hist_scat=np.zeros(28)
ang_prim=np.zeros(10); ang_scat=np.zeros(10)
ep=dict(T=[],cos=[],w=[],prim=[],kind=[])
def addep(T_,c_,w_,pr_,k_):
    ep['T'].append(T_); ep['cos'].append(c_); ep['w'].append(w_); ep['prim'].append(pr_); ep['kind'].append(np.full(T_.size,k_))
def cyl_len(p0,dd,L):
    x0,y0,z0=p0[:,0],p0[:,1],p0[:,2]; dx,dy,dz=dd[:,0],dd[:,1],dd[:,2]
    with np.errstate(divide='ignore',invalid='ignore'):
        t1=(9.9-z0)/dz; t2=(10.1-z0)/dz
    inz=(z0>=9.9)&(z0<=10.1)
    tzl=np.where(np.abs(dz)>1e-12,np.minimum(t1,t2),np.where(inz,-np.inf,np.inf))
    tzh=np.where(np.abs(dz)>1e-12,np.maximum(t1,t2),np.where(inz,np.inf,-np.inf))
    a=dx*dx+dy*dy; b=2*(x0*dx+y0*dy); c=x0*x0+y0*y0-1
    disc=b*b-4*a*c
    with np.errstate(divide='ignore',invalid='ignore'):
        sq=np.sqrt(np.maximum(disc,0))
        r1=(-b-sq)/(2*a); r2=(-b+sq)/(2*a)
    small=a<=1e-14
    trl=np.where(small,np.where(c<=0,-np.inf,np.inf),np.where(disc>=0,r1,np.inf))
    trh=np.where(small,np.where(c<=0,np.inf,-np.inf),np.where(disc>=0,r2,-np.inf))
    ta=np.maximum(np.maximum(0.0,tzl),trl); tb=np.minimum(np.minimum(L,tzh),trh)
    return np.maximum(tb-ta,0.0)
lo=np.array([-15,-15,0.]); hi=np.array([15,15,30.])
t0=time.time()
for gen in range(300):
    idx=np.where(alive)[0]
    if idx.size==0: break
    p=pos[idx]; dd=d[idx]; e=E[idx]; pr=prim[idx]
    mu=mu_tot(e)
    s=-np.log(rng.uniform(size=idx.size))/mu
    with np.errstate(divide='ignore',invalid='ignore'):
        def ex(a):
            return np.where(dd[:,a]>0,(hi[a]-p[:,a])/dd[:,a],np.where(dd[:,a]<0,(lo[a]-p[:,a])/dd[:,a],np.inf))
        dex=np.minimum(np.minimum(ex(0),ex(1)),ex(2))
    L=np.minimum(s,dex)
    dl=cyl_len(p,dd,L)
    m=dl>0
    if m.any():
        pm=pr[m]; em=e[m]; dlm=dl[m]; ddm=dd[m]
        for flag,key in [(True,'P'),(False,'S')]:
            sel=pm==flag
            if not sel.any(): continue
            es=em[sel]; ds=dlm[sel]; dz_=ddm[sel]
            Lsum[key]+=ds.sum(); ESum[key]+=(ds*es).sum()
            b=np.clip(np.searchsorted(Eb,es,side='right')-1,0,27)
            inb=es<=1.4
            (hist_prim if flag else hist_scat)[:]+=np.bincount(b[inb],weights=ds[inb],minlength=28)
            cbin=np.clip(np.searchsorted(cb,dz_[:,2],side='right')-1,0,9)
            (ang_prim if flag else ang_scat)[:]+=np.bincount(cbin,weights=ds,minlength=10)
            ce=mu_c(es); pe_=mu_pe(es); pa=mu_pair(es)
            inter[(key,'c')]+=(ds*ce).sum(); inter[(key,'pe')]+=(ds*pe_).sum()
            inter[(key,'pair')]+=(ds*pa).sum(); inter[(key,'coh')]+=(ds*mu_coh(es)).sum()
            transf[(key,'c')]+=(ds*ce*Tbar(es)).sum()
            transf[(key,'pe')]+=(ds*pe_*es).sum()
            transf[(key,'pair')]+=(ds*pa*(es-1.022)).sum()
            ns=8
            Ec=np.repeat(es,ns); dpar=np.repeat(dz_,ns,axis=0)
            wc=np.repeat(ds*ce/ns,ns)
            Ep_,cs=sample_KN(Ec,rng)
            Te=Ec-Ep_
            phi=rng.uniform(0,2*np.pi,Ec.size)
            u,vv=perp_basis(dpar); sn=np.sqrt(np.clip(1-cs**2,0,1))
            dsc=cs[:,None]*dpar+(sn*np.cos(phi))[:,None]*u+(sn*np.sin(phi))[:,None]*vv
            pev=Ec[:,None]*dpar-Ep_[:,None]*dsc
            cz=pev[:,2]/np.linalg.norm(pev,axis=1)
            addep(Te,cz,wc,np.full(Te.size,flag),0)
            # photoelectrons (direction ~ photon direction; crude)
            addep(es.copy(),dz_[:,2].copy(),ds*pe_,np.full(es.size,flag),1)
            # pair: e- (kind 2) and e+ (kind 3), uniform sharing
            tot=np.clip(es-1.022,0,None); xi=rng.uniform(size=es.size)
            addep(tot*xi,dz_[:,2].copy(),ds*pa,np.full(es.size,flag),2)
            addep(tot*(1-xi),dz_[:,2].copy(),ds*pa,np.full(es.size,flag),3)
    # interactions
    inside=s<dex
    ii=idx[inside]
    alive[idx[~inside]]=False
    if ii.size==0: continue
    eI=E[ii]
    xi=rng.uniform(size=ii.size)*mu_tot(eI)
    pc=mu_c(eI); ppe=mu_pe(eI); ppa=mu_pair(eI)
    kind=np.where(xi<pc,0,np.where(xi<pc+ppe,1,np.where(xi<pc+ppe+ppa,2,3)))
    # move
    pos[ii]=pos[ii]+s[inside][:,None]*d[ii]
    prim[ii]=False
    c_ids=ii[kind==0]
    if c_ids.size:
        Ein=E[c_ids]
        Ep_,cs=sample_KN(Ein,rng)
        phi=rng.uniform(0,2*np.pi,c_ids.size)
        u,vv=perp_basis(d[c_ids]); sn=np.sqrt(np.clip(1-cs**2,0,1))
        d[c_ids]=cs[:,None]*d[c_ids]+(sn*np.cos(phi))[:,None]*u+(sn*np.sin(phi))[:,None]*vv
        E[c_ids]=Ep_
    p_ids=ii[kind==1]; alive[p_ids]=False
    a_ids=ii[kind==2]
    if a_ids.size:
        alive[a_ids]=False
        nn=a_ids.size
        ct=rng.uniform(-1,1,nn); ph=rng.uniform(0,2*np.pi,nn); st=np.sqrt(1-ct**2)
        dv=np.column_stack([st*np.cos(ph),st*np.sin(ph),ct])
        newp=np.concatenate([pos[a_ids],pos[a_ids]]); newd=np.concatenate([dv,-dv]); newE=np.full(2*nn,0.511)
        pos=np.concatenate([pos,newp]); d=np.concatenate([d,newd]); E=np.concatenate([E,newE])
        prim=np.concatenate([prim,np.zeros(2*nn,bool)]); alive=np.concatenate([alive,np.ones(2*nn,bool)])
    r_ids=ii[kind==3]
    if r_ids.size:
        ct=rng.uniform(-1,1,r_ids.size)
        u,vv=perp_basis(d[r_ids]); phi=rng.uniform(0,2*np.pi,r_ids.size); sn=np.sqrt(np.clip(1-ct**2,0,1))
        d[r_ids]=ct[:,None]*d[r_ids]+(sn*np.cos(phi))[:,None]*u+(sn*np.sin(phi))[:,None]*vv
    # garbage: remove dead to keep arrays manageable
    if gen%5==4:
        keep=alive
        pos=pos[keep]; d=d[keep]; E=E[keep]; prim=prim[keep]; alive=alive[keep]
out=dict(inter=inter,transf=transf,Lsum=Lsum,ESum=ESum,hist_prim=hist_prim,hist_scat=hist_scat,ang_prim=ang_prim,ang_scat=ang_scat,
         ep={k:np.concatenate(v) for k,v in ep.items()},NORM=NORM,Omega=Omega,N=N,Vc=Vc)
pickle.dump(out,open(f'out_{seed}.pkl','wb'))
print("done",round(time.time()-t0,1),"s; Lsum",Lsum,"ESum",ESum)
'''
open('mc.py','w').write(mc)
import subprocess,time
t=time.time()
r=subprocess.run(['python','mc.py','1','100000'],capture_output=True,text=True)
print(r.stdout,r.stderr[-2000:],time.time()-t)
