
mc = r'''
import numpy as np, sys, time, pickle
from phys import *
trap=np.trapezoid
seed=int(sys.argv[1]); N=int(sys.argv[2])
rng=np.random.default_rng(seed)
# solid angle
ug=np.linspace(-0.05,0.05,801)
UU,VV=np.meshgrid(ug,ug)
Omega=trap(trap((1+UU**2+VV**2)**-1.5,ug,axis=1),ug)
# sample directions uniform in solid angle by rejection
def sample_dirs(n):
    out=[]
    m=0
    while m<n:
        k=2*n
        u=rng.uniform(-0.05,0.05,k); v=rng.uniform(-0.05,0.05,k)
        acc=rng.uniform(0,1,k)<(1+u*u+v*v)**-1.5
        out.append(np.stack([u[acc],v[acc]],1)); m+=acc.sum()
    return np.concatenate(out)[:n]
uv=sample_dirs(N)
d=np.column_stack([uv[:,0],uv[:,1],np.ones(N)]); d/=np.linalg.norm(d,axis=1)[:,None]
pos=np.column_stack([100*uv[:,0],100*uv[:,1],np.zeros(N)])
E=np.full(N,2.0); prim=np.ones(N,bool); alive=np.ones(N,bool)
Vc=np.pi*1.0**2*0.2
NORM=Omega*110.0**2/(N*Vc)   # fluence/Phi0 = NORM*sum(dl)
# tallies
Eb=np.array([0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4])
cb=np.array([-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1.0])
T={}
def add(key,val): T[key]=T.get(key,0.0)+val
hist_ph_prim=np.zeros(28); hist_ph_scat=np.zeros(28)
ang_prim=np.zeros(10); ang_scat=np.zeros(10)
inter={ (p,t):0.0 for p in ['P','S'] for t in ['c','pe','pair','coh']}
transf={ (p,t):0.0 for p in ['P','S'] for t in ['c','pe','pair','coh']}
Lsum={'P':0.0,'S':0.0}; ESum={'P':0.0,'S':0.0}
# electron production samples inside V
ep_T=[];ep_cos=[];ep_w=[];ep_prim=[];ep_kind=[]
def cyl_len(p0,dd,L):
    x0,y0,z0=p0[:,0],p0[:,1],p0[:,2]; dx,dy,dz=dd[:,0],dd[:,1],dd[:,2]
    lo=np.full(len(L),-np.inf); hi=np.full(len(L),np.inf)
    with np.errstate(divide='ignore',invalid='ignore'):
        t1=(9.9-z0)/dz; t2=(10.1-z0)/dz
    tzl=np.where(np.abs(dz)>1e-12,np.minimum(t1,t2),np.where((z0>=9.9)&(z0<=10.1),-np.inf,np.inf))
    tzh=np.where(np.abs(dz)>1e-12,np.maximum(t1,t2),np.where((z0>=9.9)&(z0<=10.1),np.inf,-np.inf))
    a=dx*dx+dy*dy; b=2*(x0*dx+y0*dy); c=x0*x0+y0*y0-1
    disc=b*b-4*a*c
    with np.errstate(divide='ignore',invalid='ignore'):
        sq=np.sqrt(np.maximum(disc,0))
        r1=(-b-sq)/(2*a); r2=(-b+sq)/(2*a)
    trl=np.where(a>1e-14,np.where(disc>=0,r1,np.inf),np.where(c<=0,-np.inf,np.inf))
    trh=np.where(a>1e-14,np.where(disc>=0,r2,-np.inf),np.where(c<=0,np.inf,-np.inf))
    ta=np.maximum.reduce([np.zeros(len(L)),tzl,trl]); tb=np.minimum.reduce([L,tzh,trh])
    return np.maximum(tb-ta,0.0), ta, tb
lo=np.array([-15,-15,0.]); hi=np.array([15,15,30.])
new=[]  # annihilation photons etc. appended
t0=time.time()
for gen in range(200):
    idx=np.where(alive)[0]
    if idx.size==0: break
    p=pos[idx]; dd=d[idx]; e=E[idx]; pr=prim[idx]
    mu=mu_tot(e)
    s=-np.log(rng.uniform(size=idx.size))/mu
    with np.errstate(divide='ignore',invalid='ignore'):
        tx=np.where(dd[:,0]>0,(hi[0]-p[:,0])/dd[:,0],np.where(dd[:,0]<0,(lo[0]-p[:,0])/dd[:,0],np.inf))
        ty=np.where(dd[:,1]>0,(hi[1]-p[:,1])/dd[:,1],np.where(dd[:,1]<0,(lo[1]-p[:,1])/dd[:,1],np.inf))
        tz=np.where(dd[:,2]>0,(hi[2]-p[:,2])/dd[:,2],np.where(dd[:,2]<0,(lo[2]-p[:,2])/dd[:,2],np.inf))
    dex=np.minimum(np.minimum(tx,ty),tz)
    L=np.minimum(s,dex)
    # tally in V
    dl,ta,tb=cyl_len(p,dd,L)
    m=dl>0
    if m.any():
        pm=pr[m]; em=e[m]; dlm=dl[m]; mum=mu[m]; ddm=dd[m]; ids=idx[m]
        # fluence
        for flag,key in [(True,'P'),(False,'S')]:
            sel=pm==flag
            if sel.any():
                Lsum[key]+=dlm[sel].sum(); ESum[key]+=(dlm[sel]*em[sel]).sum()
                bins=np.clip(np.searchsorted(Eb,em[sel],side='right')-1,0,27)
                inb=(em[sel]<=1.4)
                h=np.bincount(bins[inb],weights=dlm[sel][inb],minlength=28)
                (hist_ph_prim if flag else hist_ph_scat)[:]+=h
                cbin=np.clip(np.searchsorted(cb,ddm[sel][:,2],side='right')-1,0,9)
                (ang_prim if flag else ang_scat)[:]+=np.bincount(cbin,weights=dlm[sel],minlength=10)
                # interactions expected
                ce=mu_c(em[sel]); pe_=mu_pe(em[sel]); pa=mu_pair(em[sel]); co=mu_coh(em[sel])
                k=key
                inter[(k,'c')]+= (dlm[sel]*ce).sum(); inter[(k,'pe')]+=(dlm[sel]*pe_).sum()
                inter[(k,'pair')]+=(dlm[sel]*pa).sum(); inter[(k,'coh')]+=(dlm[sel]*co).sum()
                transf[(k,'c')]+=(dlm[sel]*ce*Tbar(em[sel])/em[sel]*em[sel]/em[sel]*0+dlm[sel]*ce*(em[sel]-em[sel]*(1-0))*0).sum()  # placeholder
                tc=ce*Tbar(em[sel]); 
                transf[(k,'c')]+=(dlm[sel]*tc).sum()
                transf[(k,'pe')]+=(dlm[sel]*pe_*em[sel]).sum()
                transf[(k,'pair')]+=(dlm[sel]*pa*(em[sel]-1.022)).sum()
                # electron production sampling
                ns=8
                sub_idx=np.where(sel)[0]
                ww=dlm[sel]
                # Compton electrons
                Ec=np.repeat(em[sel],ns); wc=np.repeat(ww*ce/ns,1)
                Ep_,cs=sample_KN(Ec,rng)
                Te=Ec-Ep_
                # electron direction cos wrt z
                dpar=np.repeat(ddm[sel],ns,axis=0)
                phi=rng.uniform(0,2*np.pi,Ec.size)
                u,vv=perp_basis(dpar)
                sn=np.sqrt(np.clip(1-cs**2,0,1))
                dsc=cs[:,None]*dpar+(sn*np.cos(phi))[:,None]*u+(sn*np.sin(phi))[:,None]*vv
                pe_vec=Ec[:,None]*dpar-Ep_[:,None]*dsc
                cz=pe_vec[:,2]/np.linalg.norm(pe_vec,axis=1)
                ep_T.append(Te); ep_cos.append(cz); ep_w.append(wc); ep_prim.append(np.full(Te.size,flag)); ep_kind.append(np.zeros(Te.size,int))
                # photoelectrons
                wpe=ww*pe_
                ep_T.append(em[sel]); ep_cos.append(ddm[sel][:,2]); ep_w.append(wpe); ep_prim.append(np.full(sel.sum(),flag)); ep_kind.append(np.full(sel.sum(),1))
                # pair electrons (e- and e+ each half share), two electrons per event
                wpa=ww*pa
                if (pa>0).any():
                    xi=rng.uniform(size=sel.sum())
                    tot=(em[sel]-1.022).clip(0)
                    ep_T.append(tot*xi); ep_cos.append(ddm[sel][:,2]); ep_w.append(wpa); ep_prim.append(np.full(sel.sum(),flag)); ep_kind.append(np.full(sel.sum(),2))
                    ep_T.append(tot*(1-xi)); ep_cos.append(ddm[sel][:,2]); ep_w.append(wpa); ep_prim.append(np.full(sel.sum(),flag)); ep_kind.append(np.full(sel.sum(),3))
    # process interactions
    inside=s<dex
    ii=idx[inside]
    if ii.size==0:
        alive[idx]=False
        continue
    # choose process
    eI=E[ii]; muI=mu_tot(eI); xi=rng.uniform(size=ii.size)*muI
    pc=mu_c(eI); ppe=mu_pe(eI); ppa=mu_pair(eI)
    kind=np.where(xi<pc,0,np.where(xi<pc+ppe,1,np.where(xi<pc+ppe+ppa,2,3)))
    newpos=pos[ii]+s[inside][:,None]*d[ii]
    pos[ii]=newpos
    alive[np.setdiff1d(idx,ii)]=False
    # Compton
    c_ids=ii[kind==0]
    if c_ids.size:
        Ein=E[c_ids]
        Ep_,cs=sample_KN(Ein,rng)
        phi=rng.uniform(0,2*np.pi,c_ids.size)
        dsc,_,_=(None,None,None)
        u,vv=perp_basis(d[c_ids]); sn=np.sqrt(np.clip(1-cs**2,0,1))
        dn=cs[:,None]*d[c_ids]+(sn*np.cos(phi))[:,None]*u+(sn*np.sin(phi))[:,None]*vv
        d[c_ids]=dn; E[c_ids]=Ep_; prim[c_ids]=False
    # photoelectric
    p_ids=ii[kind==1]
    if p_ids.size:
        alive[p_ids]=False
    # pair
    a_ids=ii[kind==2]
    if a_ids.size:
        alive[a_ids]=False
        # two annihilation photons back to back, isotropic
        nn=a_ids.size
        ct=rng.uniform(-1,1,nn); ph=rng.uniform(0,2*np.pi,nn)
        st=np.sqrt(1-ct**2)
        dv=np.column_stack([st*np.cos(ph),st*np.sin(ph),ct])
        new.append((pos[a_ids].copy(),dv,np.full(nn,0.511)))
        new.append((pos[a_ids].copy(),-dv,np.full(nn,0.511)))
    # Rayleigh
    r_ids=ii[kind==3]
    if r_ids.size:
        ct=rng.uniform(-1,1,r_ids.size)
        # forward-biased simple Rayleigh angle sampling (1+c^2) pdf
        u,vv=perp_basis(d[r_ids]); phi=rng.uniform(0,2*np.pi,r_ids.size); sn=np.sqrt(np.clip(1-ct**2,0,1))
        d[r_ids]=ct[:,None]*d[r_ids]+(sn*np.cos(phi))[:,None]*u+(sn*np.sin(phi))[:,None]*vv
        prim[r_ids]=False
    # Compton and Rayleigh remain alive; those with small E
    # photoelectric/pair kinds handled
# gather annihilation photons: process as additional histories (simplified: tracked separately)
# ---- second pass for annihilation photons (scattered, from pairs) ----
if new:
    P=np.concatenate([x[0] for x in new]); Dd=np.concatenate([x[1] for x in new]); Ee=np.concatenate([x[2] for x in new])
else:
    P=np.zeros((0,3));Dd=np.zeros((0,3));Ee=np.zeros(0)
out=dict(T=T,inter=inter,transf=transf,Lsum=Lsum,ESum=ESum,hist_ph_prim=hist_ph_prim,hist_ph_scat=hist_ph_scat,ang_prim=ang_prim,ang_scat=ang_scat,
         ep_T=np.concatenate(ep_T),ep_cos=np.concatenate(ep_cos),ep_w=np.concatenate(ep_w),ep_prim=np.concatenate(ep_prim),ep_kind=np.concatenate(ep_kind),
         NORM=NORM,Omega=Omega,N=N,Vc=Vc,ann_pos=P,ann_dir=Dd,ann_E=Ee)
pickle.dump(out,open(f'out_{seed}.pkl','wb'))
print("done",time.time()-t0,"Omega",Omega, "Lsum",Lsum, "ann",P.shape)
'''
open('mc.py','w').write(mc)
