
code = r'''
import numpy as np, time
rng = np.random.default_rng(2024)
RE=2.8179403e-13; NE=3.3428e23; ME=0.51099891
def sig_KN(E):
    k=E/ME; L=np.log(1+2*k)
    return 2*np.pi*RE**2*((1+k)/k**2*(2*(1+k)/(1+2*k)-L/k)+L/(2*k)-(1+3*k)/(1+2*k)**2)
def mu_c(E): return NE*sig_KN(E)
def mu_R(E): return 0.2*(0.01/E)**1.5
def mu_pe(E): return 4.9*(0.01/E)**3.2
def mu_p(E):
    x=np.clip((E-1.022)/0.978,0,None); return 0.0002*x**2
def mu_tot(E): return mu_c(E)+mu_R(E)+mu_pe(E)+mu_p(E)
LO=np.array([-15,-15,0.]); HI=np.array([15,15,30.])
CZ0,CZ1,CR=9.9,10.1,1.0; RB=np.sqrt(1+0.01); CEN=np.array([0,0,10.])

def cyl_chord(o,D):
    n=len(o)
    a=D[:,0]**2+D[:,1]**2; b=2*(o[:,0]*D[:,0]+o[:,1]*D[:,1]); c=o[:,0]**2+o[:,1]**2-CR**2
    ap=a>1e-14
    disc=b*b-4*a*c
    sq=np.sqrt(np.where(disc>=0,disc,0))
    with np.errstate(divide='ignore',invalid='ignore'):
        r1=(-b-sq)/(2*a); r2=(-b+sq)/(2*a)
    lo=np.minimum(r1,r2); hi=np.maximum(r1,r2)
    ok_c = np.where(ap, disc>=0, c<=0)
    s1=np.where(ap,lo,-np.inf); s2=np.where(ap,hi,np.inf)
    Dz=D[:,2]; oz=o[:,2]
    with np.errstate(divide='ignore',invalid='ignore'):
        t1=(CZ0-oz)/Dz; t2=(CZ1-oz)/Dz
    big=np.abs(Dz)>1e-14
    zl=np.where(big,np.minimum(t1,t2),np.where((oz>=CZ0)&(oz<=CZ1),-np.inf,np.inf))
    zh=np.where(big,np.maximum(t1,t2),np.where((oz>=CZ0)&(oz<=CZ1),np.inf,-np.inf))
    sin_=np.maximum(s1,zl); sout=np.minimum(s2,zh)
    hit=ok_c&(sout>sin_)&(sout>0)
    return hit, np.maximum(sin_,0), sout

def exit_dist(p,D):
    with np.errstate(divide='ignore',invalid='ignore'):
        tp=np.where(D>0,(HI-p)/D,np.inf); tm=np.where(D<0,(LO-p)/D,np.inf)
    return np.min(np.minimum(tp,tm),axis=1)

def rotate(axis,c,phi):
    a=np.where(np.abs(axis[:,0:1])<0.9,np.array([[1.,0,0]]),np.array([[0,1.,0]]))
    e1=np.cross(axis,a); e1/=np.linalg.norm(e1,axis=1,keepdims=True)
    e2=np.cross(axis,e1)
    s=np.sqrt(np.clip(1-c*c,0,None))
    return c[:,None]*axis+s[:,None]*(np.cos(phi)[:,None]*e1+np.sin(phi)[:,None]*e2)

def kn_eps(E):
    k=E/ME; emin=1/(1+2*k); eps=np.empty_like(E)
    todo=np.arange(len(E))
    while len(todo)>0:
        kk=k[todo]; em=emin[todo]
        e=np.exp(rng.uniform(size=len(todo))*np.log(em))
        c=1-(1/e-1)/kk; s2=1-c*c
        acc=rng.uniform(size=len(todo))<=(1+e*e-e*s2)/2
        eps[todo[acc]]=e[acc]
        todo=todo[~acc]
    return eps

def sample_cos_thomson(n):
    out=np.empty(n); todo=np.arange(n)
    while len(todo):
        c=rng.uniform(-1,1,len(todo)); acc=rng.uniform(size=len(todo))<(1+c*c)/2
        out[todo[acc]]=c[acc]; todo=todo[~acc]
    return out

def sample_photo_cos(T):
    b=np.sqrt(1-(ME/(T+ME))**2)
    grid=np.linspace(-1,1,401)
    M=np.max((1-grid[None,:]**2)/(1-b[:,None]*grid[None,:])**4,axis=1)*1.001
    out=np.empty(len(T)); todo=np.arange(len(T))
    while len(todo):
        c=rng.uniform(-1,1,len(todo))
        f=(1-c*c)/(1-b[todo]*c)**4
        acc=rng.uniform(size=len(todo))*M[todo]<f
        out[todo[acc]]=c[acc]; todo=todo[~acc]
    return out

NAMES=np.array(['compton','photoelectric','pair','rayleigh'])
def make_events(Ein,W):
    n=len(Ein)
    mc=mu_c(Ein); mr=mu_R(Ein); mpe=mu_pe(Ein); mp=mu_p(Ein); mt=mc+mr+mpe+mp
    xi=rng.uniform(size=n)*mt
    proc=np.where(xi<mc,0,np.where(xi<mc+mr,3,np.where(xi<mc+mr+mpe,1,2)))
    ev={'W':W,'process':NAMES[proc],'e_in':Ein,'e_out':np.full(n,np.nan),'cos_s':np.full(n,np.nan),
        't_e':np.zeros(n),'cos_e':np.full(n,np.nan),'t_pos':np.full(n,np.nan)}
    m=proc==0
    if m.any():
        E=Ein[m]; eps=kn_eps(E); k=E/ME
        c=1-(1/eps-1)/k
        Eo=eps*E
        pe_x=E-Eo*c; pe_mag=np.sqrt(np.maximum(E*E+Eo*Eo-2*E*Eo*c,1e-30))
        ev['e_out'][m]=Eo; ev['cos_s'][m]=c; ev['t_e'][m]=E-Eo
        ev['cos_e'][m]=np.clip(pe_x/pe_mag,-1,1)
    m=proc==1
    if m.any():
        ev['t_e'][m]=Ein[m]; ev['cos_e'][m]=sample_photo_cos(Ein[m])
    m=proc==2
    if m.any():
        Tt=Ein[m]-2*ME; f=rng.uniform(size=m.sum())
        te=Tt*f; ev['t_e'][m]=te; ev['t_pos'][m]=Tt-te
        th=np.abs(rng.normal(0,1,m.sum()))*ME/(te+ME)
        ev['cos_e'][m]=np.cos(np.clip(th,0,np.pi))
    m=proc==3
    if m.any():
        c=sample_cos_thomson(m.sum()); ev['e_out'][m]=Ein[m]; ev['cos_s'][m]=c; ev['t_e'][m]=0.0
    return ev

def nee_channels(x,inc,E):
    M=len(E)
    vec=CEN[None,:]-x; d=np.linalg.norm(vec,axis=1)
    axis=vec/np.maximum(d,1e-12)[:,None]
    full=d<=RB
    cosa=np.sqrt(np.clip(1-(RB/np.maximum(d,1e-12))**2,0,None))
    cmin=np.where(full,-1.0,cosa)
    OmB=np.where(full,4*np.pi,2*np.pi*(1-cosa))
    res=[]
    mt=mu_tot(E)
    xi=rng.uniform(size=M); phi=rng.uniform(0,2*np.pi,M)
    c=cmin+(1-cmin)*xi
    Od=rotate(axis,c,phi)
    hit,sin_,sout=cyl_chord(x,Od)
    cos_th=np.clip(np.sum(Od*inc,axis=1),-1,1)
    k=E/ME
    eps=1/(1+k*(1-cos_th)); Ep=eps*E
    dsig=RE**2/2*eps**2*(eps+1/eps-(1-cos_th*cos_th))
    mtp=mu_tot(Ep)
    T=np.exp(-mtp*sin_); P=1-np.exp(-mtp*np.clip(sout-sin_,0,None))
    W=np.where(hit,NE*dsig/mt*OmB*T*P,0.0)
    res.append((W,Ep,hit))
    pR=3/(16*np.pi)*(1+cos_th*cos_th)
    T=np.exp(-mt*sin_); P=1-np.exp(-mt*np.clip(sout-sin_,0,None))
    W=np.where(hit,mu_R(E)/mt*pR*OmB*T*P,0.0)
    res.append((W,E.copy(),hit))
    Ea=np.full(M,ME); mta=mu_tot(Ea)
    T=np.exp(-mta*sin_); P=1-np.exp(-mta*np.clip(sout-sin_,0,None))
    W=np.where(hit,mu_p(E)/mt*2/(4*np.pi)*OmB*T*P,0.0)
    res.append((W,Ea,hit))
    return res

def simulate(N):
    ncand=int(N*1.1)+1000
    us=[];vs=[];tot=0
    while tot<N:
        uu=rng.uniform(-0.05,0.05,2*N); vv=rng.uniform(-0.05,0.05,2*N)
        acc=rng.uniform(size=2*N)<(1+uu*uu+vv*vv)**(-1.5)
        us.append(uu[acc]); vs.append(vv[acc]); tot+=acc.sum()
    u=np.concatenate(us)[:N]; v=np.concatenate(vs)[:N]
    D=np.stack([u,v,np.ones(N)],1); D/=np.linalg.norm(D,axis=1,keepdims=True)
    p0=np.zeros((N,3)); p0[:,0]=100/D[:,2]*D[:,0]; p0[:,1]=100/D[:,2]*D[:,1]
    hit,sin_,sout=cyl_chord(p0,D)
    mu2=mu_tot(np.array([2.0]))[0]
    T=np.exp(-mu2*sin_); P=1-np.exp(-mu2*np.clip(sout-sin_,0,None))
    Wp=np.where(hit,T*P,0.0)
    m=Wp>0
    prim=make_events(np.full(m.sum(),2.0),Wp[m])
    nee=[]  # list of dicts (W,Ep)
    pos=p0.copy(); dr=D.copy(); E=np.full(N,2.0)
    alive=np.ones(N,bool)
    ncoll=0
    while alive.any():
        idx=np.nonzero(alive)[0]
        mt=mu_tot(E[idx])
        s=-np.log(rng.uniform(size=len(idx)))/mt
        sb=exit_dist(pos[idx],dr[idx])
        coll=s<sb
        alive[idx[~coll]]=False
        ci=idx[coll]; sc=s[coll]
        if len(ci)==0: break
        x=pos[ci]+sc[:,None]*dr[ci]
        pos[ci]=x
        ncoll+=len(ci)
        Ec=E[ci]; inc=dr[ci].copy()
        for (W,Ep,h) in nee_channels(x,inc,Ec):
            if h.any(): nee.append((W[h],Ep[h]))
        mt=mu_tot(Ec); mc=mu_c(Ec); mr=mu_R(Ec); mp=mu_p(Ec)
        xi=rng.uniform(size=len(ci))*mt
        isC=xi<mc; isR=(xi>=mc)&(xi<mc+mr); isP=(xi>=mc+mr)&(xi<mc+mr+mp)
        isPE=xi>=mc+mr+mp
        alive[ci[isPE]]=False
        alive[ci[isP]]=False
        pidx=np.nonzero(isP)[0]
        if len(pidx):
            xp=x[pidx]
            g=rng.normal(size=(len(pidx),3)); g/=np.linalg.norm(g,axis=1,keepdims=True)
            pos=np.concatenate([pos,xp,xp]); dr=np.concatenate([dr,g,-g]); E=np.concatenate([E,np.full(2*len(pidx),ME)])
            alive=np.concatenate([alive,np.ones(2*len(pidx),bool)])
        cidx=ci[isC]
        if len(cidx):
            Ein=E[cidx]; eps=kn_eps(Ein); k=Ein/ME; cth=1-(1/eps-1)/k
            ph=rng.uniform(0,2*np.pi,len(cidx))
            dr[cidx]=rotate(dr[cidx],cth,ph); E[cidx]=eps*Ein
        ridx=ci[isR]
        if len(ridx):
            cth=sample_cos_thomson(len(ridx)); ph=rng.uniform(0,2*np.pi,len(ridx))
            dr[ridx]=rotate(dr[ridx],cth,ph)
        low=alive&(E<0.01)
        alive[low]=False
    W_nee=np.concatenate([a[0] for a in nee]); E_nee=np.concatenate([a[1] for a in nee])
    m=W_nee>0
    nev=make_events(E_nee[m],W_nee[m])
    return prim,nev,ncoll
'''
open('mc.py','w').write(code)
import time
t0=time.time()
import importlib, sys
sys.path.insert(0,'.')
import mc
importlib.reload(mc)
prim,nev,ncoll=mc.simulate(200000)
print("time",time.time()-t0,"ncoll",ncoll)
print("prim n",len(prim['W']),"W",prim['W'].sum())
print("nee n",len(nev['W']),"W",nev['W'].sum())
