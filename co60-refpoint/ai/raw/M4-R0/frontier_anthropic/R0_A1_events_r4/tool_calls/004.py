
code = r'''
import numpy as np, time, json, sys
rng=np.random.default_rng(int(sys.argv[1]) if len(sys.argv)>1 else 2024)
re=2.8179403e-13; ne=3.3428e23; me=0.51099895
def sigma_KN(E):
    k=E/me
    t1=(1+k)/k**2*(2*(1+k)/(1+2*k)-np.log(1+2*k)/k)
    t2=np.log(1+2*k)/(2*k)-(1+3*k)/(1+2*k)**2
    return 2*np.pi*re**2*(t1+t2)
def mu_parts(E):
    sKN=ne*sigma_KN(E)
    coh=sKN*0.032*(E/0.1)**(-1.2)
    pe=0.0028*(E/0.1)**(-3.0)
    pair=np.where(E>1.022, 0.0005*np.clip((E-1.022)/0.978,0,None)**2, 0.0)
    return sKN,coh,pe,pair
# PE envelope table
Etab=np.geomspace(0.01,1.4,300)
def sauter_f(c,E):
    g=1+E/me; b=np.sqrt(1-1/g**2)
    return (1-c*c)/(1-b*c)**4
cg=np.linspace(-1,1,2001)
fenv_tab=np.array([sauter_f(cg,E).max() for E in Etab])*1.05
def sample_KN(E):
    k=E/me; n=len(E); out=np.empty(n); todo=np.arange(n)
    while todo.size>0:
        kk=k[todo]; m=rng.uniform(-1,1,todo.size)
        eps=1/(1+kk*(1-m)); f=eps**2*(eps+1/eps-(1-m*m))
        acc=rng.uniform(0,2,todo.size)<f
        out[todo[acc]]=m[acc]; todo=todo[~acc]
    return out
def sample_thomson(n):
    out=np.empty(n); todo=np.arange(n)
    while todo.size>0:
        m=rng.uniform(-1,1,todo.size)
        acc=rng.uniform(0,2,todo.size)<(1+m*m)
        out[todo[acc]]=m[acc]; todo=todo[~acc]
    return out
def sample_sauter(E):
    n=len(E); out=np.empty(n); todo=np.arange(n)
    env=np.interp(E,Etab,fenv_tab)
    while todo.size>0:
        m=rng.uniform(-1,1,todo.size)
        Et=E[todo]; g=1+Et/me; b=np.sqrt(1-1/g**2)
        f=(1-m*m)/(1-b*m)**4
        acc=rng.uniform(0,1,todo.size)*env[todo]<f
        out[todo[acc]]=m[acc]; todo=todo[~acc]
    return out
def rotate(d,c,phi):
    n=d.shape[0]
    a=np.zeros_like(d); use_x=np.abs(d[:,0])<0.9
    a[use_x,0]=1.0; a[~use_x,1]=1.0
    e1=np.cross(d,a); e1/=np.linalg.norm(e1,axis=1)[:,None]
    e2=np.cross(d,e1)
    s=np.sqrt(np.clip(1-c*c,0,None))
    return c[:,None]*d+s[:,None]*(np.cos(phi)[:,None]*e1+np.sin(phi)[:,None]*e2)
def iso_dirs(n):
    c=rng.uniform(-1,1,n); phi=rng.uniform(0,2*np.pi,n)
    s=np.sqrt(1-c*c)
    return np.stack([s*np.cos(phi),s*np.sin(phi),c],axis=1)
def s_exit_box(p,d):
    lo=np.array([-15.,-15.,0.]); hi=np.array([15.,15.,30.])
    with np.errstate(divide='ignore',invalid='ignore'):
        t_hi=np.where(d>0,(hi-p)/d,np.inf)
        t_lo=np.where(d<0,(lo-p)/d,np.inf)
    return np.minimum(t_hi.min(axis=1),t_lo.min(axis=1))

# slab geometry
ZS1,ZS2=4.9,5.1; RC=1.0
def slab_interval(p,d):
    x,y,z=p[:,0],p[:,1],p[:,2]; ux,uy,uz=d[:,0],d[:,1],d[:,2]
    with np.errstate(divide='ignore',invalid='ignore'):
        sz1=(ZS1-z)/uz; sz2=(ZS2-z)/uz
    lo_z=np.minimum(sz1,sz2); hi_z=np.maximum(sz1,sz2)
    lo_z=np.where(np.abs(uz)<1e-14,-np.inf,lo_z); hi_z=np.where(np.abs(uz)<1e-14,np.inf,hi_z)
    # if uz==0 and z in slab: whole line OK; else none
    inside_z=(z>=ZS1)&(z<=ZS2)
    lo_z=np.where((np.abs(uz)<1e-14)&~inside_z,np.inf,lo_z)
    a=ux*ux+uy*uy; b=2*(x*ux+y*uy); c=x*x+y*y-RC*RC
    with np.errstate(divide='ignore',invalid='ignore'):
        disc=b*b-4*a*c
        sq=np.sqrt(np.clip(disc,0,None))
        c1=(-b-sq)/(2*a); c2=(-b+sq)/(2*a)
    lo_c=np.where(a>1e-14,np.minimum(c1,c2),np.where(c<=0,-np.inf,np.inf))
    hi_c=np.where(a>1e-14,np.maximum(c1,c2),np.where(c<=0,np.inf,-np.inf))
    lo_c=np.where((a>1e-14)&(disc<0),np.inf,lo_c)
    sa=np.maximum(np.maximum(lo_z,lo_c),0.0)
    sb=np.minimum(hi_z,hi_c)
    valid=sb>sa
    return sa,sb,valid

# candidate storage
C={k:[] for k in ['proc','Ein','Eout','cosS','te','cose','tpos','w','prim']}
def add(proc,Ein,Eout,cosS,te,cose,tpos,w,prim):
    C['proc'].append(np.full(len(w),proc,dtype=np.int8)); C['Ein'].append(Ein); C['Eout'].append(Eout)
    C['cosS'].append(cosS); C['te'].append(te); C['cose'].append(cose); C['tpos'].append(tpos)
    C['w'].append(w); C['prim'].append(prim)
PROC={'compton':0,'rayleigh':1,'photoelectric':2,'pair':3}

def nee(E,pos,d,prim):
    mu=sum(mu_parts(E)[0:1])+0  # placeholder
    sK,coh,pe,pr=mu_parts(E); mu=sK+coh+pe+pr
    sa,sb,valid=slab_interval(pos,d)
    idx=np.nonzero(valid)[0]
    if idx.size==0: return
    Ev=E[idx]; mv=mu[idx]
    q=np.exp(-mv*sa[idx])*(1-np.exp(-mv*(sb[idx]-sa[idx])))
    pC=sK[idx]/mv; pR=coh[idx]/mv; pP=pe[idx]/mv; pPa=pr[idx]/mv
    dv=d[idx]; pv=prim[idx]
    # Compton
    cc=sample_KN(Ev); kk=Ev/me
    Ep=Ev/(1+kk*(1-cc))
    ce=(Ev-Ep*cc)/np.sqrt(Ev**2+Ep**2-2*Ev*Ep*cc)
    m=q*pC
    add(PROC['compton'],Ev,Ep,cc,Ev-Ep,np.clip(ce,-1,1),np.full(idx.size,np.nan),m,pv)
    # Rayleigh
    cr=sample_thomson(idx.size)
    m=q*pR
    add(PROC['rayleigh'],Ev,Ev,cr,np.zeros(idx.size),np.full(idx.size,np.nan),np.full(idx.size,np.nan),m,pv)
    # photoelectric
    cp=sample_sauter(Ev)
    m=q*pP
    add(PROC['photoelectric'],Ev,np.full(idx.size,np.nan),np.full(idx.size,np.nan),Ev,cp,np.full(idx.size,np.nan),m,pv)
    # pair
    ok=Ev>1.022
    if ok.any():
        j=np.nonzero(ok)[0]
        Ej=Ev[j]; avail=Ej-2*me
        fr=rng.uniform(0,1,j.size)
        Tp=fr*avail; Te=avail-Tp
        th=(me/(Te+me))*np.abs(rng.normal(0,1,j.size))
        cos_e=np.cos(np.clip(th,0,np.pi))
        m=(q*pPa)[j]
        add(PROC['pair'],Ej,np.full(j.size,np.nan),np.full(j.size,np.nan),Te,cos_e,Tp,m,pv[j])
    # store also per-photon weights for normalization check
    C_stats['nee_sum']+=float(q.sum())
    C_stats['nee_prim']+=float(q[pv].sum())

C_stats={'nee_sum':0.0,'nee_prim':0.0}

def run(N0, Ebins=(1.1732,1.3325)):
    # source photons
    n_acc=0; txs=[];tys=[]
    while n_acc<N0:
        M=int(N0*1.2)+100
        tx=rng.uniform(-0.05,0.05,M); ty=rng.uniform(-0.05,0.05,M)
        acc=rng.uniform(0,1,M)<(1+tx*tx+ty*ty)**(-1.5)
        txs.append(tx[acc]); tys.append(ty[acc]); n_acc+=acc.sum()
    tx=np.concatenate(txs)[:N0]; ty=np.concatenate(tys)[:N0]
    nrm=np.sqrt(tx*tx+ty*ty+1)
    d=np.stack([tx/nrm,ty/nrm,1/nrm],axis=1)
    pos=np.stack([tx*100.0,ty*100.0,np.zeros(N0)],axis=1)
    E=np.where(rng.uniform(0,1,N0)<0.5,Ebins[0],Ebins[1])
    prim=np.ones(N0,dtype=bool)
    gen=0; t0=time.time()
    while E.size>0:
        nee(E,pos,d,prim)
        sK,coh,pe,pr=mu_parts(E); mu=sK+coh+pe+pr
        s=-np.log(rng.uniform(0,1,E.size))/mu
        sx=s_exit_box(pos,d)
        alive=s<sx
        E=E[alive]; d=d[alive]; pos=pos[alive]+s[alive,None]*d if False else pos[alive]
        sK,coh,pe,pr=sK[alive],coh[alive],pe[alive],pr[alive]; mu=mu[alive]; s=s[alive]
        pos=pos+s[:,None]*d
        prim=np.zeros(E.size,dtype=bool)  # after first interaction; primary flag handled below
        u=rng.uniform(0,1,E.size)
        pC=sK/mu; pRl=coh/mu; pPe=pe/mu
        isC=u<pC
        isR=(~isC)&(u<pC+pRl)
        isP=(~isC)&(~isR)&(u<pC+pRl+pPe)
        isPair=(~isC)&(~isR)&(~isP)
        newE=[];newd=[];newp=[]
        # Compton
        if isC.any():
            i=np.nonzero(isC)[0]
            c=sample_KN(E[i]); Ep=E[i]/(1+E[i]/me*(1-c))
            phi=rng.uniform(0,2*np.pi,i.size)
            nd=rotate(d[i],c,phi)
            keep=Ep>=EC
            newE.append(Ep[keep]); newd.append(nd[keep]); newp.append(pos[i][keep])
        if isR.any():
            i=np.nonzero(isR)[0]
            c=sample_thomson(i.size); phi=rng.uniform(0,2*np.pi,i.size)
            nd=rotate(d[i],c,phi)
            newE.append(E[i]); newd.append(nd); newp.append(pos[i])
        if isPair.any():
            i=np.nonzero(isPair)[0]
            dd=iso_dirs(i.size)
            newE.append(np.full(i.size,0.511)); newd.append(dd); newp.append(pos[i])
            newE.append(np.full(i.size,0.511)); newd.append(-dd); newp.append(pos[i])
        if len(newE)==0: break
        E=np.concatenate(newE); d=np.concatenate(newd); pos=np.concatenate(newp)
        # secondary annihilation/scattered photons: treated as "scattered"
        gen+=1
        if gen>200: break
    return

'''
print(len(code))
