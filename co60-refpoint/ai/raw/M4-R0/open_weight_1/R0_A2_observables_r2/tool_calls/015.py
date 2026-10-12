import numpy as np

def load_xcom():
    with open('/tmp/data/xcom_water.txt') as f:
        lines = [l for l in f if not l.startswith('#') and l.strip()]
    x = np.loadtxt(lines)
    return x[:,0], x[:,1], x[:,2], x[:,3], x[:,4], x[:,5], x[:,6]
E_x, mu_coh, mu_incoh, mu_pe, mu_pairn, mu_paire, mu_tot = load_xcom()
def mu_interp(E, col):
    return np.exp(np.interp(np.log(E), np.log(E_x), np.log(col)))
def mu_total(E): return mu_interp(E, mu_tot)
def mu_incoh_E(E): return mu_interp(E, mu_incoh)
def mu_coh_E(E): return mu_interp(E, mu_coh)
def mu_pe_E(E): return mu_interp(E, mu_pe)
def mu_pair_E(E): return mu_interp(E, mu_pairn+mu_paire)

mc2=0.51099895; re2=7.9408e-26
def sample_compton(E, u, rng):
    a = E/mc2; n=E.size
    cosT=np.empty(n); done=np.zeros(n,bool)
    while not done.all():
        c=rng.uniform(-1,1,n)
        r_=1.0/(1.0+a*(1.0-c)); br=r_**2*(r_+1.0/r_-(1.0-c*c))
        take=(~done)&(rng.uniform(0,2,n)<br); cosT[take]=c[take]; done[take]=True
    Ep=E/(1.0+a*(1.0-cosT)); sinT=np.sqrt(1.0-cosT**2); phi=rng.uniform(0,2*np.pi,n)
    ux,uy,uz=u[:,0],u[:,1],u[:,2]
    cross=np.stack([-uy,ux,np.zeros(n)],axis=1); nrm=np.linalg.norm(cross,axis=1)
    par=nrm<1e-8; cross[par]=[1,0,0]; nrm[par]=1.0
    e1=cross/nrm[:,None]; e2=np.cross(u,e1)
    ud=cosT[:,None]*u+(np.sin(phi)[:,None]*e2+np.cos(phi)[:,None]*e1)*sinT[:,None]
    return Ep, ud/np.linalg.norm(ud,axis=1)[:,None]

def cyl_intersect(xa, ua, z0, z1):
    aq=ua[:,0]**2+ua[:,1]**2; bq=2.0*(xa[:,0]*ua[:,0]+xa[:,1]*ua[:,1]); cq=xa[:,0]**2+xa[:,1]**2-1.0
    par=aq<1e-14; tin=np.full(aq.shape,-np.inf); tout=np.full(aq.shape,np.inf); ok=np.ones(aq.shape,bool)
    npar=~par; disc=bq[npar]**2-4*aq[npar]*cq[npar]; pos=disc>=0; sq=np.sqrt(np.maximum(disc,0))
    t1=(-bq[npar]-sq)/(2*aq[npar]); t2=(-bq[npar]+sq)/(2*aq[npar]); ti=np.minimum(t1,t2); to=np.maximum(t1,t2)
    tin_np=tin[npar]; tout_np=tout[npar]; tin_np[~pos]=np.inf; tout_np[~pos]=-np.inf
    tin_np[pos]=ti[pos]; tout_np[pos]=to[pos]; tin[npar]=tin_np; tout[npar]=tout_np
    parrad=(xa[par,0]**2+xa[par,1]**2)<=1.0; ok[par]=parrad
    with np.errstate(divide='ignore',invalid='ignore'):
        tza=np.where(ua[:,2]>0,(z0-xa[:,2])/ua[:,2],np.where(ua[:,2]<0,(z0-xa[:,2])/ua[:,2],np.inf))
        tzb=np.where(ua[:,2]>0,(z1-xa[:,2])/ua[:,2],np.where(ua[:,2]<0,(z1-xa[:,2])/ua[:,2],np.inf))
    tzlo=np.minimum(tza,tzb); tzhi=np.maximum(tza,tzb)
    tlo=np.maximum(tin,tzlo); thi=np.minimum(tout,tzhi)
    valid=ok&(thi>tlo)&(thi>0)
    return tlo,thi,valid

# ---- minimal full loop, with prints
rng=np.random.default_rng(3); n=100000
ph_edges=np.array([0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4])
ct_edges=np.array([-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1])
score=np.zeros((28,2,10)); Vcyl=np.pi*0.2
E=np.where(rng.random(n)<0.5,1.1732,1.3325)
xs=rng.uniform(-5,5,n); ys=rng.uniform(-5,5,n)
d=np.stack([xs,ys,np.full(n,100.0)],axis=1); nrm=np.linalg.norm(d,axis=1); u=d/nrm[:,None]
x=np.stack([xs,ys,np.zeros(n)],axis=1)
prim=np.ones(n,bool); alive=np.ones(n,bool); mu=mu_total(E)
gen=0
while alive.any():
    gen+=1
    idx=np.where(alive)[0]; m=idx.size
    Ea=E[idx]; ua=u[idx]; xa=x[idx]; mua=mu[idx]; pra=prim[idx]
    tlo,thi,valid=cyl_intersect(xa,ua,4.9,5.1)
    t_in=np.maximum(tlo,0.0); L=np.maximum(thi-t_in,0.0)
    contrib=np.where(valid,np.exp(-mua*t_in)*L/Vcyl,0.0)
    cs=contrib.sum()
    if gen==1:
        print("gen1 contrib.sum:", cs)
    if cs>0:
        Eb=np.clip(np.searchsorted(ph_edges,Ea,side='right')-1,0,27)
        cb=np.clip(np.searchsorted(ct_edges,ua[:,2],side='right')-1,0,9)
        pr=pra.astype(int)
        np.add.at(score,(Eb,pr,cb),contrib)
    if gen==1:
        print("  after add: primary total", score[:,0,:].sum(), " scattered total", score[:,1,:].sum())
    # transport
    s=-np.log(rng.random(m))/mua
    with np.errstate(divide='ignore',invalid='ignore'):
        tz=np.where(ua[:,2]>0,(30.0-xa[:,2])/ua[:,2],np.where(ua[:,2]<0,(0.0-xa[:,2])/ua[:,2],np.inf))
        txp=np.where(ua[:,0]>0,(15.0-xa[:,0])/ua[:,0],np.where(ua[:,0]<0,(-15.0-xa[:,0])/ua[:,0],np.inf))
        typ=np.where(ua[:,1]>0,(15.0-xa[:,1])/ua[:,1],np.where(ua[:,1]<0,(-15.0-xa[:,1])/ua[:,1],np.inf))
    texit=np.minimum(tz,np.minimum(txp,typ)); seg=np.minimum(s,texit)
    inter=(s<texit-1e-12); ids=idx[inter]
    if ids.size>0:
        Es=E[ids]; us=u[ids]; xs_=x[ids]
        xnew=xs_+us*seg[inter][:,None]
        p_incoh=mu_incoh_E(Es); p_coh=mu_coh_E(Es); p_pe=mu_pe_E(Es); p_pair=mu_pair_E(Es); p_tot=mu[ids]
        r=rng.random(ids.size)*p_tot
        is_incoh=r<p_incoh; is_coh=(r>=p_incoh)&(r<p_incoh+p_coh); is_pe=(r>=p_incoh+p_coh)&(r<p_incoh+p_coh+p_pe); is_pair=~(is_incoh|is_coh|is_pe)
        if is_incoh.any():
            Enew,unew=sample_compton(Es[is_incoh],us[is_incoh],rng)
            E[ids[is_incoh]]=Enew; u[ids[is_incoh]]=unew; prim[ids[is_incoh]]=False
            x[ids[is_incoh]]=xnew[is_incoh]; mu[ids[is_incoh]]=mu_total(Enew)
        if is_coh.any():
            ct=rng.uniform(0.999,1.0,is_coh.sum()); st=np.sqrt(1-ct*ct); phi=rng.uniform(0,2*np.pi,is_coh.sum())
            uc=us[is_coh]; ex=np.stack([-uc[:,1],uc[:,0],np.zeros(uc.shape[0])],axis=1); nm=np.linalg.norm(ex,axis=1)
            ex[nm<1e-8]=[1,0,0]; nm[nm<1e-8]=1; e1=ex/nm[:,None]; e2=np.cross(uc,e1)
            unew=ct[:,None]*uc+(np.cos(phi)[:,None]*e1+np.sin(phi)[:,None]*e2)*st[:,None]
            unew=unew/np.linalg.norm(unew,axis=1)[:,None]
            u[ids[is_coh]]=unew; prim[ids[is_coh]]=False; x[ids[is_coh]]=xnew[is_coh]
        dead=is_pe|is_pair
        if dead.any(): alive[ids[dead]]=False
    exited=~inter
    if exited.any(): alive[idx[exited]]=False
    if gen==1:
        print("  gen1 remaining alive:", alive.sum(), "primary among alive:", prim[alive].sum())
print("final primary/Phi0:", score[:,0,:].sum()/n*110.25, "scattered/Phi0:", score[:,1,:].sum()/n*110.25)