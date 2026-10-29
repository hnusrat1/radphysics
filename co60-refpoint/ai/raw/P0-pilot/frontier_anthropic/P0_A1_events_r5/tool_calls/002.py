
def sample_cos_thomson(n):
    out=np.empty(n); todo=np.arange(n)
    while len(todo):
        c=rng.uniform(-1,1,len(todo)); acc=rng.uniform(size=len(todo))<(1+c*c)/2
        out[todo[acc]]=c[acc]; todo=todo[~acc]
    return out

def sample_photo_cos(T):
    # Sauter-like angular distribution  sin^2/(1-beta cos)^4 ; rejection
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

def make_events(Ein,W):
    n=len(Ein)
    mc=mu_c(Ein); mr=mu_R(Ein); mpe=mu_pe(Ein); mp=mu_p(Ein); mt=mc+mr+mpe+mp
    xi=rng.uniform(size=n)*mt
    proc=np.where(xi<mc,0,np.where(xi<mc+mr,3,np.where(xi<mc+mr+mpe,1,2)))  # 0 compton,1 photo,2 pair,3 rayleigh
    names=np.array(['compton','photoelectric','pair','rayleigh'])
    ev={'W':W,'process':names[proc],'e_in':Ein,'e_out':np.full(n,np.nan),'cos_s':np.full(n,np.nan),
        't_e':np.zeros(n),'cos_e':np.full(n,np.nan),'t_pos':np.full(n,np.nan)}
    # compton
    m=proc==0
    if m.any():
        E=Ein[m]; eps=kn_eps(E,rng); k=E/ME
        c=1-(1/eps-1)/k
        Eo=eps*E
        pe_x=E-Eo*c; pe_mag=np.sqrt(E*E+Eo*Eo-2*E*Eo*c)
        ev['e_out'][m]=Eo; ev['cos_s'][m]=c; ev['t_e'][m]=E-Eo
        ev['cos_e'][m]=np.clip(pe_x/pe_mag,-1,1)
    m=proc==1
    if m.any():
        ev['t_e'][m]=Ein[m]; ev['cos_e'][m]=sample_photo_cos(Ein[m])
    m=proc==2
    if m.any():
        Tt=Ein[m]-2*ME; f=rng.uniform(size=m.sum())
        te=Tt*f; ev['t_e'][m]=te; ev['t_pos'][m]=Tt-te
        th=np.abs(rng.normal(0,1,m.sum()))*ME/(te+ME)*1.0
        ev['cos_e'][m]=np.cos(np.clip(th,0,np.pi))
    m=proc==3
    if m.any():
        c=sample_cos_thomson(m.sum()); ev['e_out'][m]=Ein[m]; ev['cos_s'][m]=c; ev['t_e'][m]=0.0
    return ev

def chord_from(x,dirs):
    return cyl_chord(x,dirs)

def nee_channels(x,inc,E,rng):
    """x (M,3) collision points, inc (M,3) incident dirs, E (M,) energies.
    returns list of (W,Eprime) arrays for compton, rayleigh, pair-annihilation channels"""
    M=len(E)
    vec=CEN[None,:]-x; d=np.linalg.norm(vec,axis=1)
    axis=vec/d[:,None]
    full=d<=RB
    cosa=np.sqrt(np.clip(1-(RB/np.maximum(d,1e-12))**2,0,None))
    cmin=np.where(full,-1.0,cosa)
    OmB=np.where(full,4*np.pi,2*np.pi*(1-cosa))
    res=[]
    mt=mu_tot(E)
    xi=rng.uniform(size=M); phi=rng.uniform(0,2*np.pi,M)
    c=cmin+(1-cmin)*xi
    Od=rotate(axis,c,phi)
    hit,sin_,sout=chord_from(x,Od)
    cos_th=np.sum(Od*inc,axis=1)
    k=E/ME
    # Compton
    cth=np.clip(cos_th,-1,1)
    eps=1/(1+k*(1-cth)); Ep=eps*E
    dsig=RE**2/2*eps**2*(eps+1/eps-(1-cth*cth))
    mtp=mu_tot(Ep)
    T=np.exp(-mtp*sin_); P=1-np.exp(-mtp*np.clip(sout-sin_,0,None))
    W=np.where(hit,NE*dsig/mt*OmB*T*P,0.0)
    res.append(('c',W,Ep,hit))
    # Rayleigh
    pR=3/(16*np.pi)*(1+cth*cth)
    mtr=mt
    T=np.exp(-mtr*sin_); P=1-np.exp(-mtr*np.clip(sout-sin_,0,None))
    W=np.where(hit,mu_R(E)/mt*pR*OmB*T*P,0.0)
    res.append(('r',W,E.copy(),hit))
    # pair annihilation photons (two, isotropic), E'=0.511
    Ea=np.full(M,ME)
    mta=mu_tot(Ea)
    T=np.exp(-mta*sin_); P=1-np.exp(-mta*np.clip(sout-sin_,0,None))
    W=np.where(hit,mu_p(E)/mt*2/(4*np.pi)*OmB*T*P,0.0)
    res.append(('a',W,Ea,hit))
    return res

def simulate(N, batch=200000):
    ev_all=[]
    # ---- primary forced events
    # pyramid sampling uniform in solid angle
    u=np.empty(0); v=np.empty(0)
    us=[];vs=[]; tot=0
    while tot<N:
        uu=rng.uniform(-0.05,0.05,2*N); vv=rng.uniform(-0.05,0.05,2*N)
        acc=rng.uniform(size=2*N)<(1+uu*uu+vv*vv)**(-1.5)
        us.append(uu[acc]); vs.append(vv[acc]); tot+=acc.sum()
    u=np.concatenate(us)[:N]; v=np.concatenate(vs)[:N]
    D=np.stack([u,v,np.ones(N)],1); D/=np.linalg.norm(D,axis=1,keepdims=True)
    p0=np.zeros((N,3)); p0[:,0]=100/D[:,2]*D[:,0]; p0[:,1]=100/D[:,2]*D[:,1]
    hit,sin_,sout=chord_from(p0,D)
    # shift: sin_ measured from p0 (phantom entry)
    mu2=mu_tot(np.array(2.0))
    T=np.exp(-mu2*sin_); P=1-np.exp(-mu2*np.clip(sout-sin_,0,None))
    Wp=np.where(hit,T*P,0.0)
    m=Wp>0
    prim=make_events(np.full(m.sum(),2.0),Wp[m])
    ev_all.append(prim)
    # ---- analog transport of all photons, NEE from each collision
    pos=p0.copy(); dr=D.copy(); E=np.full(N,2.0)
    nee_list=[]
    alive=np.ones(N,bool)
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
        Ec=E[ci]; inc=dr[ci]
        for (ch,W,Ep,h) in nee_channels(x,inc,Ec,rng):
            if h.any():
                nee_list.append((ch,W[h],Ep[h]))
        # analog interaction
        mt=mu_tot(Ec); mc=mu_c(Ec); mr=mu_R(Ec); mp=mu_p(Ec)
        xi=rng.uniform(size=len(ci))*mt
        isC=xi<mc; isR=(xi>=mc)&(xi<mc+mr); isP=(xi>=mc+mr)&(xi<mc+mr+mp)
        isPE=xi>=mc+mr+mp
        alive[ci[isPE]]=False
        # pair: kill photon, create two annihilation photons
        pidx=ci[isP]; alive[pidx]=False
        if len(pidx):
            xp=x[isP]
            cdir=rng.uniform(-1,1,len(pidx)); ph=rng.uniform(0,2*np.pi,len(pidx))
            d1=rotate(np.tile([[0,0,1.]],(len(pidx),1)),cdir,ph)
            # isotropic direction: use rotate with random axis: simpler build via random normal
            g=rng.normal(size=(len(pidx),3)); g/=np.linalg.norm(g,axis=1,keepdims=True)
            new_pos=np.concatenate([xp,xp]); new_dir=np.concatenate([g,-g])
            new_E=np.full(2*len(pidx),ME)
            pos=np.concatenate([pos,new_pos]); dr=np.concatenate([dr,new_dir]); E=np.concatenate([E,new_E])
            alive=np.concatenate([alive,np.ones(2*len(pidx),bool)])
        # Compton / Rayleigh updates
        cidx=ci[isC]
        if len(cidx):
            Ein=E[cidx]; eps=kn_eps(Ein,rng); k=Ein/ME; cth=1-(1/eps-1)/k
            ph=rng.uniform(0,2*np.pi,len(cidx))
            dr[cidx]=rotate(dr[cidx],cth,ph); E[cidx]=eps*Ein
        ridx=ci[isR]
        if len(ridx):
            cth=sample_cos_thomson(len(ridx)); ph=rng.uniform(0,2*np.pi,len(ridx))
            dr[ridx]=rotate(dr[ridx],cth,ph)
        # energy cut
        low=alive & (E<0.01)
        alive[low]=False
        # ensure arrays length consistent
    return ev_all, nee_list, prim

t0=time.time()
ev_all,nee_list,prim=simulate(200000)
print("time",time.time()-t0, "prim events",len(prim['W']), "sum prim W", prim['W'].sum())
print(len(nee_list))
