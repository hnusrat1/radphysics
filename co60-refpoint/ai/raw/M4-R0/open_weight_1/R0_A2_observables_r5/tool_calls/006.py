import numpy as np, time

xcom = np.loadtxt('/tmp/data/xcom_water.txt', comments='#')
E_x = xcom[:,0]; mu_coh=xcom[:,1]; mu_inc=xcom[:,2]; mu_pe=xcom[:,3]
mu_pair=xcom[:,4]+xcom[:,5]; mu_tot=xcom[:,6]
LE = np.log(E_x)
def mut_at(E, mu):
    return np.exp(np.interp(np.log(np.clip(E,1e-6,20.)), LE, np.log(mu)))

m_e = 0.5109989461

def sample_compton(E, rng):
    E = np.asarray(E, dtype=np.float64); n=E.size
    alpha = E/m_e
    K=10
    mu_c = rng.uniform(-1,1,(n,K)); u_c = rng.random((n,K))
    d = 1.0 + alpha[:,None]*(1.0-mu_c)
    p = ((1+mu_c*mu_c)/2.0 + (alpha[:,None]**2*(1-mu_c)**2)/(2.0*d)) / (d*d)
    accept = u_c < p; has = accept.any(axis=1)
    mu = mu_c[np.arange(n), np.argmax(accept,axis=1)]
    bad = ~has; cnt=0
    while bad.any() and cnt<50:
        nb=bad.sum()
        mc=rng.uniform(-1,1,(nb,K)); uc=rng.random((nb,K))
        dd=1.0+alpha[bad][:,None]*(1.0-mc)
        pp=((1+mc*mc)/2.0+(alpha[bad][:,None]**2*(1-mc)**2)/(2.0*dd))/(dd*dd)
        acc=uc<pp; h2=acc.any(axis=1)
        mu[bad]=mc[np.arange(nb), np.argmax(acc,axis=1)]
        bad[bad]=~h2; cnt+=1
    if bad.any(): mu[bad]=rng.uniform(-1,1,bad.sum())
    Ep = E/(1.0+alpha*(1.0-mu))
    pe = np.sqrt(np.maximum(E*E+Ep*Ep-2*E*Ep*mu,0))
    cos_e = (E-Ep*mu)/pe
    return mu, Ep, cos_e, E-Ep

def cyl_seg_length(p0, d, s, z0=4.9, z1=5.1, R=1.0):
    x0=p0[:,0]; y0=p0[:,1]; z0v=p0[:,2]
    dx=d[:,0]; dy=d[:,1]; dz=d[:,2]
    dzs = np.where(dz==0, 1e-12, dz)
    t1=(z0-z0v)/dzs; t2=(z1-z0v)/dzs
    tmin=np.minimum(t1,t2); tmax=np.maximum(t1,t2)
    tin=np.maximum(0.0,tmin); tout=np.minimum(s,tmax)
    a=dx*dx+dy*dy; a=np.maximum(a,1e-16)
    b=2*(x0*dx+y0*dy); c=x0*x0+y0*y0-R*R
    disc=b*b-4*a*c
    ok=disc>0
    sq=np.sqrt(np.where(ok,disc,0.0))
    t1c=(-b-sq)/(2*a); t2c=(-b+sq)/(2*a)
    tinc=np.maximum(np.where(ok,t1c,0.0),0.0)
    toutc=np.minimum(np.where(ok,t2c,s),s)
    tin_=np.maximum(tin,tinc); tout_=np.minimum(tout,toutc)
    L=np.maximum(0.0,tout_-tin_)
    return L

def photon_mc(N, seed=42, region=4.5, region_zmax=13.0, verbose=True):
    rng = np.random.default_rng(seed)
    # energies: half each line
    E = np.where(rng.random(N)<0.5, 1.1732, 1.3325).astype(np.float64)
    u = rng.uniform(-5,5,N); v = rng.uniform(-5,5,N)
    L = np.sqrt(u*u+v*v+100.0*100.0)
    dx = u/L; dy=v/L; dz=100.0/L
    dirv = np.column_stack([dx,dy,dz])
    # start at surface z=0
    pos = np.zeros((N,3)); pos[:,0]=u; pos[:,1]=v  # x,y at surface, z=0
    interact = np.zeros(N, dtype=np.int32)
    alive = np.ones(N, dtype=bool)
    
    # tally arrays
    nb_ph=28; ph_edges=np.arange(0,1.401,0.05)
    fl_prim=np.zeros(nb_ph); fl_scat=np.zeros(nb_ph)
    ang_prim=np.zeros(10); ang_scat=np.zeros(10)
    cos_edges=np.linspace(-1,1,11)
    
    # interaction tallies at 5-cm scoring cylinder
    # process: 0 coherent,1 incoherent,2 photo,3 pair
    int_all=np.zeros(4); int_prim=np.zeros(4); int_scat=np.zeros(4)
    etr_all=np.zeros(4); etr_prim=np.zeros(4); etr_scat=np.zeros(4)
    # Layer3 lists
    L3_Te=[]; L3_cos=[]; L3_parent=[]  # parent 1=primary photon,0=scattered
    # electron source lists (region)
    src_pos=[]; src_dir=[]; src_T=[]; src_par=[]; src_kind=[]
    
    tstart=time.time()
    generation=0
    while alive.any() and generation<40:
        generation+=1
        n = alive.sum()
        Ea = E[alive]; pa = pos[alive]; da = dirv[alive]; ic = interact[alive]
        mu = mut_at(Ea, mu_tot)
        s = -np.log(np.maximum(rng.random(n),1e-30))/mu
        # distance to cube boundary
        # z: +30, 0
        tzp = np.where(da[:,2]>0, (30-pa[:,2])/np.where(da[:,2]>0,da[:,2],1), 1e30)
        tzn = np.where(da[:,2]<0, (0-pa[:,2])/np.where(da[:,2]<0,da[:,2],-1), 1e30)
        txn = np.where(da[:,0]<0, (-15-pa[:,0])/np.where(da[:,0]<0,da[:,0],-1), 1e30)
        txp = np.where(da[:,0]>0, (15-pa[:,0])/np.where(da[:,0]>0,da[:,0],1), 1e30)
        tyn = np.where(da[:,1]<0, (-15-pa[:,1])/np.where(da[:,1]<0,da[:,1],-1), 1e30)
        typ = np.where(da[:,1]>0, (15-pa[:,1])/np.where(da[:,1]>0,da[:,1],1), 1e30)
        db = np.minimum.reduce([tzp,tzn,txn,txp,tyn,typ])
        # segment length inside phantom for scoring/interaction
        seg_ok = s <= db   # interacts inside
        # tally photon fluence in scoring cylinder for all segments (length min(s,db))
        segL = np.minimum(s, db)
        Lcyl = cyl_seg_length(pa, da, segL)
        # energy bin
        binidx = np.searchsorted(ph_edges, Ea, 'right')-1
        binidx = np.clip(binidx, 0, nb_ph-1)
        cosb = np.searchsorted(cos_edges, da[:,2], 'right')-1
        cosb = np.clip(cosb,0,9)
        prim = (ic==0)
        # add totals
        np.add.at(fl_prim, binidx[prim], Lcyl[prim])
        np.add.at(fl_scat, binidx[~prim], Lcyl[~prim])
        np.add.at(ang_prim, cosb[prim], Lcyl[prim])
        np.add.at(ang_scat, cosb[~prim], Lcyl[~prim])
        
        # interactions
        if seg_ok.any():
            idx_int = np.where(alive)[0][seg_ok]
            ii = np.where(seg_ok)[0]
            # interaction point
            p_int = pa[ii] + da[ii]*s[ii]
            E_int = Ea[ii]; ic_int = ic[ii]
            mu_i = mu[ii]
            # sample process
            rp = rng.random(len(ii))*mu_i
            muc = mut_at(E_int, mu_coh); mui = mut_at(E_int, mu_inc); mup = mut_at(E_int, mu_pe); mupair = mut_at(E_int, mu_pair)
            c1 = muc; c2=c1+mui; c3=c2+mup; c4=c3+mupair
            proc = np.zeros(len(ii), dtype=np.int32)
            proc[rp>=c3] = 2  # photoelectric (for rp in [c2,c3)) -> wait order
            # digitize properly:
            proc = np.digitize(rp, [c1,c2,c3])  # 0 coh,1 inc,2 pe,3 pair
            proc = np.where(rp>=c4, 3, proc)
            prim_part = (ic_int==0)
            # tally interactions in 5-cm scoring cylinder
            in_sc = (p_int[:,2]>=4.9)&(p_int[:,2]<=5.1)&(p_int[:,0]**2+p_int[:,1]**2<=1.0)
            if in_sc.any():
                psub = proc[in_sc]; prsub = prim_part[in_sc]
                np.add.at(int_all, psub, 1)
                np.add.at(int_prim, psub[prsub], 1)
                np.add.at(int_scat, psub[~prsub], 1)
                # energy transfer
                Etr = np.where(psub==1, E_int[in_sc]-mut_at(E_int[in_sc],0)*0+ (E_int[in_sc] - E_int[in_sc]/(1+(E_int[in_sc]/m_e)*(1-sample_compton(E_int[in_sc],rng)[0]))), 0.0)
                # simpler: compute below properly
            # electron sources for region (also for Layer3 when in scoring cylinder)
            in_reg = (p_int[:,2]>=0)&(p_int[:,2]<=region_zmax)&(p_int[:,0]**2+p_int[:,1]**2<=region**2)
            # Compton recoil direction/energy
            # (compute for all interacting)
            # handle processes
            # Compton
            m_coh = proc==0; m_inc = proc==1; m_pe = proc==2; m_pair = proc==3
            if m_inc.any():
                mu_c, Ep, cos_e, Te = sample_compton(E_int[m_inc], rng)
                # new photon state for continuing photons
                # store continuing photon info (will set below)
            # This temp approach needs restructuring: collect continuation arrays
        break  # placeholder
    
    print('placeholder ok')
    return

photon_mc(1000)