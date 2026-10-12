import numpy as np, time, os

xcom = np.loadtxt('/tmp/data/xcom_water.txt', comments='#')
E_x = xcom[:,0]; mu_coh=xcom[:,1]; mu_inc=xcom[:,2]; mu_pe=xcom[:,3]
mu_pair=xcom[:,4]+xcom[:,5]; mu_tot=xcom[:,6]
LE = np.log(E_x)
def mut_at(E, mu):
    mu = np.maximum(mu, 1e-30)
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

def photon_mc(N, seed=42, region=4.5, region_zmax=13.0, save_src=True, verbose=True):
    rng = np.random.default_rng(seed)
    E = np.where(rng.random(N)<0.5, 1.1732, 1.3325).astype(np.float64)
    u = rng.uniform(-5,5,N); v = rng.uniform(-5,5,N)
    L = np.sqrt(u*u+v*v+100.0*100.0)
    dirv = np.column_stack([u/L, v/L, 100.0/L])
    pos = np.zeros((N,3)); pos[:,0]=u; pos[:,1]=v
    interact = np.zeros(N, dtype=np.int32)
    alive = np.ones(N, dtype=bool)

    nb_ph=28; ph_edges=np.linspace(0,1.4,29)
    fl_prim=np.zeros(nb_ph); fl_scat=np.zeros(nb_ph)
    ang_prim=np.zeros(10); ang_scat=np.zeros(10)
    cos_edges=np.linspace(-1,1,11)

    int_all=np.zeros(4); int_prim=np.zeros(4); int_scat=np.zeros(4)
    etr_all=np.zeros(4); etr_prim=np.zeros(4); etr_scat=np.zeros(4)
    L3_Te=[]; L3_cos=[]; L3_parent=[]
    src_pos=[]; src_dir=[]; src_T=[]; src_par=[]; src_kind=[]
    n_interact=0
    tstart=time.time()
    generation=0
    while alive.any() and generation<80:
        generation+=1
        n = alive.sum()
        Ea = E[alive]; pa = pos[alive]; da = dirv[alive]; ic = interact[alive]
        mu = mut_at(Ea, mu_tot)
        s = -np.log(np.maximum(rng.random(n),1e-30))/mu
        tzp = np.where(da[:,2]>0, (30-pa[:,2])/np.maximum(da[:,2],1e-12), 1e30)
        tzn = np.where(da[:,2]<0, (0-pa[:,2])/np.minimum(da[:,2],-1e-12), 1e30)
        txn = np.where(da[:,0]<0, (-15-pa[:,0])/np.minimum(da[:,0],-1e-12), 1e30)
        txp = np.where(da[:,0]>0, (15-pa[:,0])/np.maximum(da[:,0],1e-12), 1e30)
        tyn = np.where(da[:,1]<0, (-15-pa[:,1])/np.minimum(da[:,1],-1e-12), 1e30)
        typ = np.where(da[:,1]>0, (15-pa[:,1])/np.maximum(da[:,1],1e-12), 1e30)
        db = np.minimum.reduce([tzp,tzn,txn,txp,tyn,typ])
        segL = np.minimum(s, db)
        Lcyl = cyl_seg_length(pa, da, segL)
        binidx = np.clip(np.searchsorted(ph_edges, Ea, 'right')-1, 0, nb_ph-1)
        cosb = np.clip(np.searchsorted(cos_edges, da[:,2], 'right')-1, 0, 9)
        prim = (ic==0)
        np.add.at(fl_prim, binidx[prim], Lcyl[prim])
        np.add.at(fl_scat, binidx[~prim], Lcyl[~prim])
        np.add.at(ang_prim, cosb[prim], Lcyl[prim])
        np.add.at(ang_scat, cosb[~prim], Lcyl[~prim])

        seg_ok = s <= db
        cont_pos=[]; cont_dir=[]; cont_E=[]; cont_ic=[]
        if seg_ok.any():
            ii = np.where(seg_ok)[0]
            p_int = pa[ii] + da[ii]*s[ii][:,None]
            E_int = Ea[ii]; ic_int = ic[ii]; da_int = da[ii]
            mu_i = mu[ii]
            rp = rng.random(len(ii))*mu_i
            muc = mut_at(E_int, mu_coh); mui = mut_at(E_int, mu_inc); mup = mut_at(E_int, mu_pe); mupair = mut_at(E_int, mu_pair)
            c1=muc; c2=c1+mui; c3=c2+mup; c4=c3+mupair
            proc = np.where(rp<c1, 0, np.where(rp<c2, 1, np.where(rp<c3, 2, 3)))
            proc = np.where(rp>=c4, 3, proc)
            prim_part = (ic_int==0)

            in_sc = (p_int[:,2]>=4.9)&(p_int[:,2]<=5.1)&(p_int[:,0]**2+p_int[:,1]**2<=1.0)
            in_reg = (p_int[:,2]>=0)&(p_int[:,2]<=region_zmax)&(p_int[:,0]**2+p_int[:,1]**2<=region**2)

            newE = E_int.copy(); newdir = da_int.copy()

            m_inc = proc==1
            if m_inc.any():
                mu_c, Ep, cos_e, Te = sample_compton(E_int[m_inc], rng)
                newE[m_inc] = Ep
                cth = mu_c; sth = np.sqrt(np.maximum(1-cth*cth,1e-30))
                a = np.column_stack([-da_int[m_inc][:,1], da_int[m_inc][:,0], np.zeros(len(E_int[m_inc]))])
                na = np.sqrt(a[:,0]**2+a[:,1]**2)
                bad = na<1e-6
                a[bad] = [1,0,0]
                na = np.sqrt(a[:,0]**2+a[:,1]**2)
                a = a/na[:,None]
                b = np.cross(da_int[m_inc], a)
                phi = rng.uniform(0,2*np.pi, len(E_int[m_inc]))
                newdir[m_inc] = da_int[m_inc]*cth[:,None] + sth[:,None]*(a*np.cos(phi)[:,None]+b*np.sin(phi)[:,None])
                newdir[m_inc] /= np.sqrt((newdir[m_inc]**2).sum(axis=1))[:,None]
                dir_e2 = da_int[m_inc]*cos_e[:,None] + np.sqrt(np.maximum(1-cos_e*cos_e,1e-30))[:,None]*(a*np.cos(phi+np.pi)[:,None]+b*np.sin(phi+np.pi)[:,None])
                dir_e2 /= np.sqrt((dir_e2**2).sum(axis=1))[:,None]
                # electron sources in region
                sel = in_reg[m_inc]
                if sel.any():
                    src_pos.append(p_int[m_inc][sel]); src_dir.append(dir_e2[sel].astype(np.float32))
                    src_T.append(Te[sel].astype(np.float32)); src_par.append(prim_part[m_inc][sel].astype(np.int8))
                    src_kind.append(np.zeros(sel.sum(), dtype=np.int8))
                # Layer3 / energy transfer in scoring cylinder
                ssel = in_sc[m_inc]
                if ssel.any():
                    L3_Te.append(Te[ssel]); L3_cos.append(cos_e[ssel]); L3_parent.append(prim_part[m_inc][ssel].astype(np.int8))
                    etr_all[1]+= (E_int[m_inc][ssel]-newE[m_inc][ssel]).sum()
                    pp = prim_part[m_inc][ssel]
                    etr_prim[1]+= (E_int[m_inc][ssel][pp]-newE[m_inc][ssel][pp]).sum()
                    etr_scat[1]+= (E_int[m_inc][ssel][~pp]-newE[m_inc][ssel][~pp]).sum()
            m_pe = proc==2
            if m_pe.any():
                Te_pe = E_int[m_pe]
                cth = rng.uniform(-1,1,len(E_int[m_pe])); sth = np.sqrt(np.maximum(1-cth*cth,0))
                phi = rng.uniform(0,2*np.pi,len(E_int[m_pe]))
                dir_e = np.column_stack([sth*np.cos(phi), sth*np.sin(phi), cth])
                sel = in_reg[m_pe]
                if sel.any():
                    src_pos.append(p_int[m_pe][sel]); src_dir.append(dir_e[sel].astype(np.float32))
                    src_T.append(Te_pe[sel].astype(np.float32)); src_par.append(prim_part[m_pe][sel].astype(np.int8))
                    src_kind.append(np.zeros(sel.sum(), dtype=np.int8))
                ssel = in_sc[m_pe]
                if ssel.any():
                    L3_Te.append(Te_pe[ssel]); L3_cos.append(dir_e[ssel][:,2]); L3_parent.append(prim_part[m_pe][ssel].astype(np.int8))
                    etr_all[2]+= Te_pe[ssel].sum()
                    pp=prim_part[m_pe][ssel]
                    etr_prim[2]+= Te_pe[ssel][pp].sum(); etr_scat[2]+= Te_pe[ssel][~pp].sum()
            m_pair = proc==3
            if m_pair.any():
                Tpair = E_int[m_pair]-1.022
                Te1 = Tpair/2; Te2=Tpair/2
                cth = 1 - 0.5*rng.random(len(E_int[m_pair]))**2
                sth = np.sqrt(np.maximum(1-cth*cth,0)); phi = rng.uniform(0,2*np.pi,len(E_int[m_pair]))
                dir1 = np.column_stack([sth*np.cos(phi), sth*np.sin(phi), cth])
                dir2 = np.column_stack([-sth*np.cos(phi), -sth*np.sin(phi), -cth])
                sel = in_reg[m_pair]
                if sel.any():
                    src_pos.append(np.concatenate([p_int[m_pair][sel],p_int[m_pair][sel]]))
                    src_dir.append(np.concatenate([dir1[sel],dir2[sel]]).astype(np.float32))
                    src_T.append(np.concatenate([Te1[sel],Te2[sel]]).astype(np.float32))
                    src_par.append(np.concatenate([prim_part[m_pair][sel],prim_part[m_pair][sel]]).astype(np.int8))
                    src_kind.append(np.concatenate([np.zeros(sel.sum()),np.ones(sel.sum())]).astype(np.int8))
                ssel = in_sc[m_pair]
                if ssel.any():
                    L3_Te.append(np.concatenate([Te1[ssel],Te2[ssel]]))
                    L3_cos.append(np.concatenate([dir1[ssel][:,2],dir2[ssel][:,2]]))
                    L3_parent.append(np.concatenate([prim_part[m_pair][ssel],prim_part[m_pair][ssel]]).astype(np.int8))
                    etr_all[3]+= Tpair[ssel].sum()
                    pp=prim_part[m_pair][ssel]
                    etr_prim[3]+= Tpair[ssel][pp].sum(); etr_scat[3]+= Tpair[ssel][~pp].sum()
                if True:
                    npa=m_pair.sum()
                    cth=rng.uniform(-1,1,npa); sth=np.sqrt(np.maximum(1-cth*cth,0)); phi=rng.uniform(0,2*np.pi,npa)
                    nvec=np.column_stack([sth*np.cos(phi), sth*np.sin(phi), cth])
                    cont_pos.append(np.concatenate([p_int[m_pair],p_int[m_pair]]))
                    cont_dir.append(np.concatenate([nvec,-nvec]))
                    cont_E.append(np.full(2*npa,0.511))
                    cont_ic.append(ic_int[m_pair][np.repeat(np.arange(npa),2)]+1)
            # coherent: nothing (continues same direction/energy)

            if in_sc.any():
                psub = proc[in_sc]; prsub = prim_part[in_sc]
                np.add.at(int_all, psub, 1)
                np.add.at(int_prim, psub[prsub], 1)
                np.add.at(int_scat, psub[~prsub], 1)

            cont_photons = (proc==1)|(proc==0)
            if cont_photons.any():
                cont_pos.append(p_int[cont_photons]); cont_dir.append(newdir[cont_photons])
                cont_E.append(newE[cont_photons]); cont_ic.append(ic_int[cont_photons]+1)
            n_interact += len(ii)

        if cont_pos:
            newpos = np.concatenate(cont_pos); newdir=np.concatenate(cont_dir)
            newE = np.concatenate(cont_E); newic=np.concatenate(cont_ic).astype(np.int32)
            keep = newE>=0.02
            pos = newpos[keep]; dirv=newdir[keep]; E=newE[keep]; interact=newic[keep]
            alive = np.ones(len(E), dtype=bool)
        else:
            alive[:] = False
        if verbose and generation%5==0:
            print(f'gen {generation}: n={n}, alive next={alive.sum()}, elapsed {time.time()-tstart:.1f}s')
    if verbose:
        print(f'finished generations={generation}, elapsed {time.time()-tstart:.1f}s')

    def cat(l):
        return np.concatenate(l) if l else np.zeros(0)
    result = dict(
        fl_prim=fl_prim, fl_scat=fl_scat, ang_prim=ang_prim, ang_scat=ang_scat,
        int_all=int_all, int_prim=int_prim, int_scat=int_scat,
        etr_all=etr_all, etr_prim=etr_prim, etr_scat=etr_scat,
        L3_Te=cat(L3_Te), L3_cos=cat(L3_cos), L3_parent=cat(L3_parent).astype(np.int8),
        N_emitted=N, n_interact=n_interact
    )
    if save_src:
        result['src'] = dict(pos=cat(src_pos).astype(np.float32), dir=cat(src_dir).astype(np.float32),
                             T=cat(src_T).astype(np.float32), par=cat(src_par).astype(np.int8),
                             kind=cat(src_kind).astype(np.int8))
    return result

t0=time.time()
res = photon_mc(200000, seed=42, verbose=True)
print('total time', time.time()-t0)
print('L1 prim fluence raw (cm):', res['fl_prim'].sum(), 'scat:', res['fl_scat'].sum())
print('int_all', res['int_all'], 'etr', res['etr_all'])
print('src counts', len(res['src']['T']) if 'src' in res else 0)
print('L3 n', len(res['L3_Te']))