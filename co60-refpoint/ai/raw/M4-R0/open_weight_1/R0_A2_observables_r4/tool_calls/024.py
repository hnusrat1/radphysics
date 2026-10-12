# Electron MC transport - test run
import numpy as np, time

def run_electron_mc(z0, N_el, seed, src, R_lat=2.5, zr=1.2):
    rng=np.random.default_rng(seed)
    # sample sources
    pos,d,E0,parent,Egamma,ispos,w_el,N_birth = sample_sources(z0,src,N_el,rng,R_lat=R_lat,zr=zr)
    maxN=N_el+8*N_el
    P=np.zeros((maxN,3)); Dd=np.zeros((maxN,3)); En=np.zeros(maxN)
    isdelta=np.zeros(maxN,bool); isposi=np.zeros(maxN,bool)
    Prnt=np.zeros(maxN)  # 0 primary parent, 1 scattered parent
    Al=np.zeros(maxN,bool)
    P[:N_el]=pos; Dd[:N_el]=d; En[:N_el]=E0; Prnt[:N_el]=parent; Al[:N_el]=True
    nactive=N_el; free=list(range(N_el,maxN))
    # tallies (per sampled electron, weight later)
    flu=np.zeros(nEB); ang=np.zeros(nCT); flu_delta=np.zeros(nEB)
    sumChord=0.0; sumE=0.0; sumScol=0.0; sumSrad=0.0; sumSrest=0.0
    trackend=0.0; dose_prim=0.0; dose_sec=0.0
    V= np.pi*1.0**2*0.2
    zc=100.0+z0
    it=0; steps=0; n_delta_created=0
    t0=time.time()
    while np.any(Al):
        idx=np.nonzero(Al)[0]
        if len(idx)==0: break
        Ei=En[idx]
        St=Srest(Ei)+Srad_f(Ei)
        dE_step=0.02*Ei
        ds=np.minimum(dE_step/np.maximum(St,1e-6), (Ei-Delta)/np.maximum(St,1e-6))
        ds=np.maximum(ds,1e-6)
        p0=P[idx]; dd=Dd[idx]
        p1=p0+dd*ds[:,None]
        ch=cyl_chord(p0,p1,1.0,zc,0.1)
        # fluence scoring (electrons only, exclude positrons)
        el_mask=~isposi[idx]
        if ch.sum()>0:
            ebin=np.searchsorted(eb_edges,Ei,side='right')-1
            cbin=np.searchsorted(ct_edges,dd[:,2],side='right')-1
            isd=isdelta[idx]
            for b in range(nEB):
                mm=(ebin==b)&(ch>0)
                if mm.any():
                    mm_e=mm&el_mask
                    flu[b]+=np.sum(ch[mm_e])
                    mm_d=mm&el_mask&isd
                    flu_delta[b]+=np.sum(ch[mm_d])
            for b in range(nCT):
                mm=(cbin==b)&(ch>0)&el_mask
                if mm.any():
                    ang[b]+=np.sum(ch[mm])
            sumChord+=np.sum(ch[el_mask])
            sumE+=np.sum(ch[el_mask]*Ei[el_mask])
            sumScol+=np.sum(ch[el_mask]*Scol_f(Ei[el_mask]))
            sumSrad+=np.sum(ch[el_mask]*Srad_f(Ei[el_mask]))
        # dose (all charged incl positrons): continuous restricted
        Srest_i=Srest(Ei)
        dch=ch*Srest_i   # MeV (per unit mass? ch[cm]*Srest[MeV cm^2/g]*rho... careful: ch*Srest*rho /V/rho = ch*Srest/V MeV/g)
        sumSrest+=np.sum(ch*Srest_i)
        # dose by parent
        pr=Prnt[idx]
        dose_prim+=np.sum(ch[pr==0]*Srest_i[pr==0])
        dose_sec+=np.sum(ch[pr==1]*Srest_i[pr==1])
        # delta creation
        lam=Ndel(Ei)*ds
        nd=rng.poisson(lam)
        totd=int(nd.sum())
        if totd>0:
            if len(free)<totd+10:
                ext=max(totd+100,100000)
                P=np.vstack([P,np.zeros((ext,3))]); Dd=np.vstack([Dd,np.zeros((ext,3))])
                En=np.concatenate([En,np.zeros(ext)]); isdelta=np.concatenate([isdelta,np.zeros(ext,bool)])
                isposi=np.concatenate([isposi,np.zeros(ext,bool)]); Prnt=np.concatenate([Prnt,np.zeros(ext)])
                Al=np.concatenate([Al,np.zeros(ext,bool)])
                free.extend(range(maxN,maxN+ext)); maxN+=ext
            # for each electron with nd>0
            reps=np.repeat(np.arange(len(idx)),nd)
            Td=sample_delta_energy(Ei[reps],len(reps),rng)
            cd=cos_delta_fn(Td,Ei[reps])
            az=rng.uniform(0,2*np.pi,len(reps))
            sd=np.sqrt(np.maximum(0,1-cd*cd))
            # direction relative to primary dir
            d0=Dd[reps]
            # build basis
            aux=np.broadcast_to(np.array([0.,0.,1.]),(len(reps),3)).copy()
            aux[np.abs(d0[:,2])>0.9]=np.array([1.,0.,0.])
            e1=np.cross(aux,d0); n1=np.sqrt(np.sum(e1*e1,axis=1,keepdims=True)); e1=e1/np.maximum(n1,1e-12)
            e2=np.cross(d0,e1)
            dd_new=cd[:,None]*d0+sd[:,None]*(np.cos(az)[:,None]*e1+np.sin(az)[:,None]*e2)
            # position at current location (p0) - use start of step
            for j in range(len(reps)):
                fi=free.pop()
                P[fi]=p0[reps[j]]; Dd[fi]=dd_new[j]; En[fi]=Td[j]
                isdelta[fi]=True; isposi[fi]=False; Prnt[fi]=pr[reps[j]]; Al[fi]=True
                if fi>=nactive: nactive=fi+1
                n_delta_created+=1
        # multiple scattering
        sig=np.array([highland_sigma(e,ds[j]) for j,e in enumerate(Ei)])
        dth=rng.normal(0,sig)
        dph=rng.uniform(0,2*np.pi,len(idx))
        newd=rotate_e(dd,dth,dph)
        # energy update
        E_new=Ei - ds*(Srest(Ei)+Srad_f(Ei)) - 0.0
        # subtract delta energies: need sum of Td per original electron
        if totd>0:
            Esum=np.zeros(len(idx))
            np.add.at(Esum,reps,Td)
            E_new-=Esum
        # move
        P[idx]=p1; Dd[idx]=newd; En[idx]=E_new
        # boundary/termination
        outside=(P[idx,2]<100.0)|(P[idx,2]>130.0)|(np.abs(P[idx,0])>15.0)|(np.abs(P[idx,1])>15.0)
        below=En[idx]<Delta
        # track-end: deposit residual if inside voxel
        in_vox=(np.abs(P[idx,2]-zc)<0.1)&(P[idx,0]**2+P[idx,1]**2<1.0)
        tr_end=En[idx]
        trackend+=np.sum(tr_end[below&in_vox])
        dead=outside|below
        Al[idx[dead]]=False
        it+=1
        steps+=len(idx)
        if it>500: break
    # finalize
    res=dict(N_el=N_el, w_el=w_el, N_birth=N_birth, flu=flu, ang=ang, flu_delta=flu_delta,
             sumChord=sumChord, sumE=sumE, sumScol=sumScol, sumSrad=sumSrad, sumSrest=sumSrest,
             trackend=trackend, dose_prim=dose_prim, dose_sec=dose_sec, it=it, steps=steps,
             n_delta=n_delta_created, time=time.time()-t0)
    return res

def rotate_e(d,th,ph):
    aux=np.broadcast_to(np.array([0.,0.,1.]),(len(d),3)).copy()
    aux[np.abs(d[:,2])>0.9]=np.array([1.,0.,0.])
    e1=np.cross(aux,d); n1=np.sqrt(np.sum(e1*e1,axis=1,keepdims=True)); e1=e1/np.maximum(n1,1e-12)
    e2=np.cross(d,e1)
    return np.cos(th)[:,None]*d+np.sin(th)[:,None]*(np.cos(ph)[:,None]*e1+np.sin(ph)[:,None]*e2)

def cos_delta_fn(T,E):
    return np.sqrt(np.maximum(0,T*(E+2*mec2)/(E*(T+2*mec2))))

# quick test
if __name__=='__main__':
    rph=np.load('/tmp/photon_mc_final.npy',allow_pickle=True).item()
    src=build_source_fns(rph)
    res=run_electron_mc(5.0,20000,seed=101,src=src)
    print("time",res['time'],"it",res['it'],"steps",res['steps'],"ndelta",res['n_delta'])
    print("N_birth per Phi0 in region:",res['N_birth'],"w_el",res['w_el'])
    V=np.pi*0.2
    flu_tot = res['sumChord']/V*res['w_el']
    print("electron fluence per Phi0:",flu_tot)
    D=res['sumSrest']/V*res['w_el']*1.602e-10 + res['trackend']/V*res['w_el']*1.602e-10
    print("D per Phi0 [Gy cm^2]:",D)
    print("Kcol expectation ~4.9e-12")
    print("dose_prim frac:",res['dose_prim']/(res['dose_prim']+res['dose_sec']))
    print("mean e energy:",res['sumE']/res['sumChord'])
    print("Srad/Scol ratio:",res['sumSrad']/res['sumScol'])
    print("delta fluence fraction:",res['flu_delta'].sum()/res['flu'].sum())
    print("angular:",res['ang']/res['ang'].sum())
    np.save('/tmp/test_emc.npy',res,allow_pickle=True)