
mc = r'''
import numpy as np, time
from physics import *
def run(N, zpts, seed=7, Emin=0.005, store_main=None):
    rng=np.random.default_rng(seed)
    npt=len(zpts)
    # raw accumulators (w*score sums)
    A_E=np.zeros((npt,len(Eb)-1)); A_c=np.zeros((npt,len(cb)-1))
    A_tot=np.zeros(npt); A_Ef=np.zeros(npt)
    A_K=np.zeros(npt); A_Kc=np.zeros(npt)
    A_int=np.zeros((npt,4)); A_tr=np.zeros((npt,4))  # processes: C,pe,pair,R
    A_intE=np.zeros(npt)
    store=[]  # for main point contributions (E, rhat(3), wscore)
    # histories
    u=rng.uniform(-0.05,0.05,N); v=rng.uniform(-0.05,0.05,N)
    w=(1+u*u+v*v)**(-1.5)
    d=np.stack([u,v,np.ones(N)],axis=1); d/=np.linalg.norm(d,axis=1)[:,None]
    pos=np.stack([100*u,100*v,np.zeros(N)],axis=1)
    E=lines[(rng.random(N)<0.5).astype(int)]
    hid=np.arange(N)
    box_lo=np.array([-15,-15,0.]); box_hi=np.array([15,15,30.])
    P=np.array([[0,0,z] for z in zpts])
    gen=0; nColl=0
    while E.size>0:
        gen+=1
        mu=mutot(E)
        s=-np.log(rng.random(E.size))/mu
        # distance to exit box
        with np.errstate(divide='ignore',invalid='ignore'):
            tx=np.where(d>0,(box_hi-pos)/d,np.where(d<0,(box_lo-pos)/d,np.inf))
        texit=np.min(tx,axis=1)
        inside=s<texit
        pos=pos[inside]; d=d[inside]; E=E[inside]; w=w[inside]; hid=hid[inside]; mu=mu[inside]
        s=s[inside]
        if E.size==0: break
        X=pos+s[:,None]*d
        pos=X
        nColl+=E.size
        # process selection
        xi=rng.random(E.size)*mu
        pC=muC(E); ppe=mupe(E); ppair=mupair(E); pR=muR(E)
        isC=xi<pC
        isPE=(xi>=pC)&(xi<pC+ppe)
        isPair=(xi>=pC+ppe)&(xi<pC+ppe+ppair)
        isR=xi>=pC+ppe+ppair
        # ---- NEE scoring for C, R, pair annihilation
        for j in range(npt):
            rv=P[j][None,:]-X
            r=np.linalg.norm(rv,axis=1)
            rh=rv/r[:,None]
            cpsi=np.sum(d*rh,axis=1)
            # Compton
            kk=E/me
            ep=1/(1+kk*(1-cpsi))
            Ep=E*ep
            dsC=0.5*re**2*ep**2*(ep+1/ep-(1-cpsi**2))
            pC_sr=dsC/np.maximum(KN_sigma(E),1e-40)
            attC=np.exp(-mutot(np.maximum(Ep,Emin))*r)
            scC=pC_sr*attC/r**2
            # Rayleigh
            scR=3*(1+cpsi**2)/(16*np.pi)*np.exp(-mutot(E)*r)/r**2
            scR=np.where(isR,scR,0.0)
            scC=np.where(isC,scC,0.0)
            # annihilation (pair events)
            attA=np.exp(-mutot(np.array([0.511]))[0]*r)
            scA=np.where(isPair,2*attA/(4*np.pi*r**2),0.0)
            # accumulate (raw, weighted by w)
            wC=w*scC; wR=w*scR; wA=w*scA
            # energy-binned
            Eprime_C=Ep
            for (sc,Ee,tag) in [(wC,Eprime_C,0),(wR,E,3),(wA,np.full(E.size,0.511),2)]:
                m=sc>0
                if m.sum()==0: continue
                hb=np.clip(np.digitize(Ee[m],Eb)-1,0,len(Eb)-2)
                # binning by E bins
                np.add.at(A_E[j],hb,sc[m])
                cz=rh[m,2]
                cbin=np.clip(np.digitize(cz,cb)-1,0,len(cb)-2)
                np.add.at(A_c[j],cbin,sc[m])
                A_tot[j]+=sc[m].sum()
                A_Ef[j]+=np.sum(sc[m]*Ee[m])
                mu_i=np.array([0,0,0,0.0])
                # energy-transfer and kerma
                if tag==0:
                    A_K[j]+=np.sum(sc[m]*(muC(Ee[m])*Tmean(Ee[m])+mupe(Ee[m])*Ee[m]))
                    A_Kc[j]+=np.sum(sc[m]*KC(Ee[m]))
                    A_int[j,0]+=np.sum(sc[m]*muC(Ee[m])); A_tr[j,0]+=np.sum(sc[m]*muC(Ee[m])*Tmean(Ee[m]))
                    A_int[j,1]+=np.sum(sc[m]*mupe(Ee[m])); A_tr[j,1]+=np.sum(sc[m]*mupe(Ee[m])*Ee[m])
                    A_int[j,2]+=np.sum(sc[m]*mupair(Ee[m])); A_tr[j,2]+=np.sum(sc[m]*mupair(Ee[m])*(Ee[m]-2*me))
                elif tag==3:
                    A_int[j,3]+=np.sum(sc[m]*muR(Ee[m]))
                elif tag==2:
                    A_K[j]+=0.0
                    A_Kc[j]+=0.0
                if j==store_main:
                    store.append((Ee[m].copy(), rh[m].copy(), sc[m].copy(), np.full(m.sum(),tag)))
        # ---- update photon states
        # Compton
        newE=E.copy(); newd=d.copy(); alive=np.ones(E.size,bool)
        ci=np.where(isC)[0]
        if ci.size>0:
            eps,cc=sample_KN(E[ci],rng)
            phi=2*np.pi*rng.random(ci.size)
            newd[ci]=rotate(d[ci],cc,phi)
            newE[ci]=E[ci]*eps
        ri=np.where(isR)[0]
        if ri.size>0:
            # Rayleigh direction sampling p(c) ~ 1+c^2 by rejection
            cs=[]
            c=np.empty(ri.size); idx=np.arange(ri.size)
            while idx.size>0:
                cc=2*rng.random(idx.size)-1
                acc=rng.random(idx.size)*2<(1+cc**2)
                c[idx[acc]]=cc[acc]; idx=idx[~acc]
            phi=2*np.pi*rng.random(ri.size)
            newd[ri]=rotate(d[ri],c,phi)
        pi_=np.where(isPair)[0]
        alive[isPE]=False
        alive[isPair]=False
        # pair -> two annihilation photons
        if pi_.size>0:
            nd=isotropic(pi_.size,rng)
            Xa=X[pi_]
            hpos=np.concatenate([Xa,Xa]); hdir=np.concatenate([nd,-nd])
            hE=np.full(2*pi_.size,0.511); hw=np.concatenate([w[pi_],w[pi_]]); hh=np.concatenate([hid[pi_],hid[pi_]])
        else:
            hpos=np.zeros((0,3)); hdir=np.zeros((0,3)); hE=np.zeros(0); hw=np.zeros(0); hh=np.zeros(0,int)
        # low energy cut
        alive &= newE>Emin
        # new states
        pos=np.concatenate([X[alive],hpos]); d=np.concatenate([newd[alive],hdir])
        E=np.concatenate([newE[alive],hE]); w=np.concatenate([w[alive],hw]); hid=np.concatenate([hid[alive],hh])
        # boundaries: pos on boundary may need small nudge; not needed
        if gen>200: break
    return dict(A_E=A_E,A_c=A_c,A_tot=A_tot,A_Ef=A_Ef,A_K=A_K,A_Kc=A_Kc,A_int=A_int,A_tr=A_tr,store=store,nColl=nColl,N=N)
'''
open('/tmp/work/mc.py','w').write(mc)
import importlib, sys, time
sys.path.insert(0,'/tmp/work')
import mc as M
t=time.time()
res=M.run(20000, [5.0,0.5], seed=3, store_main=0)
print(time.time()-t, res['nColl'], res['A_tot'])
