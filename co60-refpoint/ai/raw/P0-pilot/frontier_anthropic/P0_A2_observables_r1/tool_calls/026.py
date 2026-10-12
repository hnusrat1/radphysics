
code_dk = r'''
import numpy as np, sys
sys.path.insert(0,'/tmp/work')
from phys import *
from electrons import *   # EBe, moller tables, sample_delta_T, lam_of_E, _orth, DEL, X0, ME
ZB=np.arange(-6.0,8.0001,0.05)   # depth bins (cm) for deposit kernel
def transport_depth(T0,u0,w0,rng,maxit=5000):
    E=T0.copy(); u=u0.copy(); w=w0.copy(); isd=np.zeros(len(E),bool); Z=np.zeros(len(E))
    hist=np.zeros(len(ZB)+1)
    def add(zv,vals):
        b=np.clip(np.floor((zv-ZB[0])/0.05).astype(int),0,len(ZB)-1)
        np.add.at(hist,b,vals)
    it=0
    # low-energy sources deposit locally
    m=E<DEL
    if m.any(): add(Z[m],w[m]*E[m]); E=E[~m];u=u[~m];w=w[~m];isd=isd[~m];Z=Z[~m]
    while len(E)>0 and it<maxit:
        it+=1
        S=Stot(E)
        dE=0.03*E
        ds=dE/S
        zc=Z+0.5*ds*u[:,2]
        lam=lam_of_E(E)*ds
        nd=rng.poisson(lam)
        idx=np.repeat(np.arange(len(E)),nd)
        Tsum=np.zeros(len(E))
        if len(idx)>0:
            Tp=sample_delta_T(E[idx],rng)
            np.add.at(Tsum,idx,Tp)
            cth=np.sqrt(np.clip(Tp*(E[idx]+2*ME)/(E[idx]*(Tp+2*ME)),0,1))
            phi=rng.uniform(0,2*np.pi,len(idx))
            ud=rotate(u[idx],cth,phi)
            pp=np.sqrt(E*(E+2*ME)); pd=np.sqrt(Tp*(Tp+2*ME))
            pvec=np.zeros_like(u); np.add.at(pvec,idx,pd[:,None]*ud)
            newp=pp[:,None]*u-pvec
            nrm=np.linalg.norm(newp,axis=1)
            u=np.where((nrm>1e-12)[:,None], newp/np.maximum(nrm,1e-300)[:,None], u)
            Dw=w[idx]; Dz=zc[idx]
        dep=np.maximum(dE-Tsum,0)
        add(zc,w*dep)
        Enew=E-dE
        Znew=Z+ds*u[:,2]
        x=ds
        pc=np.sqrt(E*(E+2*ME)); beta=pc/(E+ME)
        th0=13.6/(beta*pc)*np.sqrt(np.maximum(x,1e-30)/X0)*(1+0.038*np.log(np.maximum(x,1e-30)/X0))
        th0=np.clip(th0,0,np.pi)
        e1,e2=_orth(u)
        g1=rng.normal(size=len(E))*th0; g2=rng.normal(size=len(E))*th0
        un=u+g1[:,None]*e1+g2[:,None]*e2
        u=un/np.linalg.norm(un,axis=1,keepdims=True)
        E=Enew; Z=Znew
        # remove stopped
        st=E<DEL
        if st.any():
            add(Z[st],w[st]*np.maximum(E[st],0))
        keep=~st
        E=E[keep];u=u[keep];w=w[keep];isd=isd[keep];Z=Z[keep]
        if len(idx)>0:
            Tn=Tp; 
            # new deltas with T>=DEL
            E=np.concatenate([E,Tn]); u=np.concatenate([u,ud]); w=np.concatenate([w,Dw]); isd=np.concatenate([isd,np.ones(len(Tn),bool)]); Z=np.concatenate([Z,Dz])
    return hist[:len(ZB)]
'''
open('/tmp/work/depthk.py','w').write(code_dk)
import sys, importlib, time, pickle
sys.path.insert(0,'/tmp/work')
import numpy as np
import sources, depthk
from phys import mu_all
seg=pickle.load(open('/tmp/work/photon_tallies.pkl','rb'))['seg']
Phi_p=float(np.exp(-mu_all(np.array([2.0]))[4][0]*10))
rng=np.random.default_rng(5)
cat=sources.build_sources(seg,4000000,rng,Phi_p)
# subsample for speed: use all
t=time.time()
rng2=np.random.default_rng(7)
# transport all sources (electrons and positrons treated alike)
N=len(cat['T'])
hist=depthk.transport_depth(cat['T'],cat['u'],cat['w'],rng2)
print("time",time.time()-t, "sum hist",hist.sum(), "sum expected", (cat['w']*cat['T']).sum())
pickle.dump(dict(hist=hist,ZB=depthk.ZB),open('/tmp/work/kern.pkl','wb'))
