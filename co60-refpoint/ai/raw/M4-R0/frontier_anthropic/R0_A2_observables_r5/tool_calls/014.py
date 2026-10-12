
test = r'''
import numpy as np
from mcmod import *
def analog_slab(Nh, rng, zlo=4.9, zhi=5.1):
    E0 = LINES[rng.integers(0,2,Nh)]
    u = (rng.random(Nh)-0.5)*0.1; v=(rng.random(Nh)-0.5)*0.1
    A=0.01; W = A*(1+u*u+v*v)**-1.5/Omega
    dn=np.sqrt(1+u*u+v*v); d=np.column_stack([u/dn,v/dn,1/dn]); pos=np.column_stack([100*u,100*v,np.zeros(Nh)])
    E=E0.copy(); Wc=W.copy(); TL=0.0; prim_TL=0.0
    isprim=np.ones(Nh,bool)
    BL=np.array([-15.,-15.,0.]); BH=np.array([15.,15.,30.])
    gen=0
    while E.size>0 and gen<300:
        gen+=1
        with np.errstate(divide='ignore',invalid='ignore'):
            tx=np.where(d>1e-14,(BH-pos)/d,np.where(d<-1e-14,(BL-pos)/d,np.inf))
        tex=np.min(tx,axis=1); mu=mu_tot(E); s=-np.log(rng.random(E.size))/mu
        seg=np.minimum(s,tex)
        # slab overlap
        dz=d[:,2]; z0=pos[:,2]
        with np.errstate(divide='ignore',invalid='ignore'):
            ta=(zlo-z0)/dz; tb=(zhi-z0)/dz
        t1=np.minimum(ta,tb); t2=np.maximum(ta,tb)
        t1=np.where(np.abs(dz)>1e-14,t1,np.where((z0>=zlo)&(z0<=zhi),-np.inf,np.inf))
        t2=np.where(np.abs(dz)>1e-14,t2,np.where((z0>=zlo)&(z0<=zhi),np.inf,-np.inf))
        ov=np.clip(np.minimum(seg,t2)-np.maximum(0,t1),0,None)
        TLc=np.sum(Wc*ov*(~isprim)); TLp=np.sum(Wc*ov*isprim)
        TL+=TLc; prim_TL+=TLp
        inside=s<tex
        pos=pos[inside]+ (s[inside])[:,None]*d[inside]; d=d[inside]; E=E[inside]; Wc=Wc[inside]; isprim=isprim[inside]
        if E.size==0: break
        r=rng.random(E.size)*mu[inside]
        muc=mu_coh(E); mui=mu_inc(E); mupe=mu_pe(E); mup=mu_pair(E); mut=muc+mui+mupe+mup
        is_c=r<mui; is_r=(~is_c)&(r<muc+mui); is_pe=(~is_c)&(~is_r)&(r<muc+mui+mupe); is_pa=(~is_c)&(~is_r)&(~is_pe)
        newpos=[];newd=[];newE=[];newW=[];newp=[]
        if is_c.any():
            idx=np.where(is_c)[0]; ep,cs_=sample_kn(E[idx],rng)
            newpos.append(pos[idx]); newd.append(rot_dir(d[idx],cs_,rng)); newE.append(E[idx]*ep); newW.append(Wc[idx]); newp.append(np.zeros(idx.size,bool))
        if is_r.any():
            idx=np.where(is_r)[0]; cs_=sample_rayleigh_cos(idx.size,rng)
            newpos.append(pos[idx]); newd.append(rot_dir(d[idx],cs_,rng)); newE.append(E[idx]); newW.append(Wc[idx]); newp.append(np.zeros(idx.size,bool))
        if is_pa.any():
            idx=np.where(is_pa)[0]; idx=idx[E[idx]>1.022]
            if idx.size:
                d1=iso_dir(idx.size,rng)
                for sg in [1,-1]:
                    newpos.append(pos[idx]); newd.append(sg*d1); newE.append(np.full(idx.size,0.511)); newW.append(Wc[idx]); newp.append(np.zeros(idx.size,bool))
        if not newE: break
        pos=np.concatenate(newpos); d=np.concatenate(newd); E=np.concatenate(newE); Wc=np.concatenate(newW); isprim=np.concatenate(newp)
        keep=E>ECUT; pos=pos[keep]; d=d[keep]; E=E[keep]; Wc=Wc[keep]; isprim=isprim[keep]
    return TL/Nh, prim_TL/Nh
def nee_slab(Nh, rng, zlo=4.9, zhi=5.1):
    # same transport but NEE into the slab (lateral: whole box), returns scattered fluence (per src)
    E0 = LINES[rng.integers(0,2,Nh)]
    u = (rng.random(Nh)-0.5)*0.1; v=(rng.random(Nh)-0.5)*0.1
    A=0.01; W = A*(1+u*u+v*v)**-1.5/Omega
    dn=np.sqrt(1+u*u+v*v); d=np.column_stack([u/dn,v/dn,1/dn]); pos=np.column_stack([100*u,100*v,np.zeros(Nh)])
    E=E0.copy(); Wc=W.copy(); S=0.0
    BL=np.array([-15.,-15.,0.]); BH=np.array([15.,15.,30.])
    gen=0
    while E.size>0 and gen<300:
        gen+=1
        with np.errstate(divide='ignore',invalid='ignore'):
            tx=np.where(d>1e-14,(BH-pos)/d,np.where(d<-1e-14,(BL-pos)/d,np.inf))
        tex=np.min(tx,axis=1); mu=mu_tot(E); s=-np.log(rng.random(E.size))/mu
        inside=s<tex
        pos=pos[inside]; d=d[inside]; E=E[inside]; Wc=Wc[inside]; mu=mu[inside]; s=s[inside]
        if E.size==0: break
        x=pos+s[:,None]*d; n=E.size
        muc=mu_coh(E); mui=mu_inc(E); mupe=mu_pe(E); mup=mu_pair(E); mut=muc+mui+mupe+mup; k=E/mc2
        # slab y: x,y uniform over box cross-section, z uniform in slab
        yy=np.column_stack([-15+30*rng.random(n), -15+30*rng.random(n), zlo+(zhi-zlo)*rng.random(n)])
        rv=yy-x; R=np.linalg.norm(rv,axis=1); om=rv/R[:,None]; c=np.sum(d*om,axis=1)
        epsC=1/(1+k*(1-c)); E2=E*epsC; s2=1-c*c
        pK=re**2/2*epsC**2*(epsC+1/epsC-s2)/sigKN(E)
        vC=Wc*(mui/mut)*pK*np.exp(-mu_tot(E2)*R)/R**2
        pR=3*(1+c*c)/(16*PI); vR=Wc*(muc/mut)*pR*np.exp(-mu*R)/R**2
        vP=np.where(E>1.022,Wc*(mup/mut)*2*(1/(4*PI))*np.exp(-mu_tot(0.511)*R)/R**2,0.0)
        S+= (vC.sum()+vR.sum()+vP.sum())*30*30*0.2   # volume-average estimator: mean fluence * slab volume
        r=rng.random(n)*mut
        is_c=r<mui; is_r=(~is_c)&(r<muc+mui); is_pe=(~is_c)&(~is_r)&(r<muc+mui+mupe); is_pa=(~is_c)&(~is_r)&(~is_pe)
        newpos=[];newd=[];newE=[];newW=[]
        if is_c.any():
            idx=np.where(is_c)[0]; ep,cs_=sample_kn(E[idx],rng)
            newpos.append(x[idx]); newd.append(rot_dir(d[idx],cs_,rng)); newE.append(E[idx]*ep); newW.append(Wc[idx])
        if is_r.any():
            idx=np.where(is_r)[0]; cs_=sample_rayleigh_cos(idx.size,rng)
            newpos.append(x[idx]); newd.append(rot_dir(d[idx],cs_,rng)); newE.append(E[idx]); newW.append(Wc[idx])
        if is_pa.any():
            idx=np.where(is_pa)[0]; idx=idx[E[idx]>1.022]
            if idx.size:
                d1=iso_dir(idx.size,rng)
                for sg in [1,-1]:
                    newpos.append(x[idx]); newd.append(sg*d1); newE.append(np.full(idx.size,0.511)); newW.append(Wc[idx])
        if not newE: break
        pos=np.concatenate(newpos); d=np.concatenate(newd); E=np.concatenate(newE); Wc=np.concatenate(newW)
        keep=E>ECUT; pos=pos[keep]; d=d[keep]; E=E[keep]; Wc=Wc[keep]
    return S/Nh/(30*30*0.2)*(30*30*0.2)  # mean-fluence*volume-> compare with track length
'''
open('/tmp/work/test_slab.py','w').write(test)
import sys; sys.path.insert(0,'/tmp/work')
import importlib, test_slab
importlib.reload(test_slab)
rng=np.random.default_rng(5)
t0=time.time()
TLc, TLp = test_slab.analog_slab(40000, rng)
print("analog slab track per src (scat, prim):", TLc, TLp, "time", time.time()-t0)
