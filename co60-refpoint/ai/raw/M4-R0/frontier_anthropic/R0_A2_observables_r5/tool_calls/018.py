
mcana = r'''
import numpy as np
from mcmod import *

PH_EDGES=np.array([0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4])
EL_EDGES=np.array([0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.4])
CO_EDGES=np.array([-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1.0])
FINE=np.logspace(-3,np.log10(1.4),241)
NFINE=240
ZB=np.linspace(0,10,101)          # z-bins for depth profile (0.1 cm)
LAT=4.0                           # lateral half-size of depth-profile region
VCYL=np.pi*1.0**2*0.2
ZV0,ZV1=4.9,5.1

def mu_tr_all(E): return mu_tr_c(E)+mu_tr_pe(E)+mu_tr_pair(E)
def mu_en_all(E): return mu_en_c(E)+mu_en_pe(E)+mu_en_pair(E)

def seg_overlap_cyl(p0,d,s):
    """length of segment p0+t d (t in [0,s]) inside cylinder r<1, z in [ZV0,ZV1]"""
    dz=d[:,2]
    with np.errstate(divide='ignore',invalid='ignore'):
        ta=(ZV0-p0[:,2])/dz; tb=(ZV1-p0[:,2])/dz
    tz1=np.where(np.abs(dz)>1e-12,np.minimum(ta,tb),np.where((p0[:,2]>=ZV0)&(p0[:,2]<=ZV1),-np.inf,np.inf))
    tz2=np.where(np.abs(dz)>1e-12,np.maximum(ta,tb),np.where((p0[:,2]>=ZV0)&(p0[:,2]<=ZV1),np.inf,-np.inf))
    a=d[:,0]**2+d[:,1]**2; b=2*(p0[:,0]*d[:,0]+p0[:,1]*d[:,1]); c=p0[:,0]**2+p0[:,1]**2-1.0
    disc=b*b-4*a*c
    with np.errstate(divide='ignore',invalid='ignore'):
        sq=np.sqrt(np.maximum(disc,0))
        r1=(-b-sq)/(2*a); r2=(-b+sq)/(2*a)
    rlo=np.where(a>1e-14, np.where(disc>=0, np.minimum(r1,r2), np.inf), np.where(c<0,-np.inf,np.inf))
    rhi=np.where(a>1e-14, np.where(disc>=0, np.maximum(r1,r2), -np.inf), np.where(c<0,np.inf,-np.inf))
    lo=np.maximum.reduce([np.zeros_like(s), tz1, rlo])
    hi=np.minimum.reduce([s, tz2, rhi])
    return np.clip(hi-lo,0,None)

def seg_overlap_bins(p0,d,s):
    """n x 100 array: overlap of segment with depth bins (|x|,|y|<LAT)."""
    n=p0.shape[0]
    out=np.zeros((n,100))
    # lateral interval
    lat_lo=np.zeros(n); lat_hi=s.copy()
    ok=np.ones(n,bool)
    for ax in (0,1):
        dd=d[:,ax]; pp=p0[:,ax]
        with np.errstate(divide='ignore',invalid='ignore'):
            ta=(-LAT-pp)/dd; tb=(LAT-pp)/dd
        t1=np.where(np.abs(dd)>1e-12,np.minimum(ta,tb),np.where(np.abs(pp)<=LAT,-np.inf,np.inf))
        t2=np.where(np.abs(dd)>1e-12,np.maximum(ta,tb),np.where(np.abs(pp)<=LAT,np.inf,-np.inf))
        lat_lo=np.maximum(lat_lo,t1); lat_hi=np.minimum(lat_hi,t2)
    dz=d[:,2]; z0=p0[:,2]
    with np.errstate(divide='ignore',invalid='ignore'):
        ta=(ZB[None,:-1]-z0[:,None])/dz[:,None]; tb=(ZB[None,1:]-z0[:,None])/dz[:,None]
    zl=np.minimum(ta,tb); zh=np.maximum(ta,tb)
    dzz=np.abs(dz)[:,None]>1e-12
    inz=((z0>=0)&(z0<=10))[:,None]
    zl=np.where(dzz,zl,np.where((z0[:,None]>=ZB[None,:-1])&(z0[:,None]<=ZB[None,1:]),-np.inf,np.inf))
    zh=np.where(dzz,zh,np.where((z0[:,None]>=ZB[None,:-1])&(z0[:,None]<=ZB[None,1:]),np.inf,-np.inf))
    lo=np.maximum(np.maximum(zl,lat_lo[:,None]),0.0)
    hi=np.minimum(np.minimum(zh,lat_hi[:,None]),s[:,None])
    return np.clip(hi-lo,0,None)

def new_tally():
    return dict(nh=0.0, TLV=0.0, arr_spec=np.zeros(28), arr_cos=np.zeros(10), arr_E=0.0, arr_tot=0.0,
      cnt_c=0.0, cnt_pe=0.0, cnt_pair=0.0, cnt_r=0.0, etr_c=0.0, etr_pe=0.0, etr_pair=0.0, etr_r=0.0,
      K_sc=0.0, Kcol_sc=0.0, TLV_cnt=0.0,
      el_T_all=np.zeros(25), el_C_all=np.zeros(10), el_w_all=0.0, el_wT_all=0.0,
      q_fine_all=np.zeros(NFINE), q_fine_em=np.zeros(NFINE),
      q_fine_all_cos=np.zeros((NFINE,10)), q_fine_em_cos=np.zeros((NFINE,10)),
      zb_tl=np.zeros(100), zb_kcol=np.zeros(100), ncoll=0, nseg=0, TLV_by_cos=np.zeros(10))

def tally_electrons(tal, lst):
    for (T,cz,w,ispos) in lst:
        tal['el_w_all']+=w.sum(); tal['el_wT_all']+=(w*T).sum()
        h,_=np.histogram(T,bins=EL_EDGES,weights=w); tal['el_T_all']+=h
        hc,_=np.histogram(cz,bins=CO_EDGES,weights=w); tal['el_C_all']+=hc
        ok=(T>=FINE[0])&(T<FINE[-1])
        fb=np.clip(np.digitize(T,FINE)-1,0,NFINE-1)
        cb=np.clip(np.digitize(cz,CO_EDGES)-1,0,9)
        np.add.at(tal['q_fine_all'],fb[ok],w[ok])
        np.add.at(tal['q_fine_all_cos'],(fb[ok],cb[ok]),w[ok])
        m=ok&(~ispos)
        np.add.at(tal['q_fine_em'],fb[m],w[m])
        np.add.at(tal['q_fine_em_cos'],(fb[m],cb[m]),w[m])

def chunk(Nh, rng, tal, ksamp=8):
    tal['nh']+=Nh
    E0 = LINES[rng.integers(0,2,Nh)]
    u = (rng.random(Nh)-0.5)*0.1; v=(rng.random(Nh)-0.5)*0.1
    A=0.01
    W = A*(1+u*u+v*v)**-1.5/Omega
    dn = np.sqrt(1+u*u+v*v)
    d = np.column_stack([u/dn, v/dn, 1/dn])
    pos = np.column_stack([100*u, 100*v, np.zeros(Nh)])
    E=E0.copy(); Wc=W.copy(); scat=np.zeros(Nh,bool)
    gen=0
    while E.size>0 and gen<400:
        gen+=1
        with np.errstate(divide='ignore',invalid='ignore'):
            tx = np.where(d>1e-14, (BOX_HI-pos)/d, np.where(d<-1e-14,(BOX_LO-pos)/d, np.inf))
        tex = np.min(tx,axis=1)
        mu = mu_tot(E)
        s = -np.log(rng.random(E.size))/mu
        coll = s < tex
        seg = np.where(coll, s, tex)
        # ---- analog track-length tallies for scattered photons over this segment ----
        m = scat
        if m.any():
            p0=pos[m]; dd=d[m]; ss=seg[m]; ww=Wc[m]; EE=E[m]
            ov = seg_overlap_cyl(p0,dd,ss)
            hit = ov>0
            if hit.any():
                wv = ww[hit]*ov[hit]          # track length weight (per history)
                Eh=EE[hit]; dh=dd[hit]
                tal['TLV']+=wv.sum(); tal['nseg']+=hit.sum()
                tal['TLV_by_cos']+=np.histogram(dh[:,2],bins=CO_EDGES,weights=wv)[0]
                valv = wv/VCYL                  # contribution to V-averaged fluence
                tal['arr_spec']+=np.histogram(Eh,bins=PH_EDGES,weights=valv)[0]
                tal['arr_cos']+=np.histogram(dh[:,2],bins=CO_EDGES,weights=valv)[0]
                tal['arr_E']+=np.sum(valv*Eh); tal['arr_tot']+=np.sum(valv)
                tal['cnt_c']+=np.sum(valv*mu_inc(Eh)); tal['cnt_pe']+=np.sum(valv*mu_pe(Eh))
                tal['cnt_pair']+=np.sum(valv*mu_pair(Eh)); tal['cnt_r']+=np.sum(valv*mu_coh(Eh))
                tal['etr_c']+=np.sum(valv*mu_tr_c(Eh)); tal['etr_pe']+=np.sum(valv*mu_tr_pe(Eh)); tal['etr_pair']+=np.sum(valv*mu_tr_pair(Eh))
                tal['K_sc']+=np.sum(valv*Eh*mu_tr_all(Eh)); tal['Kcol_sc']+=np.sum(valv*Eh*mu_en_all(Eh))
                lst = electrons_from_arrivals(np.tile(Eh,ksamp), np.tile(dh,(ksamp,1)), np.tile(valv/ksamp,ksamp), rng, False)
                tally_electrons(tal, lst)
            # depth-profile region (|x|,|y|<LAT, z 0..10)
            ob = seg_overlap_bins(p0,dd,ss)
            sel = ob.sum(axis=1)>0
            if sel.any():
                obs=ob[sel]; ws=ww[sel]; Es=EE[sel]
                tl_bins = (obs*ws[:,None]).sum(axis=0)
                tal['zb_tl']+=tl_bins
                tal['zb_kcol']+=(obs*(ws*Es*mu_en_all(Es))[:,None]).sum(axis=0)
        # ---- move photons ----
        pos = pos + seg[:,None]*d
        tal['ncoll']+=int(coll.sum())
        # photons leaving are dropped
        alive = coll
        pos=pos[alive]; d=d[alive]; E=E[alive]; Wc=Wc[alive]
        if E.size==0: break
        n=E.size
        muc=mu_coh(E); mui=mu_inc(E); mupe=mu_pe(E); mup=mu_pair(E); mut=muc+mui+mupe+mup
        r = rng.random(n)*mut
        is_c = r < mui
        is_r = (~is_c) & (r < muc+mui)
        is_pe = (~is_c)&(~is_r)&(r < muc+mui+mupe)
        is_pa = (~is_c)&(~is_r)&(~is_pe)
        newpos=[];newd=[];newE=[];newW=[];news=[]
        if is_c.any():
            idx=np.where(is_c)[0]
            ep, cs_ = sample_kn(E[idx], rng)
            newpos.append(pos[idx]); newd.append(rot_dir(d[idx], cs_, rng)); newE.append(E[idx]*ep); newW.append(Wc[idx]); news.append(np.ones(idx.size,bool))
        if is_r.any():
            idx=np.where(is_r)[0]
            cs_=sample_rayleigh_cos(idx.size, rng)
            newpos.append(pos[idx]); newd.append(rot_dir(d[idx], cs_, rng)); newE.append(E[idx]); newW.append(Wc[idx]); news.append(np.ones(idx.size,bool))
        if is_pa.any():
            idx=np.where(is_pa)[0]
            idx=idx[E[idx]>1.022]
            if idx.size:
                d1=iso_dir(idx.size,rng)
                newpos.append(pos[idx]); newd.append(d1); newE.append(np.full(idx.size,0.511)); newW.append(Wc[idx]); news.append(np.ones(idx.size,bool))
                newpos.append(pos[idx]); newd.append(-d1); newE.append(np.full(idx.size,0.511)); newW.append(Wc[idx]); news.append(np.ones(idx.size,bool))
        if not newE: break
        pos=np.concatenate(newpos); d=np.concatenate(newd); E=np.concatenate(newE); Wc=np.concatenate(newW); scat=np.concatenate(news)
        keep = E>ECUT
        pos=pos[keep]; d=d[keep]; E=E[keep]; Wc=Wc[keep]; scat=scat[keep]
    return tal
'''
open('/tmp/work/mcana.py','w').write(mcana)
import sys, time; sys.path.insert(0,'/tmp/work')
import importlib, mcana
importlib.reload(mcana)
from mcana import *
t0=time.time()
tal=new_tally(); rng=np.random.default_rng(2)
tal=chunk(50000, rng, tal)
print("time",time.time()-t0,"nseg",tal['nseg'],"TLV per src",tal['TLV']/tal['nh'], "scat fluence/Phi0", tal['TLV']/tal['nh']/VCYL/Phi0_src)
