
mcrun = r'''
import numpy as np
from mcmod import *

# bins
PH_EDGES=np.array([0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4])
EL_EDGES=np.array([0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.4])
CO_EDGES=np.array([-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1.0])
FINE=np.logspace(-3,np.log10(1.4),241)   # fine T grid for electron source histograms
ZEDGES=np.linspace(0,10,101)

def new_tally():
    return dict(
      arr_spec=np.zeros(28), arr_cos=np.zeros(10), arr_E=0.0, arr_tot=0.0,
      cnt_c=0.0, cnt_pe=0.0, cnt_pair=0.0, cnt_r=0.0,
      etr_c=0.0, etr_pe=0.0, etr_pair=0.0,
      K_sc=0.0, Kcol_sc=0.0,
      el_T_all=np.zeros(25), el_C_all=np.zeros(10), el_w_all=0.0, el_wT_all=0.0,
      el_T_prim=np.zeros(25), el_w_prim=0.0, el_wT_prim=0.0,
      q_fine_all=np.zeros(240), q_fine_em=np.zeros(240),
      q_fine_em_cos=np.zeros((240,10)), q_fine_all_cos=np.zeros((240,10)),
      zb_fl=np.zeros(100), zb_kcol=np.zeros(100),
      nh=0, nscat_coll=0)

def tally_electrons(tal, lst, norm, prim):
    for (T,cz,w,ispos,tag) in lst:
        w=w*norm
        # all electrons (for energy budget and L3)
        tal['el_w_all']+=w.sum(); tal['el_wT_all']+=(w*T).sum()
        h,_=np.histogram(T,bins=EL_EDGES,weights=w); tal['el_T_all']+=h
        hc,_=np.histogram(cz,bins=CO_EDGES,weights=w); tal['el_C_all']+=hc
        fb=np.clip(np.digitize(T,FINE)-1,0,239)
        ok=(T>=FINE[0])&(T<FINE[-1])
        np.add.at(tal['q_fine_all'],fb[ok],w[ok])
        cb=np.clip(np.digitize(cz,CO_EDGES)-1,0,9)
        np.add.at(tal['q_fine_all_cos'],(fb[ok],cb[ok]),w[ok])
        if prim:
            tal['el_w_prim']+=w.sum(); tal['el_wT_prim']+=(w*T).sum()
            h,_=np.histogram(T,bins=EL_EDGES,weights=w); tal['el_T_prim']+=h
        if not ispos.all():
            m=~ispos
            np.add.at(tal['q_fine_em'],fb[ok&m],w[ok&m])
            np.add.at(tal['q_fine_em_cos'],(fb[ok&m],cb[ok&m]),w[ok&m])

def chunk(Nh, rng, tal):
    """Run one chunk of Nh source photons. Weights normalised per source photon (divide by total N later)."""
    tal['nh']+=Nh
    Ntot_norm = 1.0   # we accumulate sum over histories; divide by total Nh at the end
    E0 = LINES[rng.integers(0,2,Nh)]
    u = (rng.random(Nh)-0.5)*0.1; v=(rng.random(Nh)-0.5)*0.1
    A=0.01
    W = A*(1+u*u+v*v)**-1.5/Omega          # weight per history (solid-angle uniform emission)
    dn = np.sqrt(1+u*u+v*v)
    d = np.column_stack([u/dn, v/dn, 1/dn])
    pos = np.column_stack([100*u, 100*v, np.zeros(Nh)])
    E=E0.copy(); Wc=W.copy()
    # first move to phantom entrance is at z=0 already
    gen=0
    while E.size>0:
        gen+=1
        # distance to box exit
        with np.errstate(divide='ignore',invalid='ignore'):
            tx = np.where(d>1e-14, (BOX_HI-pos)/d, np.where(d<-1e-14,(BOX_LO-pos)/d, np.inf))
        tex = np.min(tx,axis=1)
        mu = mu_tot(E)
        s = -np.log(rng.random(E.size))/mu
        inside = s < tex
        pos=pos[inside]; d=d[inside]; E=E[inside]; Wc=Wc[inside]; mu=mu[inside]; s=s[inside]
        if E.size==0: break
        x = pos + s[:,None]*d
        # ---- NEE scoring for all collisions ----
        n=E.size
        muc=mu_coh(E); mui=mu_inc(E); mupe=mu_pe(E); mup=mu_pair(E)
        mut=muc+mui+mupe+mup
        # V detector
        yV = cyl_sample(rng, n, 4.9, 5.1)
        rv = yV-x; R=np.linalg.norm(rv,axis=1); om=rv/R[:,None]
        c = np.sum(d*om,axis=1)
        k=E/mc2
        epsC = 1/(1+k*(1-c)); E2=E*epsC; s2=1-c*c
        dsdO = re**2/2*epsC**2*(epsC+1/epsC-s2)
        pK = dsdO/sigKN(E)
        valC = Wc*(mui/mut)*pK*np.exp(-mu_tot(E2)*R)/R**2 /n * 0 + Wc*(mui/mut)*pK*np.exp(-mu_tot(E2)*R)/R**2
        pR = 3*(1+c*c)/(16*PI)
        valR = Wc*(muc/mut)*pR*np.exp(-mu*R)/R**2
        valP = np.where(E>1.022, Wc*(mup/mut)*2*(1/(4*PI))*np.exp(-mu_tot(0.511)*R)/R**2, 0.0)
        # arrivals (per history weight already in Wc; divide by total histories at end)
        # Compton arrival
        for (Ea,va) in [(E2,valC),(E,valR),(np.full(n,0.511),valP)]:
            ok = va>0
            if not ok.any(): continue
            Ea_=Ea[ok]; va_=va[ok]; oo=om[ok]
            tal['arr_spec']+=np.histogram(Ea_,bins=PH_EDGES,weights=va_)[0]
            tal['arr_cos']+=np.histogram(oo[:,2],bins=CO_EDGES,weights=va_)[0]
            tal['arr_E']+=np.sum(va_*Ea_); tal['arr_tot']+=np.sum(va_)
            tal['K_sc']+=np.sum(va_*Ea_*mu_tr_c(Ea_)*0)  # placeholder, set below
            # interaction-process bookkeeping (rate per volume = sum val*mu_proc(E'))
            tal['cnt_c']+=np.sum(va_*mu_inc(Ea_)) if Ea is E2 else 0.0
            tal['cnt_r']+=np.sum(va_*mu_coh(Ea_)) if Ea is E else 0.0
            tal['cnt_pair']+=np.sum(va_*mu_pair(Ea_)) if Ea is not E2 and Ea is not E else 0.0
            tal['cnt_pe']+=np.sum(va_*mu_pe(Ea_)) if Ea is E2 else 0.0
            tal['etr_c']+=np.sum(va_*mu_tr_c(Ea_)) if Ea is E2 else 0.0
            tal['etr_pe']+=np.sum(va_*mu_tr_pe(Ea_)) if Ea is E2 else 0.0
            tal['etr_pair']+=np.sum(va_*mu_tr_pair(Ea_)) if (Ea is not E2 and Ea is not E) else 0.0
            tal['K_sc']+=np.sum(va_*Ea_*(mu_tr_c(Ea_) if Ea is E2 else (mu_tr_pair(Ea_) if Ea is not E else 0.0)))
        # electrons from arrivals (Compton/photo/pair from each scattered arrival)
        ok=valC>0
        lst=[]
        lst += electrons_from_arrivals(E2[ok], om[ok], valC[ok], rng, False)  # compton+photo+pair from Compton-arrivals
        ok=valR>0
        lst += electrons_from_arrivals(E[ok], om[ok], valR[ok], rng, False)
        ok=valP>0
        lst += electrons_from_arrivals(np.full(ok.sum(),0.511), om[ok], valP[ok], rng, False)
        tally_electrons(tal, lst, 1.0, False)
        # column detector for depth profile
        yC = cyl_sample(rng,n,0.0,10.0)
        rc = yC-x; Rc=np.linalg.norm(rc,axis=1); omc=rc/Rc[:,None]; cc=np.sum(d*omc,axis=1)
        epsCc=1/(1+k*(1-cc)); E2c=E*epsCc; s2c=1-cc*cc
        pKc=re**2/2*epsCc**2*(epsCc+1/epsCc-s2c)/sigKN(E)
        vC = Wc*(mui/mut)*pKc*np.exp(-mu_tot(E2c)*Rc)/Rc**2
        pRc=3*(1+cc*cc)/(16*PI)
        vR = Wc*(muc/mut)*pRc*np.exp(-mu*Rc)/Rc**2
        vP = np.where(E>1.022, Wc*(mup/mut)*2*(1/(4*PI))*np.exp(-mu_tot(0.511)*Rc)/Rc**2, 0.0)
        zb = np.clip(np.digitize(yC[:,2],ZEDGES)-1,0,99)
        for (Ea,va) in [(E2c,vC),(E,vR),(np.full(n,0.511),vP)]:
            np.add.at(tal['zb_fl'], zb, va)
            np.add.at(tal['zb_kcol'], zb, va*Ea*mu_en_c(Ea) if False else va*Ea*(mu_en_c(Ea)))
        # ---- propagate the actual photons ----
        r = rng.random(n)*mut
        is_c = r < mui
        is_r = (~is_c) & (r < muc+mui)
        is_pe = (~is_c)&(~is_r)&(r < muc+mui+mupe)
        is_pa = (~is_c)&(~is_r)&(~is_pe)
        newpos=[];newd=[];newE=[];newW=[]
        # Compton
        if is_c.any():
            idx=np.where(is_c)[0]
            ep, cs_ = sample_kn(E[idx], rng)
            dn_ = rot_dir(d[idx], cs_, rng)
            newpos.append(x[idx]); newd.append(dn_); newE.append(E[idx]*ep); newW.append(Wc[idx])
        if is_r.any():
            idx=np.where(is_r)[0]
            cs_=sample_rayleigh_cos(idx.size, rng)
            dn_=rot_dir(d[idx], cs_, rng)
            newpos.append(x[idx]); newd.append(dn_); newE.append(E[idx]); newW.append(Wc[idx])
        if is_pa.any():
            idx=np.where(is_pa)[0]
            idx=idx[E[idx]>1.022]
            if idx.size:
                d1=iso_dir(idx.size,rng)
                newpos.append(x[idx]); newd.append(d1); newE.append(np.full(idx.size,0.511)); newW.append(Wc[idx])
                newpos.append(x[idx]); newd.append(-d1); newE.append(np.full(idx.size,0.511)); newW.append(Wc[idx])
        if newE:
            pos=np.concatenate(newpos); d=np.concatenate(newd); E=np.concatenate(newE); Wc=np.concatenate(newW)
            keep = E>ECUT
            pos=pos[keep]; d=d[keep]; E=E[keep]; Wc=Wc[keep]
        else:
            break
        if gen>200: break
    return tal
'''
open('/tmp/work/mcrun.py','w').write(mcrun)
print("written")
