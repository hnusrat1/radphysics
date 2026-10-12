
mcrun = r'''
import numpy as np
from mcmod import *

PH_EDGES=np.array([0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4])
EL_EDGES=np.array([0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.4])
CO_EDGES=np.array([-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1.0])
FINE=np.logspace(-3,np.log10(1.4),241)
ZEDGES=np.linspace(0,10,101)
NFINE=240

def new_tally():
    return dict(
      arr_spec=np.zeros(28), arr_cos=np.zeros(10), arr_E=0.0, arr_tot=0.0,
      cnt_c=0.0, cnt_pe=0.0, cnt_pair=0.0, cnt_r=0.0,
      etr_c=0.0, etr_pe=0.0, etr_pair=0.0,
      K_sc=0.0, Kcol_sc=0.0,
      el_T_all=np.zeros(25), el_C_all=np.zeros(10), el_w_all=0.0, el_wT_all=0.0,
      q_fine_all=np.zeros(NFINE), q_fine_em=np.zeros(NFINE),
      q_fine_all_cos=np.zeros((NFINE,10)), q_fine_em_cos=np.zeros((NFINE,10)),
      zb_fl=np.zeros(100), zb_kcol=np.zeros(100), nh=0, ncoll=0)

def mu_tr_all(E): return mu_tr_c(E)+mu_tr_pe(E)+mu_tr_pair(E)
def mu_en_all(E): return mu_en_c(E)+mu_en_pe(E)+mu_en_pair(E)

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

def chunk(Nh, rng, tal):
    tal['nh']+=Nh
    E0 = LINES[rng.integers(0,2,Nh)]
    u = (rng.random(Nh)-0.5)*0.1; v=(rng.random(Nh)-0.5)*0.1
    A=0.01
    W = A*(1+u*u+v*v)**-1.5/Omega
    dn = np.sqrt(1+u*u+v*v)
    d = np.column_stack([u/dn, v/dn, 1/dn])
    pos = np.column_stack([100*u, 100*v, np.zeros(Nh)])
    E=E0.copy(); Wc=W.copy()
    gen=0
    while E.size>0 and gen<300:
        gen+=1
        with np.errstate(divide='ignore',invalid='ignore'):
            tx = np.where(d>1e-14, (BOX_HI-pos)/d, np.where(d<-1e-14,(BOX_LO-pos)/d, np.inf))
        tex = np.min(tx,axis=1)
        mu = mu_tot(E)
        s = -np.log(rng.random(E.size))/mu
        inside = s < tex
        pos=pos[inside]; d=d[inside]; E=E[inside]; Wc=Wc[inside]; mu=mu[inside]; s=s[inside]
        if E.size==0: break
        x = pos + s[:,None]*d
        n=E.size; tal['ncoll']+=n
        muc=mu_coh(E); mui=mu_inc(E); mupe=mu_pe(E); mup=mu_pair(E)
        mut=muc+mui+mupe+mup
        k=E/mc2
        # ---------- NEE to scoring cylinder ----------
        yV = cyl_sample(rng, n, 4.9, 5.1)
        rv = yV-x; R=np.linalg.norm(rv,axis=1); om=rv/R[:,None]
        c = np.sum(d*om,axis=1)
        epsC = 1/(1+k*(1-c)); E2=E*epsC; s2=1-c*c
        pK = re**2/2*epsC**2*(epsC+1/epsC-s2)/sigKN(E)
        valC = Wc*(mui/mut)*pK*np.exp(-mu_tot(E2)*R)/R**2
        pR = 3*(1+c*c)/(16*PI)
        valR = Wc*(muc/mut)*pR*np.exp(-mu_tot(E)*R)/R**2
        valP = np.where(E>1.022, Wc*(mup/mut)*2*(1/(4*PI))*np.exp(-mu_tot(0.511)*R)/R**2, 0.0)
        Ea = np.concatenate([E2,E,np.full(n,0.511)])
        va = np.concatenate([valC,valR,valP])
        oa = np.concatenate([om,om,om])
        ok = va>0
        Ea=Ea[ok]; va=va[ok]; oa=oa[ok]
        if Ea.size:
            tal['arr_spec']+=np.histogram(Ea,bins=PH_EDGES,weights=va)[0]
            tal['arr_cos']+=np.histogram(oa[:,2],bins=CO_EDGES,weights=va)[0]
            tal['arr_E']+=np.sum(va*Ea); tal['arr_tot']+=np.sum(va)
            tal['cnt_c']+=np.sum(va*mu_inc(Ea)); tal['cnt_pe']+=np.sum(va*mu_pe(Ea))
            tal['cnt_pair']+=np.sum(va*mu_pair(Ea)); tal['cnt_r']+=np.sum(va*mu_coh(Ea))
            tal['etr_c']+=np.sum(va*mu_tr_c(Ea)); tal['etr_pe']+=np.sum(va*mu_tr_pe(Ea)); tal['etr_pair']+=np.sum(va*mu_tr_pair(Ea))
            tal['K_sc']+=np.sum(va*Ea*mu_tr_all(Ea)); tal['Kcol_sc']+=np.sum(va*Ea*mu_en_all(Ea))
            # electrons from all scattered arrivals (Compton, photo, pair)
            lst = electrons_from_arrivals(Ea, oa, va, rng, False)
            tally_electrons(tal, lst)
        # ---------- column detector (depth profile) ----------
        yC = cyl_sample(rng,n,0.0,10.0)
        rc = yC-x; Rc=np.linalg.norm(rc,axis=1); omc=rc/Rc[:,None]; cc=np.sum(d*omc,axis=1)
        epsc=1/(1+k*(1-cc)); E2c=E*epsc; s2c=1-cc*cc
        pKc=re**2/2*epsc**2*(epsc+1/epsc-s2c)/sigKN(E)
        vC = Wc*(mui/mut)*pKc*np.exp(-mu_tot(E2c)*Rc)/Rc**2
        pRc=3*(1+cc*cc)/(16*PI)
        vR = Wc*(muc/mut)*pRc*np.exp(-mu_tot(E)*Rc)/Rc**2
        vP = np.where(E>1.022, Wc*(mup/mut)*2*(1/(4*PI))*np.exp(-mu_tot(0.511)*Rc)/Rc**2, 0.0)
        zb = np.clip(np.digitize(yC[:,2],ZEDGES)-1,0,99)
        Ec=np.concatenate([E2c,E,np.full(n,0.511)]); vc=np.concatenate([vC,vR,vP]); zz=np.concatenate([zb,zb,zb])
        np.add.at(tal['zb_fl'], zz, vc)
        np.add.at(tal['zb_kcol'], zz, vc*Ec*mu_en_all(Ec))
        # ---------- propagate photons ----------
        r = rng.random(n)*mut
        is_c = r < mui
        is_r = (~is_c) & (r < muc+mui)
        is_pe = (~is_c)&(~is_r)&(r < muc+mui+mupe)
        is_pa = (~is_c)&(~is_r)&(~is_pe)
        newpos=[];newd=[];newE=[];newW=[]
        if is_c.any():
            idx=np.where(is_c)[0]
            ep, cs_ = sample_kn(E[idx], rng)
            newpos.append(x[idx]); newd.append(rot_dir(d[idx], cs_, rng)); newE.append(E[idx]*ep); newW.append(Wc[idx])
        if is_r.any():
            idx=np.where(is_r)[0]
            cs_=sample_rayleigh_cos(idx.size, rng)
            newpos.append(x[idx]); newd.append(rot_dir(d[idx], cs_, rng)); newE.append(E[idx]); newW.append(Wc[idx])
        if is_pa.any():
            idx=np.where(is_pa)[0]
            idx=idx[E[idx]>1.022]
            if idx.size:
                d1=iso_dir(idx.size,rng)
                newpos.append(x[idx]); newd.append(d1); newE.append(np.full(idx.size,0.511)); newW.append(Wc[idx])
                newpos.append(x[idx]); newd.append(-d1); newE.append(np.full(idx.size,0.511)); newW.append(Wc[idx])
        if not newE: break
        pos=np.concatenate(newpos); d=np.concatenate(newd); E=np.concatenate(newE); Wc=np.concatenate(newW)
        keep = E>ECUT
        pos=pos[keep]; d=d[keep]; E=E[keep]; Wc=Wc[keep]
    return tal
'''
open('/tmp/work/mcrun.py','w').write(mcrun)
# add electrons_from_arrivals signature: (Ep, om, val, rng, tag) -> returns list of (T,cos,w,ispos)
s=open('/tmp/work/mcmod.py').read()
s=s.replace("out.append((T, cose, val*mi, np.zeros(n,bool), tag))","out.append((T, cose, val*mi, np.zeros(n,bool)))")
s=s.replace("out.append((Ep.copy(), om[:,2].copy(), val*mp, np.zeros(n,bool), tag))","out.append((Ep.copy(), om[:,2].copy(), val*mp, np.zeros(n,bool)))")
s=s.replace("out.append((U*Tp, om[ok,2], w, np.zeros(ok.sum(),bool), tag))","out.append((U*Tp, om[ok,2], w, np.zeros(ok.sum(),bool)))")
s=s.replace("out.append(((1-U)*Tp, om[ok,2], w, np.ones(ok.sum(),bool), tag)) # positron","out.append(((1-U)*Tp, om[ok,2], w, np.ones(ok.sum(),bool)))")
open('/tmp/work/mcmod.py','w').write(s)
import importlib, mcmod, mcrun
importlib.reload(mcmod); importlib.reload(mcrun)
from mcrun import *
import time
t0=time.time()
tal=new_tally(); rng=np.random.default_rng(1)
tal=chunk(20000, rng, tal)
print("time", time.time()-t0, "ncoll", tal['ncoll'], "nh", tal['nh'])
print("arr_tot", tal['arr_tot']/tal['nh'], "Omega*...")
