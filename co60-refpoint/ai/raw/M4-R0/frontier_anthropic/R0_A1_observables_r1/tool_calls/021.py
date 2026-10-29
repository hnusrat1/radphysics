
final = r'''
import numpy as np, sys, json
sys.path.insert(0,'/tmp/work')
from phys import re, mc2, ne, Lines, mu_c, mu_pe, mu_pair, mu_coh, mu_tot, ETR, OMEGA
from elec import S_col_BS, S_rad_w, S_res, ne_air, beta2
from anal import PE, EE, CE, build
from anal2 import slowdown
GY=1.602176634e-10
def compute(N, seed, Nel=200000):
    B=build(N,seed,Nel=Nel)
    R=B['R']; E=B['E']; W=B['W']; u=B['u']; Pw=B['Pw']
    out={}
    prim=Pw.sum(); scat=W.sum()
    out['L1_primary_fluence_per_phi0']=prim
    out['L1_total_fluence_per_phi0']=prim+scat
    out['L1_scattered_fraction_fluence']=scat/(prim+scat)
    ep=np.sum(Pw*Lines); es=np.sum(W*E)
    out['L1_scattered_fraction_energy_fluence']=es/(ep+es)
    out['L1_mean_photon_energy_MeV']=(ep+es)/(prim+scat)
    sp=np.zeros(28); idx=np.searchsorted(PE,Lines,side='right')-1
    for i,p in zip(idx,Pw): sp[i]+=p
    out['L1_photon_fluence_spectrum_primary']=sp.tolist()
    ss,_=np.histogram(E,bins=PE,weights=W); out['L1_photon_fluence_spectrum_scattered']=ss.tolist()
    out['L1_angular_primary']=[0.0]*9+[1.0]
    ca,_=np.histogram(u[:,2],bins=CE,weights=W); out['L1_angular_scattered']=(ca/ca.sum()).tolist()
    # ---- Layer 2
    def counts(Ev,w):
        return {'c':np.sum(w*mu_c(Ev)),'p':np.sum(w*mu_pe(Ev)),'pair':np.sum(w*mu_pair(Ev)),'r':np.sum(w*mu_coh(Ev))}
    def etr(Ev,w):
        return {'c':np.sum(w*ETR(Ev)),'p':np.sum(w*mu_pe(Ev)*Ev),'pair':np.sum(w*mu_pair(Ev)*np.clip(Ev-1.022,0,None)),'r':0.0}
    kn={'c':'compton','p':'photoelectric','pair':'pair','r':'rayleigh'}
    def nrm(d):
        s=sum(d.values()); return {kn[k]:float(d[k]/s) for k in d}
    Cp=counts(Lines,Pw); Cs=counts(E,W)
    Ct={k:Cp[k]+Cs[k] for k in Cp}
    out['L2_interaction_fractions']=nrm(Ct)
    Ep_=etr(Lines,Pw); Es_=etr(E,W); Et={k:Ep_[k]+Es_[k] for k in Ep_}
    out['L2_energy_transfer_fractions']=nrm(Et)
    out['L2_interaction_fractions_primary']=nrm(Cp)
    out['L2_interaction_fractions_scattered']=nrm(Cs)
    out['L2_scattered_share_of_interactions']=float(sum(Cs.values())/sum(Ct.values()))
    # ---- Layer 3 / 4 from electron records
    T=R['T']; D=R['D']; w=R['w']; par=R['parent']; proc=R['proc']; ch=R['charge']
    ok=w>0
    def hist_norm(vals,wts,bins):
        h,_=np.histogram(vals,bins=bins,weights=wts); return h/h.sum()
    inb=(T>=0.01)&(T<=1.4)
    out['L3_initial_spectrum']=hist_norm(T[inb&ok],w[inb&ok],EE).tolist()
    pp=(par==1)
    out['L3_initial_spectrum_parent_primary']=hist_norm(T[inb&ok&pp],w[inb&ok&pp],EE).tolist()
    out['L3_mean_initial_energy_MeV']=float(np.sum(w*T)/np.sum(w))
    out['L3_mean_initial_energy_parent_primary_MeV']=float(np.sum((w*T)[pp])/np.sum(w[pp]))
    out['L3_initial_direction']=hist_norm(D[:,2][ok],w[ok],CE).tolist()
    # ---- Layer 4: electrons (charge -1) above 10 keV
    em=(ch<0)&(T>=0.01)
    Tg=np.geomspace(0.01,1.4,500)
    Ts=np.sort(T[em]); ws=w[em][np.argsort(T[em])]
    cw=np.cumsum(ws[::-1])[::-1]   # sum of weights with T>=Ts[i]
    # N_gamma(>T): sum of weights with T_i > T
    pos=np.searchsorted(Ts,Tg,side='right')
    Ngam=np.where(pos<len(Ts), cw[np.minimum(pos,len(Ts)-1)], 0.0)
    Psi,Psid,Qd,Nd,Stot=slowdown(Tg,Ngam,None)
    def integ(f,a,b):
        m=(Tg>=a)&(Tg<=b)
        return np.trapezoid(f[m],Tg[m]) if m.sum()>1 else 0.0
    # Bin spectrum from Psi
    spec=[]
    for a,b in zip(EE[:-1],EE[1:]):
        spec.append(integ(Psi,max(a,Tg[0]),min(b,Tg[-1])))
    spec=np.array(spec)
    out['L4_electron_fluence_spectrum']=spec.tolist()
    tot=integ(Psi,Tg[0],Tg[-1])
    out['L4_electron_fluence_per_phi0']=float(tot)
    out['L4_delta_fraction']=float(integ(Psid,Tg[0],Tg[-1])/tot)
    out['L4_mean_electron_energy_MeV']=float(np.trapezoid(Tg*Psi,Tg)/tot)
    # angular distribution of electrons (photon-born directions used as proxy)
    ang=hist_norm(D[:,2][em],w[em],CE)
    out['L4_angular']=ang.tolist()
    Scol_t=np.array([S_col_BS(x,75e-6,ne,True) for x in Tg]); Srad_t=S_rad_w(Tg)
    out['L4_radiative_to_collision']=float(np.trapezoid(Psi*Srad_t,Tg)/np.trapezoid(Psi*Scol_t,Tg))
    # ---- Layer 5
    Ktr=np.sum(w*T)     # all electrons/positrons initial kinetic energy (MeV/g per Phi0)
    Ktr=float(Ktr)
    # Ktr from electron records counts all electrons; check vs ETR-based K
    K_etr=float(np.sum(Pw*ETR(Lines))+np.sum(W*ETR(E))+np.sum(Pw*mu_pe(Lines)*Lines)+np.sum(W*mu_pe(E)*E)
                +np.sum(Pw*mu_pair(Lines)*np.clip(Lines-1.022,0,None))+np.sum(W*mu_pair(E)*np.clip(E-1.022,0,None)))
    K=K_etr
    Krad=float(np.trapezoid(Psi*Srad_t,Tg))
    Kcol=K-Krad
    g=Krad/K
    Kscat=float(np.sum(W*ETR(E))+np.sum(W*mu_pe(E)*E)+np.sum(W*mu_pair(E)*np.clip(E-1.022,0,None)))
    out['L5_K_over_phi0_Gy_cm2']=K*GY
    out['L5_Kcol_over_phi0_Gy_cm2']=Kcol*GY
    out['L5_D_over_phi0_Gy_cm2']=Kcol*GY
    out['L5_beta']=1.0
    out['L5_g']=g
    out['L5_dose_fraction_parent_scattered']=Kscat/K
    # Spencer-Attix water-to-air, Delta=10 keV
    Dl=0.01
    m=Tg>=Dl
    Sw=np.array([S_res(x,Dl,'w') for x in Tg]); Sa=np.array([S_res(x,Dl,'a') for x in Tg])
    psiD=np.interp(Dl,Tg,Psi)
    numer=np.trapezoid((Psi*Sw)[m],Tg[m])+Dl*psiD*S_res(Dl,Dl,'w')
    denom=np.trapezoid((Psi*Sa)[m],Tg[m])+Dl*psiD*S_res(Dl,Dl,'a')
    out['L5_s_w_air']=float(numer/denom)
    out['_diag']=dict(Ktr_records=Ktr, K_etr=K, Krad=Krad, Ngam_total=float(Ngam[0]), Psi_at_10keV=float(Psi[0]), Psid_frac=float(out['L4_delta_fraction']))
    return out, B
'''
open('/tmp/work/final.py','w').write(final)
import sys; sys.path.insert(0,'/tmp/work')
import final, time, json
t=time.time()
out,B=final.compute(100000,11,Nel=20000)
print(time.time()-t)
for k,v in out.items():
    if isinstance(v,list): print(k, np.round(v,5))
    else: print(k, v)
