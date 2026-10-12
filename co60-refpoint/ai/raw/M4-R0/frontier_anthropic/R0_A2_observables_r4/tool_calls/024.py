
ana3 = r'''
import sys; sys.path.insert(0,'/tmp/work')
import numpy as np
import ana2 as A
from ana2 import *
# ---------- L4 angular
def delta_hist(T,cp,wL,rng):
    ok=(T>=2*DELTA_)
    h=np.zeros(10)
    if not ok.any(): return h
    T0=T[ok]; wl=wL[ok]; cpp=cp[ok]
    beta2=1-1/(1+T0/mc2)**2
    Nrate=2*np.pi*re**2*mc2*ne/beta2*(1/DELTA_-2/T0)
    u_=rng.random(T0.size)
    Td=1/(1/DELTA_-u_*(1/DELTA_-2/T0))
    Ld_=np.maximum(R_T(Td)-R_T(DELTA_),0)
    wd=wl*Nrate*Ld_
    cr=np.sqrt(np.clip(Td*(T0+2*mc2)/(T0*(Td+2*mc2)),0,1))
    phi2=rng.uniform(0,2*np.pi,T0.size)
    cbeam=cpp*cr+np.sqrt(np.maximum(0,1-cpp*cpp))*np.sqrt(np.maximum(0,1-cr*cr))*np.cos(phi2)
    cbd=np.clip(np.searchsorted(EDGES_CT,cbeam,side='right')-1,0,9)
    return np.bincount(cbd,weights=wd,minlength=10)
# primary sampled electrons already: Tpc, cpc, Wp
Lp_p=np.maximum(R_T(np.maximum(Tpc,1e-4))-R_T(DELTA_),0)*(Tpc>=DELTA_)
hA_p=np.bincount(np.clip(np.searchsorted(EDGES_CT,cpc,side='right')-1,0,9),weights=Wp*Lp_p,minlength=10)
hB_p=delta_hist(Tpc,cpc,Wp,rng)
hA=hA_p+SM['L4_hC']*F
hB=hB_p+SM['D4_hC']*F
dl=L4['L4_delta_fraction']
ang=(1-dl)*hA/hA.sum()+dl*hB/hB.sum()
L4['L4_angular']=ang/ang.sum()
print('hA',hA/hA.sum()); print('hB',hB/hB.sum())
# ---------- Spencer-Attix water/air
ZA_w=0.55508; ZA_a=0.49919; Iw=75e-6; Ia=85.7e-6
Sw=Scol_f(Tg)
gam=1+Tg/mc2; b2=1-1/gam**2
Xw=Sw*b2/(2*np.pi*re**2*mc2*ne)
ratio=(ZA_a/ZA_w)*(Xw+2*np.log(Iw/Ia))/Xw     # S_air/S_water (approx Bethe ratio)
Lw=LDelta          # restricted collision stopping (water), MeV/cm
La=Lw*ratio
Phi_=Phi
mD=Tg>=DELTA_
PhiD=np.interp(DELTA_,Tg,Phi)
Sw_D=Scol_f(DELTA_); Sa_D=Sw_D*np.interp(DELTA_,Tg,ratio)
num=integ(Phi*Lw)+DELTA_*PhiD*Sw_D
den=integ(Phi*La)+DELTA_*PhiD*Sa_D
s_wa=num/den
print('s_wa',s_wa, 'ratio at 0.1,0.3,1 MeV', np.interp([0.1,0.3,1.0],Tg,ratio))
L5={}
L5['L5_s_w_air']=s_wa
# ---------- L5 kerma / dose
prim_Kcol=sum(phi[i]*(mu_inc(LN[i])*LN[i]*kcf_f(LN[i])+mu_pe(LN[i])*LN[i]*(1-Y_T(LN[i]))) for i in range(2))
prim_K=sum(phi[i]*(mu_inc(LN[i])*LN[i]*ftr_f(LN[i])+mu_pe(LN[i])*LN[i]) for i in range(2))
sc_Kcol=SM['kcol'][4]*F; sc_K=SM['kgross'][4]*F
Kcol=prim_Kcol+sc_Kcol; K=prim_K+sc_K
print('Kcol',Kcol,'K',K,'prim',prim_Kcol,'scat',sc_Kcol)
L5['L5_Kcol_over_phi0_Gy_cm2']=Kcol*1.602176634e-10*1e0  # MeV/g -> Gy with unit Phi0 cm^2
L5['L5_K_over_phi0_Gy_cm2']=K*1.602176634e-10
L5['L5_g']=1-Kcol/K
L5['L5_dose_fraction_parent_scattered']=sc_Kcol/Kcol
# depth dependence of Kcol: primary + scattered per depth
depth_list=[0.25,0.5,0.75,1.0,5.0]
Kd=[]
for j,z in enumerate(depth_list):
    pz=sum(0.5*np.exp(-mu_tot(LN[i])*z)*(mu_inc(LN[i])*LN[i]*kcf_f(LN[i])+mu_pe(LN[i])*LN[i]*(1-Y_T(LN[i]))) for i in range(2))
    Kd.append(pz+SM['kcol'][j]*F)
Kd=np.array(Kd)
print('Kcol vs depth',Kd, Kd/Kd[-1])
# beta estimate: D/Kcol = 1 + m1*|dlnK/dz| ; m1 = mean forward displacement of deposits
slope=np.log(Kd[3]/Kd[4])/(5.0-1.0)   # between 1 and 5 cm (per cm, negative)
slope5 = np.gradient(Kd,np.array(depth_list))[-1]/Kd[-1]
print('dlnK/dz at 5cm (1-5 cm avg)',slope,' local',slope5)
# mean forward displacement: energy-weighted cos(theta) * projected range (approx 0.4*CSDA range)
# electrons: primary sampled, weights = Wp * T*(1-Y) etc
Rp=R_T(np.maximum(Tpc,1e-4))
cavg=np.sum(Wp*cpc*Rp*(Tpc-0*Tpc))/np.sum(Wp*Rp)
m1=0.4*cavg
print('mean cos (primary electrons, track weighted)',cavg,' m1 (cm)',m1)
beta=1+m1*abs(slope)
L5['L5_D_over_phi0_Gy_cm2']=beta*Kcol*1.602176634e-10
L5['L5_beta']=beta
L5['L5_PDD5_percent']=100*Kd[4]/np.max(Kd[:4])
print('PDD5 from grid', 100*Kd[4]/np.max(Kd[:4]), ' dmax index', np.argmax(Kd[:4]))
'''
open('/tmp/work/ana3.py','w').write(ana3)
import sys,time; sys.path.insert(0,'/tmp/work')
import ana3
print({k:v for k,v in ana3.L1.items() if not hasattr(v,'__len__')})
