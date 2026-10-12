
main = r'''
import numpy as np, sys
sys.path.insert(0,'/tmp/work')
import lib
from lib import *
from ana import *
trapz=np.trapezoid
Hsum,Ntot,dep=load_H()
idx={z:i for i,z in enumerate(dep)}
i5=idx[5.0]
W5=arrivals(Hsum[i5],Ntot)            # (280,10) scattered fluence per Phi0 per cell
EL=np.array([1.1732,1.3325])
Wp=0.5*np.exp(-mutot(EL)*5.0)          # primary per line /Phi0 at 5 cm (105 cm reference)
Phi_p=Wp.sum(); Phi_s=W5.sum()
out={}
# ---------- Layer 1
Ecoarse_edges=np.arange(29)*0.05
spec_p=np.zeros(28); spec_p[23]=Wp[0]; spec_p[26]=Wp[1]
spec_s=W5.sum(axis=1).reshape(28,10).sum(axis=1)
ang_s=W5.sum(axis=0)
ang_p=np.zeros(10); ang_p[9]=Phi_p
Phi_tot=Phi_p+Phi_s
EF_p=np.sum(Wp*EL); EF_s=np.sum(Ecen*W5.sum(axis=1)); 
out['L1_primary_fluence_per_phi0']=Phi_p
out['L1_total_fluence_per_phi0']=Phi_tot
out['L1_scattered_fraction_fluence']=Phi_s/Phi_tot
out['L1_scattered_fraction_energy_fluence']=EF_s/(EF_s+EF_p)
out['L1_mean_photon_energy_MeV']=(EF_s+EF_p)/Phi_tot
out['L1_photon_fluence_spectrum_primary']=spec_p.tolist()
out['L1_photon_fluence_spectrum_scattered']=spec_s.tolist()
out['L1_angular_primary']=(ang_p/Phi_p).tolist()
out['L1_angular_scattered']=(ang_s/Phi_s).tolist()
# ---------- Layer 2 : interaction rates per Phi0
def tmeanC(E):
    E=np.atleast_1d(E).astype(float)
    res=np.zeros(len(E))
    for n_,e in enumerate(E):
        mug,w=KNw(e); Ep=e/(1+e/mec2*(1-mug)); res[n_]=np.sum(w*(e-Ep))
    return res
rates={'compton':0.0,'photoelectric':0.0,'pair':0.0,'rayleigh':0.0}
ktr={'compton':0.0,'photoelectric':0.0,'pair':0.0,'rayleigh':0.0}
rprim=dict(rates); kprim=dict(ktr); rsc=dict(rates); ksc=dict(ktr)
for E,W in zip(EL,Wp):
    e=np.array([E])
    r_c=W*mu_inc(e)[0]; r_pe=W*mu_pe(e)[0]; r_p=W*mu_pair(e)[0]; r_r=W*mu_coh(e)[0]
    tc=tmeanC(E)[0]
    rprim['compton']+=r_c; rprim['photoelectric']+=r_pe; rprim['pair']+=r_p; rprim['rayleigh']+=r_r
    kprim['compton']+=r_c*tc; kprim['photoelectric']+=r_pe*E; kprim['pair']+=r_p*(E-1.022)
ET=np.unique(Ecen)
tcs=tmeanC(Ecen)
for i in range(nE):
    E=Ecen[i]; w=W5[i].sum()
    if w<=0: continue
    e=np.array([E])
    r_c=w*mu_inc(e)[0]; r_pe=w*mu_pe(e)[0]; r_p=w*mu_pair(e)[0]; r_r=w*mu_coh(e)[0]
    rsc['compton']+=r_c; rsc['photoelectric']+=r_pe; rsc['pair']+=r_p; rsc['rayleigh']+=r_r
    ksc['compton']+=r_c*tcs[i]; ksc['photoelectric']+=r_pe*E; ksc['pair']+=r_p*max(E-1.022,0)
def norm(d):
    s=sum(d.values()); return {k:v/s for k,v in d.items()}
rtot={k:rprim[k]+rsc[k] for k in rprim}; ktot={k:kprim[k]+ksc[k] for k in kprim}
out['L2_interaction_fractions']=norm(rtot)
out['L2_energy_transfer_fractions']=norm(ktot)
out['L2_interaction_fractions_primary']=norm(rprim)
out['L2_interaction_fractions_scattered']=norm(rsc)
out['L2_scattered_share_of_interactions']=sum(rsc.values())/sum(rtot.values())
Kall=sum(ktot.values()); Kscat=sum(ksc.values())
# ---------- Layer 3/4 electron sources
Wcells=W5.copy()
src,Tg,dT=electron_sources(Wcells, list(zip(EL,Wp)))
Qp=src['prim'][0]; Qs=src['scat'][0]
Qtot=Qp+Qs
Cp=src['prim'][1]; Cs=src['scat'][1]
Ctot=Cp+Cs
# electron bins
EDG=np.array([0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.4])
kedg=np.round(EDG/dT).astype(int)
def rebin(h):
    return np.array([h[kedg[i]:kedg[i+1]].sum() for i in range(25)])
Tc=Tg  # centres
def mean_T(h): return np.sum(h*Tc)/np.sum(h)
out['L3_initial_spectrum']=(rebin(Qtot)/rebin(Qtot).sum()).tolist()
out['L3_initial_spectrum_parent_primary']=(rebin(Qp)/rebin(Qp).sum()).tolist()
out['L3_mean_initial_energy_MeV']=mean_T(Qtot)
out['L3_mean_initial_energy_parent_primary_MeV']=mean_T(Qp)
out['L3_initial_direction']=(Ctot/Ctot.sum()).tolist()
# ---------- Layer 4 : slowing-down with knock-on electrons
Qdens=Qtot/dT                       # per MeV, per Phi0
Tk=Tg
Stk=Stot_w(Tk); Sck=Scol_w(Tk); Srk=Srad_w(Tk)
kmin=5
def sigM(Tp,eps):
    tau=Tp/mec2; b2=tau*(tau+2)/(tau+1)**2
    br=1/eps**2+1/(Tp-eps)**2-(2*tau+1)/((tau+1)**2*eps*(Tp-eps))+(tau/(tau+1))**2/Tp**2
    return 2*np.pi*re**2*mec2/b2*br
Phid=np.zeros(700); Rdens=np.zeros(700)
Ncum=0.0; NRcum=0.0; NQcum=0.0
for k in range(699,kmin-1,-1):
    # knock-on source at T_k from parents with T_j >= 2 T_k (already computed, j>k)
    js=np.arange(k+1,700)
    js=js[Tk[js]>=2*Tk[k]]
    if len(js): Rdens[k]=ne*np.sum(Phid[js]*sigM(Tk[js],Tk[k])*dT)
    NRcum+=Rdens[k]*dT; NQcum+=Qdens[k]*dT
    Ncum=NRcum+NQcum
    Phid[k]=Ncum/Stk[k]
    PhiR_k=NRcum/Stk[k]
# separate delta part
PhiR=np.zeros(700); NRc=0.0; NQc=0.0
for k in range(699,kmin-1,-1):
    pass
# recompute delta-split cleanly
NQc_arr=np.zeros(701); NRc_arr=np.zeros(701)
for k in range(699,kmin-1,-1):
    NQc_arr[k]=NQc_arr[k+1]+Qdens[k]*dT
    NRc_arr[k]=NRc_arr[k+1]+Rdens[k]*dT
PhiTot=(NQc_arr[:700]+NRc_arr[:700])/Stk
PhiD=NRc_arr[:700]/Stk
sel=np.arange(kmin,700)
fl=PhiTot[sel]*dT
L4tot=fl.sum()
out['L4_electron_fluence_per_phi0']=L4tot
# spectrum bins
Phi_bins=np.array([PhiTot[kedg[i]:kedg[i+1]].sum()*dT for i in range(25)])
out['L4_electron_fluence_spectrum']=Phi_bins.tolist()
out['L4_delta_fraction']=PhiD[sel].sum()*dT/L4tot
out['L4_mean_electron_energy_MeV']=np.sum(fl*Tk[sel])/L4tot
out['L4_radiative_to_collision']=np.sum(PhiTot[sel]*Srk[sel]*dT)/np.sum(PhiTot[sel]*Sck[sel]*dT)
out['L4_angular']=(Ctot/Ctot.sum()).tolist()
# ---------- Layer 5
Yk=Yield(Tk)
g=np.sum(Qtot*Tk*Yk)/np.sum(Qtot*Tk)
Kq=np.sum(Qtot*Tk)
out['L5_g']=g
Kall_s=Kall; Kcol=Kall*(1-g)
F=1.602e-10
out['L5_K_over_phi0_Gy_cm2']=Kall*F
out['L5_Kcol_over_phi0_Gy_cm2']=Kcol*F
out['L5_D_over_phi0_Gy_cm2']=Kcol*F
out['L5_beta']=1.0
out['L5_dose_fraction_parent_scattered']=Kscat/Kall
# checks
chk_ratio=Kq/Kall
CPE_check=np.sum(PhiTot[sel]*Sck[sel]*dT)*F/(Kcol*F)
# ---------- Spencer-Attix water->air
Dl=0.01
ZAw,ZAa=0.55508,0.49919; Iw,Ia=75e-6,85.7e-6
def Fminus(tau):
    b2=tau*(tau+2)/(tau+1)**2
    return 1-b2+(tau*tau/8-(2*tau+1)*np.log(2))/(tau+1)**2
def bethe_br(T,I,delta):
    tau=T/mec2; b2=tau*(tau+2)/(tau+1)**2
    return np.log(tau*tau*(tau+2)/(2*(I/mec2)**2))+Fminus(tau)-delta
def delta_w(T): return np.interp(np.log(T),np.log(ST[:,0]),ST[:,6])  # density effect water (ESTAR)
fA=ZAa/ZAw*bethe_br(Tk,Ia,0.0)/bethe_br(Tk,Iw,delta_w(Tk))
# restricted water
ee_grid=lambda T: np.linspace(Dl,max(T/2,Dl*1.0001),400)
LD=np.zeros(700)
for k in range(kmin,700):
    T=Tk[k]
    if T/2>Dl:
        eg=np.linspace(Dl,T/2,400)
        LD[k]=Sck[k]-ne*trapz(eg*sigM(T,eg),eg)
    else:
        LD[k]=Sck[k]
LDa=LD*fA
PhiD_delta=PhiTot[kmin]  # placeholder
TE_w=Dl*np.interp(Dl,Tk,PhiTot)*Scol_w(np.array([Dl]))[0]
TE_a=Dl*np.interp(Dl,Tk,PhiTot)*Scol_w(np.array([Dl]))[0]*fA[kmin]
num=np.sum(PhiTot[sel]*LD[sel]*dT)+TE_w
den=np.sum(PhiTot[sel]*LDa[sel]*dT)+TE_a
out['L5_s_w_air']=num/den
out['_s_noTE']=np.sum(PhiTot[sel]*LD[sel]*dT)/np.sum(PhiTot[sel]*LDa[sel]*dT)
out['_CPE_check_sumPhiScol_over_Kcol']=CPE_check
out['_K_from_Q_over_K']=chk_ratio
out['_Phi_p']=Phi_p; out['_Phi_s']=Phi_s
# ---------- depth dose profile (K_col) for PDD
prof={}
for z in DEPTHS:
    jz=idx[z]
    Wz=arrivals(Hsum[jz],Ntot)
    Kz=0.0
    for E,W in zip(EL,0.5*np.exp(-mutot(EL)*z)*(105/(100+z))**2):
        Kz+=W*mu_inc(np.array([E]))[0]*tmeanC(E)[0]+W*mu_pe(np.array([E]))[0]*E+W*mu_pair(np.array([E]))[0]*max(E-1.022,0)
    for i in range(nE):
        w=Wz[i].sum()
        if w<=0: continue
        E=Ecen[i]; e=np.array([E])
        Kz+=w*(mu_inc(e)[0]*tcs[i]+mu_pe(e)[0]*E+mu_pair(e)[0]*max(E-1.022,0))
    prof[z]=Kz*(1-g)
out['_profile_Kcol']={str(k):v for k,v in prof.items()}
dmax=max(prof,key=prof.get)
out['_dmax']=dmax
out['L5_PDD5_percent']=100*prof[5.0]/prof[dmax]
'''
open('/tmp/work/main.py','w').write(main)
print(subprocess.run([sys.executable,'-c','print(1)'],capture_output=True,text=True).stdout)
import glob; print(glob.glob('/tmp/work/H_*.npz'))
