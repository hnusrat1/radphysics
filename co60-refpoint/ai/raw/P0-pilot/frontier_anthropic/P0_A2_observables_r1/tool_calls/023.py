
final2 = r'''
import sys, pickle
sys.path.insert(0,'/tmp/work')
import numpy as np
import phys, deposit, electrons, sources
from phys import mu_all, Yrad, Stot, Scol, Srad
S_phi0=0.009975072683099257*110.0**2; V=np.pi*1.0**2*0.2; Ntot=4000000
D=pickle.load(open('/tmp/work/photon_tallies.pkl','rb'))
tot=D['tot']; seg=D['seg']
ER=pickle.load(open('/tmp/work/L4.pkl','rb'))
EBe=electrons.EBe
CB=np.array([-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1.0])
Phi_p=float(np.exp(-mu_all(np.array([2.0]))[4][0]*10))
rng=np.random.default_rng(5)
cat=sources.build_sources(seg,Ntot,rng,Phi_p)
T=cat['T']; w=cat['w']; uz=cat['u'][:,2]; ps=cat['ps']; par=cat['par']; q=cat['q']
def binspec(mask):
    tt=T[mask]; ww=w[mask]
    b=np.digitize(tt,EBe)-1
    good=(b>=0)&(b<25)&(tt>=0.01)
    s=np.zeros(25); np.add.at(s,b[good],ww[good])
    return s/s.sum()
L3_spec=binspec(np.ones(len(T),bool))
L3_spec_p=binspec(par==0)
L3_mean=float((w*T).sum()/w.sum())
L3_mean_p=float((w*T)[par==0].sum()/w[par==0].sum())
cb=np.clip(np.digitize(uz,CB)-1,0,9)
dirw=np.zeros(10); np.add.at(dirw,cb,w); L3_dir=dirw/dirw.sum()
Y=Yrad(np.maximum(T,0.01))
K_all=1.602e-10*(w*T).sum()
Kcol_all=1.602e-10*(w*T*(1-Y)).sum()
g=1-Kcol_all/K_all
dosefrac_scat=float((w*T*(1-Y))[ps].sum()/(w*T*(1-Y)).sum())
dep2=deposit.transfer_coeffs(np.array([2.0]))
ZD=np.arange(0.25,20.0001,0.25)
Kd=tot['Kd']; k10=np.argmin(np.abs(ZD-10.0))
print("K/Phi0",K_all,"Kcol/Phi0",Kcol_all,"g",g,"dose frac scat",dosefrac_scat)
print("Kcol prim: sources",1.602e-10*(w*T*(1-Y))[par==0].sum(),"analytic",1.602e-10*Phi_p*float(dep2['col'][0]))
print("Kcol scat at 10cm: tally",1.602e-10*Kd[k10],"sources",1.602e-10*(w*T*(1-Y))[ps].sum())
print("K analytic total at 10cm approx: prim tr",1.602e-10*Phi_p*float(dep2['tr'][0]), "scat tr tally", 1.602e-10*(tot['tr_sc'].sum()))
# ---- Layer 4
tb=ER['tb']; tcos=ER['tcos']; tt_=ER['tot']
L4_fl=float(tt_); L4_spec=list(map(float,tb))
L4_delfrac=float(ER['delt']/ER['tot'])
L4_meanE=float(ER['sumE']/ER['tot'])
L4_ang=list(map(float,tcos/tcos.sum()))
L4_rc=float(ER['Sr']/ER['Sc'])
# ---- Spencer-Attix s_w,air with Delta=10 keV
Dl=0.01
ZA_w=0.55508; ZA_a=0.49919
def Bethe_unres(Tk,ZA,I_eV,delta):
    tau=Tk/0.51099895; b2=1-(1/(tau+1))**2
    I=I_eV*1e-6/0.51099895
    F=1-b2+(tau*tau/8-(2*tau+1)*np.log(2))/(tau+1)**2
    return 0.153536*ZA/b2*(np.log(tau*tau*(tau+2)/(2*I*I))+F-delta)
def restricted(Tk,ZA,I_eV,delta,Dcut=Dl):
    tau=Tk/0.51099895; b2=1-(1/(tau+1))**2
    S=Bethe_unres(Tk,ZA,I_eV,delta)
    eps_c=Dcut/Tk
    if eps_c>=0.5: return S
    ee=np.linspace(eps_c,0.5,4000)
    f=1/ee**2+1/(1-ee)**2+(tau/(tau+2))**2-(2*tau+1)/((tau+1)**2)/(ee*(1-ee))
    loss=0.153536*ZA/b2*np.trapz(ee*f,ee)
    return S-loss
# validate water unrestricted Bethe vs ESTAR
for Tk in [0.1,1.0,2.0]:
    dl_water=float(phys.Scol(np.array([Tk]))[0]) if False else 0
Ecols=np.array([0.1,0.5,1.0,2.0])
import phys as P
print("Bethe water check", [round(float(Bethe_unres(t,ZA_w,75.0,0.0)),4) for t in Ecols], "ESTAR", [round(float(P.Scol(np.array([t]))[0]),4) for t in Ecols])
Ebin_mid=0.5*(EBe[:-1]+EBe[1:])
psi_bin=tb/np.diff(EBe)
Lw=np.array([restricted(t,ZA_w,75.0,0.0) for t in Ebin_mid])
La=np.array([restricted(t,ZA_a,85.7,0.0) for t in Ebin_mid])
# use ESTAR-based unrestricted water for Lw: scale correction factor
corrw=np.array([float(P.Scol(np.array([t]))[0])/Bethe_unres(t,ZA_w,75.0,0.0) for t in Ebin_mid])
Lw_est=np.array([float(P.Scol(np.array([t]))[0])-(Bethe_unres(t,ZA_w,75.0,0.0)-restricted(t,ZA_w,75.0,0.0)) for t in Ebin_mid])
dE=np.diff(EBe)
num=np.sum(psi_bin*Lw_est*dE); den=np.sum(psi_bin*La*dE)
# track-end term: Delta * psi(Delta) * S_col(Delta); psi(Delta) from first bin average
psiD=psi_bin[0]
Sw_D=float(P.Scol(np.array([Dl]))[0]); Sa_D=Bethe_unres(Dl,ZA_a,85.7,0.0)
Tend_w=Dl*psiD*Sw_D; Tend_a=Dl*psiD*Sa_D
s_wa=(num+Tend_w)/(den+Tend_a)
s_wa_noend=num/den
print("s_w,air (bins 0.01-1.4) with track-end", s_wa, "without", s_wa_noend, "Tend contribution", Tend_w/num)
print("L_w bins sample", Lw_est[:3], La[:3])
# ---- PDD
muT=mu2=float(mu_all(np.array([2.0]))[4][0])
Phi_pz=np.exp(-muT*ZD)
Kcol_z=1.602e-10*(Phi_pz*float(dep2['col'][0])+Kd)
imax=np.argmax(Kcol_z)
k5=np.argmin(np.abs(ZD-5.0))
PDD5=100*Kcol_z[k5]/Kcol_z[imax]
print("dmax (cm)",ZD[imax],"Kcol max",Kcol_z[imax],"Kcol(5)",Kcol_z[k5],"PDD5",PDD5)
print("Kcol depth profile:",[ (round(z,2),float(f'{Kcol_z[i]:.4e}')) for i,z in enumerate(ZD) if i%4==0])
out=dict(L3_initial_spectrum=list(map(float,L3_spec)), L3_initial_spectrum_parent_primary=list(map(float,L3_spec_p)),
 L3_mean_initial_energy_MeV=L3_mean, L3_mean_initial_energy_parent_primary_MeV=L3_mean_p, L3_initial_direction=list(map(float,L3_dir)),
 L4_electron_fluence_per_phi0=L4_fl, L4_electron_fluence_spectrum=L4_spec, L4_delta_fraction=L4_delfrac, L4_mean_electron_energy_MeV=L4_meanE,
 L4_angular=L4_ang, L4_radiative_to_collision=L4_rc,
 L5_K_over_phi0_Gy_cm2=float(K_all), L5_Kcol_over_phi0_Gy_cm2=float(Kcol_all), L5_D_over_phi0_Gy_cm2=float(Kcol_all),
 L5_beta=1.0, L5_g=float(g), L5_dose_fraction_parent_scattered=dosefrac_scat, L5_s_w_air=float(s_wa), L5_PDD5_percent=float(PDD5),
 dmax_cm=float(ZD[imax]))
pickle.dump(out,open('/tmp/work/final2.pkl','wb'))
print({k:(v if not isinstance(v,list) else np.round(v,5).tolist()) for k,v in out.items() if k not in ['L3_initial_spectrum','L3_initial_spectrum_parent_primary','L4_electron_fluence_spectrum']})
'''
open('/tmp/work/final2.py','w').write(final2)
import subprocess
r=subprocess.run(['python3','/tmp/work/final2.py'],capture_output=True,text=True,cwd='/tmp/work')
print(r.stdout[-6000:], r.stderr[-3000:])
