
code_final = r'''
import sys, pickle
sys.path.insert(0,'/tmp/work')
import numpy as np
import phys, deposit, electrons, sources
from phys import mu_all, Yrad, Stot, Scol, Srad
S_phi0=0.009975072683099257*110.0**2; V=np.pi*1.0**2*0.2; Ntot=4000000
D=pickle.load(open('/tmp/work/photon_tallies.pkl','rb'))
tot=D['tot']; seg=D['seg']
L4=pickle.load(open('/tmp/work/L4.pkl','rb'))
EB=np.array([0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4])
CB=np.array([-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1.0])
EBe=electrons.EBe
Phi_p=float(np.exp(-mu_all(np.array([2.0]))[4][0]*10))
mu2=[float(x[0]) for x in mu_all(np.array([2.0]))]
coh2,inc2,pe2,pr2,mt2=mu2
El=seg['E']; ll=seg['l']; dl=seg['d']
# ---- scattered spectrum & angular (recomputed from segments)
ebin=np.digitize(El,EB)-1
spec_s=np.zeros(28); ok=(ebin>=0)&(ebin<28); np.add.at(spec_s,ebin[ok],ll[ok])
spec_s*=S_phi0/V/Ntot
Fs=ll.sum()*S_phi0/V/Ntot
Fs_check=float(tot['Fl_sc'])
EFs=(ll*El).sum()*S_phi0/V/Ntot
ang=np.zeros(10); cb=np.clip(np.digitize(dl[:,2],CB)-1,0,9); np.add.at(ang,cb,ll)
ang_s=ang/ang.sum()
print("Fs MC (recomputed)",Fs,"vs earlier",Fs_check,"spec sum",spec_s.sum(),"over1.4",ll[El>1.4].sum()*S_phi0/V/Ntot)
Ftot=Phi_p+Fs
EFp=Phi_p*2.0
L1=dict(L1_primary_fluence_per_phi0=Phi_p, L1_total_fluence_per_phi0=Ftot, L1_scattered_fraction_fluence=Fs/Ftot,
        L1_scattered_fraction_energy_fluence=EFs/(EFs+EFp), L1_mean_photon_energy_MeV=(EFs+EFp)/Ftot,
        L1_photon_fluence_spectrum_primary=[0.0]*28, L1_photon_fluence_spectrum_scattered=list(spec_s),
        L1_angular_primary=[0.0]*9+[1.0], L1_angular_scattered=list(ang_s))
print(L1)
# ---- Layer 2
prim_counts=np.array([coh2,inc2,pe2,pr2])*Phi_p
scat_counts=np.array(tot['mu_sc'])
allc=prim_counts+scat_counts
dep2=deposit.transfer_coeffs(np.array([2.0]))
prim_tr=np.array([0.0, float(dep2['tr_c'][0])*Phi_p, float(dep2['tr_pe'][0])*Phi_p, float(dep2['tr_pa'][0])*Phi_p])
scat_tr=np.array([0.0, tot['tr_sc'][0], tot['tr_sc'][1], tot['tr_sc'][2]])
ctr=prim_tr+scat_tr
keys=['rayleigh','compton','photoelectric','pair']
def fd(v): v=np.asarray(v,float); return {k:float(x) for k,x in zip(keys,v/v.sum())}
L2=dict(L2_interaction_fractions=fd(allc), L2_energy_transfer_fractions=fd(ctr),
        L2_interaction_fractions_primary=fd(prim_counts), L2_interaction_fractions_scattered=fd(scat_counts),
        L2_scattered_share_of_interactions=float(scat_counts.sum()/allc.sum()))
print(L2)
'''
open('/tmp/work/final1.py','w').write(code_final)
import subprocess
r=subprocess.run(['python3','/tmp/work/final1.py'],capture_output=True,text=True,cwd='/tmp/work')
print(r.stdout[-4000:], r.stderr[-3000:])
