
import sys, importlib, pickle
sys.path.insert(0,'/tmp/work')
import numpy as np
import phys, deposit
D=pickle.load(open('/tmp/work/photon_tallies.pkl','rb'))
tot=D['tot']; seg=D['seg']; nproc=D['nproc']
L4=pickle.load(open('/tmp/work/L4.pkl','rb'))
from phys import mu_all, Yrad, Stot
Phi_p=np.exp(-mu_all(np.array([2.0]))[4][0]*10)
mu2=mu_all(np.array([2.0]))
coh2,inc2,pe2,pr2,mt2=[float(x[0]) for x in mu2]
# ---------- Layer 1
EB=np.array([0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4])
spec_p=np.zeros(28)
spec_s=tot['spec']
Fp=Phi_p; Fs=float(tot['Fl_sc'])
Ftot=Fp+Fs
EFs=float(tot['EFl'])
EFp=Fp*2.0
L1=dict(
 prim=Fp, total=Ftot, scat_frac=Fs/Ftot, scat_E_frac=EFs/(EFs+EFp), meanE=(EFs+EFp)/Ftot)
print(L1)
ang_s=tot['ang']/tot['ang'].sum()
print("spec scat sum (bins)", spec_s.sum(), "spec>1.4",tot['spec_over'], "check Fs", spec_s.sum()+tot['spec_over'])
print("angular scat", np.round(ang_s,4))
# ---------- Layer 2
prim_counts=np.array([coh2,inc2,pe2,pr2])*Fp
scat_counts=np.array(tot['mu_sc'])
allc=prim_counts+scat_counts
frac=allc/allc.sum()
# energy transfer
dep2=deposit.transfer_coeffs(np.array([2.0]))
prim_tr=np.array([0.0, float(dep2['tr_c'][0]), float(dep2['tr_pe'][0]), float(dep2['tr_pa'][0])])*Fp
# note: tr arrays ordering: compton, pe, pair
prim_tr=np.array([0.0, float(dep2['tr_c'][0])*Fp, float(dep2['tr_pe'][0])*Fp, float(dep2['tr_pa'][0])*Fp])
scat_tr=np.array([0.0, tot['tr_sc'][0], tot['tr_sc'][1], tot['tr_sc'][2]])
# order: [rayleigh, compton, pe, pair]
ct=prim_tr+scat_tr
print("prim counts (coh,inc,pe,pair)",prim_counts)
print("scat counts (coh,inc,pe,pair)",scat_counts)
print("interaction fractions  [coh,comp,pe,pair]", np.round(frac,6))
print("energy transfer (coh=0): ", ct/ct.sum())
print("primary interaction fractions", prim_counts/prim_counts.sum())
print("scattered interaction fractions", scat_counts/scat_counts.sum())
print("scattered share", scat_counts.sum()/allc.sum())
