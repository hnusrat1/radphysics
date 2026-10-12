
s=open('/tmp/work/full.py').read()
s=s.replace("MKERN=0.10","MKERN=0.08")
open('/tmp/work/full.py','w').write(s)
import subprocess, pickle, json, numpy as np
r=subprocess.run(['python3','/tmp/work/full.py'],capture_output=True,text=True,cwd='/tmp/work')
print(r.stderr[-500:])
exec(open('/tmp/work/ana2.py').read().split("# ---------- L1")[0]) if False else None
out=pickle.load(open('/tmp/work/out.pkl','rb'))
# add L1, L2 items computed previously (recompute quickly)
import sys; sys.path.insert(0,'/tmp/work')
d_=pickle.load(open('/tmp/work/tot.pkl','rb')); tot=d_['tot']; Ntot=d_['N']; zpts=d_['zpts']; im=zpts.index(5.0)
import physics as P
SC=P.SF/Ntot
prim_w=0.5*np.exp(-np.array([P.mutot(E) for E in P.lines])*5.0)
L={}
prim_tot=prim_w.sum(); scat=SC*tot['A_tot'][im]
L['L1_primary_fluence_per_phi0']=float(prim_tot)
L['L1_total_fluence_per_phi0']=float(prim_tot+scat)
L['L1_scattered_fraction_fluence']=float(scat/(prim_tot+scat))
Eprim=np.sum(prim_w*P.lines); Escat=SC*tot['A_Ef'][im]
L['L1_scattered_fraction_energy_fluence']=float(Escat/(Eprim+Escat))
L['L1_mean_photon_energy_MeV']=float((Eprim+Escat)/(prim_tot+scat))
sp=np.zeros(28); sp[23]=prim_w[0]; sp[26]=prim_w[1]
L['L1_photon_fluence_spectrum_primary']=sp.tolist()
L['L1_photon_fluence_spectrum_scattered']=(SC*tot['A_E'][im]).tolist()
ap=np.zeros(10); ap[9]=1.0; L['L1_angular_primary']=ap.tolist()
ac=SC*tot['A_c'][im]; L['L1_angular_scattered']=(ac/ac.sum()).tolist()
ints_p=np.zeros(4); tr_p=np.zeros(4)
for w_,E in zip(prim_w,P.lines):
    ints_p+=w_*np.array([P.muC(E),P.mupe(E),P.mupair(E),P.muR(E)])
    tr_p+=w_*np.array([P.muC(E)*P.Tmean(E),P.mupe(E)*E,P.mupair(E)*(E-2*P.me),0.0])
ints_s=SC*tot['A_int'][im]; tr_s=SC*tot['A_tr'][im]
names=['compton','photoelectric','pair','rayleigh']
dct=lambda v: {n:float(x) for n,x in zip(names,np.asarray(v)/np.sum(v))}
L['L2_interaction_fractions']=dct(ints_p+ints_s)
L['L2_energy_transfer_fractions']=dct(tr_p+tr_s)
L['L2_interaction_fractions_primary']=dct(ints_p)
L['L2_interaction_fractions_scattered']=dct(ints_s)
L['L2_scattered_share_of_interactions']=float(ints_s.sum()/(ints_s.sum()+ints_p.sum()))
for k in ['L3_initial_spectrum','L3_initial_spectrum_parent_primary','L3_mean_initial_energy_MeV','L3_mean_initial_energy_parent_primary_MeV','L3_initial_direction',
          'L4_electron_fluence_per_phi0','L4_electron_fluence_spectrum','L4_delta_fraction','L4_mean_electron_energy_MeV','L4_angular','L4_radiative_to_collision',
          'L5_D_over_phi0_Gy_cm2','L5_K_over_phi0_Gy_cm2','L5_Kcol_over_phi0_Gy_cm2','L5_beta','L5_g','L5_dose_fraction_parent_scattered','L5_s_w_air','L5_PDD5_percent']:
    L[k]=out[k]
L={k:(float(v) if isinstance(v,(np.floating,)) else v) for k,v in L.items()}
for k,v in L.items():
    if isinstance(v,float): print(k, f"{v:.6g}")
print("dmax", out['_dmax'], "Kcol_z", {k:round(v,5) for k,v in out['_Kcol_z'].items() if k in (0.05,0.5,1.0,5.0)})
pickle.dump(L,open('/tmp/work/final.pkl','wb'))
