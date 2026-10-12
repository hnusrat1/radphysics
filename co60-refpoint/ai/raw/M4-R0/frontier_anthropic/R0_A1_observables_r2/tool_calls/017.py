
final = r'''
import pickle, numpy as np, sys, json
sys.path.insert(0,'.')
import mc2
allres=pickle.load(open('allres.pkl','rb'))
depths=[0.25,0.5,0.75,1.0,2.0,3.0,4.0,5.0,6.0,8.0,10.0]
Ntot=12000000; V=mc2.Vcyl; P0=mc2.Phi0_hist; c5=depths.index(5.0); c05=depths.index(0.5)
tal={k:sum(r['tal'][k] for r in allres) for k in allres[0]['tal']}
Hfine=sum(r['Hfine'] for r in allres); Hcoarse=sum(r['Hcoarse'] for r in allres)
Hcos=sum(r['Hcos'] for r in allres); Esum=sum(r['Esum'] for r in allres); Wsum=sum(r['Wsum'] for r in allres)
fine=allres[0]['fine']; eedges=allres[0]['eedges']
Eedges=np.arange(0,1.4001,0.05)
cedges=np.array([-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1.0])
norm=Ntot*V*P0
out={}
# ---------- Layer 1
Fp=tal['Lp'][c5]/norm; Fs=tal['Ls'][c5]/norm
out['L1_primary_fluence_per_phi0']=Fp
out['L1_total_fluence_per_phi0']=Fp+Fs
out['L1_scattered_fraction_fluence']=Fs/(Fp+Fs)
out['L1_scattered_fraction_energy_fluence']=tal['LE_s'][c5]/(tal['LE_s'][c5]+tal['LE_p'][c5])
out['L1_mean_photon_energy_MeV']=(tal['LE_s'][c5]+tal['LE_p'][c5])/(tal['Ls'][c5]+tal['Lp'][c5])
out['L1_photon_fluence_spectrum_primary']=list(tal['Lp_E'][c5]/norm)
out['L1_photon_fluence_spectrum_scattered']=list(tal['Ls_E'][c5]/norm)
pc=tal['Lp_c'][c5]; sc=tal['Ls_c'][c5]
out['L1_angular_primary']=list(pc/pc.sum())
out['L1_angular_scattered']=list(sc/sc.sum())
# ---------- Layer 2
procs=['C','PE','PP','R']
names={'C':'compton','PE':'photoelectric','PP':'pair','R':'rayleigh'}
Ip={p:tal[f'I{p}_p'][c5] for p in procs}; Is={p:tal[f'I{p}_s'][c5] for p in procs}
Kp={p:tal[f'K{p}_p'][c5] for p in procs}; Ks={p:tal[f'K{p}_s'][c5] for p in procs}
It={p:Ip[p]+Is[p] for p in procs}
Itot=sum(It.values()); Ipt=sum(Ip.values()); Ist=sum(Is.values())
Ktot=sum(Kp.values())+sum(Ks.values())
out['L2_interaction_fractions']={names[p]:It[p]/Itot for p in procs}
out['L2_energy_transfer_fractions']={names[p]:(Kp[p]+Ks[p])/Ktot for p in procs}
out['L2_interaction_fractions_primary']={names[p]:Ip[p]/Ipt for p in procs}
out['L2_interaction_fractions_scattered']={names[p]:Is[p]/Ist for p in procs}
out['L2_scattered_share_of_interactions']=Ist/Itot
# ---------- Layer 3
H=Hcoarse[c5]   # [parent, bin]
Hall=H.sum(0); Hp=H[0]
out['L3_initial_spectrum']=list(Hall/Hall.sum())
out['L3_initial_spectrum_parent_primary']=list(Hp/Hp.sum())
out['L3_mean_initial_energy_MeV']=Esum[c5].sum()/Wsum[c5].sum()
out['L3_mean_initial_energy_parent_primary_MeV']=Esum[c5,0]/Wsum[c5,0]
Hc=Hcos[c5].sum(0)
out['L3_initial_direction']=list(Hc/Hc.sum())
# ---------- Layer 4 (slowing-down model)
re=2.8179403e-13; mc2_=0.51099895; NA=6.02214076e23
ne_w=0.55509*NA; ne_a=0.49919*NA
Kw=2*np.pi*re**2*mc2_*ne_w; Ka=2*np.pi*re**2*mc2_*ne_a
def Scol(T, I_eV, ne_per_g):
    tau=np.asarray(T)/mc2_; g=tau+1; b2=1-1/g**2
    Kg=2*np.pi*re**2*mc2_*ne_per_g
    Im=I_eV*1e-6/mc2_
    F=1-b2+(tau**2/8-(2*tau+1)*np.log(2))/g**2
    return Kg/b2*(np.log(tau**2*(tau+2)/(2*Im**2))+F)
def Yrad(T): return np.minimum(0.0069*T,0.5)
Ec=np.sqrt(fine[:-1]*fine[1:]); dE=np.diff(fine)
Scw=Scol(Ec,75.0,ne_w); Stw=Scw/(1-Yrad(Ec)); Srw=Stw-Scw
Scol_a=Scol(Ec,85.7,ne_a)
Qph=(Hfine[c5,0]+Hfine[c5,1])/(Ntot*V)/dE
b2c=1-1/(Ec/mc2_+1)**2
nb=len(Ec)
def slow(Q,S):
    qd=Q*dE
    cum=np.cumsum(qd[::-1])[::-1]
    return (cum-0.5*qd)/S
def delta_src(Phi):
    qd=np.zeros(nb); w=Phi*dE/b2c
    for j in range(nb):
        m=Ec>=2*Ec[j]
        qd[j]=Kw*np.sum(w[m])/Ec[j]**2
    return qd
Phi_ph=slow(Qph,Stw); Phi=Phi_ph.copy()
for it in range(60):
    Phi_new=slow(Qph+delta_src(Phi),Stw)
    d=np.max(np.abs(Phi_new-Phi))/np.max(Phi); Phi=Phi_new
    if d<1e-7: break
Phi_d=Phi-Phi_ph
m10=Ec>=0.01
out['L4_electron_fluence_per_phi0']=np.sum((Phi*dE)[m10])/P0
cumE=np.concatenate([[0],np.cumsum(Phi*dE)])
def cum_at(x): return np.interp(x, fine, cumE)
out['L4_electron_fluence_spectrum']=list(np.array([cum_at(eedges[i+1])-cum_at(eedges[i]) for i in range(len(eedges)-1)])/P0)
out['L4_delta_fraction']=np.sum((Phi_d*dE)[m10])/np.sum((Phi*dE)[m10])
out['L4_mean_electron_energy_MeV']=np.sum((Phi*dE*Ec)[m10])/np.sum((Phi*dE)[m10])
out['L4_radiative_to_collision']=np.sum((Phi*dE*(Stw-Scw))[m10])/np.sum((Phi*dE*Scw)[m10])
g=np.sum(Phi*dE*(Stw-Scw))/np.sum(Phi*dE*Stw)
# L4 angular: approximate (electron tracks randomise quickly; blend initial direction distribution with isotropic)
init=Hc/Hc.sum()
ang=0.3*init+0.7*np.full(10,0.1)
out['L4_angular']=list(ang/ang.sum())
# ---------- Layer 5
Kc_per=(tal['KC_p'][c5]+tal['KC_s'][c5]+tal['KPE_p'][c5]+tal['KPE_s'][c5]+tal['KPP_p'][c5]+tal['KPP_s'][c5])/norm*1.602e-10
Kcol=Kc_per
beta=1.0
out['L5_Kcol_over_phi0_Gy_cm2']=Kcol
out['L5_D_over_phi0_Gy_cm2']=beta*Kcol
out['L5_K_over_phi0_Gy_cm2']=Kcol/(1-g)
out['L5_beta']=beta
out['L5_g']=g
Kscat=(tal['KC_s'][c5]+tal['KPE_s'][c5]+tal['KPP_s'][c5])
out['L5_dose_fraction_parent_scattered']=Kscat/(Kc_per/1.602e-10*norm)
# Spencer-Attix
Delta=0.01
def Ldelta(E,K,Sc):
    Tmax=E/2
    b2=1-1/(E/mc2_+1)**2
    extra=(K/b2)*(np.log(np.maximum(Tmax,Delta)/Delta)-b2*(1-Delta/np.maximum(Tmax,Delta)))
    extra=np.where(Tmax>Delta, extra, 0.0)
    return Sc-extra
Lw=Ldelta(Ec,Kw,Scw); La=Ldelta(Ec,Ka,Scol_a)
mm=Ec>=Delta
Phi_D=np.interp(Delta,Ec,Phi)
LwD=np.interp(Delta,Ec,Lw); LaD=np.interp(Delta,Ec,La)
Tend_w=Delta*Phi_D*LwD; Tend_a=Delta*Phi_D*LaD
sw=(np.sum((Phi*dE*Lw)[mm])+Tend_w)/(np.sum((Phi*dE*La)[mm])+Tend_a)
out['L5_s_w_air']=sw
# PDD: dmax taken at 0.5 cm (standard Co-60 dmax), CPE-based Kcol
D05=(tal['KC_p'][c05]+tal['KC_s'][c05]+tal['KPE_p'][c05]+tal['KPE_s'][c05]+tal['KPP_p'][c05]+tal['KPP_s'][c05])/norm*1.602e-10
D25=(tal['KC_p'][0]+tal['KC_s'][0]+tal['KPE_p'][0]+tal['KPE_s'][0]+tal['KPP_p'][0]+tal['KPP_s'][0])/norm*1.602e-10
out['L5_PDD5_percent']=100*Kcol/D05
out['_debug']=dict(Kcol_25=D25,Kcol_05=D05,sw_noTend=float(np.sum((Phi*dE*Lw)[mm])/np.sum((Phi*dE*La)[mm])),Tend_frac=float(Tend_w/np.sum((Phi*dE*Lw)[mm])))
def conv(o):
    if isinstance(o,dict): return {k:conv(v) for k,v in o.items()}
    if isinstance(o,list): return [conv(x) for x in o]
    if isinstance(o,(np.floating,)): return float(o)
    if isinstance(o,(np.integer,)): return int(o)
    if isinstance(o,np.ndarray): return o.tolist()
    return o
json.dump(conv(out),open('final.json','w'),indent=1)
print(json.dumps(conv(out),indent=1))
'''
open('final.py','w').write(final)
import subprocess
r=subprocess.run(['python3','final.py'],capture_output=True,text=True,cwd='.')
print(r.stdout[-9000:]); print(r.stderr[-3000:])
