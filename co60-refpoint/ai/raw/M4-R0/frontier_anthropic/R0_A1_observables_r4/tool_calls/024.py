
asm = r'''
import sys, json; sys.path.insert(0,'/tmp/work')
import pickle, numpy as np, mc as MC
Tc,kerma,flu,ev,Ntot=pickle.load(open('/tmp/work/comb.pkl','rb'))
L3=pickle.load(open('/tmp/work/L3.pkl','rb'))
L45=pickle.load(open('/tmp/work/L45.pkl','rb'))
VOL=MC.VOL; F0=MC.F0; conv=1.602176634e-10; norm=F0/(Ntot*VOL)
cos_edges=np.array([-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1.0])
Lp=Tc['Lp']*norm; Ls=Tc['Ls']*norm
LEp=Tc['LEp']*norm; LEs=Tc['LEs']*norm
spP=Tc['Lp_E']*norm; spS=Tc['Ls_E']*norm
procs=['C','pe','pair','coh']
ip={p:Tc['intp_'+p] for p in procs}; is_={p:Tc['ints_'+p] for p in procs}
itot={p:ip[p]+is_[p] for p in procs}
def frac(d):
    s=sum(d.values()); return {k:float(v/s) for k,v in d.items()}
fa=frac(itot); fp=frac(ip); fs=frac(is_)
trt={'C':Tc['trp_C']+Tc['trs_C'],'pe':Tc['trp_pe']+Tc['trs_pe'],'pair':Tc['trp_pair']+Tc['trs_pair'],'coh':0.0}
ft=frac(trt)
def m4(f): return {'compton':f['C'],'photoelectric':f['pe'],'pair':f['pair'],'rayleigh':f['coh']}
# L4 angular: heuristic forward-tilted distribution p(c) ~ (1 + a c)
T=ev[:,0]; ce=ev[:,1]; w=ev[:,2]
m=(T>=0.01)
cbar=np.sum((w*ce)[m])/np.sum(w[m])
a=3*0.5*cbar    # track-averaged mean cos taken as half the initial mean
c_lo=cos_edges[:-1]; c_hi=cos_edges[1:]
L4ang=0.5*(c_hi-c_lo)+ (a/4)*(c_hi**2-c_lo**2)
L4ang=L4ang/L4ang.sum()
spec4=L45['spec']
out={
 "method":"",
 "data_sources":"",
 "L1_primary_fluence_per_phi0": float(Lp),
 "L1_total_fluence_per_phi0": float(Lp+Ls),
 "L1_scattered_fraction_fluence": float(Ls/(Lp+Ls)),
 "L1_scattered_fraction_energy_fluence": float(LEs/(LEp+LEs)),
 "L1_mean_photon_energy_MeV": float((LEp+LEs)/(Lp+Ls)),
 "L1_photon_fluence_spectrum_primary": [float(x) for x in spP],
 "L1_photon_fluence_spectrum_scattered": [float(x) for x in spS],
 "L1_angular_primary": [float(x) for x in Tc['Lp_c']/Tc['Lp_c'].sum()],
 "L1_angular_scattered": [float(x) for x in Tc['Ls_c']/Tc['Ls_c'].sum()],
 "L2_interaction_fractions": m4(fa),
 "L2_energy_transfer_fractions": m4(ft),
 "L2_interaction_fractions_primary": m4(fp),
 "L2_interaction_fractions_scattered": m4(fs),
 "L2_scattered_share_of_interactions": float(sum(is_.values())/sum(itot.values())),
 "L3_initial_spectrum": L3['L3_initial_spectrum'],
 "L3_initial_spectrum_parent_primary": L3['L3_initial_spectrum_parent_primary'],
 "L3_mean_initial_energy_MeV": L3['L3_mean_initial_energy_MeV'],
 "L3_mean_initial_energy_parent_primary_MeV": L3['L3_mean_initial_energy_parent_primary_MeV'],
 "L3_initial_direction": L3['L3_initial_direction'],
 "L4_electron_fluence_per_phi0": float(L45['Lfl']),
 "L4_electron_fluence_spectrum": [float(x) for x in spec4],
 "L4_delta_fraction": float(L45['delta']),
 "L4_mean_electron_energy_MeV": float(L45['meanE']),
 "L4_angular": [float(x) for x in L4ang],
 "L4_radiative_to_collision": float(L45['rc']),
 "L5_D_over_phi0_Gy_cm2": float(L45['Kcol']),
 "L5_K_over_phi0_Gy_cm2": float(L45['K']),
 "L5_Kcol_over_phi0_Gy_cm2": float(L45['Kcol']),
 "L5_beta": 1.00,
 "L5_g": float(L45['g']),
 "L5_dose_fraction_parent_scattered": float((Tc['trs_C']+Tc['trs_pe']+Tc['trs_pair'])/(Tc['trp_C']+Tc['trs_C']+Tc['trp_pe']+Tc['trs_pe']+Tc['trp_pair']+Tc['trs_pair'])),
 "L5_s_w_air": float(L45['sw']),
 "L5_PDD5_percent": float(L45['pdd5'])
}
print(json.dumps({k:(v if not isinstance(v,list) else [round(x,6) for x in v]) for k,v in out.items() if k not in ('method','data_sources')},indent=1))
pickle.dump(out,open('/tmp/work/final.pkl','wb'))
print('cbar',cbar,'a',a, 'sum L4ang', L4ang.sum(), 'spec4 sum', sum(spec4))
'''
open('/tmp/work/asm.py','w').write(asm)
import subprocess
r=subprocess.run(['python','/tmp/work/asm.py'],capture_output=True,text=True,cwd='/tmp/work'); print(r.stdout[-6000:], r.stderr[-2000:])
