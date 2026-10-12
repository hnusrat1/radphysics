
import sys; sys.path.insert(0,'/tmp/work')
import pickle, numpy as np, mc as MC, phys
Tc,kerma,flu,ev,Ntot=pickle.load(open('/tmp/work/comb.pkl','rb'))
VOL=MC.VOL; F0=MC.F0
norm=F0/(Ntot*VOL)     # converts track-sum per source photon to fluence/Phi0
Eph=np.arange(0,1.4001,0.05)
cos_edges=np.array([-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1.0])
out={}
Lp=Tc['Lp']*norm; Ls=Tc['Ls']*norm
out['L1_primary_fluence_per_phi0']=Lp
out['L1_total_fluence_per_phi0']=Lp+Ls
out['L1_scattered_fraction_fluence']=Ls/(Lp+Ls)
LEp=Tc['LEp']*norm; LEs=Tc['LEs']*norm
out['L1_scattered_fraction_energy_fluence']=LEs/(LEp+LEs)
out['L1_mean_photon_energy_MeV']=(LEp+LEs)/(Lp+Ls)
# spectra
spP=Tc['Lp_E']*norm; spS=Tc['Ls_E']*norm
# primary analytic check
print('primary spectrum nonzero bins:', [(round(Eph[i],2),round(spP[i],4)) for i in range(28) if spP[i]>0])
print('scattered total', spS.sum(), 'primary total',spP.sum())
out['L1_photon_fluence_spectrum_primary']=spP.tolist()
out['L1_photon_fluence_spectrum_scattered']=spS.tolist()
out['L1_angular_primary']=(Tc['Lp_c']/Tc['Lp_c'].sum()).tolist()
out['L1_angular_scattered']=(Tc['Ls_c']/Tc['Ls_c'].sum()).tolist()
# interactions
procs=['C','pe','pair','coh']
ip={p:Tc['intp_'+p] for p in procs}; is_={p:Tc['ints_'+p] for p in procs}
itot={p:ip[p]+is_[p] for p in procs}
def frac(d): 
    s=sum(d.values()); return {k:v/s for k,v in d.items()}
fa=frac(itot); out['L2_interaction_fractions']={'compton':fa['C'],'photoelectric':fa['pe'],'pair':fa['pair'],'rayleigh':fa['coh']}
trp={'C':Tc['trp_C'],'pe':Tc['trp_pe'],'pair':Tc['trp_pair'],'coh':0.0}
trs={'C':Tc['trs_C'],'pe':Tc['trs_pe'],'pair':Tc['trs_pair'],'coh':0.0}
trt={k:trp[k]+trs[k] for k in trp}
ft=frac(trt); out['L2_energy_transfer_fractions']={'compton':ft['C'],'photoelectric':ft['pe'],'pair':ft['pair'],'rayleigh':0.0}
fp=frac(ip); out['L2_interaction_fractions_primary']={'compton':fp['C'],'photoelectric':fp['pe'],'pair':fp['pair'],'rayleigh':fp['coh']}
fs=frac(is_); out['L2_interaction_fractions_scattered']={'compton':fs['C'],'photoelectric':fs['pe'],'pair':fs['pair'],'rayleigh':fs['coh']}
out['L2_scattered_share_of_interactions']=sum(is_.values())/sum(itot.values())
for k in ['L1_primary_fluence_per_phi0','L1_total_fluence_per_phi0','L1_scattered_fraction_fluence','L1_scattered_fraction_energy_fluence','L1_mean_photon_energy_MeV']:
    print(k,out[k])
print(out['L2_interaction_fractions'], out['L2_energy_transfer_fractions'])
print(out['L2_interaction_fractions_primary'], out['L2_interaction_fractions_scattered'], out['L2_scattered_share_of_interactions'])
print('primary ang',np.round(out['L1_angular_primary'],4))
print('scat ang',np.round(out['L1_angular_scattered'],4))
