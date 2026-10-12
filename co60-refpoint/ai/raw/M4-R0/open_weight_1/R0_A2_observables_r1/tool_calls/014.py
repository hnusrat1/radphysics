import numpy as np, math
A = np.load('/tmp/mc_tallies_5e6.npy', allow_pickle=True)
flu_p,flu_s,inter_a,inter_p,inter_s,trans,e_all,e_prim,e_cos,e_sum,e_sum_prim,slab_src,Ntot = A
print('N', Ntot)
V_cyl=math.pi*0.2; PHI0=1/10.5**2
ph_edges=np.array([0,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,1,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4])
el_edges=np.array([0.01,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,1,1.05,1.1,1.15,1.2,1.4])
cos_edges=np.linspace(-1,1,11)
Emid=(ph_edges[:-1]+ph_edges[1:])/2
Eemid=(el_edges[:-1]+el_edges[1:])/2
# fluence spectra per Phi0 (dimensionless per bin)
fp = flu_p.sum(1)/Ntot/V_cyl/PHI0
fs = flu_s.sum(1)/Ntot/V_cyl/PHI0
fp_tot = fp.sum(); fs_tot=fs.sum(); ftot=fp_tot+fs_tot
print('L1_primary_fluence_per_phi0', fp_tot)
print('L1_total_fluence_per_phi0', ftot)
print('L1_scattered_fraction_fluence', fs_tot/ftot)
Ef_p=(Emid*fp).sum(); Ef_s=(Emid*fs).sum()
print('L1_scattered_fraction_energy_fluence', Ef_s/(Ef_p+Ef_s))
print('L1_mean_photon_energy_MeV', (Emid*(fp+fs)).sum()/ftot)
print('photon spectrum primary (28):'); print(fp)
print('photon spectrum scattered (28):'); print(fs)
# angular
ap = flu_p.sum(0)/flu_p.sum()
as_ = flu_s.sum(0)/flu_s.sum()
print('L1_angular_primary (sum=%.4f):'%ap.sum()); print(ap)
print('L1_angular_scattered (sum=%.4f):'%as_.sum()); print(as_)
# Layer 2
names=['rayleigh','compton','photoelectric','pair']
print('L2_interaction_fractions', dict(zip(names,inter_a/inter_a.sum())))
print('L2_energy_transfer_fractions', dict(zip(names,trans/trans.sum())))
print('L2_interaction_fractions_primary', dict(zip(names,inter_p/inter_p.sum())))
print('L2_interaction_fractions_scattered', dict(zip(names,inter_s/inter_s.sum())))
print('L2_scattered_share_of_interactions', inter_s.sum()/inter_a.sum())
print('inter_a', inter_a, 'inter_p', inter_p, 'inter_s', inter_s, 'trans', trans)
# Layer 3
tot_e = e_all.sum(); tot_ep=e_prim.sum()
print('L3_initial_spectrum (sum=%.4f):'%(e_all/e_all.sum()).sum()); print(e_all/e_all.sum())
print('L3_initial_spectrum_parent_primary (sum=%.4f):'%(e_prim/e_prim.sum()).sum()); print(e_prim/e_prim.sum())
print('L3_mean_initial_energy_MeV', e_sum/tot_e)
print('L3_mean_initial_energy_parent_primary_MeV', e_sum_prim/tot_ep)
print('L3_initial_direction (sum=%.4f):'%(e_cos/e_cos.sum()).sum()); print(e_cos/e_cos.sum())
print('slab_src total', slab_src.sum())