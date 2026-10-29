import sys, os
sys.path.insert(0, os.getcwd())
import numpy as np
import importlib, mc_photon
importlib.reload(mc_photon)
res = mc_photon.run_mc(50000, seed=11)
print('time', round(res['t'],1))
print('N', res['N'])
print('flu pri total (cm)', res['flu_pri_ph'].sum(), 'flu sct total', res['flu_sct_ph'].sum())
# per phi0
print('primary flu/phi0', res['flu_pri_ph'].sum()/mc_photon.Vcyl/res['N']*mc_photon.Phi0_area)
print('scattered flu/phi0', res['flu_sct_ph'].sum()/mc_photon.Vcyl/res['N']*mc_photon.Phi0_area)
print('interactions', res['n_int'], 'pri', res['n_int_pri'], 'sct', res['n_int_sct'])
print('N_int_cyl', res['N_int_cyl'], 'scat share', res['n_scat_inter']/max(res['N_int_cyl'],1))
print('E_tr total', round(sum(res['E_tr_by_proc'].values()),4), 'scat E_tr', round(res['E_tr_scat'],4))
print('N_el_e', res['N_el_e'], 'N_el_pos', res['N_el_pos'])
print('time/10k', round(res['t']/5,3))