import sys, os
sys.path.insert(0, os.getcwd())
import mc_photon as mcp
res = mcp.run_mc(50000, seed=11)
print('time', round(res['t'],1))
print('N', res['N'])
print('flu pri total (cm)', res['flu_pri_ph'].sum(), 'flu sct total', res['flu_sct_ph'].sum())
print('interactions', res['n_int'], 'pri', res['n_int_pri'], 'sct', res['n_int_sct'])
print('N_int_cyl', res['N_int_cyl'], 'scat share', res['n_scat_inter']/max(res['N_int_cyl'],1))
print('E_tr total', sum(res['E_tr_by_proc'].values()), 'scat E_tr', res['E_tr_scat'])
print('N_el_e', res['N_el_e'], 'N_el_pos', res['N_el_pos'])
print('time/10k', round(res['t']/5,2))