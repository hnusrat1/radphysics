import sys, os
sys.path.insert(0, os.getcwd())
import numpy as np, importlib, mc_photon
importlib.reload(mc_photon)
res = mc_photon.run_mc(50000, seed=2)
print('time', round(res['t'],2))
print('depth_tr sum (total E_tr in axis cyl)', res['depth_tr'].sum(), 'nonzero bins', (res['depth_tr']>0).sum())
print('E_tr total from proc', sum(res['E_tr_by_proc'].values()))
print('prim/sct flu', res['flu_pri_ph'].sum(), res['flu_sct_ph'].sum())
print('N_int_cyl', res['N_int_cyl'])