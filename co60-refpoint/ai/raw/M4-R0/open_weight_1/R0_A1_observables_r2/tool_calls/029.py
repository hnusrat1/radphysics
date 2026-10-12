import sys, os
sys.path.insert(0, os.getcwd())
src = open('mc_photon.py').read()
src = src.replace("np.add.at(depth_tr, zb, tr[ir])", "np.add.at(depth_tr, zb, (Epair-1.022)[ir])")
open('mc_photon.py','w').write(src)
import importlib, mc_photon
importlib.reload(mc_photon)
import numpy as np
res = mc_photon.run_mc(50000, seed=2)
print('time', round(res['t'],2))
print('depth_tr sum', res['depth_tr'].sum(), 'nonzero', (res['depth_tr']>0).sum())
print('E_tr sum proc', sum(res['E_tr_by_proc'].values()))
print('N_int_cyl', res['N_int_cyl'])
print('first bins', np.round(res['depth_tr'][:60],3))