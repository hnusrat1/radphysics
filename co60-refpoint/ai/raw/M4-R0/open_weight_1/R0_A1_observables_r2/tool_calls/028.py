import sys, os
sys.path.insert(0, os.getcwd())
src = open('mc_photon.py').read()

# add incyl_r after incyl_pt line
src = src.replace("        incyl_pt = (p1x*p1x+p1y*p1y <= CYLR*CYLR) & (p1z>=CYLZ0)&(p1z<=CYLZ1) & ~outside\n        N_int_cyl += int(incyl_pt.sum())",
                  "        incyl_pt = (p1x*p1x+p1y*p1y <= CYLR*CYLR) & (p1z>=CYLZ0)&(p1z<=CYLZ1) & ~outside\n        incyl_r = (p1x*p1x+p1y*p1y <= CYLR*CYLR) & ~outside\n        N_int_cyl += int(incyl_pt.sum())")

# fix compton depth
src = src.replace("                zb = np.clip((p1[ic][:,2]/0.2).astype(int),0,149)\n                np.add.at(depth_tr, zb, Tsub)",
                  "                ir = incyl_r[comp_a]\n                if ir.any():\n                    zb = np.clip((p1[comp_a][ir][:,2]/0.2).astype(int),0,149)\n                    np.add.at(depth_tr, zb, T_e[ir])")
# fix photo depth
src = src.replace("                zb = np.clip((p1[ic][:,2]/0.2).astype(int),0,149)\n                np.add.at(depth_tr, zb, Tsub[ic])",
                  "                ir = incyl_r[pho_a]\n                if ir.any():\n                    zb = np.clip((p1[pho_a][ir][:,2]/0.2).astype(int),0,149)\n                    np.add.at(depth_tr, zb, Tsub[ir])")
# fix pair depth
src = src.replace("                zb = np.clip((p1[ic][:,2]/0.2).astype(int),0,149)\n                np.add.at(depth_tr, zb, tr)",
                  "                ir = incyl_r[pri_a]\n                if ir.any():\n                    zb = np.clip((p1[pri_a][ir][:,2]/0.2).astype(int),0,149)\n                    np.add.at(depth_tr, zb, tr[ir])")
open('mc_photon.py','w').write(src)
import importlib, mc_photon
importlib.reload(mc_photon)
res = mc_photon.run_mc(50000, seed=2)
print('time', round(res['t'],2))
print('depth_tr sum', res['depth_tr'].sum(), 'nonzero', (res['depth_tr']>0).sum())
print('E_tr sum proc', sum(res['E_tr_by_proc'].values()))
print('N_int_cyl', res['N_int_cyl'])
import numpy as np
print('first few depth bins', np.round(res['depth_tr'][:60],3))