import sys, time, numpy as np
sys.path.insert(0,'/tmp')
import photonmc, importlib
importlib.reload(photonmc)
res = photonmc.photon_mc(200000, seed=123, verbose=True, debug=False)
print('ang_prim', res['ang_prim']/res['ang_prim'].sum())
print('scat frac flu', res['fl_scat'].sum()/(res['fl_prim'].sum()+res['fl_scat'].sum()))
print('int_all', res['int_all'], 'int_prim', res['int_prim'], 'int_scat', res['int_scat'])
print('etr_all', res['etr_all'])
# mean energies
E_edges=np.linspace(0,1.4,29); Ec=(E_edges[:-1]+E_edges[1:])/2
flt=res['fl_prim']+res['fl_scat']
meanE=(Ec*flt).sum()/flt.sum()
print('mean photon E', meanE)
print('src count', len(res['src']['T']))
print('L3 mean Te', res['L3_Te'].mean() if len(res['L3_Te']) else None, 'n', len(res['L3_Te']))