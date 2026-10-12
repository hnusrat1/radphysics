import sys, time, numpy as np, pickle
sys.path.insert(0,'/tmp')
import photonmc, importlib
importlib.reload(photonmc)

t0=time.time()
res = photonmc.photon_mc(6000000, seed=20240517, verbose=True, debug=False)
print('elapsed', time.time()-t0)
# save
with open('/tmp/photon_results.pkl','wb') as f:
    pickle.dump({k:v for k,v in res.items() if k!='src'}, f)
np.savez_compressed('/tmp/photon_sources.npz', pos=res['src']['pos'], dir=res['src']['dir'],
                    T=res['src']['T'], par=res['src']['par'], kind=res['src']['kind'])
print('saved. src count', len(res['src']['T']), 'L3', len(res['L3_Te']))
print('fl_prim sum', res['fl_prim'].sum(), 'fl_scat sum', res['fl_scat'].sum())
print('int_all', res['int_all'], 'int_prim', res['int_prim'], 'int_scat', res['int_scat'])
print('etr_all', res['etr_all'])