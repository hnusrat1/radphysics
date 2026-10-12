import importlib, sys, time, numpy as np, pickle
sys.path.insert(0,'/tmp')
import photonmc
importlib.reload(photonmc)
pd_depths=[0.2,0.35,0.5,0.65,0.8,1.0,1.5,2,3,4,5,6,8,10]
t0=time.time()
res = photonmc.photon_mc(6000000, seed=20240517, verbose=True, debug=False, pd_depths=pd_depths)
print('elapsed', time.time()-t0)
with open('/tmp/photon_results.pkl','wb') as f:
    pickle.dump({k:v for k,v in res.items() if k!='src'}, f)
np.savez_compressed('/tmp/photon_sources.npz', pos=res['src']['pos'], dir=res['src']['dir'],
                    T=res['src']['T'], par=res['src']['par'], kind=res['src']['kind'])
print('saved. src', len(res['src']['T']), 'L3', len(res['L3_Te']))
print('fl_prim', res['fl_prim'].sum(), 'fl_scat', res['fl_scat'].sum())
print('int_all', res['int_all'], 'int_prim', res['int_prim'], 'int_scat', res['int_scat'])
print('etr_all', res['etr_all'])
print('Kdep', res['Kdep'])