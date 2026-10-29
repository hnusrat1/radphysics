import importlib, sys, time
sys.path.insert(0,'/tmp')
import photonmc
importlib.reload(photonmc)
t0=time.time()
res = photonmc.photon_mc(50000, seed=42, verbose=False, debug=True)
print('time', time.time()-t0)
print('primL', res['fl_prim'].sum(), 'scatL', res['fl_scat'].sum())
print('n_cyl_seg prim/scat', res['n_cyl_seg_prim'], res['n_cyl_seg_scat'])