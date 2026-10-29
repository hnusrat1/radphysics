import importlib, sys, time
sys.path.insert(0,'/tmp')
import photonmc_dbg
importlib.reload(photonmc_dbg)
photonmc_dbg.PAIRDBG=True
t0=time.time()
res = photonmc_dbg.photon_mc(3000000, seed=99, verbose=False, debug=False)
print('elapsed', time.time()-t0)
print('int_all', res['int_all'])
print('etr_all', res['etr_all'])