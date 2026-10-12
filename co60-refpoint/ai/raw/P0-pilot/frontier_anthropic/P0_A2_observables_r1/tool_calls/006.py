
import sys, time, importlib
sys.path.insert(0,'/tmp/work')
import phys, photon
t=time.time()
res=photon.run_photons(100000, seed=1)
print(time.time()-t, len(res['E']), res['nproc'], res['iters'], res['f'].mean())
