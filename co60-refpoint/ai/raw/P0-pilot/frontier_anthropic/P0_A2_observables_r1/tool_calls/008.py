
import sys, time, importlib
sys.path.insert(0,'/tmp/work')
import phys, photon, track
import numpy as np
Om=0.009975072683099257; S_phi0=Om*110.0**2; V=np.pi*1.0**2*0.2
mu_p=phys.mu_all(np.array([2.0]))[4][0]; Phi_p=np.exp(-mu_p*10.0)
print("Phi_p/Phi0", Phi_p, "S_phi0", S_phi0)
t=time.time()
res=photon.run_photons(400000, seed=2)
print("time",time.time()-t, len(res['E']))
lin=track.cyl_track(res['r'],res['d'],res['l'],10.0)
sel=lin>0
Fsc=(lin*res['f']).sum()/res['N']*S_phi0/V
print("scattered fluence/Phi0",Fsc, "nseg hits", sel.sum(), "hits scattered", (sel&res['f']).sum())
