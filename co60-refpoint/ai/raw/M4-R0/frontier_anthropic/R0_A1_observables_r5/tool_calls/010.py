
import numpy as np, sys, time
sys.path.insert(0,'/tmp/work')
import sim, mc, phys
V=np.pi*1.0**2*0.2
N=2000000
t=time.time()
seg,dep=sim.run(N,7)
print("time",time.time()-t)
prim=seg['prim']
ratio=lambda tl: tl.sum()/V/N * mc.Omega*105**2
print("primary ratio MC", ratio(seg['tl'][prim]))
mu1=mc.mu_c(1.1732)+mc.mu_pe(1.1732)+mc.mu_coh(1.1732)
mu2=mc.mu_c(1.3325)+mc.mu_pe(1.3325)+mc.mu_coh(1.3325)
print("analytic", 0.5*(np.exp(-mu1*5)+np.exp(-mu2*5)), mu1, mu2)
print("total ratio", ratio(seg['tl']), "scattered", ratio(seg['tl'][~prim]))
