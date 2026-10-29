
import sys, importlib, time, pickle
sys.path.insert(0,'/tmp/work')
import numpy as np
import phys, electrons, sources
D=pickle.load(open('/tmp/work/photon_tallies.pkl','rb'))
seg=D['seg']; tot=D['tot']
Ntot=4000000
rng=np.random.default_rng(5)
Phi_p=np.exp(-phys.mu_all(np.array([2.0]))[4][0]*10)
cat=sources.build_sources(seg,Ntot,rng,Phi_p)
ele=(cat['q']<0)&(cat['T']>=0.01)
print("electron sources (e-, T>=0.01):",ele.sum(), "of", len(cat['T']))
# uses weights; note electrons below 10 keV not tracked (contribute to deposit only)
t=time.time()
r=electrons.transport(cat['T'][ele],cat['u'][ele],cat['w'][ele],cat['ps'][ele],rng)
print("time",time.time()-t,"iters",r['it'])
pickle.dump(r,open('/tmp/work/L4.pkl','wb'))
print("total track/Phi0",r['tot'],"delta track",r['delt'])
