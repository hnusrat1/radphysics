
import sys; sys.path.insert(0,'/tmp/work')
import numpy as np, elec, mell
B=np.array([[1.0,1.0,0,1.0,0]])
r=elec.run_e(B,1.0,2000,3)
print(r['dep_tot']/2000, r['fl_tot'], r['fl_del']/r['fl_tot'], r['steps'])
# check sampling of delta energies: mean eps at T=1
e=mell.sample_delta_eps(np.full(100000,1.0),np.random.uniform(size=100000)); print(e.mean(), e.max())
print(mell.sig_delta(1.0))
