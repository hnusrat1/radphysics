
import sys; sys.path.insert(0,'/tmp/work')
import numpy as np, elec, mell
# single 1 MeV electron test, no deltas (monkeypatch sig_delta to 0)
elec.sig_delta=lambda T: np.zeros_like(np.asarray(T,float))
B=np.array([[1.0,1.0,0,1.0,0]])
r=elec.run_e(B,1.0,1000,3)
print(r['dep_tot']/1000, r['T0s']/1000, r['fl_tot'], r['steps'])
# expected track length (CSDA range)
import xs; print(np.interp(1.0,xs.ET,xs.RANGE))
