
import numpy as np, pickle
out=pickle.load(open('/tmp/work/out.pkl','rb'))
print(np.array(out['L3_initial_spectrum'])[-5:], np.array(out['L3_initial_spectrum_parent_primary'])[-5:])
exec(open('/tmp/work/physics.py').read())
print(Tmax:=2*1.3325**2/(0.51099895+2*1.3325), 2*1.1732**2/(0.51099895+2*1.1732))
