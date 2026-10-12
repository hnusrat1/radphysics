
import sys; sys.path.insert(0,'/tmp/work')
import phys, numpy as np
Et=np.array([0.02,0.1,0.3,0.5,1.0,1.1732,1.3325])
for k,v in phys.mu_parts(Et).items(): print(k, np.round(v,5))
print('Om',phys.Om,'F0',phys.F0)
