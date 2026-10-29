
import sys, time, pickle; sys.path.insert(0,'/tmp/work')
import numpy as np, mcph, mcph2
t=time.time(); r=mcph2.run(1000000,11); print(time.time()-t)
pickle.dump(r,open('/tmp/work/run1.pkl','wb'))
print(r['flu_E'].sum(axis=1)/1e6, r['intr'], r['births'].shape)
