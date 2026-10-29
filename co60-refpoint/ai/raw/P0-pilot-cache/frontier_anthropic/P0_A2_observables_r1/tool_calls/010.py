
import sys, importlib, time
sys.path.insert(0,'/tmp/work')
import mcph, mcph2
t=time.time(); r=mcph2.run(20000,1); print(time.time()-t)
print(r['flu_E'].sum(axis=1), r['intr'], r['births'].shape, r['Qslab'][:5])
