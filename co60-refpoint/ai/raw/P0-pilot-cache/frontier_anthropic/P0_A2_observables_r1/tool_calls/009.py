
s=open('/tmp/work/mcph2.py').read().replace('np.trapz','np.trapezoid')
s=s.replace("""    for k in mc_:""","""    for k in mc_:""")
open('/tmp/work/mcph2.py','w').write(s)
s2=open('/tmp/work/mcph.py').read().replace('np.trapz','np.trapezoid'); open('/tmp/work/mcph.py','w').write(s2)
import importlib
importlib.reload(mcph); importlib.reload(mcph2)
t=time.time(); r=mcph2.run(20000,1); print(time.time()-t)
print(r['flu_E'].sum(axis=1), r['intr'], r['births'].shape, r['Qslab'][:5])
