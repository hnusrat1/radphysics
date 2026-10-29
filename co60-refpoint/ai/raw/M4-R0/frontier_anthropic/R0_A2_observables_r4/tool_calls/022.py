
src=open('/tmp/work/ana2.py').read()
src=src.replace("""    Mmat[j,:]=np.where(sel, ne*np.array([moller_dsig(T,Tp) if T>=2*Tp else 0.0 for T in Tg]),0.0)""",
"""    Mmat[j,:]=np.where(sel, ne*moller_dsig(Tg,Tp),0.0)""")
open('/tmp/work/ana2.py','w').write(src)
import sys,time; sys.path.insert(0,'/tmp/work')
t=time.time()
import ana2
print('time',time.time()-t)
