
dbg = open('/tmp/work/dbg.py').read().replace('np.trapz','np.trapezoid')
open('/tmp/work/dbg.py','w').write(dbg)
o,e=runpy('/tmp/work/dbg.py'); print(o,e)
