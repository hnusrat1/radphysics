
s=open('/tmp/ana.py').read().replace('np.trapz','np.trapezoid')
open('/tmp/ana.py','w').write(s)
exec(s)
