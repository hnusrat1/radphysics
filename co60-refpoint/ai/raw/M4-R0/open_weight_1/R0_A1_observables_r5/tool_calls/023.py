import sys,os
sys.path.insert(0,'/tmp/work'); os.chdir('/tmp/work')
s=open('mcbeam.py').read()
s=s.replace("if not hasattr(photon_mc,'debug'):","if not hasattr(photon_mc,'debug') or photon_mc.debug is None:")
open('mcbeam.py','w').write(s)
print('ok')