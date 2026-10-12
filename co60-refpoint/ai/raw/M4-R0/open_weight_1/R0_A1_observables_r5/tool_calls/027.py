
import sys, os
sys.path.insert(0,'/tmp/work'); os.chdir('/tmp/work')
s=open('mcbeam.py').read()
s=s.replace("photon_mc.debug = dict(ntrk=0, ntrkp=0, ntrks=0, sumtrk=0.0, sfrac=0, sfracN=0, ssum=0.0)",
"photon_mc.debug = dict(ntrk=0, ntrkp=0, ntrks=0, sumtrk=0.0, sfrac=0, sfracN=0, ssum=0.0, itrkp={})")
s=s.replace("photon_mc.debug['ntrkp']+=int((has_trk&(pra==1)).sum());","photon_mc.debug['ntrkp']+=int((has_trk&(pra==1)).sum());\n            photon_mc.debug['itrkp'][niter]=photon_mc.debug['itrkp'].get(niter,0)+int((has_trk&(pra==1)).sum());")
open('mcbeam.py','w').write(s)
print('ok')